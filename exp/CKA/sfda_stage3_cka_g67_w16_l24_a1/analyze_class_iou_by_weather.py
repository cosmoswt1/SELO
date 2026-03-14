#!/usr/bin/env python3
"""Analyze class-wise IoU deltas by weather for a Stage3 CKA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from datasets import ACDCEvalDataset  # noqa: E402
from datasets.acdc import get_acdc_class_names  # noqa: E402
from eval_stage3_cka import ACDCTestWithGTDataset, CONDITIONS, fast_hist  # noqa: E402
from model_cka_v1 import Stage3CKAModel  # noqa: E402


STRUCTURE_CLASSES = {
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "vegetation",
    "terrain",
    "sky",
}
SMALL_OBJECT_CLASSES = {
    "pole",
    "traffic light",
    "traffic sign",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}


def check_gpu_or_exit() -> None:
    try:
        smi = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True)
        first = smi.stdout.splitlines()[0] if smi.stdout else "nvidia-smi OK"
        print(f"[GPU 체크] {first}")
    except Exception as exc:
        print("[GPU 체크 실패] nvidia-smi 실행 실패", file=sys.stderr)
        print(f"에러: {exc}", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print('  python -c "import torch; print(torch.cuda.is_available())"', file=sys.stderr)
        print("해결: GPU 환경에서 다시 실행하세요.", file=sys.stderr)
        raise SystemExit(1)

    if not torch.cuda.is_available():
        print("[GPU 체크 실패] torch.cuda.is_available() == False", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print('  python -c "import torch; print(torch.cuda.is_available())"', file=sys.stderr)
        print("해결: conda selo 환경의 CUDA torch 설치/가용성을 확인하세요.", file=sys.stderr)
        raise SystemExit(1)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("class_iou_by_weather")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze class-wise IoU deltas by weather.")
    p.add_argument("--acdc_root", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--test_gt_dir", type=str, default="")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--resize", type=int, default=1080)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--segformer_model", type=str, default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--dino_model", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--dino_layer", type=int, default=24)
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--adapter_bottleneck", type=int, default=0)
    p.add_argument("--gate_bias_init", type=float, default=-4.0)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--focus_conditions", nargs="+", default=["rain", "snow"])
    p.add_argument("--topk", type=int, default=5)
    return p.parse_args()


def load_model(args: argparse.Namespace, logger: logging.Logger) -> Stage3CKAModel:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    bottleneck = int(args.adapter_bottleneck) if int(args.adapter_bottleneck) > 0 else int(
        ckpt_args.get("adapter_bottleneck", 128)
    )
    gate_bias_init = float(ckpt_args.get("gate_bias_init", args.gate_bias_init))
    dino_layer = int(ckpt_args.get("dino_layer", args.dino_layer))

    device = torch.device("cuda")
    model = Stage3CKAModel(
        segformer_model=args.segformer_model,
        dino_model=args.dino_model,
        dino_layer=dino_layer,
        num_classes=int(args.num_classes),
        adapter_bottleneck=bottleneck,
        gate_bias_init=gate_bias_init,
        enable_dino=False,
    ).to(device)
    adapter_sd = ckpt["stage3_adapter"]
    missing_adapter, unexpected_adapter = model.stage3_adapter.load_state_dict(adapter_sd, strict=False)
    if missing_adapter or unexpected_adapter:
        logger.warning(
            "stage3_adapter load_state_dict(strict=False): missing=%s unexpected=%s",
            missing_adapter,
            unexpected_adapter,
        )
    model.stage3_gate.load_state_dict(ckpt["stage3_gate"], strict=True)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.ckpt)
    return model


def build_dataset(args: argparse.Namespace) -> Dataset:
    if args.split == "test" and args.test_gt_dir:
        return ACDCTestWithGTDataset(
            acdc_root=args.acdc_root,
            test_gt_dir=args.test_gt_dir,
            conditions=list(args.conditions),
            resize=int(args.resize),
        )
    return ACDCEvalDataset(
        root=args.acdc_root,
        split=args.split,
        conditions=list(args.conditions),
        resize=int(args.resize),
    )


def iou_from_hist(hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tp = torch.diag(hist)
    gt = hist.sum(dim=1)
    pred = hist.sum(dim=0)
    denom = gt + pred - tp
    iou = torch.full_like(tp, float("nan"), dtype=torch.float32)
    valid = denom > 0
    iou[valid] = tp[valid] / denom[valid] * 100.0
    return iou, gt, pred


def class_group(class_name: str) -> str:
    if class_name in STRUCTURE_CLASSES:
        return "large_structure"
    if class_name in SMALL_OBJECT_CLASSES:
        return "small_object"
    return "other"


def write_results_csv(out_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "condition",
        "num_images",
        "class_id",
        "class_name",
        "class_group",
        "iou_base",
        "iou_adapt",
        "delta_abs",
        "gt_pixels",
        "pred_pixels_base",
        "pred_pixels_adapt",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_group_summary_csv(out_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "condition",
        "num_images",
        "class_group",
        "num_valid_classes",
        "mean_delta_abs",
        "median_delta_abs",
        "max_delta_abs",
        "min_delta_abs",
        "top_positive_classes",
        "top_negative_classes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: float) -> str:
    return "NaN" if (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def _fmt3(v: float) -> str:
    return "NaN" if (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def summarize_condition(rows: list[dict], condition: str, focus_topk: int) -> tuple[dict, str]:
    cond_rows = [r for r in rows if r["condition"] == condition]
    group_rows: list[dict] = []
    lines: list[str] = []
    valid_rows = [r for r in cond_rows if not math.isnan(float(r["delta_abs"]))]

    for group_name in ["large_structure", "small_object"]:
        grow = [r for r in valid_rows if r["class_group"] == group_name]
        deltas = [float(r["delta_abs"]) for r in grow]
        if deltas:
            sorted_pos = sorted(grow, key=lambda x: float(x["delta_abs"]), reverse=True)[:focus_topk]
            sorted_neg = sorted(grow, key=lambda x: float(x["delta_abs"]))[:focus_topk]
            entry = {
                "condition": condition,
                "num_images": cond_rows[0]["num_images"] if cond_rows else 0,
                "class_group": group_name,
                "num_valid_classes": len(deltas),
                "mean_delta_abs": float(sum(deltas) / len(deltas)),
                "median_delta_abs": float(torch.tensor(deltas).median().item()),
                "max_delta_abs": float(max(deltas)),
                "min_delta_abs": float(min(deltas)),
                "top_positive_classes": ", ".join(f"{r['class_name']}({_fmt3(float(r['delta_abs']))})" for r in sorted_pos),
                "top_negative_classes": ", ".join(f"{r['class_name']}({_fmt3(float(r['delta_abs']))})" for r in sorted_neg),
            }
        else:
            entry = {
                "condition": condition,
                "num_images": cond_rows[0]["num_images"] if cond_rows else 0,
                "class_group": group_name,
                "num_valid_classes": 0,
                "mean_delta_abs": float("nan"),
                "median_delta_abs": float("nan"),
                "max_delta_abs": float("nan"),
                "min_delta_abs": float("nan"),
                "top_positive_classes": "",
                "top_negative_classes": "",
            }
        group_rows.append(entry)

    big = next(x for x in group_rows if x["class_group"] == "large_structure")
    small = next(x for x in group_rows if x["class_group"] == "small_object")
    focus_sorted = sorted(valid_rows, key=lambda x: float(x["delta_abs"]), reverse=True)
    focus_sorted_neg = sorted(valid_rows, key=lambda x: float(x["delta_abs"]))
    overall_top_pos = ", ".join(f"{r['class_name']}({_fmt3(float(r['delta_abs']))})" for r in focus_sorted[:focus_topk]) or "없음"
    overall_top_neg = ", ".join(f"{r['class_name']}({_fmt3(float(r['delta_abs']))})" for r in focus_sorted_neg[:focus_topk]) or "없음"
    lines.append(f"### {condition}")
    lines.append("")
    lines.append("| group | mean_delta_abs | median_delta_abs | max_delta_abs | min_delta_abs |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| large_structure | {_fmt3(float(big['mean_delta_abs']))} | {_fmt3(float(big['median_delta_abs']))} | {_fmt3(float(big['max_delta_abs']))} | {_fmt3(float(big['min_delta_abs']))} |"
    )
    lines.append(
        f"| small_object | {_fmt3(float(small['mean_delta_abs']))} | {_fmt3(float(small['median_delta_abs']))} | {_fmt3(float(small['max_delta_abs']))} | {_fmt3(float(small['min_delta_abs']))} |"
    )
    lines.append("")
    lines.append(
        f"- large_structure positive: {big['top_positive_classes'] or '없음'}"
    )
    lines.append(
        f"- large_structure negative: {big['top_negative_classes'] or '없음'}"
    )
    lines.append(
        f"- small_object positive: {small['top_positive_classes'] or '없음'}"
    )
    lines.append(
        f"- small_object negative: {small['top_negative_classes'] or '없음'}"
    )
    lines.append(
        f"- overall top positive classes: {overall_top_pos}"
    )
    lines.append(
        f"- overall top negative classes: {overall_top_neg}"
    )
    if not math.isnan(float(big["mean_delta_abs"])) and not math.isnan(float(small["mean_delta_abs"])):
        if float(big["mean_delta_abs"]) > float(small["mean_delta_abs"]) + 0.01:
            judgement = "texture stabilization 쪽 신호가 더 강함"
        elif float(small["mean_delta_abs"]) > float(big["mean_delta_abs"]) + 0.01:
            judgement = "semantic recovery 쪽 신호가 더 강함"
        else:
            judgement = "texture stabilization과 semantic recovery가 비슷한 크기로 함께 나타남"
        lines.append(
            f"- 해석: large_structure 평균 delta={_fmt3(float(big['mean_delta_abs']))}, "
            f"small_object 평균 delta={_fmt3(float(small['mean_delta_abs']))} 이므로 {judgement}."
        )
    lines.append("")
    return {"large_structure": big, "small_object": small}, "\n".join(lines)


def write_summary_md(
    out_path: Path,
    *,
    args: argparse.Namespace,
    group_rows: list[dict],
    focus_sections: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Class-wise IoU By Weather")
    lines.append("")
    lines.append("## Preprocessing / Label Mapping / Resize-Crop Rule")
    lines.append("")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Resize rule: shorter side -> `{args.resize}`, then eval path에서 32배수 pad")
    lines.append("- Crop rule: 없음 (evaluation에서는 random crop 미사용)")
    lines.append("- Label mapping: Cityscapes trainIds (`0..18`), ignore=`255`")
    lines.append("- Comparison: baseline(segformer base) vs adapted(Stage3 CKA adapter)")
    lines.append(
        "- Group split: `large_structure` = road/sidewalk/building/wall/fence/vegetation/terrain/sky, "
        "`small_object` = pole/traffic light/traffic sign/person/rider/car/truck/bus/train/motorcycle/bicycle"
    )
    lines.append("")
    lines.append("## Weather Overview")
    lines.append("")
    lines.append("| condition | class_group | num_valid_classes | mean_delta_abs | median_delta_abs | max_delta_abs | min_delta_abs |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in group_rows:
        lines.append(
            f"| {row['condition']} | {row['class_group']} | {row['num_valid_classes']} | "
            f"{_fmt3(float(row['mean_delta_abs']))} | {_fmt3(float(row['median_delta_abs']))} | "
            f"{_fmt3(float(row['max_delta_abs']))} | {_fmt3(float(row['min_delta_abs']))} |"
        )
    lines.append("")
    lines.append("## Rain / Snow Focus")
    lines.append("")
    lines.extend(focus_sections)
    lines.append("## Next Action")
    lines.append("")
    lines.append("- `rain/snow`에서 delta가 가장 큰 클래스만 골라, boundary 오류인지 texture 오류인지 qualitative panel 1회 추가 점검.")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_gpu_or_exit()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "eval.log")
    logger.info("Args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    model = load_model(args, logger)
    dataset = build_dataset(args)
    if len(dataset) == 0:
        logger.error("No images with GT found for split=%s", args.split)
        return 1

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True,
    )

    class_names = get_acdc_class_names()
    n_class = int(args.num_classes)
    device = torch.device("cuda")
    conditions = list(args.conditions)
    cond_hist_base = {c: torch.zeros(n_class, n_class, device=device) for c in conditions}
    cond_hist_adapt = {c: torch.zeros(n_class, n_class, device=device) for c in conditions}
    cond_n = {c: 0 for c in conditions}

    for batch in tqdm(loader, desc="ClassIoUByWeather"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.no_grad():
            out_base = model(
                images,
                adapter_enabled=False,
                use_dino=False,
                compute_logits=True,
                need_stage4_anchor=False,
            )
            out_adapt = model(
                images,
                adapter_enabled=True,
                use_dino=False,
                compute_logits=True,
                need_stage4_anchor=False,
            )
            logits_base = out_base["logits"]
            logits_adapt = out_adapt["logits"]

        logits_base = F.interpolate(logits_base, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        logits_adapt = F.interpolate(logits_adapt, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds_base = logits_base.argmax(dim=1)
        preds_adapt = logits_adapt.argmax(dim=1)

        for i in range(images.shape[0]):
            cond = str(batch["condition"][i])
            label = labels[i]
            mask = label != 255
            hist_base = fast_hist(preds_base[i][mask], label[mask], n_class)
            hist_adapt = fast_hist(preds_adapt[i][mask], label[mask], n_class)
            if cond in cond_hist_base:
                cond_hist_base[cond] += hist_base
                cond_hist_adapt[cond] += hist_adapt
                cond_n[cond] += 1

    results_rows: list[dict] = []
    for cond in conditions:
        hist_base = cond_hist_base[cond]
        hist_adapt = cond_hist_adapt[cond]
        iou_base, gt_base, pred_base = iou_from_hist(hist_base)
        iou_adapt, _gt_adapt, pred_adapt = iou_from_hist(hist_adapt)
        for class_id, class_name in enumerate(class_names):
            base_val = float(iou_base[class_id].item())
            adapt_val = float(iou_adapt[class_id].item())
            delta = float(adapt_val - base_val) if not (math.isnan(base_val) or math.isnan(adapt_val)) else float("nan")
            results_rows.append(
                {
                    "condition": cond,
                    "num_images": int(cond_n[cond]),
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "class_group": class_group(class_name),
                    "iou_base": base_val,
                    "iou_adapt": adapt_val,
                    "delta_abs": delta,
                    "gt_pixels": int(gt_base[class_id].item()),
                    "pred_pixels_base": int(pred_base[class_id].item()),
                    "pred_pixels_adapt": int(pred_adapt[class_id].item()),
                }
            )

    write_results_csv(out_dir / "results.csv", results_rows)

    group_rows: list[dict] = []
    focus_sections: list[str] = []
    focus_sections_map: dict[str, str] = {}
    for cond in conditions:
        group_map, section = summarize_condition(results_rows, cond, int(args.topk))
        focus_sections_map[cond] = section
        group_rows.append(group_map["large_structure"])
        group_rows.append(group_map["small_object"])

    write_group_summary_csv(out_dir / "group_summary.csv", group_rows)
    write_summary_md(
        out_dir / "summary.md",
        args=args,
        group_rows=group_rows,
        focus_sections=[focus_sections_map[fc] for fc in args.focus_conditions if fc in focus_sections_map],
    )

    metrics = {
        "conditions": conditions,
        "focus_conditions": list(args.focus_conditions),
        "class_names": class_names,
        "num_images_by_condition": {c: int(cond_n[c]) for c in conditions},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved: summary.md, results.csv, group_summary.csv, eval.log, metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
