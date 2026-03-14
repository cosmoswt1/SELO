#!/usr/bin/env python3
"""Evaluate stage3 CKA adapter and export summary/results/log artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from datasets import ACDCEvalDataset  # noqa: E402
from datasets.acdc import get_acdc_class_colors  # noqa: E402
from model_cka_v1 import Stage3CKAModel  # noqa: E402


CONDITIONS = ["fog", "night", "rain", "snow"]


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
    logger = logging.getLogger("stage3_cka_eval")
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
    p = argparse.ArgumentParser(description="Evaluate stage3 CKA adapter")
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
    return p.parse_args()


class ACDCTestWithGTDataset(Dataset):
    """Test split dataset using external GT root with trainId labels."""

    CONDITIONS = CONDITIONS

    def __init__(
        self,
        *,
        acdc_root: str,
        test_gt_dir: str,
        conditions: list[str],
        resize: int,
    ):
        super().__init__()
        self.conditions = conditions
        self.resize = int(resize)
        self.acdc_root = Path(acdc_root)
        self.test_gt_dir = Path(test_gt_dir)
        self.samples: list[tuple[Path, Path, str]] = []

        rgb_base = self._find_rgb_base(self.acdc_root)
        for cond in self.conditions:
            rgb_dir = rgb_base / cond / "test"
            if not rgb_dir.exists():
                continue
            for img_path in sorted(rgb_dir.rglob("*_rgb_anon.png")):
                rel = img_path.relative_to(rgb_dir)
                gt_name = img_path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
                gt_path = self.test_gt_dir / cond / "test" / rel.parent / gt_name
                if gt_path.exists():
                    self.samples.append((img_path, gt_path, cond))

    @staticmethod
    def _find_rgb_base(acdc_root: Path) -> Path:
        candidates = [
            acdc_root / "rgb_anon_trainvaltest" / "rgb_anon",
            acdc_root / "rgb_anon",
        ]
        for c in candidates:
            if c.exists():
                return c
        for p in acdc_root.rglob("rgb_anon"):
            if p.is_dir():
                return p
        raise FileNotFoundError(f"rgb_anon base not found under: {acdc_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, gt_path, cond = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        label = Image.open(gt_path)

        w, h = image.size
        if h < w:
            new_h = self.resize
            new_w = int(round(w * self.resize / h))
        else:
            new_w = self.resize
            new_h = int(round(h * self.resize / w))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        label = label.resize((new_w, new_h), Image.NEAREST)

        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], padding_mode="reflect")
            label = TF.pad(label, [0, 0, pad_w, pad_h], fill=255)

        x = TF.to_tensor(image)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        y = torch.from_numpy(np.array(label)).long()
        return {
            "image": x,
            "label": y,
            "condition": cond,
            "path": str(img_path),
        }


def fast_hist(pred: torch.Tensor, label: torch.Tensor, n_class: int) -> torch.Tensor:
    mask = (label >= 0) & (label < n_class)
    return torch.bincount(
        n_class * label[mask].int() + pred[mask].int(),
        minlength=n_class * n_class,
    ).reshape(n_class, n_class).float()


def miou_from_hist(hist: torch.Tensor) -> float:
    iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + 1e-6)
    valid = hist.sum(dim=1) > 0
    if not valid.any():
        return float("nan")
    return float(iou[valid].mean().item() * 100.0)


def _denorm(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1)
    return (y.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _colorize(mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(colors))
    out[valid] = colors[mask[valid]]
    return out


def _fmt(v: float) -> str:
    return "NaN" if (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def _delta_rel_pct(base: float, adapt: float) -> float:
    if math.isnan(base) or math.isnan(adapt):
        return float("nan")
    return float((adapt - base) / max(abs(base), 1e-6) * 100.0)


def write_results_csv(
    out_csv: Path,
    split: str,
    overall_n: int,
    overall_base: float,
    overall_adapt: float,
    cond_n: dict,
    cond_base: dict,
    cond_adapt: dict,
) -> None:
    del split
    lines = ["group_type,group_name,num_images,miou_base,miou_adapt,delta_abs,delta_rel_pct"]
    overall_delta = float(overall_adapt - overall_base) if not (math.isnan(overall_base) or math.isnan(overall_adapt)) else float("nan")
    lines.append(
        f"overall,all,{overall_n},{_fmt(overall_base)},{_fmt(overall_adapt)},{_fmt(overall_delta)},{_fmt(_delta_rel_pct(overall_base, overall_adapt))}"
    )
    for cond in CONDITIONS:
        n = int(cond_n.get(cond, 0))
        mb = float(cond_base.get(cond, float("nan")))
        ma = float(cond_adapt.get(cond, float("nan")))
        d = float(ma - mb) if not (math.isnan(mb) or math.isnan(ma)) else float("nan")
        dr = _delta_rel_pct(mb, ma)
        lines.append(f"condition,{cond},{n},{_fmt(mb)},{_fmt(ma)},{_fmt(d)},{_fmt(dr)}")
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(
    out_md: Path,
    *,
    args: argparse.Namespace,
    overall_n: int,
    overall_base: float,
    overall_adapt: float,
    cond_n: dict,
    cond_base: dict,
    cond_adapt: dict,
    failure_rows: list[dict],
    no_gt_test: bool,
) -> None:
    lines: list[str] = []
    lines.append("# Stage3 CKA Evaluation Summary")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Conditions: `{', '.join(args.conditions)}`")
    lines.append(f"- Num images with GT: **{overall_n}**")
    lines.append("")
    lines.append("## Preprocessing / Label Mapping / Resize-Crop Rule")
    lines.append(f"- Resize short side: `{args.resize}`")
    lines.append("- Resize rule: keep aspect ratio")
    lines.append("- Pad rule: pad to multiple of 32 (`reflect` for image, `255` for ignore label)")
    lines.append("- Crop rule: 없음 (evaluation에서는 random crop 미사용)")
    lines.append("- Label mapping: Cityscapes trainIds (`0..18`), ignore=`255`")
    if args.split == "test":
        if args.test_gt_dir:
            lines.append(f"- test GT source: `{args.test_gt_dir}`")
        else:
            lines.append("- test GT source: ACDC default path (없으면 No GT로 처리)")
    lines.append("")
    lines.append("## mIoU (%)")
    lines.append("| group | num_images | baseline | adapted | delta(abs) | delta(rel, %) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    overall_delta = float(overall_adapt - overall_base) if not (math.isnan(overall_base) or math.isnan(overall_adapt)) else float("nan")
    lines.append(
        f"| overall | {overall_n} | {_fmt(overall_base)} | {_fmt(overall_adapt)} | {_fmt(overall_delta)} | {_fmt(_delta_rel_pct(overall_base, overall_adapt))} |"
    )
    for cond in CONDITIONS:
        mb = float(cond_base.get(cond, float('nan')))
        ma = float(cond_adapt.get(cond, float('nan')))
        d = float(ma - mb) if not (math.isnan(mb) or math.isnan(ma)) else float("nan")
        dr = _delta_rel_pct(mb, ma)
        lines.append(f"| {cond} | {int(cond_n.get(cond, 0))} | {_fmt(mb)} | {_fmt(ma)} | {_fmt(d)} | {_fmt(dr)} |")
    lines.append("")

    if no_gt_test:
        lines.append("## Note")
        lines.append("- test split에서 GT를 찾지 못해 `0 images / No GT`로 정상 처리했습니다.")
        lines.append("")
        lines.append("## Qualitative 실패 케이스 3장")
        lines.append("- GT가 없어 생성하지 않았습니다.")
        lines.append("")
        lines.append("## 다음 액션 (1)")
        lines.append("- `--test_gt_dir`를 지정해 test split 정량/정성 평가를 1회 수행.")
        lines.append("")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Qualitative 실패 케이스 3장")
    if not failure_rows:
        lines.append("- 실패 케이스를 수집하지 못했습니다.")
    else:
        for i, row in enumerate(failure_rows, start=1):
            lines.append(
                f"- Case {i}: mIoU(adapt)={row['miou_adapt']:.2f}, mIoU(base)={row['miou_base']:.2f}, cond={row['condition']}, "
                f"file=`{row['file']}` | 설명: {row['reason']}"
            )
    lines.append("")
    lines.append("## 다음 액션 (1)")
    lines.append("- local boundary 비율(`--boundary_ratio_local`)을 0.6→0.7로 올린 단일 ablation 1회 실행.")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_no_gt_test_artifacts(args: argparse.Namespace, out_dir: Path, logger: logging.Logger) -> None:
    logger.info("No GT found for test split. This is treated as a normal situation.")
    cond_n = {c: 0 for c in CONDITIONS}
    cond_base = {c: float("nan") for c in CONDITIONS}
    cond_adapt = {c: float("nan") for c in CONDITIONS}
    write_results_csv(out_dir / "results.csv", args.split, 0, float("nan"), float("nan"), cond_n, cond_base, cond_adapt)
    write_summary_md(
        out_md=out_dir / "summary.md",
        args=args,
        overall_n=0,
        overall_base=float("nan"),
        overall_adapt=float("nan"),
        cond_n=cond_n,
        cond_base=cond_base,
        cond_adapt=cond_adapt,
        failure_rows=[],
        no_gt_test=True,
    )
    metrics = {
        "overall_base": float("nan"),
        "overall_adapt": float("nan"),
        "overall_delta_abs": float("nan"),
        "overall_delta_rel_pct": float("nan"),
        "cond_base": {c: float("nan") for c in CONDITIONS},
        "cond_adapt": {c: float("nan") for c in CONDITIONS},
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_gpu_or_exit()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "eval.log")
    logger.info("Args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

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
    logger.info(f"Loaded checkpoint: {args.ckpt}")

    if args.split == "test" and args.test_gt_dir:
        dataset: Dataset = ACDCTestWithGTDataset(
            acdc_root=args.acdc_root,
            test_gt_dir=args.test_gt_dir,
            conditions=list(args.conditions),
            resize=int(args.resize),
        )
    else:
        dataset = ACDCEvalDataset(
            root=args.acdc_root,
            split=args.split,
            conditions=list(args.conditions),
            resize=int(args.resize),
        )

    if len(dataset) == 0:
        if args.split == "test":
            write_no_gt_test_artifacts(args, out_dir, logger)
            logger.info("Saved: summary.md, results.csv, eval.log, eval_metrics.json")
            return 0
        logger.error("No images with GT found for split=%s", args.split)
        return 1

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True,
    )

    n_class = int(args.num_classes)
    total_hist_base = torch.zeros(n_class, n_class, device=device)
    total_hist_adapt = torch.zeros(n_class, n_class, device=device)
    cond_hist_base = {c: torch.zeros(n_class, n_class, device=device) for c in CONDITIONS}
    cond_hist_adapt = {c: torch.zeros(n_class, n_class, device=device) for c in CONDITIONS}
    cond_n = {c: 0 for c in CONDITIONS}

    failure_pool: list[dict] = []
    colors = get_acdc_class_colors()
    reason_by_cond = {
        "fog": "저대비 구간에서 원거리 객체 경계가 흐려져 클래스 혼동이 발생함.",
        "night": "저조도 구간에서 도로/보도 및 동적 객체 경계가 약화됨.",
        "rain": "노면 반사와 물기 영향으로 차량/도로 구분이 불안정함.",
        "snow": "노면 질감 균질화로 보도/도로 경계와 소형 객체가 붕괴됨.",
    }

    for batch in tqdm(loader, desc="Evaluation"):
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
            path = str(batch["path"][i])
            pred_base = preds_base[i]
            pred_adapt = preds_adapt[i]
            label = labels[i]
            mask = label != 255
            hist_base = fast_hist(pred_base[mask], label[mask], n_class)
            hist_adapt = fast_hist(pred_adapt[mask], label[mask], n_class)
            total_hist_base += hist_base
            total_hist_adapt += hist_adapt
            if cond in cond_hist_base:
                cond_hist_base[cond] += hist_base
                cond_hist_adapt[cond] += hist_adapt
                cond_n[cond] += 1

            img_miou_base = miou_from_hist(hist_base)
            img_miou_adapt = miou_from_hist(hist_adapt)
            rec = {
                "miou_base": img_miou_base if not math.isnan(img_miou_base) else 999.0,
                "miou_adapt": img_miou_adapt if not math.isnan(img_miou_adapt) else 999.0,
                "image": images[i].detach().cpu(),
                "pred": pred_adapt.detach().cpu(),
                "gt": label.detach().cpu(),
                "condition": cond,
                "path": path,
            }
            if len(failure_pool) < 3:
                failure_pool.append(rec)
            else:
                # Keep only global worst-3 (lower mIoU is worse).
                max_idx = int(np.argmax([float(x["miou_adapt"]) for x in failure_pool]))
                if float(rec["miou_adapt"]) < float(failure_pool[max_idx]["miou_adapt"]):
                    failure_pool[max_idx] = rec

    overall_n = int(sum(cond_n.values()))
    overall_base = miou_from_hist(total_hist_base)
    overall_adapt = miou_from_hist(total_hist_adapt)
    cond_base = {c: miou_from_hist(cond_hist_base[c]) if cond_n[c] > 0 else float("nan") for c in CONDITIONS}
    cond_adapt = {c: miou_from_hist(cond_hist_adapt[c]) if cond_n[c] > 0 else float("nan") for c in CONDITIONS}

    # Pick worst 3 samples globally.
    failure_rows: list[dict] = []
    failure_pool = sorted(failure_pool, key=lambda x: float(x["miou_adapt"]))[:3]
    for rank, item in enumerate(failure_pool, start=1):
        img = _denorm(item["image"])
        pred = item["pred"].numpy().astype(np.int64)
        gt = item["gt"].numpy().astype(np.int64)
        pred_color = _colorize(pred, colors)
        gt_color = _colorize(gt, colors)
        panel = np.concatenate([img, pred_color, gt_color], axis=1)
        out_name = f"failure_{rank:02d}_{Path(item['path']).name}"
        out_path = qual_dir / out_name
        Image.fromarray(panel).save(out_path)
        cond = str(item["condition"])
        failure_rows.append(
            {
                "miou_base": float(item["miou_base"]),
                "miou_adapt": float(item["miou_adapt"]),
                "condition": cond,
                "file": str(out_path.relative_to(out_dir)),
                "reason": reason_by_cond.get(cond, "경계/텍스처 불확실성으로 오분류가 증가함."),
            }
        )

    write_results_csv(
        out_dir / "results.csv",
        args.split,
        overall_n,
        overall_base,
        overall_adapt,
        cond_n,
        cond_base,
        cond_adapt,
    )
    write_summary_md(
        out_md=out_dir / "summary.md",
        args=args,
        overall_n=overall_n,
        overall_base=overall_base,
        overall_adapt=overall_adapt,
        cond_n=cond_n,
        cond_base=cond_base,
        cond_adapt=cond_adapt,
        failure_rows=failure_rows,
        no_gt_test=False,
    )
    metrics = {
        "overall_base": float(overall_base),
        "overall_adapt": float(overall_adapt),
        "overall_delta_abs": float(overall_adapt - overall_base) if not (math.isnan(overall_base) or math.isnan(overall_adapt)) else float("nan"),
        "overall_delta_rel_pct": float(_delta_rel_pct(overall_base, overall_adapt)),
        "cond_base": {k: float(v) for k, v in cond_base.items()},
        "cond_adapt": {k: float(v) for k, v in cond_adapt.items()},
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Overall mIoU(base/adapt): %s / %s", _fmt(overall_base), _fmt(overall_adapt))
    logger.info(
        "Condition mIoU(base/adapt): "
        + ", ".join(f"{c}={_fmt(float(cond_base[c]))}/{_fmt(float(cond_adapt[c]))}" for c in CONDITIONS)
    )
    logger.info("Saved: summary.md, results.csv, eval.log, eval_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
