#!/usr/bin/env python3
"""Build qualitative panels for top positive/negative class deltas in rain/snow."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from datasets import ACDCEvalDataset  # noqa: E402
from datasets.acdc import get_acdc_class_colors, get_acdc_class_names  # noqa: E402
from eval_stage3_cka import CONDITIONS, ACDCTestWithGTDataset  # noqa: E402
from model_cka_v1 import Stage3CKAModel  # noqa: E402


TARGET_SPECS = [
    {"condition": "rain", "class_name": "train", "direction": "positive"},
    {"condition": "rain", "class_name": "bus", "direction": "positive"},
    {"condition": "rain", "class_name": "pole", "direction": "positive"},
    {"condition": "rain", "class_name": "person", "direction": "positive"},
    {"condition": "rain", "class_name": "truck", "direction": "negative"},
    {"condition": "rain", "class_name": "traffic sign", "direction": "negative"},
    {"condition": "snow", "class_name": "truck", "direction": "positive"},
    {"condition": "snow", "class_name": "bus", "direction": "positive"},
    {"condition": "snow", "class_name": "pole", "direction": "positive"},
    {"condition": "snow", "class_name": "person", "direction": "positive"},
    {"condition": "snow", "class_name": "bicycle", "direction": "negative"},
]


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
    logger = logging.getLogger("top_class_qual_panels")
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
    p = argparse.ArgumentParser(description="Qualitative panels for top class IoU deltas.")
    p.add_argument("--acdc_root", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--test_gt_dir", type=str, default="")
    p.add_argument("--conditions", nargs="+", default=["rain", "snow"])
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
    p.add_argument("--min_crop_size", type=int, default=256)
    p.add_argument("--tile_height", type=int, default=300)
    p.add_argument("--overall_miou_csv", type=str, default="")
    return p.parse_args()


def infer_overall_miou_csv(ckpt_path: Path) -> Path | None:
    name = ckpt_path.name
    if not name.startswith("adapter_epoch_") or not name.endswith(".pth"):
        return None
    epoch_tag = name.replace("adapter_epoch_", "").replace(".pth", "")
    candidate = ckpt_path.parent / "eval_by_epoch" / f"epoch_{epoch_tag}" / "results.csv"
    return candidate if candidate.exists() else None


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


def build_dataset(args: argparse.Namespace):
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


def _denorm(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1)
    return (y.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _blend_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _error_overlay(rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.4).astype(np.uint8)
    tp = gt_mask & pred_mask
    fp = (~gt_mask) & pred_mask
    fn = gt_mask & (~pred_mask)
    out = _blend_mask(out, tp, (60, 220, 90), alpha=0.75)
    out = _blend_mask(out, fp, (230, 70, 70), alpha=0.75)
    out = _blend_mask(out, fn, (70, 140, 245), alpha=0.75)
    return out


def _diff_overlay(rgb: np.ndarray, gt_mask: np.ndarray, base_mask: np.ndarray, adapt_mask: np.ndarray) -> np.ndarray:
    out = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    out = _blend_mask(out, gt_mask, (70, 140, 245), alpha=0.35)
    added = (~base_mask) & adapt_mask
    removed = base_mask & (~adapt_mask)
    out = _blend_mask(out, added, (250, 215, 40), alpha=0.8)
    out = _blend_mask(out, removed, (225, 80, 200), alpha=0.8)
    return out


def _draw_bbox(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(255, 220, 40), width=4)
    return np.array(img)


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    width = max(1, int(round(w * (height / h))))
    return np.array(Image.fromarray(img).resize((width, height), Image.BILINEAR))


def _concat_h(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    max_h = max(x.shape[0] for x in images)
    total_w = sum(x.shape[1] for x in images) + gap * (len(images) - 1)
    canvas = np.full((max_h, total_w, 3), 245, dtype=np.uint8)
    x0 = 0
    for img in images:
        canvas[: img.shape[0], x0 : x0 + img.shape[1]] = img
        x0 += img.shape[1] + gap
    return canvas


def _add_header(panel: np.ndarray, lines: list[str]) -> np.ndarray:
    header_h = 24 + 22 * len(lines)
    canvas = Image.new("RGB", (panel.shape[1], panel.shape[0] + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = 10
    for line in lines:
        draw.text((12, y), line, fill=(20, 20, 20))
        y += 22
    canvas.paste(Image.fromarray(panel), (0, header_h))
    return np.array(canvas)


def _crop_box(mask: np.ndarray, min_crop: int) -> tuple[int, int, int, int]:
    h, w = mask.shape
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, w, h)
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    bw = x2 - x1
    bh = y2 - y1
    side_w = max(min_crop, int(round(bw * 1.8)))
    side_h = max(min_crop, int(round(bh * 1.8)))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x1 = max(0, cx - side_w // 2)
    y1 = max(0, cy - side_h // 2)
    x2 = min(w, x1 + side_w)
    y2 = min(h, y1 + side_h)
    x1 = max(0, x2 - side_w)
    y1 = max(0, y2 - side_h)
    return (x1, y1, x2, y2)


def _crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _per_image_iou(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return float("nan")
    return float(tp / denom * 100.0)


def _dominant_effect(rec: dict) -> tuple[str, str]:
    delta_tp = int(rec["tp_adapt"]) - int(rec["tp_base"])
    delta_fp = int(rec["fp_adapt"]) - int(rec["fp_base"])
    delta_fn = int(rec["fn_adapt"]) - int(rec["fn_base"])
    pred_growth = int(rec["pred_pixels_adapt"]) - int(rec["pred_pixels_base"])
    gt_pixels = int(rec["gt_pixels"])
    if gt_pixels == 0 and delta_fp > 0:
        return "false_positive_growth", "GT가 없는 장면에서 false positive가 늘어남."
    if delta_fn < 0 and delta_fp <= 0:
        return "boundary_recovery", "FN 감소가 커서 경계/누락 복원이 주효함."
    if delta_fn < 0 and delta_tp > 0 and pred_growth > 0 and delta_tp >= max(delta_fp, 0):
        return "extent_recovery", "예측 영역이 커지며 TP가 늘어 object extent 회복이 우세함."
    if delta_fp > max(delta_tp, 0) and delta_fp > max(-delta_fn, 0):
        return "false_positive_growth", "TP 개선보다 FP 증가가 커 false positive expansion이 우세함."
    if delta_fn > 0 and delta_fn >= max(delta_tp, 0):
        return "missed_extent", "FN 증가가 커서 object extent가 줄거나 경계를 놓침."
    return "mixed_change", "TP/FP/FN 변화가 섞여 있어 mixed change로 보임."


def _positive_score(rec: dict) -> float:
    if int(rec["gt_pixels"]) <= 0:
        return -1e9
    delta_iou = float(rec["delta_iou"]) if not math.isnan(float(rec["delta_iou"])) else -999.0
    recall_gain = float(rec["recall_adapt"]) - float(rec["recall_base"]) if not math.isnan(float(rec["recall_base"])) and not math.isnan(float(rec["recall_adapt"])) else 0.0
    precision_gain = float(rec["precision_adapt"]) - float(rec["precision_base"]) if not math.isnan(float(rec["precision_base"])) and not math.isnan(float(rec["precision_adapt"])) else 0.0
    fp_penalty = max(int(rec["fp_adapt"]) - int(rec["fp_base"]), 0) / max(int(rec["gt_pixels"]), 1) * 100.0
    return float(delta_iou + recall_gain * 30.0 + precision_gain * 10.0 - fp_penalty)


def _negative_score(rec: dict) -> float:
    delta_iou = float(rec["delta_iou"]) if not math.isnan(float(rec["delta_iou"])) else 0.0
    fp_growth = max(int(rec["fp_adapt"]) - int(rec["fp_base"]), 0)
    fn_growth = max(int(rec["fn_adapt"]) - int(rec["fn_base"]), 0)
    tp_gain = max(int(rec["tp_adapt"]) - int(rec["tp_base"]), 0)
    denom = max(int(rec["gt_pixels"]) + int(rec["pred_pixels_base"]) + 1, 1)
    return float((-delta_iou) + (fp_growth + fn_growth - tp_gain) / denom * 100.0)


def _load_overall_miou_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return {row["group_name"]: row for row in csv.DictReader(f)}


def _make_panel(rec: dict, out_path: Path, tile_height: int, min_crop_size: int) -> None:
    rgb = rec["rgb"]
    gt_mask = rec["gt_mask"]
    base_mask = rec["base_mask"]
    adapt_mask = rec["adapt_mask"]
    union_mask = gt_mask | base_mask | adapt_mask
    bbox = _crop_box(union_mask, min_crop=min_crop_size)

    full_rgb = _draw_bbox(rgb, bbox)
    gt_overlay = _blend_mask(rgb, gt_mask, (70, 140, 245), alpha=0.65)
    base_overlay = _error_overlay(rgb, gt_mask, base_mask)
    adapt_overlay = _error_overlay(rgb, gt_mask, adapt_mask)
    diff_overlay = _diff_overlay(rgb, gt_mask, base_mask, adapt_mask)

    crop_imgs = [
        full_rgb,
        _crop(gt_overlay, bbox),
        _crop(base_overlay, bbox),
        _crop(adapt_overlay, bbox),
        _crop(diff_overlay, bbox),
    ]
    crop_imgs = [_resize_to_height(x, tile_height) for x in crop_imgs]
    panel = _concat_h(crop_imgs, gap=8)

    title = f"{rec['condition']} | {rec['class_name']} | {rec['direction']} | img_delta_iou={rec['delta_iou']:.2f}"
    stats = (
        f"TP {rec['tp_base']}->{rec['tp_adapt']} | FP {rec['fp_base']}->{rec['fp_adapt']} | "
        f"FN {rec['fn_base']}->{rec['fn_adapt']} | pred {rec['pred_pixels_base']}->{rec['pred_pixels_adapt']} | gt={rec['gt_pixels']}"
    )
    header = [
        title,
        stats,
        "Legend: GT target=blue, TP=green, FP=red, FN=blue, adapt-added=yellow, adapt-removed=magenta",
    ]
    panel = _add_header(panel, header)
    Image.fromarray(panel).save(out_path)


def _write_results_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "selected",
        "selection_rank",
        "condition",
        "class_name",
        "direction",
        "path",
        "class_iou_base_condition",
        "class_iou_adapt_condition",
        "class_iou_delta_condition",
        "iou_base",
        "iou_adapt",
        "delta_iou",
        "tp_base",
        "tp_adapt",
        "fp_base",
        "fp_adapt",
        "fn_base",
        "fn_adapt",
        "pred_pixels_base",
        "pred_pixels_adapt",
        "gt_pixels",
        "precision_base",
        "precision_adapt",
        "recall_base",
        "recall_adapt",
        "dominant_effect",
        "dominant_effect_kr",
        "selection_score",
        "panel_rel_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: float) -> str:
    return "NaN" if (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def _write_summary_md(
    out_path: Path,
    *,
    args: argparse.Namespace,
    overall_rows: dict[str, dict[str, str]],
    selected_rows: list[dict],
    neg_rows: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# Top Class Qualitative Panels")
    lines.append("")
    lines.append("## Preprocessing / Label Mapping / Resize-Crop Rule")
    lines.append("")
    lines.append(f"- Split: `{args.split}`")
    lines.append(f"- Conditions: `{', '.join(args.conditions)}`")
    lines.append(f"- Resize rule: shorter side -> `{args.resize}`, then eval path에서 32배수 pad")
    lines.append("- Crop rule: 없음 (evaluation에서는 random crop 미사용)")
    lines.append("- Label mapping: Cityscapes trainIds (`0..18`), ignore=`255`")
    lines.append("- Panel crop rule: target GT/pred union bbox를 기준으로 `1.8x` 확장, 최소 crop 크기 적용")
    lines.append("- Panel legend: GT target=blue, TP=green, FP=red, FN=blue, adapt-added=yellow, adapt-removed=magenta")
    lines.append("")
    lines.append("## mIoU (%)")
    lines.append("")
    lines.append("| group | num_images | baseline | adapted | delta(abs) |")
    lines.append("|---|---:|---:|---:|---:|")
    for key in ["all", "rain", "snow"]:
        row = overall_rows.get(key, {})
        group = "overall" if key == "all" else key
        lines.append(
            f"| {group} | {row.get('num_images', 'NaN')} | {row.get('miou_base', 'NaN')} | {row.get('miou_adapt', 'NaN')} | {row.get('delta_abs', 'NaN')} |"
        )
    lines.append("")
    lines.append("## Selected Panels")
    lines.append("")
    lines.append("| condition | class | selection | direction | class delta(cond) | image delta | dominant effect | panel |")
    lines.append("|---|---|---|---|---:|---:|---|---|")
    for row in selected_rows:
        panel_rel = row["panel_rel_path"]
        panel_abs = out_path.parent / panel_rel
        lines.append(
            f"| {row['condition']} | {row['class_name']} | {row['selection_rank']} | {row['direction']} | {_fmt(float(row['class_iou_delta_condition']))} | "
            f"{_fmt(float(row['delta_iou']))} | {row['dominant_effect_kr']} | [{Path(panel_rel).name}]({panel_abs}) |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for row in selected_rows:
        lines.append(
            f"- `{row['condition']} / {row['class_name']} / {row['selection_rank']}`: class IoU delta={_fmt(float(row['class_iou_delta_condition']))}, "
            f"selected image delta={_fmt(float(row['delta_iou']))}, TP `{row['tp_base']}->{row['tp_adapt']}`, "
            f"FP `{row['fp_base']}->{row['fp_adapt']}`, FN `{row['fn_base']}->{row['fn_adapt']}`. "
            f"판정: {row['dominant_effect_kr']} panel=`{row['panel_rel_path']}`"
        )
    lines.append("")
    lines.append("## Qualitative 실패 케이스 3장")
    if not neg_rows:
        lines.append("- negative target panel을 찾지 못했습니다.")
    else:
        for idx, row in enumerate(neg_rows[:3], start=1):
            panel_abs = out_path.parent / row["panel_rel_path"]
            lines.append(
                f"- Case {idx}: `{row['condition']} / {row['class_name']}` | image delta={_fmt(float(row['delta_iou']))}, "
                f"TP `{row['tp_base']}->{row['tp_adapt']}`, FP `{row['fp_base']}->{row['fp_adapt']}`, FN `{row['fn_base']}->{row['fn_adapt']}` | "
                f"설명: {row['dominant_effect_kr']} | file=[{Path(row['panel_rel_path']).name}]({panel_abs})"
            )
    lines.append("")
    lines.append("## 다음 액션 (1)")
    lines.append("- 같은 장면들에 대해 gate map 또는 confidence map을 겹쳐서, 회복/붕괴가 실제로 어느 영역에서 시작되는지 1회 확인.")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    check_gpu_or_exit()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "eval.log")
    logger.info("Args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    ckpt_path = Path(args.ckpt)
    miou_csv = Path(args.overall_miou_csv) if args.overall_miou_csv else infer_overall_miou_csv(ckpt_path)
    overall_rows = _load_overall_miou_rows(miou_csv)
    if miou_csv is not None:
        logger.info("Loaded overall mIoU csv: %s", miou_csv)

    model = load_model(args, logger)
    dataset = build_dataset(args)
    if len(dataset) == 0:
        logger.error("No images with GT found for split=%s", args.split)
        return 1

    class_names = get_acdc_class_names()
    class_colors = get_acdc_class_colors()
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    specs = []
    for spec in TARGET_SPECS:
        if spec["condition"] in args.conditions:
            spec = dict(spec)
            spec["class_id"] = class_to_id[spec["class_name"]]
            specs.append(spec)
    if not specs:
        logger.error("No target specs left after condition filtering")
        return 1

    by_key: dict[tuple[str, str, str], list[dict]] = {(s["condition"], s["class_name"], s["direction"]): [] for s in specs}
    totals: dict[tuple[str, str, str], dict[str, int]] = {
        (s["condition"], s["class_name"], s["direction"]): {
            "tp_base": 0,
            "tp_adapt": 0,
            "fp_base": 0,
            "fp_adapt": 0,
            "fn_base": 0,
            "fn_adapt": 0,
        }
        for s in specs
    }

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True,
    )
    device = torch.device("cuda")

    for batch in tqdm(loader, desc="TopClassQual"):
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
        logits_base = F.interpolate(out_base["logits"], size=labels.shape[-2:], mode="bilinear", align_corners=False)
        logits_adapt = F.interpolate(out_adapt["logits"], size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds_base = logits_base.argmax(dim=1)
        preds_adapt = logits_adapt.argmax(dim=1)

        for i in range(images.shape[0]):
            cond = str(batch["condition"][i])
            image_path = str(batch["path"][i])
            spec_list = [s for s in specs if s["condition"] == cond]
            if not spec_list:
                continue

            rgb = _denorm(images[i])
            gt = labels[i].detach().cpu().numpy()
            pred_base = preds_base[i].detach().cpu().numpy()
            pred_adapt = preds_adapt[i].detach().cpu().numpy()
            valid = gt != 255

            for spec in spec_list:
                class_id = int(spec["class_id"])
                gt_mask = (gt == class_id) & valid
                base_mask = (pred_base == class_id) & valid
                adapt_mask = (pred_adapt == class_id) & valid
                tp_base = int((gt_mask & base_mask).sum())
                tp_adapt = int((gt_mask & adapt_mask).sum())
                fp_base = int(((~gt_mask) & base_mask & valid).sum())
                fp_adapt = int(((~gt_mask) & adapt_mask & valid).sum())
                fn_base = int((gt_mask & (~base_mask)).sum())
                fn_adapt = int((gt_mask & (~adapt_mask)).sum())
                pred_pixels_base = int(base_mask.sum())
                pred_pixels_adapt = int(adapt_mask.sum())
                gt_pixels = int(gt_mask.sum())
                iou_base = _per_image_iou(tp_base, fp_base, fn_base)
                iou_adapt = _per_image_iou(tp_adapt, fp_adapt, fn_adapt)
                delta_iou = float(iou_adapt - iou_base) if not math.isnan(iou_base) and not math.isnan(iou_adapt) else float("nan")
                precision_base = _safe_div(tp_base, tp_base + fp_base)
                precision_adapt = _safe_div(tp_adapt, tp_adapt + fp_adapt)
                recall_base = _safe_div(tp_base, tp_base + fn_base)
                recall_adapt = _safe_div(tp_adapt, tp_adapt + fn_adapt)
                rec = {
                    "selected": 0,
                    "selection_rank": "",
                    "condition": cond,
                    "class_name": spec["class_name"],
                    "direction": spec["direction"],
                    "path": image_path,
                    "class_iou_base_condition": float("nan"),
                    "class_iou_adapt_condition": float("nan"),
                    "class_iou_delta_condition": float("nan"),
                    "iou_base": iou_base,
                    "iou_adapt": iou_adapt,
                    "delta_iou": delta_iou,
                    "tp_base": tp_base,
                    "tp_adapt": tp_adapt,
                    "fp_base": fp_base,
                    "fp_adapt": fp_adapt,
                    "fn_base": fn_base,
                    "fn_adapt": fn_adapt,
                    "pred_pixels_base": pred_pixels_base,
                    "pred_pixels_adapt": pred_pixels_adapt,
                    "gt_pixels": gt_pixels,
                    "precision_base": precision_base,
                    "precision_adapt": precision_adapt,
                    "recall_base": recall_base,
                    "recall_adapt": recall_adapt,
                    "selection_score": 0.0,
                    "dominant_effect": "",
                    "dominant_effect_kr": "",
                    "panel_rel_path": "",
                    "rgb": rgb,
                    "gt_mask": gt_mask,
                    "base_mask": base_mask,
                    "adapt_mask": adapt_mask,
                }
                key = (spec["condition"], spec["class_name"], spec["direction"])
                by_key[key].append(rec)
                totals[key]["tp_base"] += tp_base
                totals[key]["tp_adapt"] += tp_adapt
                totals[key]["fp_base"] += fp_base
                totals[key]["fp_adapt"] += fp_adapt
                totals[key]["fn_base"] += fn_base
                totals[key]["fn_adapt"] += fn_adapt

    selected_rows: list[dict] = []
    all_rows_for_csv: list[dict] = []
    for spec in specs:
        key = (spec["condition"], spec["class_name"], spec["direction"])
        rows = by_key[key]
        total = totals[key]
        class_iou_base = _per_image_iou(total["tp_base"], total["fp_base"], total["fn_base"])
        class_iou_adapt = _per_image_iou(total["tp_adapt"], total["fp_adapt"], total["fn_adapt"])
        class_iou_delta = float(class_iou_adapt - class_iou_base) if not math.isnan(class_iou_base) and not math.isnan(class_iou_adapt) else float("nan")
        for row in rows:
            row["class_iou_base_condition"] = class_iou_base
            row["class_iou_adapt_condition"] = class_iou_adapt
            row["class_iou_delta_condition"] = class_iou_delta
            effect_key, effect_kr = _dominant_effect(row)
            row["dominant_effect"] = effect_key
            row["dominant_effect_kr"] = effect_kr
            row["selection_score"] = _positive_score(row) if spec["direction"] == "positive" else _negative_score(row)

        fp_alert_row = None
        if spec["direction"] == "positive":
            candidate_rows = [r for r in rows if int(r["gt_pixels"]) > 0]
            candidate_rows.sort(
                key=lambda r: (float(r["selection_score"]), float(r["delta_iou"]) if not math.isnan(float(r["delta_iou"])) else -999.0, int(r["gt_pixels"])),
                reverse=True,
            )
        else:
            gt_present_rows = [r for r in rows if int(r["gt_pixels"]) > 0 and not math.isnan(float(r["delta_iou"]))]
            if gt_present_rows:
                candidate_rows = sorted(
                    gt_present_rows,
                    key=lambda r: (
                        float(r["delta_iou"]),
                        -(int(r["fp_adapt"]) - int(r["fp_base"])),
                        -float(r["selection_score"]),
                    ),
                )
            else:
                candidate_rows = [r for r in rows if int(r["gt_pixels"]) > 0 or int(r["pred_pixels_base"]) > 0 or int(r["pred_pixels_adapt"]) > 0]
                candidate_rows.sort(
                    key=lambda r: (float(r["selection_score"]), -(float(r["delta_iou"]) if not math.isnan(float(r["delta_iou"])) else 999.0), int(r["fp_adapt"]) - int(r["fp_base"])),
                    reverse=True,
                )
            fp_only_rows = [
                r
                for r in rows
                if int(r["gt_pixels"]) == 0 and (int(r["fp_adapt"]) - int(r["fp_base"])) > 0
            ]
            if fp_only_rows:
                fp_only_rows.sort(
                    key=lambda r: (int(r["fp_adapt"]) - int(r["fp_base"]), int(r["pred_pixels_adapt"])),
                    reverse=True,
                )
                fp_alert_row = fp_only_rows[0]

        if not candidate_rows:
            logger.warning("No candidate rows for %s / %s / %s", *key)
            continue

        best = candidate_rows[0]
        best["selected"] = 1
        best["selection_rank"] = "best"
        panel_name = f"{spec['condition']}_{spec['class_name'].replace(' ', '_')}_{spec['direction']}.png"
        panel_rel = Path("panels") / panel_name
        _make_panel(best, out_dir / panel_rel, tile_height=int(args.tile_height), min_crop_size=int(args.min_crop_size))
        best["panel_rel_path"] = str(panel_rel)
        selected_rows.append(best)

        if fp_alert_row is not None and fp_alert_row["path"] != best["path"]:
            fp_alert_row["selected"] = 1
            fp_alert_row["selection_rank"] = "fp_alert"
            fp_panel_name = f"{spec['condition']}_{spec['class_name'].replace(' ', '_')}_{spec['direction']}_fp_alert.png"
            fp_panel_rel = Path("panels") / fp_panel_name
            _make_panel(fp_alert_row, out_dir / fp_panel_rel, tile_height=int(args.tile_height), min_crop_size=int(args.min_crop_size))
            fp_alert_row["panel_rel_path"] = str(fp_panel_rel)
            selected_rows.append(fp_alert_row)

        for row in rows:
            row_copy = {k: v for k, v in row.items() if k not in {"rgb", "gt_mask", "base_mask", "adapt_mask"}}
            if row is best:
                row_copy["panel_rel_path"] = str(panel_rel)
            elif fp_alert_row is not None and row is fp_alert_row:
                row_copy["panel_rel_path"] = str(fp_panel_rel)
            all_rows_for_csv.append(row_copy)

    selected_rows.sort(key=lambda r: (r["condition"], r["direction"], r["class_name"]))
    neg_rows = [r for r in selected_rows if r["direction"] == "negative" and r["selection_rank"] == "best"]

    _write_results_csv(out_dir / "results.csv", all_rows_for_csv)
    _write_summary_md(
        out_dir / "summary.md",
        args=args,
        overall_rows=overall_rows,
        selected_rows=selected_rows,
        neg_rows=neg_rows,
    )
    metrics = {
        "targets": [{"condition": s["condition"], "class_name": s["class_name"], "direction": s["direction"]} for s in specs],
        "overall_miou_csv": str(miou_csv) if miou_csv is not None else "",
        "selected_panels": [
            {
                "condition": r["condition"],
                "class_name": r["class_name"],
                "direction": r["direction"],
                "panel_rel_path": r["panel_rel_path"],
                "class_iou_delta_condition": r["class_iou_delta_condition"],
                "delta_iou": r["delta_iou"],
                "dominant_effect": r["dominant_effect"],
            }
            for r in selected_rows
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved: summary.md, results.csv, eval.log, metrics.json, panels/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
