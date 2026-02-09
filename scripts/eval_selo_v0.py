#!/usr/bin/env python3
"""Evaluate SELO v0 and save qual failures + results.csv."""

import argparse
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import ACDCEvalDataset, get_acdc_class_names  # noqa: E402
from datasets.acdc import get_acdc_class_colors  # noqa: E402
from models import SeloV0Model  # noqa: E402


def setup_logger(log_path: Path):
    logger = logging.getLogger("selo_v0_eval")
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


def fast_hist(pred: torch.Tensor, label: torch.Tensor, n: int) -> torch.Tensor:
    mask = (label >= 0) & (label < n)
    return torch.bincount(
        n * label[mask].int() + pred[mask].int(),
        minlength=n ** 2
    ).reshape(n, n).float()


def miou_from_hist(hist: torch.Tensor) -> float:
    iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + 1e-6)
    valid = hist.sum(dim=1) > 0
    return (iou[valid].mean().item() * 100) if valid.any() else 0.0


def class_iou_from_hist(hist: torch.Tensor) -> np.ndarray:
    iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + 1e-6)
    return iou.cpu().numpy()


def parse_baseline(baseline_path: Path):
    text = baseline_path.read_text().strip().splitlines()
    for line in text:
        if line.startswith("| val "):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            # Split | val | overall | fog | rain | snow | night | notes |
            overall = float(parts[1])
            fog = float(parts[2])
            rain = float(parts[3])
            snow = float(parts[4])
            night = float(parts[5])
            avg = float(np.mean([fog, rain, snow, night]))
            return {
                "overall": overall,
                "fog": fog,
                "rain": rain,
                "snow": snow,
                "night": night,
                "avg": avg,
            }
    raise RuntimeError(f"baseline val row not found: {baseline_path}")


def colorize_mask(mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(colors))
    out[valid] = colors[mask[valid]]
    return out


def save_failure_cases(failure_dict, out_dir: Path):
    colors = get_acdc_class_colors()
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    for cond, items in failure_dict.items():
        cond_dir = out_dir / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        for rank, item in enumerate(sorted(items, key=lambda x: x["miou"])):
            img = item["image"].cpu().numpy().transpose(1, 2, 0)
            img = (img * std + mean) * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)

            pred = item["pred"].cpu().numpy().astype(np.int64)
            gt = item["gt"].cpu().numpy().astype(np.int64)

            pred_color = colorize_mask(pred, colors)
            gt_color = colorize_mask(gt, colors)

            concat = np.concatenate([img, pred_color, gt_color], axis=1)
            name = Path(item["path"]).name
            out_path = cond_dir / f"rank{rank+1}_miou{item['miou']:.2f}_{name}"
            Image.fromarray(concat).save(out_path)


def get_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser(description="SELO v0 evaluation")
    p.add_argument("--acdc_root", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--conditions", nargs="+", default=["fog", "night", "rain", "snow"])
    p.add_argument("--resize", type=int, default=1080)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)

    p.add_argument("--segformer_model", type=str,
                   default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--dino_model", type=str,
                   default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--adapter_hidden_ratio", type=float, default=0.25)
    p.add_argument("--adapter_scale", type=float, default=0.1)
    p.add_argument("--proj_type", type=str, default="auto", choices=["auto", "conv", "mlp"])
    p.add_argument("--proj_mlp_hidden", type=int, default=0,
                   help="proj_type=mlp일 때 hidden dim. 0이면 ckpt args에서 자동 추론.")

    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--qual_dir", type=str, required=True)
    p.add_argument("--exp_id", type=str, default="E1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--results_path", type=str, default="logs/results.csv")
    return p.parse_args()


def main():
    if not torch.cuda.is_available():
        print("[GPU 체크 실패] torch.cuda.is_available() == False", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        raise SystemExit(1)

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "eval.log")
    logger.info("Starting evaluation...")
    logger.info("Args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    device = torch.device("cuda")

    dataset = ACDCEvalDataset(
        root=args.acdc_root,
        split=args.split,
        conditions=args.conditions,
        resize=args.resize,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = SeloV0Model(
        segformer_model=args.segformer_model,
        dino_model=args.dino_model,
        num_classes=args.num_classes,
        adapter_hidden_ratio=args.adapter_hidden_ratio,
        adapter_scale=args.adapter_scale,
    ).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    proj_type = args.proj_type
    if proj_type == "auto":
        proj_type = str(ckpt_args.get("proj_type", "conv"))
    proj_hidden = int(args.proj_mlp_hidden)
    if proj_hidden <= 0:
        proj_hidden = int(ckpt_args.get("proj_mlp_hidden", 256))

    # Rebuild projector to match checkpoint if needed (avoid state_dict mismatch).
    if getattr(model.stage3_proj, "proj_type", "conv") != proj_type:
        model = SeloV0Model(
            segformer_model=args.segformer_model,
            dino_model=args.dino_model,
            num_classes=args.num_classes,
            adapter_hidden_ratio=args.adapter_hidden_ratio,
            adapter_scale=args.adapter_scale,
            proj_type=proj_type,
            proj_mlp_hidden=proj_hidden,
        ).to(device)
        model.eval()

    model.stage3_adapter.load_state_dict(ckpt["stage3_adapter"], strict=True)
    model.stage3_proj.load_state_dict(ckpt["stage3_proj"], strict=True)
    logger.info(f"Loaded adapter checkpoint: {args.ckpt}")
    logger.info(f"Projector: type={proj_type} mlp_hidden={proj_hidden}")
    try:
        s = model.stage3_adapter.scale.detach().float()
        logger.info(
            f"Adapter scale: mean={float(s.mean().item()):.6f} |max|={float(s.abs().max().item()):.6f}"
        )
    except Exception:
        logger.info("Adapter scale: <unavailable>")

    num_classes = args.num_classes
    total_hist = torch.zeros(num_classes, num_classes, device=device)
    condition_hist = defaultdict(lambda: torch.zeros(num_classes, num_classes, device=device))

    failure_dict = defaultdict(list)
    max_failures = 3

    logged_delta = False
    for batch in tqdm(loader, desc="Evaluation"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        condition = batch["condition"][0]

        with torch.no_grad():
            outputs = model(images, use_dino=False)
            logits = outputs["logits"]
            if not logged_delta:
                f_raw = outputs["stage3_raw"]
                f_adapt = outputs["stage3_adapt"]
                delta = (f_adapt - f_raw).pow(2).mean().sqrt().item()
                logger.info(f"Stage3 adapt delta (L2 mean sqrt): {delta:.6f}")
                logged_delta = True

        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1)

        for pred, label, img, path in zip(preds, labels, images, batch["path"]):
            mask = label != 255
            hist = fast_hist(pred[mask], label[mask], num_classes)
            total_hist += hist
            condition_hist[condition] += hist

            miou = miou_from_hist(hist)
            items = failure_dict[condition]
            record = {
                "miou": miou,
                "image": img.detach().cpu(),
                "pred": pred.detach().cpu(),
                "gt": label.detach().cpu(),
                "path": path,
            }
            if len(items) < max_failures:
                items.append(record)
            else:
                worst_idx = int(np.argmax([r["miou"] for r in items]))
                if miou < items[worst_idx]["miou"]:
                    items[worst_idx] = record

    overall_miou = miou_from_hist(total_hist)
    class_iou = class_iou_from_hist(total_hist)
    cond_miou = {cond: miou_from_hist(hist) for cond, hist in condition_hist.items()}
    avg = float(np.mean([cond_miou.get(c, 0.0) for c in ["fog", "rain", "snow", "night"]]))

    logger.info(f"Overall mIoU: {overall_miou:.2f}%")
    if cond_miou:
        cond_parts = [f"{cond}={cond_miou[cond]:.2f}%" for cond in sorted(cond_miou.keys())]
        logger.info("Condition mIoU: " + ", ".join(cond_parts))

    class_names = get_acdc_class_names()
    for idx, name in enumerate(class_names):
        logger.info(f"class_iou/{name}: {float(class_iou[idx]) * 100:.2f}%")

    save_failure_cases(failure_dict, Path(args.qual_dir))

    baseline = parse_baseline(REPO_ROOT / "baseline" / "summary.md")
    delta = avg - baseline["avg"]

    results_path = REPO_ROOT / args.results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header = "exp_id,commit,seed,epochs,split,overall,fog,rain,snow,night,avg,delta_baseline,run_dir"
    row = ",".join([
        args.exp_id,
        get_commit(),
        str(args.seed),
        str(args.epochs),
        args.split,
        f"{overall_miou:.2f}",
        f"{cond_miou.get('fog', 0.0):.2f}",
        f"{cond_miou.get('rain', 0.0):.2f}",
        f"{cond_miou.get('snow', 0.0):.2f}",
        f"{cond_miou.get('night', 0.0):.2f}",
        f"{avg:.2f}",
        f"{delta:.2f}",
        str(output_dir),
    ])
    if not results_path.exists():
        results_path.write_text(header + "\n" + row + "\n")
    else:
        with results_path.open("a") as f:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
