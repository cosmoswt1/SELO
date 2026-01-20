#!/usr/bin/env python3
# Evaluate (1) prediction mIoU and (2) pseudo-label quality vs ACDC GT (for analysis only).
# - Works with mmseg SegFormer config + checkpoint, and optional adapted checkpoint saved by train.py.
# - Reports: pred mIoU, pseudo mIoU (masked by conf>thr), coverage.

import os, argparse, glob, math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------- utils -----------------

def imread_rgb(path: str) -> torch.Tensor:
    from PIL import Image
    x = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1)  # [3,H,W]

def imread_gt(path: str) -> torch.Tensor:
    from PIL import Image
    y = np.array(Image.open(path), dtype=np.int64)  # [H,W]
    return torch.from_numpy(y)

def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    m = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    s = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - m) / s

def resize_keep_aspect(x: torch.Tensor, y: torch.Tensor, long_side: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [3,H,W], y: [H,W]
    _, H, W = x.shape
    s = long_side / max(H, W)
    nh, nw = int(round(H * s)), int(round(W * s))
    x2 = F.interpolate(x[None], (nh, nw), mode="bilinear", align_corners=False)[0]
    y2 = F.interpolate(y[None, None].float(), (nh, nw), mode="nearest")[0, 0].long()
    return x2, y2

def pad_to_divisor(x: torch.Tensor, y: torch.Tensor, div: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    # pad right/bottom to make H,W divisible by div
    _, H, W = x.shape
    ph = (div - (H % div)) % div
    pw = (div - (W % div)) % div
    if ph == 0 and pw == 0:
        return x, y
    x2 = F.pad(x, (0, pw, 0, ph), mode="reflect")
    y2 = F.pad(y, (0, pw, 0, ph), mode="constant", value=255)  # ignore
    return x2, y2

def fast_hist(pred: np.ndarray, gt: np.ndarray, ncls: int, ignore: int = 255) -> np.ndarray:
    # pred, gt: [H,W]
    m = (gt != ignore) & (gt >= 0) & (gt < ncls)
    if m.sum() == 0:
        return np.zeros((ncls, ncls), dtype=np.int64)
    x = ncls * gt[m].astype(np.int64) + pred[m].astype(np.int64)
    return np.bincount(x, minlength=ncls * ncls).reshape(ncls, ncls)

def miou_from_hist(hist: np.ndarray) -> float:
    # hist: [C,C]
    diag = np.diag(hist).astype(np.float64)
    denom = hist.sum(1) + hist.sum(0) - diag
    iou = diag / np.maximum(denom, 1.0)
    return float(np.nanmean(iou))


# ----------------- dataset -----------------

class ACDCEval(Dataset):
    """
    Matches:
      rgb: root/rgb_anon_trainvaltest/rgb_anon/{cond}/{split}/**/*_rgb_anon.*
      gt : root/gt_trainval/gt/{cond}/{split}/**/*_gt_labelTrainIds.png
    """
    def __init__(self, root: str, split: str, conds: List[str], long_side: int = 1024):
        self.root = root
        self.split = split
        self.conds = conds
        self.long_side = long_side

        rgb_base = os.path.join(root, "rgb_anon_trainvaltest", "rgb_anon")
        ps = []
        for c in conds:
            ps += glob.glob(os.path.join(rgb_base, c, split, "**", "*_rgb_anon.*"), recursive=True)
        self.rgb_paths = sorted([p for p in ps if os.path.isfile(p)])
        if not self.rgb_paths:
            raise RuntimeError(f"No RGB found. root={root} split={split} conds={conds}")

    def __len__(self):
        return len(self.rgb_paths)

    def _rgb_to_gt(self, rgb_path: str) -> str:
        # swap base + suffix
        gt_path = rgb_path.replace(
            os.path.join("rgb_anon_trainvaltest", "rgb_anon"),
            os.path.join("gt_trainval", "gt"),
        )
        gt_path = gt_path.replace("_rgb_anon", "_gt_labelTrainIds")
        # enforce png
        if not gt_path.endswith(".png"):
            gt_path = os.path.splitext(gt_path)[0] + ".png"
        return gt_path

    def __getitem__(self, i):
        rgb_path = self.rgb_paths[i]
        gt_path = self._rgb_to_gt(rgb_path)
        if not os.path.isfile(gt_path):
            raise RuntimeError(f"GT missing for: {rgb_path}\nExpected: {gt_path}")

        x = imread_rgb(rgb_path)     # [3,H,W]
        y = imread_gt(gt_path)       # [H,W] trainIds (0..18) or 255 ignore

        # deterministic eval transform
        x, y = resize_keep_aspect(x, y, self.long_side)
        x, y = pad_to_divisor(x, y, 32)

        return x, y, rgb_path


# ----------------- mmseg student -----------------

def build_student(cfg_path: str, ckpt_path: str, device: str):
    from mmengine.config import Config
    from mmengine.runner.checkpoint import load_checkpoint
    from mmseg.registry import MODELS

    cfg = Config.fromfile(cfg_path)
    if hasattr(cfg, "model") and isinstance(cfg.model, dict):
        cfg.model.pop("data_preprocessor", None)  # we normalize manually
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location="cpu", revise_keys=[(r"^module\.", "")])
    model.to(device).eval()
    return model

@torch.no_grad()
def forward_logits(model, x_norm_bchw: torch.Tensor) -> torch.Tensor:
    feats = model.backbone(x_norm_bchw)
    if getattr(model, "with_neck", False) and model.neck is not None:
        feats = model.neck(feats)
    logits = model.decode_head(feats)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    logits = F.interpolate(logits, x_norm_bchw.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def load_adapted_if_any(model, adapted_path: str):
    if not adapted_path:
        return
    ck = torch.load(adapted_path, map_location="cpu")
    sd = ck.get("student", None)
    if sd is None:
        # maybe direct state_dict
        sd = ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load adapted] missing={len(missing)} unexpected={len(unexpected)} from {adapted_path}")


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--conds", nargs="*", default=["fog", "night", "rain", "snow"])
    ap.add_argument("--long_side", type=int, default=1024)

    ap.add_argument("--config", default="/home/kevinlee01/miniconda3/envs/mmseg/lib/python3.8/site-packages/mmseg/.mim/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py")
    ap.add_argument("--checkpoint", default="segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth")
    ap.add_argument("--adapted", default="", help="Optional: ./work_dirs/sfda/adapt_XXXXXX.pth or adapt_final.pth")

    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--classes", type=int, default=19)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_images", type=int, default=0, help="0=all")
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    ds = ACDCEval(args.acdc_root, args.split, args.conds, long_side=args.long_side)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_student(args.config, args.checkpoint, dev)
    load_adapted_if_any(model, args.adapted)

    hist_pred = np.zeros((args.classes, args.classes), dtype=np.int64)
    hist_pseudo = np.zeros((args.classes, args.classes), dtype=np.int64)

    cov_sum = 0.0
    pix_sum = 0.0

    n_seen = 0
    for x, y, paths in dl:
        if args.max_images and n_seen >= args.max_images:
            break

        x = x.to(dev, non_blocking=True)          # [B,3,H,W] in [0,1]
        y = y.numpy()                             # [B,H,W] int64
        x_norm = imagenet_norm(x)

        with torch.cuda.amp.autocast(enabled=args.amp and str(dev).startswith("cuda")):
            logits = forward_logits(model, x_norm)  # [B,C,H,W]
            prob = logits.softmax(1)
            conf, pred = prob.max(1)                # [B,H,W]

        pred_np = pred.cpu().numpy()
        conf_np = conf.cpu().numpy()

        B = pred_np.shape[0]
        for b in range(B):
            gt = y[b]
            pr = pred_np[b]

            # (1) pred mIoU
            hist_pred += fast_hist(pr, gt, args.classes, ignore=255)

            # (2) pseudo mIoU: only on conf>thr
            mask = conf_np[b] > args.thr
            # treat non-selected as ignore
            gt_masked = gt.copy()
            gt_masked[~mask] = 255
            hist_pseudo += fast_hist(pr, gt_masked, args.classes, ignore=255)

            cov_sum += float(mask.sum())
            pix_sum += float(mask.size)

            n_seen += 1
            if n_seen % 50 == 0:
                print(f"[{n_seen}] last={paths[b]}")

    miou_pred = miou_from_hist(hist_pred)
    miou_pseudo = miou_from_hist(hist_pseudo)
    coverage = cov_sum / max(pix_sum, 1.0)

    print("==== RESULTS ====")
    print(f"split={args.split} conds={args.conds} long_side={args.long_side}")
    print(f"model_ckpt={args.checkpoint}")
    if args.adapted:
        print(f"adapted={args.adapted}")
    print(f"Pred  mIoU (all pixels):        {miou_pred*100:.2f}")
    print(f"Pseudo mIoU (conf>{args.thr}):  {miou_pseudo*100:.2f}")
    print(f"Coverage (conf>{args.thr}):     {coverage*100:.2f}%")

if __name__ == "__main__":
    main()