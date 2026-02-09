#!/usr/bin/env python3
"""
Minimal diagnostics to answer: "is training signal sane?"

Artifacts:
1) Anchor overlay on input image (top: candidate entropy low anchors)
2) Teacher entropy histogram (candidates vs selected)
3) |delta| map between stage3_raw and stage3_adapt (per-pixel over channels)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import ACDCDataset  # noqa: E402
from losses import LocalAffinityKLLoss  # noqa: E402
from models import SeloV0Model  # noqa: E402


class ACDCFullFrameDataset(Dataset):
    """ACDC images only, eval-like transform (no crop/flip), padding to 32."""

    CONDITIONS = ["fog", "night", "rain", "snow"]

    def _find_dir(self, root: Path, name: str) -> Path | None:
        direct = root / name
        if direct.exists():
            return direct
        for p in root.iterdir():
            if not p.is_dir():
                continue
            cand = p / name
            if cand.exists():
                return cand
            for q in p.iterdir():
                if not q.is_dir():
                    continue
                cand2 = q / name
                if cand2.exists():
                    return cand2
        return None

    def __init__(
        self,
        root: str,
        split: str = "val",
        conditions: list[str] | None = None,
        resize: int = 540,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.conditions = conditions if conditions is not None else list(self.CONDITIONS)
        self.resize = int(resize)

        for cond in self.conditions:
            if cond not in self.CONDITIONS:
                raise ValueError(f"Invalid condition: {cond}. Must be one of {self.CONDITIONS}")

        self.images: list[Path] = []
        self.image_conditions: list[str] = []

        rgb_root = self._find_dir(self.root, "rgb_anon_trainvaltest")
        rgb_base = (rgb_root / "rgb_anon") if rgb_root is not None else (self.root / "rgb_anon")
        print(f"[ACDCFullFrameDataset] rgb base: {rgb_base}")

        for cond in self.conditions:
            cond_dir = rgb_base / cond / split
            if not cond_dir.exists():
                print(f"Warning: {cond_dir} does not exist")
                continue
            for img_path in sorted(cond_dir.rglob("*_rgb_anon.png")):
                self.images.append(img_path)
                self.image_conditions.append(cond)

        print(f"Loaded {len(self.images)} images from ACDC {split} split (full-frame)")
        for cond in self.conditions:
            count = sum(1 for c in self.image_conditions if c == cond)
            print(f"  {cond}: {count} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        condition = self.image_conditions[idx]

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # Resize keeping aspect ratio (short side = self.resize)
        if h < w:
            new_h = self.resize
            new_w = int(w * self.resize / h)
        else:
            new_w = self.resize
            new_h = int(h * self.resize / w)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        # Pad to be divisible by 32 (eval-like)
        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], padding_mode="reflect")

        image = TF.to_tensor(image)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return {
            "image": image,
            "condition": condition,
            "path": str(img_path),
            "original_size": (h, w),
            "padded_size": (new_h + pad_h, new_w + pad_w),
        }


def parse_args():
    p = argparse.ArgumentParser(description="SELO v0 minimal diagnostics")
    p.add_argument("--acdc_root", type=str, required=True)
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--data_mode", type=str, default="train_aug", choices=["train_aug", "full_frame"],
                   help="train_aug: random crop/flip (like training); full_frame: eval-like resize+pad (no aug).")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle dataloader order (useful to sample across conditions).")
    p.add_argument("--mode", type=str, default="full", choices=["full", "delta"],
                   help="full: anchors+entropy+delta; delta: delta-only (fast, no DINO/affinity).")
    p.add_argument("--conditions", nargs="+", default=["fog", "night", "rain", "snow"])
    p.add_argument("--resize", type=int, default=1080)
    p.add_argument("--crop_size", type=int, default=1072,
                   help="Only used when --data_mode train_aug. Set <=0 to disable crop.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--steps", type=int, default=10,
                   help="Number of batches to run. Set <=0 to run all images.")
    p.add_argument("--save_anchors_n", type=int, default=10,
                   help="How many images to save anchor overlays for (full mode only). 0 disables saving.")
    p.add_argument("--save_anchors_all", action="store_true",
                   help="Save anchor overlays for all images (ignores --save_anchors_n).")

    p.add_argument("--segformer_model", type=str,
                   default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--dino_model", type=str,
                   default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--adapter_hidden_ratio", type=float, default=1.0)
    p.add_argument("--adapter_scale", type=float, default=1.0)
    p.add_argument("--proj_type", type=str, default="auto", choices=["auto", "conv", "mlp"])
    p.add_argument("--proj_mlp_hidden", type=int, default=0,
                   help="proj_type=mlp일 때 hidden dim. 0이면 ckpt args에서 자동 추론.")

    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--affinity_k", type=int, default=7)
    p.add_argument("--affinity_tau", type=float, default=0.1)
    p.add_argument("--affinity_anchors", type=int, default=512)
    p.add_argument("--affinity_candidates", type=int, default=4096)
    p.add_argument("--affinity_kcenter_top_m", type=int, default=1500)
    p.add_argument("--affinity_per_image", type=int, default=1,
                   help="1이면 이미지별로 앵커를 따로 선택합니다(권장). 0이면 배치 평균 entropy로 공통 앵커를 선택합니다.")

    # delta-map visualization knobs
    # Percentile normalization (default: linear min-max with 0..100).
    p.add_argument("--delta_p_low", type=float, default=0.0,
                   help="Lower percentile for delta-map normalization (values below become 0).")
    p.add_argument("--delta_p_high", type=float, default=100.0,
                   help="Upper percentile for delta-map normalization (values above become 1).")
    p.add_argument("--delta_gamma", type=float, default=1.0,
                   help="Gamma for delta-map intensity ( >1 emphasizes strong changes ).")
    p.add_argument("--delta_alpha_max", type=float, default=0.90,
                   help="Max overlay alpha for delta-map (0..1).")
    p.add_argument("--delta_base_dim", type=float, default=0.60,
                   help="Base image brightness multiplier under overlay (0..1).")

    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def denorm_image(x: torch.Tensor) -> np.ndarray:
    # x: [3,H,W], ImageNet norm
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1)
    y = (y * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return y


def overlay_points(img_rgb: np.ndarray, pts_yx: np.ndarray, color=(255, 0, 0), r: int = 3) -> np.ndarray:
    im = Image.fromarray(img_rgb)
    dr = ImageDraw.Draw(im)
    for y, x in pts_yx:
        x0, y0 = int(x) - r, int(y) - r
        x1, y1 = int(x) + r, int(y) + r
        dr.ellipse([x0, y0, x1, y1], outline=color, width=2)
    return np.array(im)


def save_heatmap_overlay(
    img_rgb: np.ndarray,
    heat: np.ndarray,
    out_path: Path,
    *,
    base_dim: float = 0.60,
    alpha_max: float = 0.90,
    gamma: float = 1.5,
):
    # heat: [H,W] in [0,1]
    h = np.clip(heat, 0.0, 1.0)
    if gamma != 1.0:
        # gamma>1 : suppress weak changes, emphasize strong ones
        h = np.power(h, float(gamma))

    base = img_rgb.astype(np.float32) * float(base_dim)
    overlay = np.zeros_like(base, dtype=np.float32)
    overlay[..., 0] = 255.0  # red

    a = (h * float(alpha_max)).astype(np.float32)[..., None]  # [H,W,1]
    out = (base * (1.0 - a) + overlay * a).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("[GPU 체크 실패] torch.cuda.is_available() == False", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        raise SystemExit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta").mkdir(parents=True, exist_ok=True)
    save_anchors = (args.mode == "full") and (args.save_anchors_all or (int(args.save_anchors_n) > 0))
    save_anchors_limit = None if args.save_anchors_all else int(args.save_anchors_n)
    if save_anchors:
        (out_dir / "anchors").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    if args.data_mode == "full_frame":
        dataset = ACDCFullFrameDataset(
            root=args.acdc_root,
            split=args.split,
            conditions=args.conditions,
            resize=args.resize,
        )
        shuffle = False
    else:
        crop = None
        if int(args.crop_size) > 0:
            crop = (int(args.crop_size), int(args.crop_size))
        dataset = ACDCDataset(
            root=args.acdc_root,
            split=args.split,
            conditions=args.conditions,
            resize=args.resize,
            crop_size=crop,
        )
        shuffle = True

    if args.shuffle:
        shuffle = True
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model = SeloV0Model(
        segformer_model=args.segformer_model,
        dino_model=args.dino_model,
        num_classes=args.num_classes,
        adapter_hidden_ratio=args.adapter_hidden_ratio,
        adapter_scale=args.adapter_scale,
    ).to(device)
    model.eval()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        proj_type = args.proj_type
        if proj_type == "auto":
            proj_type = str(ckpt_args.get("proj_type", "conv"))
        proj_hidden = int(args.proj_mlp_hidden)
        if proj_hidden <= 0:
            proj_hidden = int(ckpt_args.get("proj_mlp_hidden", 256))

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

    loss_fn = None
    if args.mode == "full":
        loss_fn = LocalAffinityKLLoss(
            k=args.affinity_k,
            tau=args.affinity_tau,
            anchors=args.affinity_anchors,
            candidates=args.affinity_candidates,
            kcenter_top_m=int(args.affinity_kcenter_top_m),
            per_image=(int(args.affinity_per_image) > 0),
        )

    all_cand_ent = []
    all_sel_ent = []
    saved_anchor_images = 0
    img_counter = 0

    for bi, batch in enumerate(loader):
        if int(args.steps) > 0 and bi >= int(args.steps):
            break
        images = batch["image"].to(device)
        paths = batch["path"]
        conds = batch.get("condition", ["unknown"] * int(images.shape[0]))
        b, _c, inp_h, inp_w = images.shape

        debug = None
        if args.mode == "full":
            assert loss_fn is not None
            with torch.no_grad():
                out = model(images, use_dino=True, compute_logits=False)
                f3 = out["stage3_adapt"]
                proj = model.stage3_proj(f3)
                dino_feat = out["dino_feat"]
                _loss, _stats, debug = loss_fn(proj, dino_feat, return_stats=True, return_debug=True)

            cand_ent = debug["candidate_ent"].numpy()
            sel_ent = debug["selected_ent"].numpy()
            all_cand_ent.append(cand_ent)
            all_sel_ent.append(sel_ent)
        else:
            with torch.no_grad():
                out = model(images, use_dino=False, compute_logits=False)

        # 1) Anchor overlay (optional): use selected anchors, map 67x67 -> input pixels
        sel = None
        ah = aw = None
        if save_anchors and (save_anchors_limit is None or saved_anchor_images < int(save_anchors_limit)):
            assert debug is not None
            ah, aw = debug["hw"]
            sel = debug["selected_anchors"].numpy().astype(np.int64)  # [B,M,2] (y,x)

        # 3) Delta map: |f1_adapt - f1_raw| aggregated across channels, upsample to input
        with torch.no_grad():
            f_raw = out["stage3_raw"].float()
            f_adapt = out["stage3_adapt"].float()
            delta = (f_adapt - f_raw).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
            delta_up = F.interpolate(delta, size=(inp_h, inp_w), mode="bilinear", align_corners=False)
            # Percentile normalize per-image to make changes easier to see.
            # We do it on CPU for simplicity (diag only).
            delta_cpu = delta_up.squeeze(1).detach().float().cpu().numpy()  # [B,H,W]
            delta_norm = []
            for j in range(b):
                d = delta_cpu[j]
                lo = np.percentile(d, float(args.delta_p_low))
                hi = np.percentile(d, float(args.delta_p_high))
                if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-12:
                    n = np.zeros_like(d, dtype=np.float32)
                else:
                    n = np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
                delta_norm.append(n)
            delta_norm = np.stack(delta_norm, axis=0)

        for j in range(b):
            img_rgb = denorm_image(images[j])
            name = Path(paths[j]).name
            cond = conds[j] if isinstance(conds, (list, tuple)) else str(conds)
            cond_dir = out_dir / "delta" / str(cond)
            cond_dir.mkdir(parents=True, exist_ok=True)
            save_heatmap_overlay(
                img_rgb,
                delta_norm[j],
                cond_dir / f"{img_counter:06d}_{name}",
                base_dim=float(args.delta_base_dim),
                alpha_max=float(args.delta_alpha_max),
                gamma=float(args.delta_gamma),
            )
            if sel is not None and save_anchors and (
                save_anchors_limit is None or saved_anchor_images < int(save_anchors_limit)
            ):
                a_cond_dir = out_dir / "anchors" / str(cond)
                a_cond_dir.mkdir(parents=True, exist_ok=True)
                sel_j = sel[j]
                ys = (sel_j[:, 0] + 0.5) * (inp_h / ah)
                xs = (sel_j[:, 1] + 0.5) * (inp_w / aw)
                pts = np.stack([ys, xs], axis=1)
                over = overlay_points(img_rgb, pts_yx=pts, color=(255, 0, 0), r=2)
                Image.fromarray(over).save(a_cond_dir / f"{img_counter:06d}_{name}")
                saved_anchor_images += 1
            img_counter += 1

    if args.mode != "full":
        return

    cand = np.concatenate(all_cand_ent, axis=0) if all_cand_ent else np.array([], dtype=np.float32)
    sel = np.concatenate(all_sel_ent, axis=0) if all_sel_ent else np.array([], dtype=np.float32)
    np.savetxt(out_dir / "teacher_entropy_candidates.csv", cand, delimiter=",")
    np.savetxt(out_dir / "teacher_entropy_selected.csv", sel, delimiter=",")

    # 2) entropy histogram (png if matplotlib available; otherwise just csv is saved)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433

        plt.figure(figsize=(6, 4))
        if cand.size:
            plt.hist(cand, bins=50, alpha=0.5, label="candidates", density=True)
        if sel.size:
            plt.hist(sel, bins=50, alpha=0.7, label="selected", density=True)
        plt.xlabel("teacher entropy")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "teacher_entropy_hist.png", dpi=160)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
