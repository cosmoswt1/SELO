#!/usr/bin/env python3
"""Find best DINO layer <-> SegFormer stage mapping on ACDC with CKA/SSM."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dino_teacher import DinoTeacher  # noqa: E402
from models.segformer_backbone import SegFormerBackbone  # noqa: E402


CONDITIONS = ["fog", "night", "rain", "snow"]
STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4"]
SPLIT_DIRS = ["train", "val", "test", "train_ref", "val_ref", "test_ref"]


def find_rgb_base(acdc_root: Path) -> Path:
    candidates = [
        acdc_root / "rgb_anon_trainvaltest" / "rgb_anon",
        acdc_root / "rgb_anon",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    for p in acdc_root.rglob("rgb_anon"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"rgb_anon base not found under: {acdc_root}")


@dataclass(frozen=True)
class ImageItem:
    path: Path
    condition: str
    split_dir: str
    is_ref: bool


def collect_acdc_images(
    acdc_root: Path,
    conditions: Iterable[str],
    include_ref: bool,
) -> list[ImageItem]:
    rgb_base = find_rgb_base(acdc_root)
    items: list[ImageItem] = []
    for cond in conditions:
        for split_dir in SPLIT_DIRS:
            is_ref = split_dir.endswith("_ref")
            if is_ref and not include_ref:
                continue
            folder = rgb_base / cond / split_dir
            if not folder.exists():
                continue
            pattern = "*_rgb_ref_anon.png" if is_ref else "*_rgb_anon.png"
            for path in sorted(folder.rglob(pattern)):
                items.append(
                    ImageItem(
                        path=path,
                        condition=cond,
                        split_dir=split_dir,
                        is_ref=is_ref,
                    )
                )
    return items


class ACDCAllImagesDataset(Dataset):
    def __init__(
        self,
        items: list[ImageItem],
        resize_short: int = 384,
        pad_multiple: int = 32,
        square_crop_size: int = 0,
    ):
        self.items = items
        self.resize_short = int(resize_short)
        self.pad_multiple = int(pad_multiple)
        self.square_crop_size = int(square_crop_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        image = Image.open(item.path).convert("RGB")
        w, h = image.size
        if h < w:
            new_h = self.resize_short
            new_w = int(round(w * self.resize_short / h))
        else:
            new_w = self.resize_short
            new_h = int(round(h * self.resize_short / w))
        image = image.resize((new_w, new_h), Image.BILINEAR)

        if self.square_crop_size > 0:
            # Training-like square input path: resize then center-crop to fixed square.
            target = self.square_crop_size
            if new_h < target or new_w < target:
                pad_h = max(target - new_h, 0)
                pad_w = max(target - new_w, 0)
                image = TF.pad(image, [0, 0, pad_w, pad_h], padding_mode="reflect")
                new_h += pad_h
                new_w += pad_w
            top = max((new_h - target) // 2, 0)
            left = max((new_w - target) // 2, 0)
            image = TF.crop(image, top, left, target, target)
        else:
            pad_h = (self.pad_multiple - (new_h % self.pad_multiple)) % self.pad_multiple
            pad_w = (self.pad_multiple - (new_w % self.pad_multiple)) % self.pad_multiple
            if pad_h > 0 or pad_w > 0:
                image = TF.pad(image, [0, 0, pad_w, pad_h], padding_mode="reflect")

        tensor = TF.to_tensor(image)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return {
            "image": tensor,
            "path": str(item.path),
            "condition": item.condition,
            "split_dir": item.split_dir,
            "is_ref": int(item.is_ref),
        }


def check_gpu_or_exit() -> None:
    try:
        smi = subprocess.run(["nvidia-smi"], check=True, text=True, capture_output=True)
        line = smi.stdout.splitlines()[0] if smi.stdout else "nvidia-smi OK"
        print(f"[GPU 체크] {line}")
    except Exception as exc:
        print("[GPU 체크 실패] nvidia-smi 실행 실패", file=sys.stderr)
        print(f"에러: {exc}", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        print("해결: NVIDIA 드라이버/CUDA 런타임을 확인하고 GPU가 보이는 환경에서 다시 실행하세요.", file=sys.stderr)
        raise SystemExit(1)

    if not torch.cuda.is_available():
        print("[GPU 체크 실패] torch.cuda.is_available() == False", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        print("해결: conda selo 환경에서 CUDA 지원 torch 설치 여부를 확인하세요.", file=sys.stderr)
        raise SystemExit(1)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("dino_segformer_layer_match")
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


def resize_feature_to_grid(feat: torch.Tensor, grid_size: int) -> torch.Tensor:
    # feat: [B, C, H, W]
    h, w = int(feat.shape[-2]), int(feat.shape[-1])
    if h == grid_size and w == grid_size:
        return feat
    # Downsample: adaptive average pooling.
    if h >= grid_size and w >= grid_size:
        return F.adaptive_avg_pool2d(feat, output_size=(grid_size, grid_size))
    # Upsample: copy-based expansion (no interpolation).
    return F.interpolate(feat, size=(grid_size, grid_size), mode="nearest")


def pool_tokens(feat: torch.Tensor, grid_size: int) -> torch.Tensor:
    pooled = resize_feature_to_grid(feat, grid_size)
    # [B, C, H, W] -> [B, N, C]
    return pooled.flatten(2).transpose(1, 2).contiguous()


def linear_cka(tokens_x: torch.Tensor, tokens_y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # tokens: [N, C]
    x = tokens_x - tokens_x.mean(dim=0, keepdim=True)
    y = tokens_y - tokens_y.mean(dim=0, keepdim=True)

    x_ty = x.transpose(0, 1) @ y
    hsic = torch.sum(x_ty * x_ty)

    x_tx = x.transpose(0, 1) @ x
    y_ty = y.transpose(0, 1) @ y
    denom = torch.sqrt(torch.sum(x_tx * x_tx) * torch.sum(y_ty * y_ty) + eps)
    return (hsic / denom).clamp(min=-1.0, max=1.0)


def ssm_signature(tokens: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # tokens: [N, C]
    z = F.normalize(tokens, dim=1)
    ssm = z @ z.transpose(0, 1)  # [N, N]
    n = ssm.shape[0]
    tri = torch.triu_indices(n, n, offset=1, device=ssm.device)
    v = ssm[tri[0], tri[1]]
    v = v - v.mean()
    return v / (v.norm() + eps)


def ssm_similarity(sig_a: torch.Tensor, sig_b: torch.Tensor) -> torch.Tensor:
    return (sig_a * sig_b).sum().clamp(min=-1.0, max=1.0)


@torch.no_grad()
def extract_dino_layer_features(
    teacher: DinoTeacher,
    x: torch.Tensor,
    strict_same_resolution: bool = True,
) -> tuple[list[int], list[torch.Tensor]]:
    x_aligned, h_aligned, w_aligned = teacher._align_to_patch(x, mode="resize")
    if strict_same_resolution and (h_aligned != x.shape[-2] or w_aligned != x.shape[-1]):
        raise RuntimeError(
            "DINO alignment changed input resolution. "
            f"input={tuple(x.shape[-2:])}, aligned={(h_aligned, w_aligned)}. "
            "Use an input size divisible by DINO patch size."
        )
    ph = h_aligned // teacher.patch_size
    pw = w_aligned // teacher.patch_size
    expected = ph * pw

    outs = teacher.dino(
        pixel_values=x_aligned,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outs.hidden_states  # embeddings + all blocks

    layer_ids: list[int] = []
    layer_feats: list[torch.Tensor] = []
    for idx in range(1, len(hidden_states)):  # skip embedding output(0)
        hs = hidden_states[idx]
        seq_len = hs.shape[1]
        prefix = seq_len - expected
        if prefix < 0:
            raise RuntimeError(
                f"DINO token length mismatch: seq_len={seq_len}, expected={expected}, layer={idx}"
            )
        if prefix != teacher.num_prefix_tokens:
            raise RuntimeError(
                "DINO prefix token mismatch: "
                f"layer={idx}, seq_len={seq_len}, expected_patches={expected}, "
                f"prefix={prefix}, cfg_prefix={teacher.num_prefix_tokens}"
            )
        tokens = hs[:, prefix:, :]  # [B, ph*pw, D]
        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], ph, pw)
        layer_ids.append(idx)
        layer_feats.append(feat)

    return layer_ids, layer_feats


def write_results_csv(
    out_csv: Path,
    layer_ids: list[int],
    cka_overall: np.ndarray,
    ssm_overall: np.ndarray,
    cka_by_cond: dict[str, np.ndarray],
    ssm_by_cond: dict[str, np.ndarray],
    n_overall: int,
    n_by_cond: dict[str, int],
    cka_by_split: dict[str, np.ndarray],
    ssm_by_split: dict[str, np.ndarray],
    n_by_split: dict[str, int],
    cka_by_ref: dict[str, np.ndarray],
    ssm_by_ref: dict[str, np.ndarray],
    n_by_ref: dict[str, int],
) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group_type",
                "group_name",
                "num_images",
                "dino_layer",
                "segformer_stage",
                "cka_mean",
                "ssm_mean",
            ]
        )

        def write_group(group_type: str, group_name: str, n_img: int, cka: np.ndarray, ssm: np.ndarray):
            for li, layer in enumerate(layer_ids):
                for sj, stage_name in enumerate(STAGE_NAMES):
                    writer.writerow(
                        [
                            group_type,
                            group_name,
                            n_img,
                            int(layer),
                            stage_name,
                            float(cka[li, sj]),
                            float(ssm[li, sj]),
                        ]
                    )

        write_group("overall", "all", n_overall, cka_overall, ssm_overall)
        for cond in CONDITIONS:
            if n_by_cond[cond] > 0:
                write_group("condition", cond, n_by_cond[cond], cka_by_cond[cond], ssm_by_cond[cond])
        for split_name, n_img in n_by_split.items():
            if n_img > 0:
                write_group("split", split_name, n_img, cka_by_split[split_name], ssm_by_split[split_name])
        for ref_key, n_img in n_by_ref.items():
            if n_img > 0:
                write_group("ref_flag", ref_key, n_img, cka_by_ref[ref_key], ssm_by_ref[ref_key])


def save_heatmap(
    matrix: np.ndarray,
    layer_ids: list[int],
    title: str,
    out_path: Path,
    vmin: float = -0.2,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 9.2))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("SegFormer stage")
    ax.set_ylabel("DINO layer")
    ax.set_xticks(np.arange(len(STAGE_NAMES)))
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_yticks(np.arange(len(layer_ids)))
    ax.set_yticklabels([str(x) for x in layer_ids])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_best_layer_plot(
    layer_ids: list[int],
    best_by_cka: list[int],
    best_by_ssm: list[int],
    out_path: Path,
) -> None:
    x = np.arange(len(STAGE_NAMES))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(x, best_by_cka, marker="o", linewidth=2.0, label="best by CKA")
    ax.plot(x, best_by_ssm, marker="s", linewidth=2.0, label="best by SSM")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_ylim(min(layer_ids) - 1, max(layer_ids) + 1)
    ax.set_ylabel("DINO layer")
    ax.set_title("Best-matching DINO layer per SegFormer stage")
    ax.grid(alpha=0.35, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_low_agreement_images(
    image_paths: list[str],
    image_scores: np.ndarray,
    out_dir: Path,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    if len(image_paths) == 0:
        return []
    rank = np.argsort(image_scores)[:top_k]
    selected: list[tuple[str, float]] = []
    for i, idx in enumerate(rank, start=1):
        src = Path(image_paths[idx])
        score = float(image_scores[idx])
        selected.append((str(src), score))
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        w, h = img.size
        scale = min(720.0 / max(h, w), 1.0)
        if scale < 1.0:
            img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        dst = out_dir / f"low_agreement_{i:02d}.png"
        img.save(dst)
    return selected


def compute_group_means(
    sum_cka: np.ndarray,
    sum_ssm: np.ndarray,
    n_total: int,
    sum_cka_cond: dict[str, np.ndarray],
    sum_ssm_cond: dict[str, np.ndarray],
    n_cond: dict[str, int],
    sum_cka_split: dict[str, np.ndarray],
    sum_ssm_split: dict[str, np.ndarray],
    n_split: dict[str, int],
    sum_cka_ref: dict[str, np.ndarray],
    sum_ssm_ref: dict[str, np.ndarray],
    n_ref: dict[str, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    cka_overall = sum_cka / max(n_total, 1)
    ssm_overall = sum_ssm / max(n_total, 1)
    cka_cond = {c: (sum_cka_cond[c] / max(n_cond[c], 1)) for c in CONDITIONS}
    ssm_cond = {c: (sum_ssm_cond[c] / max(n_cond[c], 1)) for c in CONDITIONS}
    cka_split = {s: (sum_cka_split[s] / max(n_split[s], 1)) for s in SPLIT_DIRS}
    ssm_split = {s: (sum_ssm_split[s] / max(n_split[s], 1)) for s in SPLIT_DIRS}
    cka_ref = {k: (sum_cka_ref[k] / max(n_ref[k], 1)) for k in ["non_ref", "ref"]}
    ssm_ref = {k: (sum_ssm_ref[k] / max(n_ref[k], 1)) for k in ["non_ref", "ref"]}
    return cka_overall, ssm_overall, cka_cond, ssm_cond, cka_split, ssm_split, cka_ref, ssm_ref


def compute_best_mapping(
    layer_ids: list[int],
    cka_overall: np.ndarray,
    ssm_overall: np.ndarray,
) -> tuple[list[int], list[float], list[int], list[float]]:
    best_cka_layer_by_stage: list[int] = []
    best_cka_score_by_stage: list[float] = []
    best_ssm_layer_by_stage: list[int] = []
    best_ssm_score_by_stage: list[float] = []
    for sj in range(4):
        li_cka = int(np.argmax(cka_overall[:, sj]))
        li_ssm = int(np.argmax(ssm_overall[:, sj]))
        best_cka_layer_by_stage.append(layer_ids[li_cka])
        best_ssm_layer_by_stage.append(layer_ids[li_ssm])
        best_cka_score_by_stage.append(float(cka_overall[li_cka, sj]))
        best_ssm_score_by_stage.append(float(ssm_overall[li_ssm, sj]))
    return (
        best_cka_layer_by_stage,
        best_cka_score_by_stage,
        best_ssm_layer_by_stage,
        best_ssm_score_by_stage,
    )


def write_running_summary(
    out_path: Path,
    n_total: int,
    n_all: int,
    best_cka_layer_by_stage: list[int],
    best_cka_score_by_stage: list[float],
    best_ssm_layer_by_stage: list[int],
    best_ssm_score_by_stage: list[float],
) -> None:
    pct = (100.0 * n_total / max(n_all, 1))
    lines = [
        "# DINO Layer ↔ SegFormer Stage Matching (RUNNING)",
        "",
        f"- Status: running",
        f"- Processed: **{n_total}/{n_all}** ({pct:.2f}%)",
        "- Streaming viz: `viz_stream/latest.png`",
        "",
        "## Current Best Layer Mapping",
        "| SegFormer Stage | Best DINO layer (CKA) | CKA | Best DINO layer (SSM) | SSM |",
        "|---|---:|---:|---:|---:|",
    ]
    for sj, stage in enumerate(STAGE_NAMES):
        lines.append(
            f"| {stage} | {best_cka_layer_by_stage[sj]} | {best_cka_score_by_stage[sj]:.6f} "
            f"| {best_ssm_layer_by_stage[sj]} | {best_ssm_score_by_stage[sj]:.6f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _denorm_image_uint8(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1)
    return (y.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _draw_anchor_dot(img_uint8: np.ndarray, px: int, py: int, r: int = 7) -> np.ndarray:
    img = Image.fromarray(img_uint8)
    dr = ImageDraw.Draw(img)
    dr.ellipse([(px - r, py - r), (px + r, py + r)], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
    return np.asarray(img)


def _token_cosine_map(tokens: torch.Tensor, anchor_idx: int, grid_size: int) -> np.ndarray:
    # tokens: [N, C], N=grid_size*grid_size
    z = F.normalize(tokens.float(), dim=1)
    anchor = z[anchor_idx : anchor_idx + 1]  # [1,C]
    sim = (z @ anchor.transpose(0, 1)).squeeze(1).reshape(grid_size, grid_size)
    sim_np = sim.detach().cpu().numpy()
    vmin, vmax = float(sim_np.min()), float(sim_np.max())
    if vmax - vmin > 1e-8:
        sim_np = (sim_np - vmin) / (vmax - vmin)
    else:
        sim_np = np.zeros_like(sim_np, dtype=np.float32)
    return sim_np


def _tokens_from_feature_for_viz(feat_3d: torch.Tensor, viz_grid_size: int) -> torch.Tensor:
    # feat_3d: [C, H, W] -> tokens [N, C], where N=viz_grid_size^2
    resized = resize_feature_to_grid(feat_3d.unsqueeze(0), viz_grid_size).squeeze(0)
    return resized.flatten(1).transpose(0, 1).contiguous()


def build_anchor_points(viz_grid_size: int, anchor_mode: str, anchor_x: float, anchor_y: float) -> list[tuple[int, int]]:
    def _ratio_to_idx(rx: float, ry: float) -> tuple[int, int]:
        gx = int(np.clip(round(rx * (viz_grid_size - 1)), 0, viz_grid_size - 1))
        gy = int(np.clip(round(ry * (viz_grid_size - 1)), 0, viz_grid_size - 1))
        return gy, gx

    if anchor_mode == "quadrant4":
        # 4분면 중심: (1/4,1/4), (3/4,1/4), (1/4,3/4), (3/4,3/4)
        return [
            _ratio_to_idx(0.25, 0.25),
            _ratio_to_idx(0.75, 0.25),
            _ratio_to_idx(0.25, 0.75),
            _ratio_to_idx(0.75, 0.75),
        ]
    return [_ratio_to_idx(anchor_x, anchor_y)]


def _grid_to_pixel(gx: int, gy: int, w: int, h: int, grid_size: int) -> tuple[int, int]:
    px = int(round((gx + 0.5) * w / grid_size))
    py = int(round((gy + 0.5) * h / grid_size))
    px = int(np.clip(px, 0, w - 1))
    py = int(np.clip(py, 0, h - 1))
    return px, py


def save_input_anchor_overlay(
    out_path: Path,
    latest_path: Path,
    image_tensor: torch.Tensor,
    anchor_points: list[tuple[int, int]],
    viz_grid_size: int,
) -> None:
    img_uint8 = _denorm_image_uint8(image_tensor)
    h, w = img_uint8.shape[:2]
    img = Image.fromarray(img_uint8)
    dr = ImageDraw.Draw(img)
    for ai, (gy, gx) in enumerate(anchor_points, start=1):
        px, py = _grid_to_pixel(gx=gx, gy=gy, w=w, h=h, grid_size=viz_grid_size)
        one = _draw_anchor_dot(np.asarray(img), px=px, py=py, r=max(6, min(h, w) // 120))
        img = Image.fromarray(one)
        dr = ImageDraw.Draw(img)
        dr.text((px + 10, py - 10), f"A{ai} ({gy},{gx})", fill=(255, 0, 0))
    img.save(out_path)
    img.save(latest_path)


def save_token_similarity_panel_4x7(
    out_path: Path,
    latest_path: Path,
    stage_feats_one: list[torch.Tensor],
    layer_feats_one: list[torch.Tensor],
    layer_ids: list[int],
    viz_grid_size: int,
    anchor_gy_gx: tuple[int, int],
    anchor_label: str,
    image_path: str,
    n_total: int,
    n_all: int,
) -> None:
    gy, gx = anchor_gy_gx
    anchor_idx = gy * viz_grid_size + gx

    stage_tokens_vis = [_tokens_from_feature_for_viz(feat, viz_grid_size) for feat in stage_feats_one]
    layer_tokens_vis = [_tokens_from_feature_for_viz(feat, viz_grid_size) for feat in layer_feats_one]

    stage_maps = [
        _token_cosine_map(tok, anchor_idx=anchor_idx, grid_size=viz_grid_size) for tok in stage_tokens_vis
    ]
    dino_maps = [_token_cosine_map(tok, anchor_idx=anchor_idx, grid_size=viz_grid_size) for tok in layer_tokens_vis]

    n_rows = 4
    n_dino_cols = int(math.ceil(len(layer_ids) / n_rows))
    n_cols = 1 + n_dino_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r in range(n_rows):
        ax0 = axes[r, 0]
        ax0.imshow(stage_maps[r], cmap="viridis", vmin=0.0, vmax=1.0)
        ax0.set_title(f"Seg {STAGE_NAMES[r]}", fontsize=9)
        ax0.scatter([gx], [gy], c="red", s=22, marker="o")
        ax0.axis("off")

        for c in range(n_dino_cols):
            li = r * n_dino_cols + c
            ax = axes[r, c + 1]
            if li < len(layer_ids):
                ax.imshow(dino_maps[li], cmap="viridis", vmin=0.0, vmax=1.0)
                ax.set_title(f"DINO L{layer_ids[li]}", fontsize=9)
            else:
                ax.imshow(np.zeros((viz_grid_size, viz_grid_size), dtype=np.float32), cmap="viridis", vmin=0.0, vmax=1.0)
                ax.set_title("-", fontsize=9)
            ax.scatter([gx], [gy], c="red", s=16, marker="o")
            ax.axis("off")

    basename = Path(image_path).name
    fig.suptitle(
        f"Token similarity map (4x7) @ {n_total}/{n_all} | grid={viz_grid_size}x{viz_grid_size} | {anchor_label}=({gy},{gx}) | {basename}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    fig.savefig(latest_path, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINO layer <-> SegFormer stage match via CKA/SSM")
    p.add_argument("--acdc_root", type=str, default="/mnt/d/ACDC")
    p.add_argument("--output_dir", type=str, default="exp/dino_segformer_layer_match/run_all")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ref_group = p.add_mutually_exclusive_group()
    ref_group.add_argument(
        "--include_ref",
        dest="include_ref",
        action="store_true",
        help="Include *_ref splits.",
    )
    ref_group.add_argument(
        "--exclude_ref",
        dest="include_ref",
        action="store_false",
        help="Exclude *_ref splits.",
    )
    p.set_defaults(include_ref=True)
    p.add_argument("--resize_short", type=int, default=1072)
    p.add_argument(
        "--square_crop_size",
        type=int,
        default=1072,
        help="Final fixed input size (HxW) fed to both SegFormer and DINO. 0 disables square crop.",
    )
    p.add_argument("--pad_multiple", type=int, default=32)
    p.add_argument("--grid_size", type=int, default=14)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--progress_log_every", type=int, default=200, help="Log progress every N images.")
    p.add_argument("--snapshot_every", type=int, default=1000, help="Write running summary/results every N images.")
    p.add_argument("--viz_every", type=int, default=200, help="Save token-map visualization every N images.")
    p.add_argument(
        "--viz_grid_size",
        type=int,
        default=0,
        help="Visualization grid size. 0이면 DINO native token grid(예: 1072->67) 사용.",
    )
    p.add_argument("--viz_max", type=int, default=0, help="Max number of streaming visualizations (0=unlimited).")
    p.add_argument(
        "--viz_anchor_mode",
        type=str,
        default="center",
        choices=["center", "quadrant4"],
        help="center: 단일 앵커(anchor_x/anchor_y), quadrant4: 4분면 중심 4개 앵커.",
    )
    p.add_argument("--anchor_x", type=float, default=0.5, help="Anchor x ratio in [0,1] on token grid.")
    p.add_argument("--anchor_y", type=float, default=0.5, help="Anchor y ratio in [0,1] on token grid.")
    p.add_argument("--amp", action="store_true", help="Enable fp16 autocast on CUDA.")
    p.add_argument(
        "--segformer_model",
        type=str,
        default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    )
    p.add_argument(
        "--dino_model",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    p.add_argument("--max_images", type=int, default=0, help="0 means all images.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.anchor_x = float(np.clip(args.anchor_x, 0.0, 1.0))
    args.anchor_y = float(np.clip(args.anchor_y, 0.0, 1.0))
    if int(args.square_crop_size) > 0 and int(args.square_crop_size) % 16 != 0:
        print(
            f"[입력 해상도 오류] square_crop_size={args.square_crop_size}는 16의 배수여야 합니다.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    check_gpu_or_exit()

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    qual_dir = output_dir / "qualitative"
    viz_dir = output_dir / "viz_stream"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    qual_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir / "eval.log")
    logger.info("Start DINO-SegFormer layer matching.")
    logger.info("Args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    acdc_root = Path(args.acdc_root)
    if not acdc_root.exists():
        logger.error(f"ACDC root not found: {acdc_root}")
        return 1

    items = collect_acdc_images(
        acdc_root=acdc_root,
        conditions=args.conditions,
        include_ref=bool(args.include_ref),
    )
    if args.max_images > 0:
        items = items[: args.max_images]

    if len(items) == 0:
        logger.error("No ACDC images found.")
        return 1

    count_by_cond = {c: 0 for c in CONDITIONS}
    count_by_split = {s: 0 for s in SPLIT_DIRS}
    count_by_ref = {"non_ref": 0, "ref": 0}
    for it in items:
        count_by_cond[it.condition] += 1
        count_by_split[it.split_dir] += 1
        count_by_ref["ref" if it.is_ref else "non_ref"] += 1

    logger.info(f"Collected images: {len(items)}")
    logger.info("Condition counts: " + ", ".join(f"{k}={v}" for k, v in count_by_cond.items()))
    logger.info("Split-dir counts: " + ", ".join(f"{k}={v}" for k, v in count_by_split.items()))
    logger.info("Ref counts: " + ", ".join(f"{k}={v}" for k, v in count_by_ref.items()))
    n_all = len(items)

    dataset = ACDCAllImagesDataset(
        items=items,
        resize_short=args.resize_short,
        pad_multiple=args.pad_multiple,
        square_crop_size=args.square_crop_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device("cuda")
    segformer = SegFormerBackbone(model_name=args.segformer_model, num_classes=19).to(device)
    teacher = DinoTeacher(model_name=args.dino_model).to(device)
    segformer.eval()
    teacher.eval()

    # Determine DINO layer count with one sample.
    with torch.no_grad():
        sample_x = dataset[0]["image"].unsqueeze(0).to(device)
        layer_ids, sample_dino_feats = extract_dino_layer_features(teacher, sample_x, strict_same_resolution=True)
    num_layers = len(layer_ids)
    if num_layers <= 0:
        logger.error("No DINO layer features extracted.")
        return 1
    dino_native_grid = int(sample_dino_feats[0].shape[-1])
    viz_grid_size = int(args.viz_grid_size) if int(args.viz_grid_size) > 0 else dino_native_grid

    logger.info(f"DINO layers used: {layer_ids[0]}..{layer_ids[-1]} (count={num_layers})")
    logger.info(
        "Input tensor resolution (shared by SegFormer/DINO): %sx%s",
        int(sample_x.shape[-2]),
        int(sample_x.shape[-1]),
    )
    logger.info(
        "Grid settings: cka_ssm_grid=%dx%d, viz_grid=%dx%d (dino_native=%d)",
        int(args.grid_size),
        int(args.grid_size),
        viz_grid_size,
        viz_grid_size,
        dino_native_grid,
    )
    logger.info("Viz anchor mode: %s", args.viz_anchor_mode)

    sum_cka = np.zeros((num_layers, 4), dtype=np.float64)
    sum_ssm = np.zeros((num_layers, 4), dtype=np.float64)
    n_total = 0

    sum_cka_cond = {c: np.zeros((num_layers, 4), dtype=np.float64) for c in CONDITIONS}
    sum_ssm_cond = {c: np.zeros((num_layers, 4), dtype=np.float64) for c in CONDITIONS}
    n_cond = {c: 0 for c in CONDITIONS}

    sum_cka_split = {s: np.zeros((num_layers, 4), dtype=np.float64) for s in SPLIT_DIRS}
    sum_ssm_split = {s: np.zeros((num_layers, 4), dtype=np.float64) for s in SPLIT_DIRS}
    n_split = {s: 0 for s in SPLIT_DIRS}

    sum_cka_ref = {k: np.zeros((num_layers, 4), dtype=np.float64) for k in ["non_ref", "ref"]}
    sum_ssm_ref = {k: np.zeros((num_layers, 4), dtype=np.float64) for k in ["non_ref", "ref"]}
    n_ref = {k: 0 for k in ["non_ref", "ref"]}

    all_image_paths: list[str] = []
    all_image_pair_cka: list[np.ndarray] = []
    all_image_pair_ssm: list[np.ndarray] = []

    amp_enabled = bool(args.amp)
    amp_dtype = torch.float16
    started_at = time.time()
    last_log_n = 0
    last_snapshot_n = 0
    last_viz_n = 0
    viz_count = 0

    # Create early placeholders so artifacts are visible immediately.
    (output_dir / "results.csv").write_text(
        "group_type,group_name,num_images,dino_layer,segformer_stage,cka_mean,ssm_mean\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        f"# DINO Layer ↔ SegFormer Stage Matching (RUNNING)\n\n- Status: running\n- Processed: **0/{n_all}** (0.00%)\n",
        encoding="utf-8",
    )

    for batch in tqdm(loader, desc="Layer matching"):
        x = batch["image"].to(device, non_blocking=True)
        conds = batch["condition"]
        split_dirs = batch["split_dir"]
        is_ref_arr = batch["is_ref"]
        paths = batch["path"]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                seg_feats = segformer.forward_encoder(x)  # 4 stages
                _, dino_feats = extract_dino_layer_features(teacher, x, strict_same_resolution=True)

        # Precompute pooled tokens/signatures
        stage_tokens = [pool_tokens(feat.float(), args.grid_size) for feat in seg_feats]
        stage_ssm = []
        for s in range(4):
            stage_ssm.append([ssm_signature(stage_tokens[s][b]) for b in range(x.shape[0])])

        layer_tokens = [pool_tokens(feat.float(), args.grid_size) for feat in dino_feats]
        layer_ssm = []
        for li in range(num_layers):
            layer_ssm.append([ssm_signature(layer_tokens[li][b]) for b in range(x.shape[0])])

        for b in range(x.shape[0]):
            pair_cka = np.zeros((num_layers, 4), dtype=np.float64)
            pair_ssm = np.zeros((num_layers, 4), dtype=np.float64)

            for li in range(num_layers):
                dino_tok = layer_tokens[li][b]
                dino_sig = layer_ssm[li][b]
                for sj in range(4):
                    seg_tok = stage_tokens[sj][b]
                    seg_sig = stage_ssm[sj][b]
                    cka_val = float(linear_cka(dino_tok, seg_tok).item())
                    ssm_val = float(ssm_similarity(dino_sig, seg_sig).item())
                    pair_cka[li, sj] = cka_val
                    pair_ssm[li, sj] = ssm_val

            cond = str(conds[b])
            split_dir = str(split_dirs[b])
            ref_key = "ref" if int(is_ref_arr[b]) == 1 else "non_ref"

            sum_cka += pair_cka
            sum_ssm += pair_ssm
            n_total += 1

            if cond in sum_cka_cond:
                sum_cka_cond[cond] += pair_cka
                sum_ssm_cond[cond] += pair_ssm
                n_cond[cond] += 1

            if split_dir in sum_cka_split:
                sum_cka_split[split_dir] += pair_cka
                sum_ssm_split[split_dir] += pair_ssm
                n_split[split_dir] += 1

            sum_cka_ref[ref_key] += pair_cka
            sum_ssm_ref[ref_key] += pair_ssm
            n_ref[ref_key] += 1

            all_image_paths.append(str(paths[b]))
            all_image_pair_cka.append(pair_cka)
            all_image_pair_ssm.append(pair_ssm)

        if (args.progress_log_every > 0) and (n_total - last_log_n >= args.progress_log_every):
            elapsed = time.time() - started_at
            ips = n_total / max(elapsed, 1e-6)
            remain = max(n_all - n_total, 0)
            eta_sec = remain / max(ips, 1e-6)
            cka_now, ssm_now, _, _, _, _, _, _ = compute_group_means(
                sum_cka=sum_cka,
                sum_ssm=sum_ssm,
                n_total=n_total,
                sum_cka_cond=sum_cka_cond,
                sum_ssm_cond=sum_ssm_cond,
                n_cond=n_cond,
                sum_cka_split=sum_cka_split,
                sum_ssm_split=sum_ssm_split,
                n_split=n_split,
                sum_cka_ref=sum_cka_ref,
                sum_ssm_ref=sum_ssm_ref,
                n_ref=n_ref,
            )
            best_cka_now, _, best_ssm_now, _ = compute_best_mapping(layer_ids, cka_now, ssm_now)
            logger.info(
                "Progress: %d/%d (%.2f%%), %.2f img/s, ETA %.1f min, best(CKA)=%s, best(SSM)=%s",
                n_total,
                n_all,
                100.0 * n_total / max(n_all, 1),
                ips,
                eta_sec / 60.0,
                ",".join(f"{st}->{ly}" for st, ly in zip(STAGE_NAMES, best_cka_now)),
                ",".join(f"{st}->{ly}" for st, ly in zip(STAGE_NAMES, best_ssm_now)),
            )
            last_log_n = n_total

        if (args.snapshot_every > 0) and (n_total - last_snapshot_n >= args.snapshot_every):
            (
                cka_now,
                ssm_now,
                cka_cond_now,
                ssm_cond_now,
                cka_split_now,
                ssm_split_now,
                cka_ref_now,
                ssm_ref_now,
            ) = compute_group_means(
                sum_cka=sum_cka,
                sum_ssm=sum_ssm,
                n_total=n_total,
                sum_cka_cond=sum_cka_cond,
                sum_ssm_cond=sum_ssm_cond,
                n_cond=n_cond,
                sum_cka_split=sum_cka_split,
                sum_ssm_split=sum_ssm_split,
                n_split=n_split,
                sum_cka_ref=sum_cka_ref,
                sum_ssm_ref=sum_ssm_ref,
                n_ref=n_ref,
            )
            (
                best_cka_now,
                best_cka_now_score,
                best_ssm_now,
                best_ssm_now_score,
            ) = compute_best_mapping(layer_ids, cka_now, ssm_now)
            write_results_csv(
                out_csv=output_dir / "results.csv",
                layer_ids=layer_ids,
                cka_overall=cka_now,
                ssm_overall=ssm_now,
                cka_by_cond=cka_cond_now,
                ssm_by_cond=ssm_cond_now,
                n_overall=n_total,
                n_by_cond=n_cond,
                cka_by_split=cka_split_now,
                ssm_by_split=ssm_split_now,
                n_by_split=n_split,
                cka_by_ref=cka_ref_now,
                ssm_by_ref=ssm_ref_now,
                n_by_ref=n_ref,
            )
            write_running_summary(
                out_path=output_dir / "summary.md",
                n_total=n_total,
                n_all=n_all,
                best_cka_layer_by_stage=best_cka_now,
                best_cka_score_by_stage=best_cka_now_score,
                best_ssm_layer_by_stage=best_ssm_now,
                best_ssm_score_by_stage=best_ssm_now_score,
            )
            logger.info("Snapshot saved: processed=%d/%d", n_total, n_all)
            last_snapshot_n = n_total

        need_viz = False
        if args.viz_every > 0:
            if viz_count == 0 and n_total > 0:
                need_viz = True
            elif n_total - last_viz_n >= args.viz_every:
                need_viz = True
        if need_viz and (args.viz_max <= 0 or viz_count < args.viz_max):
            _, _, _, _, _, _, _, _ = compute_group_means(
                sum_cka=sum_cka,
                sum_ssm=sum_ssm,
                n_total=n_total,
                sum_cka_cond=sum_cka_cond,
                sum_ssm_cond=sum_ssm_cond,
                n_cond=n_cond,
                sum_cka_split=sum_cka_split,
                sum_ssm_split=sum_ssm_split,
                n_split=n_split,
                sum_cka_ref=sum_cka_ref,
                sum_ssm_ref=sum_ssm_ref,
                n_ref=n_ref,
            )
            anchor_points = build_anchor_points(
                viz_grid_size=viz_grid_size,
                anchor_mode=str(args.viz_anchor_mode),
                anchor_x=float(args.anchor_x),
                anchor_y=float(args.anchor_y),
            )
            b_vis = 0
            stage_feats_one = [seg_feats[sj][b_vis].float().detach() for sj in range(4)]
            layer_feats_one = [dino_feats[li][b_vis].float().detach() for li in range(num_layers)]
            out_input = viz_dir / f"input_{n_total:06d}.png"
            latest_input = viz_dir / "latest_input_anchors.png"
            save_input_anchor_overlay(
                out_path=out_input,
                latest_path=latest_input,
                image_tensor=x[b_vis].detach(),
                anchor_points=anchor_points,
                viz_grid_size=viz_grid_size,
            )
            saved_paths: list[str] = []
            for ai, anchor in enumerate(anchor_points, start=1):
                out_viz = viz_dir / f"viz_{n_total:06d}_a{ai}.png"
                latest_viz = viz_dir / f"latest_a{ai}.png"
                save_token_similarity_panel_4x7(
                    out_path=out_viz,
                    latest_path=latest_viz,
                    stage_feats_one=stage_feats_one,
                    layer_feats_one=layer_feats_one,
                    layer_ids=layer_ids,
                    viz_grid_size=viz_grid_size,
                    anchor_gy_gx=anchor,
                    anchor_label=f"A{ai}",
                    image_path=str(paths[b_vis]),
                    n_total=n_total,
                    n_all=n_all,
                )
                saved_paths.append(str(out_viz))
            if len(anchor_points) > 0:
                shutil.copy2(viz_dir / "latest_a1.png", viz_dir / "latest.png")
            logger.info(
                "Viz saved: %d anchor panels, first=%s (latest -> %s, grid=%d, mode=%s)",
                len(anchor_points),
                saved_paths[0] if len(saved_paths) > 0 else "-",
                str(viz_dir / "latest.png"),
                viz_grid_size,
                args.viz_anchor_mode,
            )
            viz_count += 1
            last_viz_n = n_total

    cka_overall, ssm_overall, cka_cond, ssm_cond, cka_split, ssm_split, cka_ref, ssm_ref = compute_group_means(
        sum_cka=sum_cka,
        sum_ssm=sum_ssm,
        n_total=n_total,
        sum_cka_cond=sum_cka_cond,
        sum_ssm_cond=sum_ssm_cond,
        n_cond=n_cond,
        sum_cka_split=sum_cka_split,
        sum_ssm_split=sum_ssm_split,
        n_split=n_split,
        sum_cka_ref=sum_cka_ref,
        sum_ssm_ref=sum_ssm_ref,
        n_ref=n_ref,
    )

    # Best layer for each stage.
    (
        best_cka_layer_by_stage,
        best_cka_score_by_stage,
        best_ssm_layer_by_stage,
        best_ssm_score_by_stage,
    ) = compute_best_mapping(layer_ids, cka_overall, ssm_overall)

    # Per-image low-agreement cases (using CKA-best mapping).
    per_img_cka = np.stack(all_image_pair_cka, axis=0)  # [N, L, 4]
    per_img_ssm = np.stack(all_image_pair_ssm, axis=0)
    best_cka_idx = [layer_ids.index(x) for x in best_cka_layer_by_stage]
    image_scores = []
    for i in range(per_img_cka.shape[0]):
        vals = []
        for sj in range(4):
            li = best_cka_idx[sj]
            vals.append(0.5 * per_img_cka[i, li, sj] + 0.5 * per_img_ssm[i, li, sj])
        image_scores.append(float(np.mean(vals)))
    image_scores = np.asarray(image_scores, dtype=np.float64)
    low_cases = save_low_agreement_images(
        image_paths=all_image_paths,
        image_scores=image_scores,
        out_dir=qual_dir,
        top_k=3,
    )

    write_results_csv(
        out_csv=output_dir / "results.csv",
        layer_ids=layer_ids,
        cka_overall=cka_overall,
        ssm_overall=ssm_overall,
        cka_by_cond=cka_cond,
        ssm_by_cond=ssm_cond,
        n_overall=n_total,
        n_by_cond=n_cond,
        cka_by_split=cka_split,
        ssm_by_split=ssm_split,
        n_by_split=n_split,
        cka_by_ref=cka_ref,
        ssm_by_ref=ssm_ref,
        n_by_ref=n_ref,
    )

    save_heatmap(
        matrix=cka_overall,
        layer_ids=layer_ids,
        title=f"CKA (overall, N={n_total})",
        out_path=figure_dir / "cka_overall_heatmap.png",
        vmin=float(np.min(cka_overall) - 0.02),
        vmax=float(np.max(cka_overall) + 0.02),
    )
    save_heatmap(
        matrix=ssm_overall,
        layer_ids=layer_ids,
        title=f"SSM similarity (overall, N={n_total})",
        out_path=figure_dir / "ssm_overall_heatmap.png",
        vmin=float(np.min(ssm_overall) - 0.02),
        vmax=float(np.max(ssm_overall) + 0.02),
    )
    save_best_layer_plot(
        layer_ids=layer_ids,
        best_by_cka=best_cka_layer_by_stage,
        best_by_ssm=best_ssm_layer_by_stage,
        out_path=figure_dir / "best_layer_per_stage.png",
    )

    # Summary markdown
    lines: list[str] = []
    lines.append("# DINO Layer ↔ SegFormer Stage Matching")
    lines.append("")
    lines.append("## Preprocessing / Rules")
    lines.append(f"- Dataset: `{acdc_root}`")
    lines.append(f"- Conditions: `{args.conditions}`")
    used_splits = SPLIT_DIRS if bool(args.include_ref) else ["train", "val", "test"]
    lines.append("- Splits used: `" + ", ".join(used_splits) + "`")
    if int(args.square_crop_size) > 0:
        lines.append(
            f"- Resize/Crop: shorter side `{args.resize_short}` -> center crop `{args.square_crop_size}x{args.square_crop_size}` (SegFormer/DINO 공통 입력)"
        )
    else:
        lines.append(
            f"- Resize/Padding: shorter side `{args.resize_short}` then pad to multiple of `{args.pad_multiple}` (SegFormer/DINO 공통 입력)"
        )
    lines.append(f"- Token grid for CKA/SSM: `{args.grid_size}x{args.grid_size}`")
    lines.append(f"  - Downsample: adaptive average pooling, Upsample: nearest(copy, no interpolation)")
    lines.append(f"- Token grid for visualization: `{viz_grid_size}x{viz_grid_size}` (native DINO grid if `--viz_grid_size 0`)")
    lines.append(f"  - Downsample: adaptive average pooling, Upsample: nearest(copy, no interpolation)")
    lines.append("- Label mapping: N/A (feature similarity task, no semantic labels)")
    lines.append("- Resize-crop rule: no random crop/flip, resize + pad only")
    lines.append("")
    lines.append("## Token-Map Visualization Protocol")
    if str(args.viz_anchor_mode) == "quadrant4":
        lines.append("- 1) 앵커 토큰 4개 선택: 4분면 중심 `(1/4,1/4), (3/4,1/4), (1/4,3/4), (3/4,3/4)`")
    else:
        lines.append("- 1) 한 점(앵커 토큰) 선택: `anchor_x/anchor_y` 비율로 grid 위치 선택")
    lines.append("- 원본 입력 시각화: `viz_stream/latest_input_anchors.png`에 앵커 위치를 red-dot + 좌표 라벨(`A1..`)로 표시")
    lines.append("- 2) 코사인 유사도 계산: 같은 레이어의 모든 토큰 `z_q`와 앵커 토큰 `z_p`에 대해 `cos(z_p, z_q)`")
    lines.append("- 3) 2D heatmap: 유사도 벡터를 `[H_p, W_p]`로 reshape 후 min-max normalize, `viridis` colormap 적용")
    lines.append("- 4) 패널 레이아웃: `4x7` (좌측 열=`Seg stage1~4`, 우측 `24개 DINO layer`를 `4x6`으로 배치)")
    lines.append("")
    lines.append("## Data Count")
    lines.append(f"- Total images: **{n_total}**")
    lines.append("- By condition: " + ", ".join(f"`{k}={v}`" for k, v in n_cond.items()))
    lines.append("- By split dir: " + ", ".join(f"`{k}={v}`" for k, v in n_split.items()))
    lines.append("- By ref flag: " + ", ".join(f"`{k}={v}`" for k, v in n_ref.items()))
    lines.append("")
    lines.append("## Best Layer Mapping")
    lines.append("| SegFormer Stage | Best DINO layer (CKA) | CKA | Best DINO layer (SSM) | SSM |")
    lines.append("|---|---:|---:|---:|---:|")
    for sj, stage in enumerate(STAGE_NAMES):
        lines.append(
            f"| {stage} | {best_cka_layer_by_stage[sj]} | {best_cka_score_by_stage[sj]:.6f} "
            f"| {best_ssm_layer_by_stage[sj]} | {best_ssm_score_by_stage[sj]:.6f} |"
        )
    lines.append("")
    lines.append("## Qualitative Low-Agreement Cases (Top 3)")
    if len(low_cases) == 0:
        lines.append("- No case available.")
    else:
        for i, (path, score) in enumerate(low_cases, start=1):
            lines.append(f"- Case {i}: score={score:.6f}, path=`{path}`")
    lines.append("")
    lines.append("## Figures")
    lines.append("- `figures/cka_overall_heatmap.png`")
    lines.append("- `figures/ssm_overall_heatmap.png`")
    lines.append("- `figures/best_layer_per_stage.png`")
    lines.append("- `viz_stream/latest_input_anchors.png` / `viz_stream/input_*.png` (원본 + 레드닷 앵커)")
    lines.append("- `viz_stream/latest.png` (anchor1 최신)")
    lines.append("- `viz_stream/latest_a1.png` ~ `viz_stream/latest_a4.png` (앵커별 최신)")
    lines.append("- `viz_stream/viz_*_a*.png` (주기 저장 스냅샷)")
    lines.append("- `qualitative/low_agreement_01.png` ~ `qualitative/low_agreement_03.png`")
    lines.append("")
    lines.append("## Next Action (1)")
    lines.append(
        "- stage별로 선택된 best DINO layer를 고정해, stage feature alignment loss를 단계별 가중치로 넣은 1회 학습 실험을 수행."
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Saved: summary.md, results.csv, eval.log")
    logger.info("Saved figures in: " + str(figure_dir))
    logger.info(
        "Best mapping (CKA): "
        + ", ".join(f"{s}->{l}" for s, l in zip(STAGE_NAMES, best_cka_layer_by_stage))
    )
    logger.info(
        "Best mapping (SSM): "
        + ", ".join(f"{s}->{l}" for s, l in zip(STAGE_NAMES, best_ssm_layer_by_stage))
    )
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
