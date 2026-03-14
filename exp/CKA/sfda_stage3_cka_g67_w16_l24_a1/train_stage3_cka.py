#!/usr/bin/env python3
"""Train stage3 adapter with local-only CKA + diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from datasets import ACDCDataset  # noqa: E402
from loss_cka_v1 import Stage3CKALoss  # noqa: E402
from model_cka_v1 import Stage3CKAModel  # noqa: E402


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
        print("해결: GPU가 보이는 환경/드라이버/CUDA 상태를 먼저 확인하세요.", file=sys.stderr)
        raise SystemExit(1)

    if not torch.cuda.is_available():
        print("[GPU 체크 실패] torch.cuda.is_available() == False", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print('  python -c "import torch; print(torch.cuda.is_available())"', file=sys.stderr)
        print("해결: conda selo 환경의 CUDA torch 설치/가용성을 확인하세요.", file=sys.stderr)
        raise SystemExit(1)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("stage3_cka_train")
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
    p = argparse.ArgumentParser(description="Train stage3 adapter with local-only CKA")
    p.add_argument("--acdc_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--conditions", nargs="+", default=["fog", "night", "rain", "snow"])

    p.add_argument("--resize", type=int, default=1072)
    p.add_argument("--crop_size", type=int, default=1072)

    p.add_argument("--segformer_model", type=str, default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--dino_model", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--dino_layer", type=int, default=24)
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--adapter_bottleneck", type=int, default=128)
    p.add_argument("--gate_bias_init", type=float, default=-4.0)
    p.add_argument("--delta_out", type=float, default=0.02)
    p.add_argument("--delta_upd", type=float, default=0.01)
    p.add_argument(
        "--use_upd_loss",
        type=int,
        default=1,
        help="1: enable update-energy constraint (loss_upd + lambda_upd dual update), 0: disable completely.",
    )
    p.add_argument("--lambda_out_init", type=float, default=1.0)
    p.add_argument("--lambda_upd_init", type=float, default=1.0)
    p.add_argument("--dual_lr_out", type=float, default=0.05)
    p.add_argument("--dual_lr_upd", type=float, default=0.05)
    p.add_argument("--lambda_max", type=float, default=10.0)
    p.add_argument("--gate_detach_align", type=int, default=1)
    p.add_argument("--lambda_select", type=float, default=0.0)
    p.add_argument("--select_score_norm", type=int, default=1)
    p.add_argument("--anchor_conf_gamma", type=float, default=1.0)
    p.add_argument(
        "--anchor_conf_thresh",
        type=float,
        default=0.93,
        help="Only pixels with baseline confidence >= threshold are anchored by L_out.",
    )
    p.add_argument("--anchor_temperature", type=float, default=1.0)
    p.add_argument("--force_gate_one", type=int, default=0, help="1: use constant gate g=1 everywhere.")

    p.add_argument("--local_window_size", type=int, default=16)
    p.add_argument("--local_windows_total", type=int, default=10)
    p.add_argument("--local_windows_per_step", type=int, default=10)
    p.add_argument("--boundary_ratio_local", type=float, default=0.6)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=0, help="<=0 means full epochs")

    p.add_argument(
        "--overfit_one_batch",
        type=int,
        default=0,
        help="1: capture first micro-batch once and reuse it for all training steps.",
    )
    p.add_argument(
        "--overfit_fixed_sampling",
        type=int,
        default=0,
        help="1: fix local-window sampling indices across steps (debug/overfit).",
    )
    p.add_argument(
        "--strict_dino_resolution",
        type=int,
        default=1,
        help="1: DINO alignment must keep same resolution",
    )

    p.add_argument("--diag_heavy_interval", type=int, default=50)
    p.add_argument("--diag_den_warn", type=float, default=1e-4)
    p.add_argument("--diag_den_critical", type=float, default=1e-6)
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--eval_every_epoch", type=int, default=1)
    p.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--eval_test_gt_dir", type=str, default="")
    p.add_argument("--eval_resize", type=int, default=1080)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--eval_workers", type=int, default=0)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _denorm_image_uint8(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1)
    return (y.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def _stats_from_array(arr: np.ndarray) -> dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": 0.0,
            "p01": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1.0)),
        "p10": float(np.percentile(arr, 10.0)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _append_csv(path: Path, header: list[str], row: list[object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def _flatten_grads(grads: list[torch.Tensor | None], device: torch.device) -> torch.Tensor:
    flat = []
    for g in grads:
        if g is None:
            continue
        flat.append(g.reshape(-1))
    if not flat:
        return torch.zeros((1,), device=device, dtype=torch.float32)
    return torch.cat(flat, dim=0).detach().float()


def _save_matrix_png(mat: np.ndarray, title: str, out_path: Path, cmap: str = "coolwarm", vmin: float | None = None, vmax: float | None = None) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_heatmap_overlay(
    image_uint8: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    out_path: Path,
    *,
    cmap: str = "magma",
    alpha: float = 0.45,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    hm = np.nan_to_num(heatmap.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(image_uint8)
    im = ax.imshow(hm, cmap=cmap, alpha=float(alpha), vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_hist(values: np.ndarray, title: str, out_path: Path, bins: int = 20) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)
    values = values.astype(np.float64, copy=False)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    try:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isfinite(vmin) and np.isfinite(vmax) and (vmax > vmin):
            ax.hist(values, bins=bins)
        else:
            ax.hist(values, bins=1)
    except ValueError:
        # Extremely wide ranges can fail in numpy histogram; use log-space fallback.
        logv = np.log10(np.clip(values, 1e-30, None))
        logv = logv[np.isfinite(logv)]
        if logv.size == 0:
            logv = np.array([0.0], dtype=np.float64)
        ax.hist(logv, bins=min(int(bins), max(1, int(logv.size))))
        title = f"{title} (log10 fallback)"
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_sorted(values: np.ndarray, title: str, out_path: Path) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.array([0.0], dtype=np.float32)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    v = np.sort(values)
    ax.plot(np.arange(v.shape[0]), v)
    ax.set_title(title)
    ax.set_xlabel("sorted index")
    ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _window_values_to_map(values: np.ndarray, indices: np.ndarray, h_win: int, w_win: int) -> np.ndarray:
    out = np.full((h_win, w_win), np.nan, dtype=np.float32)
    for i, lin in enumerate(indices.tolist()):
        y = int(lin) // int(w_win)
        x = int(lin) % int(w_win)
        v = float(values[i])
        out[y, x] = v if np.isfinite(v) else np.nan
    return out


def _save_window_overlay(
    image_uint8: np.ndarray,
    indices: np.ndarray,
    den_values: np.ndarray,
    den_warn: float,
    stage3_h: int,
    stage3_w: int,
    window_size: int,
    h_win: int,
    w_win: int,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(image_uint8)
    ax.set_title("Input Overlay (red: low-den windows)")
    ax.axis("off")

    img_h, img_w = image_uint8.shape[:2]
    scale_x = float(img_w) / float(stage3_w)
    scale_y = float(img_h) / float(stage3_h)

    for i, lin in enumerate(indices.tolist()):
        wy = int(lin) // int(w_win)
        wx = int(lin) % int(w_win)
        px = wx * scale_x
        py = wy * scale_y
        pw = float(window_size) * scale_x
        ph = float(window_size) * scale_y
        den_v = float(den_values[i])
        low = bool(np.isfinite(den_v) and den_v < float(den_warn))
        color = "red" if low else "yellow"
        rect = patches.Rectangle((px, py), pw, ph, fill=False, linewidth=2.0, edgecolor=color)
        ax.add_patch(rect)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_timeline(x: list[int], y: list[float], title: str, ylabel: str, out_path: Path) -> None:
    if not x:
        return
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel("update_step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_multi_timeline(
    x: list[int],
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    if not x or not series:
        return
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    plotted = 0
    for name, y in series.items():
        if len(y) != len(x):
            continue
        ax.plot(x, y, label=name)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("update_step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted >= 2:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _selected_token_gate_stats(
    gate_map: torch.Tensor,
    use_win_idx: torch.Tensor,
    window_size: int,
) -> tuple[float, float, float]:
    # gate_map: [B,1,H,W], use_win_idx: [B,S]
    gate = gate_map.detach().float()
    idx = use_win_idx.detach().long()
    ws = int(window_size)
    unfolded = F.unfold(gate, kernel_size=ws, stride=1)  # [B, ws^2, L] since C=1
    idx_exp = idx.unsqueeze(1).expand(-1, unfolded.shape[1], -1)
    selected = torch.gather(unfolded, dim=2, index=idx_exp).reshape(-1)
    if selected.numel() == 0:
        return 0.0, 0.0, 0.0
    mean_v = float(selected.mean().item())
    p10_v = float(torch.quantile(selected, q=0.10).item())
    p90_v = float(torch.quantile(selected, q=0.90).item())
    return mean_v, p10_v, p90_v


def _save_heavy_diagnostics(
    *,
    step_dir: Path,
    trainable_params: list[torch.nn.Parameter],
    cka_local_each: torch.Tensor,
    nx_each: torch.Tensor,
    ny_each: torch.Tensor,
    den_each: torch.Tensor,
    use_win_idx: torch.Tensor,
    gate_map: torch.Tensor,
    gate_align_map: torch.Tensor,
    h_win: int,
    w_win: int,
    image_tensor: torch.Tensor,
    stage3_h: int,
    stage3_w: int,
    window_size: int,
    den_warn: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    step_dir.mkdir(parents=True, exist_ok=True)

    # Per-window scalar loss for gradient conflict: average over batch axis.
    per_window_loss = 1.0 - cka_local_each.mean(dim=0)  # [S]
    num_w = int(per_window_loss.shape[0])

    grad_vecs = []
    for wi in range(num_w):
        grads = torch.autograd.grad(
            per_window_loss[wi],
            trainable_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        grad_vecs.append(_flatten_grads(list(grads), device=per_window_loss.device))

    grad_mat = torch.stack(grad_vecs, dim=0)  # [S, D]
    grad_norms = grad_mat.norm(dim=1)

    unit = grad_mat / grad_norms.unsqueeze(1).clamp_min(float(eps))
    cos_mat = unit @ unit.transpose(0, 1)

    off_mask = ~torch.eye(num_w, device=cos_mat.device, dtype=torch.bool)
    off_vals = cos_mat[off_mask]
    neg_ratio = float((off_vals < 0.0).float().mean().item()) if off_vals.numel() > 0 else 0.0

    sum_grad = grad_mat.sum(dim=0)
    cancellation_ratio = float(sum_grad.norm().item() / grad_norms.sum().clamp_min(float(eps)).item())

    cka_w = cka_local_each.mean(dim=0).detach().float().cpu().numpy()
    nx_w = nx_each.mean(dim=0).detach().float().cpu().numpy()
    ny_w = ny_each.mean(dim=0).detach().float().cpu().numpy()
    den_w = den_each.mean(dim=0).detach().float().cpu().numpy()
    grad_norms_np = grad_norms.detach().float().cpu().numpy()
    gate_np = gate_map[0, 0].detach().float().cpu().numpy()
    gate_align_np = gate_align_map[0, 0].detach().float().cpu().numpy()

    idx0 = use_win_idx[0].detach().cpu().numpy().astype(np.int64)
    den_map = _window_values_to_map(den_w, idx0, int(h_win), int(w_win))
    cka_map = _window_values_to_map(cka_w, idx0, int(h_win), int(w_win))
    ws = int(window_size)
    gate_unfold = F.unfold(gate_map.detach().float(), kernel_size=ws, stride=1)  # [B,ws^2,L]
    gate_align_unfold = F.unfold(gate_align_map.detach().float(), kernel_size=ws, stride=1)
    idx_exp = use_win_idx.detach().long().unsqueeze(1).expand(-1, gate_unfold.shape[1], -1)
    gate_sel_mean = torch.gather(gate_unfold, dim=2, index=idx_exp).mean(dim=1)[0].detach().cpu().numpy()
    gate_align_sel_mean = torch.gather(gate_align_unfold, dim=2, index=idx_exp).mean(dim=1)[0].detach().cpu().numpy()
    gate_sel_map = _window_values_to_map(gate_sel_mean, idx0, int(h_win), int(w_win))
    gate_align_sel_map = _window_values_to_map(gate_align_sel_mean, idx0, int(h_win), int(w_win))
    image_uint8 = _denorm_image_uint8(image_tensor)

    # numeric artifacts
    with (step_dir / "window_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["window_idx", "lin_idx", "grid_y", "grid_x", "cka", "nx", "ny", "den", "grad_norm"])
        for i in range(num_w):
            lin = int(idx0[i])
            gy = lin // int(w_win)
            gx = lin % int(w_win)
            w.writerow([
                i,
                lin,
                gy,
                gx,
                float(cka_w[i]),
                float(nx_w[i]),
                float(ny_w[i]),
                float(den_w[i]),
                float(grad_norms_np[i]),
            ])

    with (step_dir / "cosine_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["window_idx"] + [f"w{j}" for j in range(num_w)]
        w.writerow(header)
        cos_np = cos_mat.detach().float().cpu().numpy()
        for i in range(num_w):
            w.writerow([f"w{i}"] + [float(cos_np[i, j]) for j in range(num_w)])

    metrics = {
        "num_windows": num_w,
        "neg_cosine_ratio": float(neg_ratio),
        "cancellation_ratio": float(cancellation_ratio),
        "den_min": float(_stats_from_array(den_w)["min"]),
        "den_mean": float(_stats_from_array(den_w)["mean"]),
        "cka_mean": float(_stats_from_array(cka_w)["mean"]),
        "den_warn_threshold": float(den_warn),
    }
    (step_dir / "grad_conflict.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # visual artifacts
    _save_matrix_png(
        cos_mat.detach().float().cpu().numpy(),
        title="Window Gradient Cosine Matrix",
        out_path=step_dir / "cosine_matrix.png",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(num_w), grad_norms_np)
    ax.set_title("Per-window Gradient Norm")
    ax.set_xlabel("window_idx")
    ax.set_ylabel("||g_i||")
    fig.tight_layout()
    fig.savefig(step_dir / "grad_norms.png", dpi=150)
    plt.close(fig)

    _save_hist(den_w, "Denominator Histogram", step_dir / "den_hist.png")
    _save_sorted(den_w, "Denominator Sorted", step_dir / "den_sorted.png")
    _save_sorted(cka_w, "Local CKA Sorted", step_dir / "cka_sorted.png")

    _save_matrix_png(den_map, "Denominator Map (window grid)", step_dir / "den_map.png", cmap="viridis")
    _save_matrix_png(cka_map, "Local CKA Map (window grid)", step_dir / "cka_map.png", cmap="viridis")
    _save_matrix_png(gate_np, "Gate Heatmap (g)", step_dir / "gate_heatmap.png", cmap="magma", vmin=0.0, vmax=1.0)
    _save_matrix_png(
        gate_align_np,
        "Gate Align Heatmap (used by L_align)",
        step_dir / "gate_align_heatmap.png",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    _save_matrix_png(
        gate_sel_map,
        "Selected Window Gate Mean (g)",
        step_dir / "gate_selected_map.png",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    _save_matrix_png(
        gate_align_sel_map,
        "Selected Window Gate Mean (g_align)",
        step_dir / "gate_align_selected_map.png",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    _save_heatmap_overlay(
        image_uint8=image_uint8,
        heatmap=gate_np,
        title="Gate Overlay (g)",
        out_path=step_dir / "gate_overlay.png",
        cmap="magma",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )
    _save_heatmap_overlay(
        image_uint8=image_uint8,
        heatmap=gate_align_np,
        title="Gate Align Overlay (g_align)",
        out_path=step_dir / "gate_align_overlay.png",
        cmap="magma",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )

    _save_window_overlay(
        image_uint8=image_uint8,
        indices=idx0,
        den_values=den_w,
        den_warn=float(den_warn),
        stage3_h=int(stage3_h),
        stage3_w=int(stage3_w),
        window_size=int(window_size),
        h_win=int(h_win),
        w_win=int(w_win),
        out_path=step_dir / "input_windows.png",
    )

    return metrics


def _build_ckpt(
    *,
    model: Stage3CKAModel,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    global_step: int,
    update_step: int,
    epoch: int,
    lambda_out: float,
    lambda_upd: float,
) -> dict[str, object]:
    return {
        "stage3_adapter": model.stage3_adapter.state_dict(),
        "stage3_gate": model.stage3_gate.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "epoch": int(epoch),
        "lambda_out": float(lambda_out),
        "lambda_upd": float(lambda_upd),
        "args": vars(args),
    }


def _save_ckpt(
    *,
    path: Path,
    model: Stage3CKAModel,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    global_step: int,
    update_step: int,
    epoch: int,
    lambda_out: float,
    lambda_upd: float,
    logger: logging.Logger,
) -> None:
    ckpt = _build_ckpt(
        model=model,
        args=args,
        optimizer=optimizer,
        scaler=scaler,
        global_step=global_step,
        update_step=update_step,
        epoch=epoch,
        lambda_out=lambda_out,
        lambda_upd=lambda_upd,
    )
    torch.save(ckpt, path)
    logger.info("Saved checkpoint: %s", str(path))


def _run_epoch_eval(
    *,
    args: argparse.Namespace,
    ckpt_path: Path,
    out_dir: Path,
    epoch: int,
    logger: logging.Logger,
) -> dict[str, float] | None:
    eval_out_dir = out_dir / "eval_by_epoch" / f"epoch_{epoch + 1:03d}"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    eval_script = Path(__file__).resolve().parent / "eval_stage3_cka.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--acdc_root",
        str(args.acdc_root),
        "--split",
        str(args.eval_split),
        "--conditions",
        *list(args.conditions),
        "--resize",
        str(int(args.eval_resize)),
        "--batch_size",
        str(int(args.eval_batch_size)),
        "--workers",
        str(int(args.eval_workers)),
        "--segformer_model",
        str(args.segformer_model),
        "--dino_model",
        str(args.dino_model),
        "--dino_layer",
        str(int(args.dino_layer)),
        "--num_classes",
        str(int(args.num_classes)),
        "--adapter_bottleneck",
        str(int(args.adapter_bottleneck)),
        "--gate_bias_init",
        str(float(args.gate_bias_init)),
        "--ckpt",
        str(ckpt_path),
        "--output_dir",
        str(eval_out_dir),
    ]
    if str(args.eval_split) == "test" and str(args.eval_test_gt_dir):
        cmd.extend(["--test_gt_dir", str(args.eval_test_gt_dir)])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("Epoch eval failed (epoch=%d, code=%d)", epoch, proc.returncode)
        if proc.stdout.strip():
            logger.error("[eval stdout]\n%s", proc.stdout.strip())
        if proc.stderr.strip():
            logger.error("[eval stderr]\n%s", proc.stderr.strip())
        return None

    metrics_path = eval_out_dir / "eval_metrics.json"
    if not metrics_path.exists():
        logger.warning("Epoch eval metrics missing: %s", str(metrics_path))
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse eval metrics (%s): %s", str(metrics_path), str(exc))
        return None
    return {
        "overall_base": float(metrics.get("overall_base", float("nan"))),
        "overall_adapt": float(metrics.get("overall_adapt", float("nan"))),
        "overall_delta_abs": float(metrics.get("overall_delta_abs", float("nan"))),
        "overall_delta_rel_pct": float(metrics.get("overall_delta_rel_pct", float("nan"))),
    }


def main() -> int:
    args = parse_args()
    check_gpu_or_exit()

    _set_seed(int(args.seed))
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "train.log")
    logger.info("Args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    diag_dir = out_dir / "diagnostics"
    heavy_dir = diag_dir / "heavy"
    plots_dir = diag_dir / "plots"
    diag_dir.mkdir(parents=True, exist_ok=True)
    heavy_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    den_csv = diag_dir / "den_stats.csv"
    cka_csv = diag_dir / "local_cka_stats.csv"
    win_csv = diag_dir / "window_sampling_stats.csv"

    dataset = ACDCDataset(
        root=args.acdc_root,
        split="train",
        conditions=args.conditions,
        resize=int(args.resize),
        crop_size=(int(args.crop_size), int(args.crop_size)),
    )
    if len(dataset) == 0:
        logger.error("No train images found in ACDC.")
        return 1

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True,
        drop_last=True,
    )
    if len(loader) == 0:
        logger.error("Train dataloader is empty (batch_size too large or dataset too small).")
        return 1

    overfit_one_batch = bool(int(args.overfit_one_batch))
    fixed_batch = None
    if overfit_one_batch:
        fixed_batch = next(iter(loader))
        fixed_paths = fixed_batch.get("path", "unknown")
        if isinstance(fixed_paths, list):
            fixed_paths_msg = ", ".join(str(p) for p in fixed_paths)
        else:
            fixed_paths_msg = str(fixed_paths)
        logger.info("Overfit-one-batch enabled. Reusing fixed batch: %s", fixed_paths_msg)

    device = torch.device("cuda")
    model = Stage3CKAModel(
        segformer_model=args.segformer_model,
        dino_model=args.dino_model,
        dino_layer=int(args.dino_layer),
        num_classes=int(args.num_classes),
        adapter_bottleneck=int(args.adapter_bottleneck),
        gate_bias_init=float(args.gate_bias_init),
        enable_dino=True,
        force_gate_one=bool(int(args.force_gate_one)),
    ).to(device)
    model.freeze_backbone()
    model.freeze_dino()

    force_gate_one = bool(int(args.force_gate_one))
    for p in model.parameters():
        p.requires_grad = False
    for p in model.stage3_adapter.parameters():
        p.requires_grad = True
    if not force_gate_one:
        for p in model.stage3_gate.parameters():
            p.requires_grad = True
    else:
        logger.info("force_gate_one=1 enabled: stage3_gate is bypassed with constant ones.")

    loss_fn = Stage3CKALoss(
        local_window_size=int(args.local_window_size),
        local_windows_total=int(args.local_windows_total),
        local_windows_per_step=int(args.local_windows_per_step),
        boundary_ratio_local=float(args.boundary_ratio_local),
        overfit_fixed_sampling=bool(int(args.overfit_fixed_sampling)),
    )

    param_groups = [{"params": model.stage3_adapter.parameters()}]
    if not force_gate_one:
        param_groups.append({"params": model.stage3_gate.parameters()})
    optimizer = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = GradScaler(device="cuda", enabled=bool(args.amp))
    accum = max(1, int(args.grad_accum_steps))

    total_steps_full = int(args.epochs) * len(loader)
    total_steps = total_steps_full if int(args.max_steps) <= 0 else min(total_steps_full, int(args.max_steps))
    logger.info("Dataset size=%d, steps_per_epoch=%d, total_steps=%d", len(dataset), len(loader), total_steps)

    global_step = 0
    update_step = 0
    start_epoch = 0
    lambda_out = float(args.lambda_out_init)
    lambda_upd = float(args.lambda_upd_init)
    use_upd_loss = bool(int(args.use_upd_loss))
    if not use_upd_loss:
        lambda_upd = 0.0
    if str(args.resume_ckpt):
        resume_path = Path(str(args.resume_ckpt))
        if not resume_path.exists():
            logger.error("resume_ckpt not found: %s", str(resume_path))
            return 1
        ckpt = torch.load(resume_path, map_location="cpu")
        model.stage3_adapter.load_state_dict(ckpt["stage3_adapter"], strict=False)
        model.stage3_gate.load_state_dict(ckpt["stage3_gate"], strict=True)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as e:
                logger.warning("Skip optimizer state load due to param-group mismatch: %s", str(e))
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = int(ckpt.get("global_step", 0))
        update_step = int(ckpt.get("update_step", 0))
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        lambda_out = float(ckpt.get("lambda_out", lambda_out))
        lambda_upd = float(ckpt.get("lambda_upd", lambda_upd))
        if not use_upd_loss:
            lambda_upd = 0.0
        logger.info(
            "Resumed from %s (start_epoch=%d, global_step=%d, update_step=%d, lambda_out=%.6f, lambda_upd=%.6f)",
            str(resume_path),
            start_epoch,
            global_step,
            update_step,
            lambda_out,
            lambda_upd,
        )

    started_at = time.time()
    model.train()
    # Frozen backbone/decoder(DINO 포함) dropout/확률층 노이즈를 막아 KL anchor 안정화.
    model.backbone.eval()
    model.freeze_dino()
    model.stage3_adapter.train()
    if not force_gate_one:
        model.stage3_gate.train()
    optimizer.zero_grad(set_to_none=True)

    hist_updates: list[int] = []
    hist_den_min: list[float] = []
    hist_den_small: list[float] = []
    hist_cka: list[float] = []
    hist_gate_mean: list[float] = []
    hist_gate_align_mean: list[float] = []
    hist_gate_local_mean: list[float] = []
    hist_gate_local_p10: list[float] = []
    hist_gate_local_p90: list[float] = []
    hist_gate_local_ratio: list[float] = []
    hist_delta_rms: list[float] = []
    hist_update_rms: list[float] = []
    hist_upd_over_delta: list[float] = []
    hist_select_score: list[float] = []

    update_loss_total_vals: list[float] = []
    update_loss_align_vals: list[float] = []
    update_loss_out_vals: list[float] = []
    update_loss_upd_vals: list[float] = []
    update_loss_select_vals: list[float] = []
    update_cka_vals: list[float] = []
    update_select_score_vals: list[float] = []
    update_div_mean_vals: list[float] = []
    update_out_metric_vals: list[float] = []
    update_upd_metric_vals: list[float] = []
    update_out_violation_vals: list[float] = []
    update_upd_violation_vals: list[float] = []
    update_den_vals: list[np.ndarray] = []
    update_nx_vals: list[np.ndarray] = []
    update_ny_vals: list[np.ndarray] = []
    update_win_iou_mean_vals: list[float] = []
    update_win_iou_max_vals: list[float] = []
    update_win_cov_vals: list[float] = []
    update_win_score_min_vals: list[float] = []
    update_win_score_med_vals: list[float] = []
    update_win_score_max_vals: list[float] = []
    update_gate_mean_vals: list[float] = []
    update_gate_align_mean_vals: list[float] = []
    update_gate_align_diff_vals: list[float] = []
    update_gate_align_absdiff_vals: list[float] = []
    update_gate_max_vals: list[float] = []
    update_delta_rms_vals: list[float] = []
    update_update_rms_vals: list[float] = []
    update_upd_over_delta_vals: list[float] = []
    update_gate_local_mean_vals: list[float] = []
    update_gate_local_p10_vals: list[float] = []
    update_gate_local_p90_vals: list[float] = []
    update_gate_local_ratio_vals: list[float] = []

    stop_training = False
    last_epoch_ran = start_epoch - 1
    eps = 1e-6
    for epoch in range(start_epoch, int(args.epochs)):
        if overfit_one_batch:
            assert fixed_batch is not None
            iterator = ((i, fixed_batch) for i in range(len(loader)))
            pbar = tqdm(iterator, total=len(loader), desc=f"Epoch {epoch} [overfit1]", file=sys.stdout)
        else:
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}", file=sys.stdout)

        for step, batch in pbar:
            if global_step >= total_steps:
                stop_training = True
                break

            images = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp)):
                    out_base = model(
                        images,
                        adapter_enabled=False,
                        return_intermediates=True,
                        use_dino=False,
                        compute_logits=True,
                        need_stage4_anchor=False,
                        strict_dino_resolution=bool(int(args.strict_dino_resolution)),
                    )
                logits0 = out_base["logits"]
                temp_mask = max(float(args.anchor_temperature), eps)
                p0_mask = torch.softmax(logits0.float() / temp_mask, dim=1)
                conf_raw_mask = p0_mask.max(dim=1).values
                # hdelta all-allow: do not freeze high-confidence pixels.
                update_mask = torch.ones_like(conf_raw_mask).unsqueeze(1)
                if update_mask.shape[-2:] != out_base["stage3_raw"].shape[-2:]:
                    update_mask = F.interpolate(
                        update_mask,
                        size=out_base["stage3_raw"].shape[-2:],
                        mode="nearest",
                    )

            with autocast(device_type="cuda", dtype=torch.float16, enabled=bool(args.amp)):
                out = model(
                    images,
                    adapter_enabled=True,
                    return_intermediates=True,
                    gate_detach_for_align=bool(int(args.gate_detach_align)),
                    update_mask=update_mask,
                    use_dino=True,
                    compute_logits=True,
                    need_stage4_anchor=False,
                    strict_dino_resolution=bool(int(args.strict_dino_resolution)),
                )

            # Keep Local CKA numerics in FP32: computing Gram/denominator under AMP can overflow.
            with autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                loss_stats = loss_fn(out["stage3_align"].float(), out["dino_feat"].float(), global_step=global_step)
                loss_align = loss_stats["loss_local"]

                temp = max(float(args.anchor_temperature), eps)
                logits0_f = logits0.float() / temp
                logits1_f = out["logits"].float() / temp
                p0 = torch.softmax(logits0_f, dim=1)
                conf_raw = p0.max(dim=1).values
                gamma = float(args.anchor_conf_gamma)
                conf = conf_raw
                if abs(gamma - 1.0) > 1e-8:
                    conf = conf.clamp_min(eps).pow(gamma)
                anchor_mask = (conf_raw >= float(args.anchor_conf_thresh)).float()
                conf = conf * anchor_mask
                log_p0 = torch.log(p0.clamp_min(eps))
                log_p1 = torch.log_softmax(logits1_f, dim=1)
                kl_map = (p0 * (log_p0 - log_p1)).sum(dim=1)
                out_metric = (conf * kl_map).sum() / (conf.sum() + eps)

                out_violation = torch.relu(out_metric - float(args.delta_out))
                loss_out = out_violation.square()
                if use_upd_loss:
                    raw_energy = out["stage3_raw"].float().square().mean().detach()
                    upd_metric = out["update"].float().square().mean() / (raw_energy + eps)
                    upd_violation = torch.relu(upd_metric - float(args.delta_upd))
                    loss_upd = upd_violation.square()
                else:
                    upd_metric = loss_align.new_zeros(())
                    upd_violation = loss_align.new_zeros(())
                    loss_upd = loss_align.new_zeros(())
                # Select objective disabled: no uncertainty/select feedback in optimization path.
                select_score = loss_align.new_zeros(())
                loss_select = loss_align.new_zeros(())

                total_loss = (
                    loss_align
                    + float(lambda_out) * loss_out
                    + float(lambda_upd) * loss_upd
                )
                loss_stats["total_loss"] = total_loss

            den_each = loss_stats["den_each"].detach().double().reshape(-1).cpu().numpy()
            nx_each = loss_stats["nx_each"].detach().double().reshape(-1).cpu().numpy()
            ny_each = loss_stats["ny_each"].detach().double().reshape(-1).cpu().numpy()
            cka_each_np = loss_stats["cka_local_each"].detach().float().reshape(-1).cpu().numpy()

            den_min_micro = float(np.min(den_each)) if den_each.size > 0 else 0.0
            den_small_micro = int(np.sum(den_each < float(args.diag_den_warn)))
            den_critical_micro = int(np.sum(den_each < float(args.diag_den_critical)))
            win_iou_mean_micro = float(loss_stats["win_iou_mean"].detach().item())
            win_iou_max_micro = float(loss_stats["win_iou_max"].detach().item())
            win_cov_micro = float(loss_stats["win_coverage_ratio"].detach().item())
            win_score_min_micro = float(loss_stats["win_score_min"].detach().item())
            win_score_med_micro = float(loss_stats["win_score_median"].detach().item())
            win_score_max_micro = float(loss_stats["win_score_max"].detach().item())
            gate = out["gate"].detach().float()
            gate_align = out["gate_align"].detach().float()
            delta_map = out["delta"].detach().float()
            update_map = out["update"].detach().float()
            gate_mean_micro = float(gate.mean().item())
            gate_align_mean_micro = float(gate_align.mean().item())
            gate_align_diff_micro = float(gate_align_mean_micro - gate_mean_micro)
            gate_align_absdiff_micro = float(abs(gate_align_diff_micro))
            gate_max_micro = float(gate.max().item())
            delta_rms_micro = float(delta_map.square().mean().sqrt().item())
            update_rms_micro = float(update_map.square().mean().sqrt().item())
            upd_over_delta_micro = float(update_rms_micro / (delta_rms_micro + eps))
            gate_local_mean_micro, gate_local_p10_micro, gate_local_p90_micro = _selected_token_gate_stats(
                gate_map=gate_align,
                use_win_idx=loss_stats["use_win_idx"],
                window_size=int(args.local_window_size),
            )
            gate_local_ratio_micro = float(gate_local_mean_micro / (gate_align_mean_micro + eps))

            update_loss_total_vals.append(float(total_loss.detach().item()))
            update_loss_align_vals.append(float(loss_align.detach().item()))
            update_loss_out_vals.append(float(loss_out.detach().item()))
            update_loss_upd_vals.append(float(loss_upd.detach().item()))
            update_loss_select_vals.append(float(loss_select.detach().item()))
            update_cka_vals.append(float(np.mean(cka_each_np)) if cka_each_np.size > 0 else 0.0)
            update_select_score_vals.append(float(select_score.detach().item()))
            update_div_mean_vals.append(float(loss_stats["div_map_mean"].detach().item()))
            update_out_metric_vals.append(float(out_metric.detach().item()))
            update_upd_metric_vals.append(float(upd_metric.detach().item()))
            update_out_violation_vals.append(float(out_violation.detach().item()))
            update_upd_violation_vals.append(float(upd_violation.detach().item()))
            update_den_vals.append(den_each)
            update_nx_vals.append(nx_each)
            update_ny_vals.append(ny_each)
            update_win_iou_mean_vals.append(win_iou_mean_micro)
            update_win_iou_max_vals.append(win_iou_max_micro)
            update_win_cov_vals.append(win_cov_micro)
            update_win_score_min_vals.append(win_score_min_micro)
            update_win_score_med_vals.append(win_score_med_micro)
            update_win_score_max_vals.append(win_score_max_micro)
            update_gate_mean_vals.append(gate_mean_micro)
            update_gate_align_mean_vals.append(gate_align_mean_micro)
            update_gate_align_diff_vals.append(gate_align_diff_micro)
            update_gate_align_absdiff_vals.append(gate_align_absdiff_micro)
            update_gate_max_vals.append(gate_max_micro)
            update_delta_rms_vals.append(delta_rms_micro)
            update_update_rms_vals.append(update_rms_micro)
            update_upd_over_delta_vals.append(upd_over_delta_micro)
            update_gate_local_mean_vals.append(gate_local_mean_micro)
            update_gate_local_p10_vals.append(gate_local_p10_micro)
            update_gate_local_p90_vals.append(gate_local_p90_micro)
            update_gate_local_ratio_vals.append(gate_local_ratio_micro)

            is_last_step = (global_step + 1) >= total_steps
            is_update_boundary = ((global_step + 1) % accum == 0) or is_last_step

            heavy_due = False
            next_update_step = update_step + 1
            if is_update_boundary:
                heavy_due = (
                    next_update_step % max(1, int(args.diag_heavy_interval)) == 0
                    or den_min_micro < float(args.diag_den_warn)
                )

            if heavy_due:
                step_dir = heavy_dir / f"step_{global_step:07d}"
                heavy_metrics = _save_heavy_diagnostics(
                    step_dir=step_dir,
                    trainable_params=[p for p in model.parameters() if p.requires_grad],
                    cka_local_each=loss_stats["cka_local_each"],
                    nx_each=loss_stats["nx_each"],
                    ny_each=loss_stats["ny_each"],
                    den_each=loss_stats["den_each"],
                    use_win_idx=loss_stats["use_win_idx"],
                    gate_map=out["gate"],
                    gate_align_map=out["gate_align"],
                    h_win=int(loss_stats["h_win"].item()),
                    w_win=int(loss_stats["w_win"].item()),
                    image_tensor=images[0],
                    stage3_h=int(out["stage3_adapt"].shape[-2]),
                    stage3_w=int(out["stage3_adapt"].shape[-1]),
                    window_size=int(args.local_window_size),
                    den_warn=float(args.diag_den_warn),
                )
                logger.info(
                    "HEAVY e=%d us=%d gs=%d loss=%.6f align=%.6f cka_local=%.6f out_metric=%.6f upd_metric=%.6f lambda_out=%.6f lambda_upd=%.6f out_violation=%.6f upd_violation=%.6f delta_rms=%.6f neg_cosine_ratio=%.6f cancellation_ratio=%.6f",
                    epoch,
                    next_update_step,
                    global_step,
                    float(total_loss.detach().item()),
                    float(loss_align.detach().item()),
                    float(loss_stats["cka_local"].item()),
                    float(out_metric.detach().item()),
                    float(upd_metric.detach().item()),
                    lambda_out,
                    lambda_upd,
                    float(out_violation.detach().item()),
                    float(upd_violation.detach().item()),
                    delta_rms_micro,
                    heavy_metrics["neg_cosine_ratio"],
                    heavy_metrics["cancellation_ratio"],
                )

            scaled = total_loss / float(accum)
            scaler.scale(scaled).backward()

            if is_update_boundary:
                scaler.unscale_(optimizer)
                trainable = list(model.stage3_adapter.parameters())
                if not force_gate_one:
                    trainable.extend(list(model.stage3_gate.parameters()))
                torch.nn.utils.clip_grad_norm_(trainable, float(args.max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                out_metric_avg = float(np.mean(update_out_metric_vals)) if update_out_metric_vals else 0.0
                upd_metric_avg = float(np.mean(update_upd_metric_vals)) if update_upd_metric_vals else 0.0
                lambda_out = float(
                    np.clip(
                        lambda_out + float(args.dual_lr_out) * (out_metric_avg - float(args.delta_out)),
                        0.0,
                        float(args.lambda_max),
                    )
                )
                if use_upd_loss:
                    lambda_upd = float(
                        np.clip(
                            lambda_upd + float(args.dual_lr_upd) * (upd_metric_avg - float(args.delta_upd)),
                            0.0,
                            float(args.lambda_max),
                        )
                    )
                else:
                    lambda_upd = 0.0

                den_concat = np.concatenate(update_den_vals, axis=0) if update_den_vals else np.zeros((0,), dtype=np.float32)
                nx_concat = np.concatenate(update_nx_vals, axis=0) if update_nx_vals else np.zeros((0,), dtype=np.float32)
                ny_concat = np.concatenate(update_ny_vals, axis=0) if update_ny_vals else np.zeros((0,), dtype=np.float32)

                den_stats = _stats_from_array(den_concat)
                cka_stats = _stats_from_array(np.asarray(update_cka_vals, dtype=np.float32))
                loss_total_mean = float(np.mean(update_loss_total_vals)) if update_loss_total_vals else 0.0
                loss_align_mean = float(np.mean(update_loss_align_vals)) if update_loss_align_vals else 0.0
                loss_out_mean = float(np.mean(update_loss_out_vals)) if update_loss_out_vals else 0.0
                loss_upd_mean = float(np.mean(update_loss_upd_vals)) if update_loss_upd_vals else 0.0
                loss_select_mean = float(np.mean(update_loss_select_vals)) if update_loss_select_vals else 0.0
                select_score_mean = float(np.mean(update_select_score_vals)) if update_select_score_vals else 0.0
                div_mean = float(np.mean(update_div_mean_vals)) if update_div_mean_vals else 0.0
                out_violation_mean = float(np.mean(update_out_violation_vals)) if update_out_violation_vals else 0.0
                upd_violation_mean = float(np.mean(update_upd_violation_vals)) if update_upd_violation_vals else 0.0
                den_small_count = int(np.sum(den_concat < float(args.diag_den_warn))) if den_concat.size > 0 else 0
                den_critical_count = int(np.sum(den_concat < float(args.diag_den_critical))) if den_concat.size > 0 else 0
                nx_min = float(np.min(nx_concat)) if nx_concat.size > 0 else 0.0
                ny_min = float(np.min(ny_concat)) if ny_concat.size > 0 else 0.0
                win_iou_mean = float(np.mean(update_win_iou_mean_vals)) if update_win_iou_mean_vals else 0.0
                win_iou_max = float(np.mean(update_win_iou_max_vals)) if update_win_iou_max_vals else 0.0
                win_cov = float(np.mean(update_win_cov_vals)) if update_win_cov_vals else 0.0
                win_score_min = float(np.mean(update_win_score_min_vals)) if update_win_score_min_vals else 0.0
                win_score_med = float(np.mean(update_win_score_med_vals)) if update_win_score_med_vals else 0.0
                win_score_max = float(np.mean(update_win_score_max_vals)) if update_win_score_max_vals else 0.0
                gate_mean_update = float(np.mean(update_gate_mean_vals)) if update_gate_mean_vals else 0.0
                gate_align_mean_update = float(np.mean(update_gate_align_mean_vals)) if update_gate_align_mean_vals else 0.0
                gate_align_diff_update = float(np.mean(update_gate_align_diff_vals)) if update_gate_align_diff_vals else 0.0
                gate_align_absdiff_update = (
                    float(np.mean(update_gate_align_absdiff_vals)) if update_gate_align_absdiff_vals else 0.0
                )
                gate_max_update = float(np.mean(update_gate_max_vals)) if update_gate_max_vals else 0.0
                delta_rms_update = float(np.mean(update_delta_rms_vals)) if update_delta_rms_vals else 0.0
                update_rms_update = float(np.mean(update_update_rms_vals)) if update_update_rms_vals else 0.0
                upd_over_delta_update = float(np.mean(update_upd_over_delta_vals)) if update_upd_over_delta_vals else 0.0
                gate_local_mean_update = float(np.mean(update_gate_local_mean_vals)) if update_gate_local_mean_vals else 0.0
                gate_local_p10_update = float(np.mean(update_gate_local_p10_vals)) if update_gate_local_p10_vals else 0.0
                gate_local_p90_update = float(np.mean(update_gate_local_p90_vals)) if update_gate_local_p90_vals else 0.0
                gate_local_ratio_update = float(np.mean(update_gate_local_ratio_vals)) if update_gate_local_ratio_vals else 0.0

                logger.info(
                    "UPDATE e=%d us=%d gs=%d loss=%.6f align=%.6f cka_local=%.6f out_metric=%.6f upd_metric=%.6f lambda_out=%.6f lambda_upd=%.6f out_violation=%.6f upd_violation=%.6f delta_rms=%.6f",
                    epoch,
                    update_step,
                    global_step,
                    loss_total_mean,
                    loss_align_mean,
                    cka_stats["mean"],
                    out_metric_avg,
                    upd_metric_avg,
                    lambda_out,
                    lambda_upd,
                    out_violation_mean,
                    upd_violation_mean,
                    delta_rms_update,
                )

                _append_csv(
                    den_csv,
                    [
                        "update_step",
                        "global_step",
                        "epoch",
                        "den_min",
                        "den_p01",
                        "den_p10",
                        "den_median",
                        "den_mean",
                        "den_max",
                        "den_small_count",
                        "den_critical_count",
                        "nx_min",
                        "ny_min",
                    ],
                    [
                        update_step,
                        global_step,
                        epoch,
                        den_stats["min"],
                        den_stats["p01"],
                        den_stats["p10"],
                        den_stats["median"],
                        den_stats["mean"],
                        den_stats["max"],
                        den_small_count,
                        den_critical_count,
                        nx_min,
                        ny_min,
                    ],
                )

                _append_csv(
                    cka_csv,
                    [
                        "update_step",
                        "global_step",
                        "epoch",
                        "loss_total_mean",
                        "loss_align_mean",
                        "loss_out_mean",
                        "loss_upd_mean",
                        "loss_select_mean",
                        "select_score_mean",
                        "div_mean",
                        "out_metric_mean",
                        "upd_metric_mean",
                        "out_violation_mean",
                        "upd_violation_mean",
                        "lambda_out",
                        "lambda_upd",
                        "gate_mean",
                        "gate_align_mean",
                        "gate_align_diff",
                        "gate_align_absdiff",
                        "gate_max",
                        "delta_rms",
                        "update_rms",
                        "upd_over_delta",
                        "gate_local_mean",
                        "gate_local_p10",
                        "gate_local_p90",
                        "gate_local_ratio",
                        "cka_local_mean",
                        "cka_local_min",
                        "cka_local_max",
                    ],
                    [
                        update_step,
                        global_step,
                        epoch,
                        loss_total_mean,
                        loss_align_mean,
                        loss_out_mean,
                        loss_upd_mean,
                        loss_select_mean,
                        select_score_mean,
                        div_mean,
                        out_metric_avg,
                        upd_metric_avg,
                        out_violation_mean,
                        upd_violation_mean,
                        lambda_out,
                        lambda_upd,
                        gate_mean_update,
                        gate_align_mean_update,
                        gate_align_diff_update,
                        gate_align_absdiff_update,
                        gate_max_update,
                        delta_rms_update,
                        update_rms_update,
                        upd_over_delta_update,
                        gate_local_mean_update,
                        gate_local_p10_update,
                        gate_local_p90_update,
                        gate_local_ratio_update,
                        cka_stats["mean"],
                        cka_stats["min"],
                        cka_stats["max"],
                    ],
                )
                _append_csv(
                    win_csv,
                    [
                        "update_step",
                        "global_step",
                        "epoch",
                        "win_iou_mean",
                        "win_iou_max",
                        "win_coverage_ratio",
                        "win_score_min",
                        "win_score_median",
                        "win_score_max",
                    ],
                    [
                        update_step,
                        global_step,
                        epoch,
                        win_iou_mean,
                        win_iou_max,
                        win_cov,
                        win_score_min,
                        win_score_med,
                        win_score_max,
                    ],
                )

                hist_updates.append(update_step)
                hist_den_min.append(float(den_stats["min"]))
                hist_den_small.append(float(den_small_count))
                hist_cka.append(float(cka_stats["mean"]))
                hist_gate_mean.append(gate_mean_update)
                hist_gate_align_mean.append(gate_align_mean_update)
                hist_gate_local_mean.append(gate_local_mean_update)
                hist_gate_local_p10.append(gate_local_p10_update)
                hist_gate_local_p90.append(gate_local_p90_update)
                hist_gate_local_ratio.append(gate_local_ratio_update)
                hist_delta_rms.append(delta_rms_update)
                hist_update_rms.append(update_rms_update)
                hist_upd_over_delta.append(upd_over_delta_update)
                hist_select_score.append(select_score_mean)

                _save_timeline(hist_updates, hist_den_min, "Denominator Min Trend", "den_min", plots_dir / "den_min_trend.png")
                _save_timeline(
                    hist_updates,
                    hist_den_small,
                    "Small Denominator Count Trend",
                    "count(<den_warn)",
                    plots_dir / "den_small_count_trend.png",
                )
                _save_timeline(hist_updates, hist_cka, "Local CKA Trend", "cka_local_mean", plots_dir / "local_cka_trend.png")
                _save_timeline(
                    hist_updates,
                    hist_select_score,
                    "Gate Select Score Trend",
                    "select_score",
                    plots_dir / "select_score_trend.png",
                )
                _save_multi_timeline(
                    hist_updates,
                    {
                        "gate_mean": hist_gate_mean,
                        "gate_align_mean": hist_gate_align_mean,
                        "gate_local_mean": hist_gate_local_mean,
                    },
                    "Gate Mean Trend",
                    "gate value",
                    plots_dir / "gate_mean_trend.png",
                )
                _save_multi_timeline(
                    hist_updates,
                    {
                        "gate_local_p10": hist_gate_local_p10,
                        "gate_local_mean": hist_gate_local_mean,
                        "gate_local_p90": hist_gate_local_p90,
                    },
                    "Selected Local-Token Gate Quantiles",
                    "gate value",
                    plots_dir / "gate_local_quantile_trend.png",
                )
                _save_multi_timeline(
                    hist_updates,
                    {
                        "delta_rms": hist_delta_rms,
                        "update_rms": hist_update_rms,
                    },
                    "Adapter vs Update RMS",
                    "rms",
                    plots_dir / "update_rms_trend.png",
                )
                _save_multi_timeline(
                    hist_updates,
                    {
                        "upd_over_delta": hist_upd_over_delta,
                        "gate_local_ratio": hist_gate_local_ratio,
                    },
                    "Suppression Ratio Trend",
                    "ratio",
                    plots_dir / "suppression_ratio_trend.png",
                )

                update_loss_total_vals.clear()
                update_loss_align_vals.clear()
                update_loss_out_vals.clear()
                update_loss_upd_vals.clear()
                update_loss_select_vals.clear()
                update_cka_vals.clear()
                update_select_score_vals.clear()
                update_div_mean_vals.clear()
                update_out_metric_vals.clear()
                update_upd_metric_vals.clear()
                update_out_violation_vals.clear()
                update_upd_violation_vals.clear()
                update_den_vals.clear()
                update_nx_vals.clear()
                update_ny_vals.clear()
                update_win_iou_mean_vals.clear()
                update_win_iou_max_vals.clear()
                update_win_cov_vals.clear()
                update_win_score_min_vals.clear()
                update_win_score_med_vals.clear()
                update_win_score_max_vals.clear()
                update_gate_mean_vals.clear()
                update_gate_align_mean_vals.clear()
                update_gate_align_diff_vals.clear()
                update_gate_align_absdiff_vals.clear()
                update_gate_max_vals.clear()
                update_delta_rms_vals.clear()
                update_update_rms_vals.clear()
                update_upd_over_delta_vals.clear()
                update_gate_local_mean_vals.clear()
                update_gate_local_p10_vals.clear()
                update_gate_local_p90_vals.clear()
                update_gate_local_ratio_vals.clear()

            if step % max(1, int(args.log_every)) == 0:
                pbar.set_postfix(loss=f"{float(total_loss.detach().item()):.4f}")

            if den_critical_micro > 0:
                logger.warning(
                    "DEN-CRITICAL gs=%d count=%d threshold=%.1e",
                    global_step,
                    den_critical_micro,
                    float(args.diag_den_critical),
                )

            global_step += 1

        last_epoch_ran = epoch
        epoch_ckpt_path = out_dir / f"adapter_epoch_{epoch + 1:03d}.pth"
        _save_ckpt(
            path=epoch_ckpt_path,
            model=model,
            args=args,
            optimizer=optimizer,
            scaler=scaler,
            global_step=global_step,
            update_step=update_step,
            epoch=epoch,
            lambda_out=lambda_out,
            lambda_upd=lambda_upd,
            logger=logger,
        )

        if int(args.eval_every_epoch) > 0 and ((epoch + 1) % int(args.eval_every_epoch) == 0):
            eval_metrics = _run_epoch_eval(
                args=args,
                ckpt_path=epoch_ckpt_path,
                out_dir=out_dir,
                epoch=epoch,
                logger=logger,
            )
            if eval_metrics is not None:
                logger.info(
                    "EVAL-EPOCH e=%d base=%.4f adapt=%.4f delta=%.4f delta_rel=%.4f%%",
                    epoch,
                    eval_metrics["overall_base"],
                    eval_metrics["overall_adapt"],
                    eval_metrics["overall_delta_abs"],
                    eval_metrics["overall_delta_rel_pct"],
                )

        if stop_training:
            break

    elapsed = time.time() - started_at
    logger.info("Training finished. elapsed_sec=%.1f", elapsed)
    final_ckpt_epoch = max(last_epoch_ran, start_epoch - 1)
    final_ckpt_path = out_dir / "adapter.pth"
    _save_ckpt(
        path=final_ckpt_path,
        model=model,
        args=args,
        optimizer=optimizer,
        scaler=scaler,
        global_step=global_step,
        update_step=update_step,
        epoch=final_ckpt_epoch,
        lambda_out=lambda_out,
        lambda_upd=lambda_upd,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
