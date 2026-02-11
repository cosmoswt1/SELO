#!/usr/bin/env python3
"""Interactive anchor picker: click on input image, then render 4x7 token-sim panel."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

import matplotlib

plt = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dino_teacher import DinoTeacher  # noqa: E402
from models.segformer_backbone import SegFormerBackbone  # noqa: E402

CONDITIONS = ["fog", "night", "rain", "snow"]
SPLIT_DIRS = ["train", "val", "test", "train_ref", "val_ref", "test_ref"]
STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4"]


def setup_matplotlib(no_gui: bool):
    global plt
    if plt is not None:
        return plt
    if no_gui:
        matplotlib.use("Agg", force=True)
    else:
        backend_ok = False
        for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
            try:
                matplotlib.use(backend, force=True)
                backend_ok = True
                break
            except Exception:
                continue
        if not backend_ok:
            raise RuntimeError(
                "GUI backend를 찾지 못했습니다. "
                "headless 환경이면 `--no_gui`를 사용하거나, 로컬 GUI 환경에서 실행하세요."
            )
    import matplotlib.pyplot as _plt

    plt = _plt
    return plt


@dataclass(frozen=True)
class ImageItem:
    path: Path
    condition: str
    split_dir: str
    is_ref: bool


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


def collect_acdc_images(acdc_root: Path, condition: str, include_ref: bool) -> list[ImageItem]:
    rgb_base = find_rgb_base(acdc_root)
    items: list[ImageItem] = []
    for split_dir in SPLIT_DIRS:
        is_ref = split_dir.endswith("_ref")
        if is_ref and not include_ref:
            continue
        folder = rgb_base / condition / split_dir
        if not folder.exists():
            continue
        pattern = "*_rgb_ref_anon.png" if is_ref else "*_rgb_anon.png"
        for path in sorted(folder.rglob(pattern)):
            items.append(
                ImageItem(
                    path=path,
                    condition=condition,
                    split_dir=split_dir,
                    is_ref=is_ref,
                )
            )
    return items


def check_gpu_or_exit(retry: int, retry_sleep_sec: float) -> None:
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
        raise SystemExit(1)

    ok = False
    for i in range(1, max(retry, 1) + 1):
        if torch.cuda.is_available():
            ok = True
            print(f"[GPU 체크] torch.cuda.is_available() == True (try {i}/{retry})")
            break
        print(f"[GPU 체크] torch.cuda.is_available() == False (try {i}/{retry})")
        if i < retry:
            time.sleep(max(retry_sleep_sec, 0.0))

    if not ok:
        print("[GPU 체크 실패] torch.cuda.is_available() == False", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        print("해결: NVIDIA 드라이버/CUDA 및 WSL GPU 상태를 확인 후 재실행하세요.", file=sys.stderr)
        raise SystemExit(1)


def load_preprocessed_image(
    path: Path,
    resize_short: int,
    square_crop_size: int,
    pad_multiple: int,
) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if h < w:
        new_h = resize_short
        new_w = int(round(w * resize_short / h))
    else:
        new_w = resize_short
        new_h = int(round(h * resize_short / w))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    if square_crop_size > 0:
        target = square_crop_size
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
        pad_h = (pad_multiple - (new_h % pad_multiple)) % pad_multiple
        pad_w = (pad_multiple - (new_w % pad_multiple)) % pad_multiple
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], padding_mode="reflect")

    image_uint8 = np.asarray(image, dtype=np.uint8)
    tensor = TF.to_tensor(image)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor, image_uint8


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
            "Use an input size divisible by patch size."
        )
    ph = h_aligned // teacher.patch_size
    pw = w_aligned // teacher.patch_size
    expected = ph * pw

    outs = teacher.dino(
        pixel_values=x_aligned,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outs.hidden_states

    layer_ids: list[int] = []
    layer_feats: list[torch.Tensor] = []
    for idx in range(1, len(hidden_states)):
        hs = hidden_states[idx]
        seq_len = hs.shape[1]
        prefix = seq_len - expected
        if prefix < 0:
            raise RuntimeError(f"DINO token mismatch: seq_len={seq_len}, expected={expected}, layer={idx}")
        if prefix != teacher.num_prefix_tokens:
            raise RuntimeError(
                "DINO prefix token mismatch: "
                f"layer={idx}, seq_len={seq_len}, expected_patches={expected}, "
                f"prefix={prefix}, cfg_prefix={teacher.num_prefix_tokens}"
            )
        tokens = hs[:, prefix:, :]
        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], ph, pw)
        layer_ids.append(idx)
        layer_feats.append(feat)
    return layer_ids, layer_feats


def resize_feature_to_grid(feat: torch.Tensor, grid_size: int) -> torch.Tensor:
    h, w = int(feat.shape[-2]), int(feat.shape[-1])
    if h == grid_size and w == grid_size:
        return feat
    if h >= grid_size and w >= grid_size:
        return F.adaptive_avg_pool2d(feat, output_size=(grid_size, grid_size))
    return F.interpolate(feat, size=(grid_size, grid_size), mode="nearest")


def tokens_from_feature_for_viz(feat_3d: torch.Tensor, viz_grid_size: int) -> torch.Tensor:
    resized = resize_feature_to_grid(feat_3d.unsqueeze(0), viz_grid_size).squeeze(0)
    return resized.flatten(1).transpose(0, 1).contiguous()


def token_cosine_map(tokens: torch.Tensor, anchor_idx: int, grid_size: int) -> np.ndarray:
    z = F.normalize(tokens.float(), dim=1)
    anchor = z[anchor_idx : anchor_idx + 1]
    sim = (z @ anchor.transpose(0, 1)).squeeze(1).reshape(grid_size, grid_size)
    sim_np = sim.detach().cpu().numpy()
    vmin, vmax = float(sim_np.min()), float(sim_np.max())
    if vmax - vmin > 1e-8:
        sim_np = (sim_np - vmin) / (vmax - vmin)
    else:
        sim_np = np.zeros_like(sim_np, dtype=np.float32)
    return sim_np


def draw_anchor_overlay(image_uint8: np.ndarray, anchors_px: list[tuple[int, int]], labels: list[str]) -> np.ndarray:
    img = Image.fromarray(image_uint8.copy())
    dr = ImageDraw.Draw(img)
    h, w = image_uint8.shape[:2]
    r = max(6, min(h, w) // 120)
    for (px, py), label in zip(anchors_px, labels):
        dr.ellipse([(px - r, py - r), (px + r, py + r)], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
        dr.text((px + 10, py - 10), label, fill=(255, 0, 0))
    return np.asarray(img)


def save_panel_4x7(
    out_path: Path,
    stage_feats_one: list[torch.Tensor],
    layer_feats_one: list[torch.Tensor],
    layer_ids: list[int],
    viz_grid_size: int,
    anchor_gy_gx: tuple[int, int],
    image_name: str,
    click_idx: int,
) -> None:
    gy, gx = anchor_gy_gx
    anchor_idx = gy * viz_grid_size + gx
    stage_tokens_vis = [tokens_from_feature_for_viz(feat, viz_grid_size) for feat in stage_feats_one]
    layer_tokens_vis = [tokens_from_feature_for_viz(feat, viz_grid_size) for feat in layer_feats_one]

    stage_maps = [token_cosine_map(tok, anchor_idx=anchor_idx, grid_size=viz_grid_size) for tok in stage_tokens_vis]
    dino_maps = [token_cosine_map(tok, anchor_idx=anchor_idx, grid_size=viz_grid_size) for tok in layer_tokens_vis]

    n_rows = 4
    n_dino_cols = int(np.ceil(len(layer_ids) / n_rows))
    n_cols = 1 + n_dino_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows))

    for r in range(n_rows):
        ax0 = axes[r, 0]
        ax0.imshow(stage_maps[r], cmap="viridis", vmin=0.0, vmax=1.0)
        ax0.scatter([gx], [gy], c="red", s=24, marker="o")
        ax0.set_title(f"Seg {STAGE_NAMES[r]}", fontsize=9)
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

    fig.suptitle(
        f"Click#{click_idx} | grid={viz_grid_size}x{viz_grid_size} | anchor=({gy},{gx}) | {image_name}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def pixel_to_grid(px: float, py: float, w: int, h: int, grid: int) -> tuple[int, int]:
    gx = int(np.clip(np.floor(px * grid / max(w, 1)), 0, grid - 1))
    gy = int(np.clip(np.floor(py * grid / max(h, 1)), 0, grid - 1))
    return gy, gx


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive anchor picker for 4x7 token similarity panel")
    p.add_argument("--acdc_root", type=str, default="/mnt/d/ACDC")
    p.add_argument("--output_dir", type=str, default="exp/dino_segformer_layer_match/interactive_picker")
    p.add_argument("--condition", type=str, default="night", choices=CONDITIONS)
    ref_group = p.add_mutually_exclusive_group()
    ref_group.add_argument("--include_ref", dest="include_ref", action="store_true")
    ref_group.add_argument("--exclude_ref", dest="include_ref", action="store_false")
    p.set_defaults(include_ref=False)
    p.add_argument("--image_path", type=str, default="", help="Optional explicit image path.")
    p.add_argument("--image_index", type=int, default=0, help="Index inside collected image list.")
    p.add_argument("--resize_short", type=int, default=1072)
    p.add_argument("--square_crop_size", type=int, default=1072)
    p.add_argument("--pad_multiple", type=int, default=32)
    p.add_argument("--viz_grid_size", type=int, default=67)
    p.add_argument("--segformer_model", type=str, default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--dino_model", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--gpu_retry", type=int, default=8)
    p.add_argument("--gpu_retry_sleep_sec", type=float, default=2.0)
    p.add_argument("--no_gui", action="store_true", help="No GUI: one-shot panel by anchor_x/anchor_y.")
    p.add_argument("--anchor_x", type=float, default=0.5)
    p.add_argument("--anchor_y", type=float, default=0.5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        setup_matplotlib(no_gui=bool(args.no_gui))
    except Exception as exc:
        print(f"[오류] matplotlib backend 초기화 실패: {exc}", file=sys.stderr)
        return 1

    check_gpu_or_exit(retry=int(args.gpu_retry), retry_sleep_sec=float(args.gpu_retry_sleep_sec))

    out_dir = Path(args.output_dir)
    viz_dir = out_dir / "viz_stream"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"[오류] image_path not found: {image_path}", file=sys.stderr)
            return 1
    else:
        items = collect_acdc_images(Path(args.acdc_root), condition=args.condition, include_ref=bool(args.include_ref))
        if len(items) == 0:
            print("[오류] 수집된 이미지가 없습니다.", file=sys.stderr)
            return 1
        if args.image_index < 0 or args.image_index >= len(items):
            print(f"[오류] image_index out of range: {args.image_index} (0..{len(items)-1})", file=sys.stderr)
            return 1
        image_path = items[args.image_index].path

    print(f"[interactive] selected image: {image_path}")

    image_tensor, image_uint8 = load_preprocessed_image(
        image_path,
        resize_short=int(args.resize_short),
        square_crop_size=int(args.square_crop_size),
        pad_multiple=int(args.pad_multiple),
    )
    h, w = image_uint8.shape[:2]
    print(f"[interactive] preprocessed image size: {h}x{w}")

    device = torch.device("cuda")
    x = image_tensor.unsqueeze(0).to(device)
    segformer = SegFormerBackbone(model_name=args.segformer_model, num_classes=19).to(device).eval()
    teacher = DinoTeacher(model_name=args.dino_model).to(device).eval()

    amp_enabled = bool(args.amp)
    amp_dtype = torch.float16
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
            stage_feats = segformer.forward_encoder(x)
            layer_ids, dino_feats = extract_dino_layer_features(teacher, x, strict_same_resolution=True)

    stage_feats_one = [stage_feats[s][0].float().detach() for s in range(4)]
    layer_feats_one = [dino_feats[i][0].float().detach() for i in range(len(layer_ids))]

    print(f"[interactive] dino layers: {layer_ids[0]}..{layer_ids[-1]} (count={len(layer_ids)})")
    print(f"[interactive] viz grid: {args.viz_grid_size}x{args.viz_grid_size}")
    print("[interactive] 클릭하면 panel_click_*.png(4x7)가 생성되고 바로 표시됩니다.")
    print("[interactive] 종료: input 창에서 q 또는 Esc")

    if args.no_gui:
        gx = int(np.clip(round(float(args.anchor_x) * (int(args.viz_grid_size) - 1)), 0, int(args.viz_grid_size) - 1))
        gy = int(np.clip(round(float(args.anchor_y) * (int(args.viz_grid_size) - 1)), 0, int(args.viz_grid_size) - 1))
        panel_path = viz_dir / "panel_click_001.png"
        save_panel_4x7(
            out_path=panel_path,
            stage_feats_one=stage_feats_one,
            layer_feats_one=layer_feats_one,
            layer_ids=layer_ids,
            viz_grid_size=int(args.viz_grid_size),
            anchor_gy_gx=(gy, gx),
            image_name=image_path.name,
            click_idx=1,
        )
        px = int(np.clip(round((gx + 0.5) * w / int(args.viz_grid_size)), 0, w - 1))
        py = int(np.clip(round((gy + 0.5) * h / int(args.viz_grid_size)), 0, h - 1))
        overlay = draw_anchor_overlay(image_uint8, anchors_px=[(px, py)], labels=[f"A1 ({gy},{gx})"])
        Image.fromarray(overlay).save(viz_dir / "latest_input_anchors.png")
        print(f"[interactive] saved: {panel_path}")
        return 0

    state = {
        "anchors_px": [],
        "anchors_grid": [],
        "click_count": 0,
        "panel_fig": None,
        "panel_ax": None,
    }

    fig_in, ax_in = plt.subplots(figsize=(7.0, 7.0))
    ax_in.imshow(image_uint8)
    ax_in.set_title("Input image: click anchor points (press q/Esc to quit)")
    ax_in.axis("off")

    def on_click(event):
        if event.inaxes != ax_in or event.xdata is None or event.ydata is None:
            return
        px = int(np.clip(round(float(event.xdata)), 0, w - 1))
        py = int(np.clip(round(float(event.ydata)), 0, h - 1))
        gy, gx = pixel_to_grid(px=px, py=py, w=w, h=h, grid=int(args.viz_grid_size))

        state["click_count"] += 1
        click_idx = int(state["click_count"])
        state["anchors_px"].append((px, py))
        state["anchors_grid"].append((gy, gx))
        labels = [f"A{i+1} ({g[0]},{g[1]})" for i, g in enumerate(state["anchors_grid"])]

        overlay = draw_anchor_overlay(image_uint8, anchors_px=state["anchors_px"], labels=labels)
        ax_in.clear()
        ax_in.imshow(overlay)
        ax_in.set_title(f"Input image | clicks={click_idx} (q/Esc to quit)")
        ax_in.axis("off")
        fig_in.canvas.draw_idle()

        out_panel = viz_dir / f"panel_click_{click_idx:03d}_g{gy:02d}_{gx:02d}.png"
        save_panel_4x7(
            out_path=out_panel,
            stage_feats_one=stage_feats_one,
            layer_feats_one=layer_feats_one,
            layer_ids=layer_ids,
            viz_grid_size=int(args.viz_grid_size),
            anchor_gy_gx=(gy, gx),
            image_name=image_path.name,
            click_idx=click_idx,
        )
        Image.fromarray(overlay).save(viz_dir / f"input_click_{click_idx:03d}.png")
        Image.fromarray(overlay).save(viz_dir / "latest_input_anchors.png")

        panel_img = np.asarray(Image.open(out_panel))
        if state["panel_fig"] is None:
            panel_fig, panel_ax = plt.subplots(figsize=(15.0, 8.2))
            state["panel_fig"] = panel_fig
            state["panel_ax"] = panel_ax
        panel_ax = state["panel_ax"]
        panel_ax.clear()
        panel_ax.imshow(panel_img)
        panel_ax.axis("off")
        panel_ax.set_title(f"4x7 panel | click={click_idx} | anchor=({gy},{gx})")
        state["panel_fig"].canvas.draw_idle()
        state["panel_fig"].show()

        Image.fromarray(panel_img).save(viz_dir / "latest_panel.png")
        print(
            f"[interactive] click={click_idx}, pixel=({px},{py}), grid=({gy},{gx}), "
            f"saved={out_panel.name}"
        )

    def on_key(event):
        if event.key in ("q", "escape"):
            plt.close("all")

    fig_in.canvas.mpl_connect("button_press_event", on_click)
    fig_in.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
