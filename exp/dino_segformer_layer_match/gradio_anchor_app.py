#!/usr/bin/env python3
"""Gradio-based browser UI: file sweep + anchor click -> 4x7 panel."""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.dino_teacher import DinoTeacher  # noqa: E402
from models.segformer_backbone import SegFormerBackbone  # noqa: E402

CONDITIONS = ["fog", "night", "rain", "snow"]
STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4"]
APP_CSS = """
html, body {
  margin: 0;
  padding: 0;
  height: 100%;
}
.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  padding: 8px 12px !important;
}
#main_row {
  min-height: calc(100vh - 96px);
}
#left_col, #right_col {
  min-height: calc(100vh - 112px);
}
#file_picker {
  min-height: 32vh;
  max-height: 40vh;
  overflow: auto;
}
#input_img {
  min-height: 46vh;
}
#panel_img {
  min-height: calc(100vh - 140px);
}
"""


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
            f"input={tuple(x.shape[-2:])}, aligned={(h_aligned, w_aligned)}"
        )
    ph = h_aligned // teacher.patch_size
    pw = w_aligned // teacher.patch_size
    expected = ph * pw

    outs = teacher.dino(pixel_values=x_aligned, output_hidden_states=True, return_dict=True)
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
                f"layer={idx}, seq_len={seq_len}, expected={expected}, prefix={prefix}"
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


def pixel_to_grid(px: float, py: float, w: int, h: int, grid: int) -> tuple[int, int]:
    gx = int(np.clip(np.floor(px * grid / max(w, 1)), 0, grid - 1))
    gy = int(np.clip(np.floor(py * grid / max(h, 1)), 0, grid - 1))
    return gy, gx


def draw_anchor_overlay(image_uint8: np.ndarray, anchors_px: list[tuple[int, int]], labels: list[str]) -> np.ndarray:
    img = Image.fromarray(image_uint8.copy())
    dr = ImageDraw.Draw(img)
    h, w = image_uint8.shape[:2]
    r = max(6, min(h, w) // 120)
    for (px, py), label in zip(anchors_px, labels):
        dr.ellipse([(px - r, py - r), (px + r, py + r)], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
        dr.text((px + 10, py - 10), label, fill=(255, 0, 0))
    return np.asarray(img)


def render_panel_4x7(
    stage_feats_one: list[torch.Tensor],
    layer_feats_one: list[torch.Tensor],
    layer_ids: list[int],
    viz_grid_size: int,
    anchor_gy_gx: tuple[int, int],
    image_name: str,
    click_idx: int,
    panel_cell_inches: float,
    panel_dpi: int,
) -> np.ndarray:
    gy, gx = anchor_gy_gx
    anchor_idx = gy * viz_grid_size + gx
    stage_tokens = [tokens_from_feature_for_viz(f, viz_grid_size) for f in stage_feats_one]
    layer_tokens = [tokens_from_feature_for_viz(f, viz_grid_size) for f in layer_feats_one]
    stage_maps = [token_cosine_map(tok, anchor_idx=anchor_idx, grid_size=viz_grid_size) for tok in stage_tokens]
    dino_maps = [token_cosine_map(tok, anchor_idx=anchor_idx, grid_size=viz_grid_size) for tok in layer_tokens]

    n_rows = 4
    n_dino_cols = int(np.ceil(len(layer_ids) / n_rows))
    n_cols = 1 + n_dino_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_cell_inches * n_cols, (panel_cell_inches - 0.15) * n_rows),
        squeeze=False,
    )

    for r in range(n_rows):
        ax0 = axes[r, 0]
        ax0.imshow(stage_maps[r], cmap="viridis", vmin=0.0, vmax=1.0)
        ax0.scatter([gx], [gy], c="red", s=24, marker="o")
        ax0.set_title(f"Seg {STAGE_NAMES[r]}", fontsize=12)
        ax0.axis("off")
        for c in range(n_dino_cols):
            li = r * n_dino_cols + c
            ax = axes[r, c + 1]
            if li < len(layer_ids):
                ax.imshow(dino_maps[li], cmap="viridis", vmin=0.0, vmax=1.0)
                ax.set_title(f"DINO L{layer_ids[li]}", fontsize=12)
            else:
                ax.imshow(np.zeros((viz_grid_size, viz_grid_size), dtype=np.float32), cmap="viridis", vmin=0.0, vmax=1.0)
                ax.set_title("-", fontsize=12)
            ax.scatter([gx], [gy], c="red", s=16, marker="o")
            ax.axis("off")
    fig.suptitle(
        f"Click#{click_idx} | grid={viz_grid_size}x{viz_grid_size} | anchor=({gy},{gx}) | {image_name}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=int(panel_dpi))
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio anchor picker app")
    p.add_argument("--acdc_root", type=str, default="/mnt/d/ACDC")
    p.add_argument("--output_dir", type=str, default="exp/dino_segformer_layer_match/gradio_picker")
    p.add_argument("--condition", type=str, default="night", choices=CONDITIONS)
    ref_group = p.add_mutually_exclusive_group()
    ref_group.add_argument("--include_ref", dest="include_ref", action="store_true")
    ref_group.add_argument("--exclude_ref", dest="include_ref", action="store_false")
    p.set_defaults(include_ref=False)
    p.add_argument("--browse_root", type=str, default="")
    p.add_argument("--image_index", type=int, default=0)
    p.add_argument("--resize_short", type=int, default=1072)
    p.add_argument("--square_crop_size", type=int, default=1072)
    p.add_argument("--pad_multiple", type=int, default=32)
    p.add_argument("--viz_grid_size", type=int, default=67)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--segformer_model", type=str, default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--dino_model", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--panel_cell_inches", type=float, default=4.2)
    p.add_argument("--panel_dpi", type=int, default=220)
    p.add_argument("--input_display_height", type=int, default=900)
    p.add_argument("--panel_display_height", type=int, default=1400)
    p.add_argument("--gpu_retry", type=int, default=8)
    p.add_argument("--gpu_retry_sleep_sec", type=float, default=2.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_gpu_or_exit(retry=int(args.gpu_retry), retry_sleep_sec=float(args.gpu_retry_sleep_sec))

    output_dir = Path(args.output_dir)
    viz_dir = output_dir / "viz_stream"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    acdc_root = Path(args.acdc_root)
    rgb_base = find_rgb_base(acdc_root)
    browse_root = Path(args.browse_root) if args.browse_root else (rgb_base / args.condition)
    if not browse_root.exists():
        print(f"[오류] browse_root not found: {browse_root}", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    segformer = SegFormerBackbone(model_name=args.segformer_model, num_classes=19).to(device).eval()
    teacher = DinoTeacher(model_name=args.dino_model).to(device).eval()
    amp_enabled = bool(args.amp)
    amp_dtype = torch.float16

    app_state = {
        "current_path": "",
        "image_uint8": None,
        "stage_feats": None,
        "layer_feats": None,
        "layer_ids": None,
        "click_count": 0,
        "anchors_px": [],
        "anchors_grid": [],
    }

    def load_selected(path_val: str, state: dict):
        if not path_val:
            return gr.update(), gr.update(), "파일을 선택하세요.", state
        p = Path(path_val)
        if p.is_dir():
            return gr.update(), gr.update(), f"폴더 선택: {p} (이미지 파일을 더블클릭하세요)", state
        if not p.exists():
            return gr.update(), gr.update(), f"파일 없음: {p}", state
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            return gr.update(), gr.update(), f"이미지 파일이 아님: {p.name}", state

        image_tensor, image_uint8 = load_preprocessed_image(
            p,
            resize_short=int(args.resize_short),
            square_crop_size=int(args.square_crop_size),
            pad_multiple=int(args.pad_multiple),
        )
        x = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
                seg_feats = segformer.forward_encoder(x)
                layer_ids, dino_feats = extract_dino_layer_features(teacher, x, strict_same_resolution=True)

        state = {
            "current_path": str(p),
            "image_uint8": image_uint8,
            "stage_feats": [seg_feats[s][0].float().detach() for s in range(4)],
            "layer_feats": [dino_feats[i][0].float().detach() for i in range(len(layer_ids))],
            "layer_ids": layer_ids,
            "click_count": 0,
            "anchors_px": [],
            "anchors_grid": [],
        }
        Image.fromarray(image_uint8).save(viz_dir / "latest_input_anchors.png")
        status = f"loaded: {p.name}\nsize={image_uint8.shape[1]}x{image_uint8.shape[0]}\n이제 input 이미지를 클릭하세요."
        return image_uint8, None, status, state

    def click_on_input(img_val, state: dict, evt: gr.SelectData):
        if state is None or not state.get("current_path"):
            return gr.update(), gr.update(), "먼저 파일을 선택하세요.", state
        if state.get("image_uint8") is None:
            return gr.update(), gr.update(), "이미지가 로드되지 않았습니다.", state

        x = y = None
        idx = getattr(evt, "index", None)
        if isinstance(idx, (tuple, list)) and len(idx) >= 2:
            x = float(idx[0])
            y = float(idx[1])
        if x is None or y is None:
            return gr.update(), gr.update(), "클릭 좌표를 읽지 못했습니다.", state

        image_uint8 = state["image_uint8"]
        h, w = image_uint8.shape[:2]
        px = int(np.clip(round(x), 0, w - 1))
        py = int(np.clip(round(y), 0, h - 1))
        gy, gx = pixel_to_grid(px=px, py=py, w=w, h=h, grid=int(args.viz_grid_size))

        state["click_count"] += 1
        click_idx = int(state["click_count"])
        state["anchors_px"].append((px, py))
        state["anchors_grid"].append((gy, gx))

        labels = [f"A{i+1} ({g[0]},{g[1]})" for i, g in enumerate(state["anchors_grid"])]
        overlay = draw_anchor_overlay(image_uint8, state["anchors_px"], labels)
        Image.fromarray(overlay).save(viz_dir / "latest_input_anchors.png")
        Image.fromarray(overlay).save(viz_dir / f"input_click_{click_idx:04d}.png")

        panel = render_panel_4x7(
            stage_feats_one=state["stage_feats"],
            layer_feats_one=state["layer_feats"],
            layer_ids=state["layer_ids"],
            viz_grid_size=int(args.viz_grid_size),
            anchor_gy_gx=(gy, gx),
            image_name=Path(state["current_path"]).name,
            click_idx=click_idx,
            panel_cell_inches=float(args.panel_cell_inches),
            panel_dpi=int(args.panel_dpi),
        )
        panel_name = f"panel_click_{click_idx:04d}_g{gy:02d}_{gx:02d}.png"
        Image.fromarray(panel).save(viz_dir / panel_name)
        Image.fromarray(panel).save(viz_dir / "latest_panel.png")

        status = (
            f"click={click_idx}, pixel=({px},{py}), grid=({gy},{gx})\n"
            f"panel={viz_dir / panel_name}"
        )
        return overlay, panel, status, state

    with gr.Blocks(title="DINO-SegFormer Anchor Picker (Gradio)", css=APP_CSS) as demo:
        gr.Markdown(
            "## DINO-SegFormer Anchor Picker (Gradio)\n"
            "- 왼쪽 1열: 폴더/파일 선택 + 입력 이미지(앵커 클릭)\n"
            "- 오른쪽 1열: 4x7 패널 크게 표시\n"
        )
        with gr.Row(elem_id="main_row"):
            with gr.Column(scale=1, min_width=460, elem_id="left_col"):
                file_explorer = gr.FileExplorer(
                    root_dir=str(browse_root),
                    glob="**/*",
                    file_count="single",
                    label=f"Browse Root: {browse_root}",
                    elem_id="file_picker",
                )
                input_img = gr.Image(
                    type="numpy",
                    label="Input + red anchors",
                    interactive=True,
                    height=int(args.input_display_height),
                    elem_id="input_img",
                )
                status_box = gr.Textbox(label="Status", lines=5)
            with gr.Column(scale=3, min_width=980, elem_id="right_col"):
                panel_img = gr.Image(
                    type="numpy",
                    label="4x7 panel",
                    height=int(args.panel_display_height),
                    elem_id="panel_img",
                )
        state_comp = gr.State(app_state)

        file_explorer.change(
            fn=load_selected,
            inputs=[file_explorer, state_comp],
            outputs=[input_img, panel_img, status_box, state_comp],
        )
        input_img.select(
            fn=click_on_input,
            inputs=[input_img, state_comp],
            outputs=[input_img, panel_img, status_box, state_comp],
        )

    print(f"[gradio] output_dir={output_dir}")
    print(f"[gradio] browse_root={browse_root}")
    print(f"[gradio] URL=http://{args.host}:{args.port}")
    demo.queue().launch(server_name=args.host, server_port=args.port, inbrowser=False, share=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
