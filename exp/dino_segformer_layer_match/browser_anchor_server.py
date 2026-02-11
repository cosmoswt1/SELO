#!/usr/bin/env python3
"""Browser-based anchor picker with 4x7 token similarity panel."""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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
SPLIT_DIRS = ["train", "val", "test", "train_ref", "val_ref", "test_ref"]
STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4"]


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


def image_to_data_url(img_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_panel_4x7(
    stage_feats_one: list[torch.Tensor],
    layer_feats_one: list[torch.Tensor],
    layer_ids: list[int],
    viz_grid_size: int,
    anchor_gy_gx: tuple[int, int],
    image_name: str,
    click_idx: int,
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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


class BrowserState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.output_dir = Path(args.output_dir)
        self.viz_dir = self.output_dir / "viz_stream"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        self.items = collect_acdc_images(Path(args.acdc_root), args.condition, bool(args.include_ref))
        if len(self.items) == 0:
            raise RuntimeError("No images found for selected condition/split.")

        self.device = torch.device("cuda")
        self.segformer = SegFormerBackbone(model_name=args.segformer_model, num_classes=19).to(self.device).eval()
        self.teacher = DinoTeacher(model_name=args.dino_model).to(self.device).eval()

        self.amp_enabled = bool(args.amp)
        self.amp_dtype = torch.float16

        self.current_idx = -1
        self.current_item: ImageItem | None = None
        self.current_image_uint8: np.ndarray | None = None
        self.current_stage_feats: list[torch.Tensor] = []
        self.current_layer_feats: list[torch.Tensor] = []
        self.layer_ids: list[int] = []
        self.click_count = 0
        self.anchors_px: list[tuple[int, int]] = []
        self.anchors_grid: list[tuple[int, int]] = []

        idx = int(np.clip(args.image_index, 0, len(self.items) - 1))
        self.select_index(idx)

    def list_payload(self) -> dict:
        files = []
        for i, it in enumerate(self.items):
            files.append(
                {
                    "idx": i,
                    "name": it.path.name,
                    "path": str(it.path),
                    "split": it.split_dir,
                    "is_ref": int(it.is_ref),
                }
            )
        return {
            "total": len(self.items),
            "current_idx": self.current_idx,
            "files": files,
        }

    def _compute_features(self, image_tensor: torch.Tensor) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        x = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                seg_feats = self.segformer.forward_encoder(x)
                layer_ids, dino_feats = extract_dino_layer_features(self.teacher, x, strict_same_resolution=True)
        stage_feats_one = [seg_feats[s][0].float().detach() for s in range(4)]
        layer_feats_one = [dino_feats[i][0].float().detach() for i in range(len(layer_ids))]
        return layer_ids, stage_feats_one, layer_feats_one

    def _save_overlay(self, overlay: np.ndarray) -> None:
        Image.fromarray(overlay).save(self.viz_dir / "latest_input_anchors.png")
        Image.fromarray(overlay).save(self.viz_dir / f"input_click_{self.click_count:04d}.png")

    def select_index(self, idx: int) -> dict:
        with self.lock:
            idx = int(np.clip(idx, 0, len(self.items) - 1))
            item = self.items[idx]
            image_tensor, image_uint8 = load_preprocessed_image(
                item.path,
                resize_short=int(self.args.resize_short),
                square_crop_size=int(self.args.square_crop_size),
                pad_multiple=int(self.args.pad_multiple),
            )
            layer_ids, stage_feats_one, layer_feats_one = self._compute_features(image_tensor)

            self.current_idx = idx
            self.current_item = item
            self.current_image_uint8 = image_uint8
            self.layer_ids = layer_ids
            self.current_stage_feats = stage_feats_one
            self.current_layer_feats = layer_feats_one
            self.click_count = 0
            self.anchors_px = []
            self.anchors_grid = []

            Image.fromarray(image_uint8).save(self.viz_dir / "latest_input_anchors.png")
            Image.fromarray(image_uint8).save(self.viz_dir / f"input_idx_{idx:05d}.png")

            return {
                "ok": True,
                "idx": self.current_idx,
                "total": len(self.items),
                "image_path": str(item.path),
                "image_name": item.path.name,
                "image_data_url": image_to_data_url(image_uint8),
                "overlay_data_url": image_to_data_url(image_uint8),
                "panel_data_url": "",
                "message": f"selected idx={idx} ({item.path.name})",
            }

    def click_anchor(self, px: float, py: float) -> dict:
        with self.lock:
            if self.current_image_uint8 is None or self.current_item is None:
                return {"ok": False, "message": "no image selected"}

            h, w = self.current_image_uint8.shape[:2]
            px_i = int(np.clip(round(float(px)), 0, w - 1))
            py_i = int(np.clip(round(float(py)), 0, h - 1))
            gy, gx = pixel_to_grid(px=px_i, py=py_i, w=w, h=h, grid=int(self.args.viz_grid_size))

            self.click_count += 1
            self.anchors_px.append((px_i, py_i))
            self.anchors_grid.append((gy, gx))
            labels = [f"A{i+1} ({g[0]},{g[1]})" for i, g in enumerate(self.anchors_grid)]

            overlay = draw_anchor_overlay(self.current_image_uint8, self.anchors_px, labels)
            self._save_overlay(overlay)

            panel = render_panel_4x7(
                stage_feats_one=self.current_stage_feats,
                layer_feats_one=self.current_layer_feats,
                layer_ids=self.layer_ids,
                viz_grid_size=int(self.args.viz_grid_size),
                anchor_gy_gx=(gy, gx),
                image_name=self.current_item.path.name,
                click_idx=self.click_count,
            )
            panel_name = f"panel_click_{self.click_count:04d}_idx{self.current_idx:05d}_g{gy:02d}_{gx:02d}.png"
            Image.fromarray(panel).save(self.viz_dir / panel_name)
            Image.fromarray(panel).save(self.viz_dir / "latest_panel.png")

            return {
                "ok": True,
                "idx": self.current_idx,
                "click_count": self.click_count,
                "pixel": [px_i, py_i],
                "grid": [gy, gx],
                "image_name": self.current_item.path.name,
                "overlay_data_url": image_to_data_url(overlay),
                "panel_data_url": image_to_data_url(panel),
                "panel_file": str(self.viz_dir / panel_name),
                "message": f"click={self.click_count}, pixel=({px_i},{py_i}), grid=({gy},{gx})",
            }


HTML_PAGE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>DINO-SegFormer Anchor Browser</title>
  <style>
    :root { --bg:#f4f7fb; --panel:#ffffff; --line:#dbe3ef; --ink:#1b2638; --muted:#4f6078; --accent:#0057b8; }
    html, body { margin:0; padding:0; background:var(--bg); color:var(--ink); font-family: "Iosevka Aile", "IBM Plex Sans KR", sans-serif; }
    .root { display:flex; height:100vh; gap:12px; padding:12px; box-sizing:border-box; }
    .left { width:32%; min-width:320px; background:var(--panel); border:1px solid var(--line); border-radius:10px; display:flex; flex-direction:column; }
    .right { flex:1; background:var(--panel); border:1px solid var(--line); border-radius:10px; display:flex; flex-direction:column; }
    .head { padding:10px 12px; border-bottom:1px solid var(--line); display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .head b { color:var(--accent); }
    .meta { font-size:12px; color:var(--muted); margin-left:auto; }
    .files { overflow:auto; padding:6px; }
    .file { padding:8px; border:1px solid transparent; border-radius:8px; cursor:default; user-select:none; }
    .file:hover { border-color:var(--line); background:#f8fbff; }
    .file.sel { border-color:var(--accent); background:#edf4ff; }
    .file .idx { color:var(--accent); font-weight:700; margin-right:8px; }
    .file .name { font-weight:600; }
    .file .path { font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .viewer { display:grid; grid-template-columns: 1fr 1fr; gap:10px; padding:10px; overflow:auto; }
    .box { border:1px solid var(--line); border-radius:8px; padding:8px; background:#fff; }
    .box h4 { margin:0 0 8px 0; font-size:13px; color:var(--accent); }
    .box img { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }
    .status { padding:8px 12px; border-top:1px solid var(--line); font-size:13px; color:var(--muted); white-space:pre-wrap; }
    button, input { border:1px solid var(--line); background:#fff; border-radius:8px; padding:6px 10px; font-size:13px; }
    button:hover { border-color:var(--accent); }
    @media (max-width: 1000px) { .root { flex-direction:column; height:auto; } .left { width:100%; min-width:0; } .viewer { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="root">
    <section class="left">
      <div class="head">
        <b>Input Folder Sweep</b>
        <button id="prevBtn">Prev</button>
        <button id="nextBtn">Next</button>
        <input id="jumpIdx" type="number" min="0" value="0" style="width:80px"/>
        <button id="jumpBtn">Go</button>
        <span class="meta" id="metaText">loading...</span>
      </div>
      <div class="files" id="fileList"></div>
    </section>
    <section class="right">
      <div class="head">
        <b>Double-click file to open</b>
        <span class="meta">input 이미지 클릭 -> 4x7 panel 갱신</span>
      </div>
      <div class="viewer">
        <div class="box">
          <h4>Input + Anchors (red dots)</h4>
          <img id="inputImage" alt="input"/>
        </div>
        <div class="box">
          <h4>4x7 Panel (Seg 4 + DINO 24)</h4>
          <img id="panelImage" alt="panel"/>
        </div>
      </div>
      <div class="status" id="statusText">ready</div>
    </section>
  </div>
<script>
  let files = [];
  let currentIdx = 0;
  const fileListEl = document.getElementById("fileList");
  const inputImg = document.getElementById("inputImage");
  const panelImg = document.getElementById("panelImage");
  const statusEl = document.getElementById("statusText");
  const metaEl = document.getElementById("metaText");

  function setStatus(msg) { statusEl.textContent = msg || ""; }
  function setMeta() { metaEl.textContent = `idx=${currentIdx} / total=${files.length}`; }

  function renderFileList() {
    fileListEl.innerHTML = "";
    files.forEach((f) => {
      const div = document.createElement("div");
      div.className = "file" + (f.idx === currentIdx ? " sel" : "");
      div.dataset.idx = String(f.idx);
      div.innerHTML = `<div><span class="idx">#${f.idx}</span><span class="name">${f.name}</span></div><div class="path">${f.path}</div>`;
      div.ondblclick = () => selectIndex(f.idx);
      div.onclick = () => {
        document.getElementById("jumpIdx").value = String(f.idx);
      };
      fileListEl.appendChild(div);
    });
  }

  async function apiGet(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`GET ${url} failed: ${r.status}`);
    return r.json();
  }

  async function apiPost(url, payload) {
    const r = await fetch(url, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
    if (!r.ok) throw new Error(`POST ${url} failed: ${r.status}`);
    return r.json();
  }

  async function loadList() {
    const d = await apiGet("/api/list");
    files = d.files || [];
    currentIdx = d.current_idx || 0;
    renderFileList();
    setMeta();
  }

  async function selectIndex(idx) {
    const d = await apiGet(`/api/select?idx=${idx}`);
    if (!d.ok) throw new Error(d.message || "select failed");
    currentIdx = d.idx;
    inputImg.src = d.overlay_data_url || d.image_data_url;
    panelImg.src = "";
    document.getElementById("jumpIdx").value = String(currentIdx);
    renderFileList();
    setMeta();
    setStatus(d.message + "\n" + d.image_path);
  }

  async function clickAnchor(ev) {
    if (!inputImg.src) return;
    const rect = inputImg.getBoundingClientRect();
    const xDisp = ev.clientX - rect.left;
    const yDisp = ev.clientY - rect.top;
    const x = xDisp * (inputImg.naturalWidth / rect.width);
    const y = yDisp * (inputImg.naturalHeight / rect.height);
    const d = await apiPost("/api/click", {x, y});
    if (!d.ok) throw new Error(d.message || "click failed");
    inputImg.src = d.overlay_data_url;
    panelImg.src = d.panel_data_url;
    setStatus(d.message + "\n" + d.panel_file);
  }

  document.getElementById("prevBtn").onclick = () => selectIndex(Math.max(0, currentIdx - 1));
  document.getElementById("nextBtn").onclick = () => selectIndex(Math.min(files.length - 1, currentIdx + 1));
  document.getElementById("jumpBtn").onclick = () => {
    const v = Number(document.getElementById("jumpIdx").value || "0");
    if (!Number.isFinite(v)) return;
    selectIndex(Math.max(0, Math.min(files.length - 1, Math.floor(v))));
  };
  inputImg.onclick = (ev) => clickAnchor(ev).catch((e) => setStatus(String(e)));

  (async () => {
    try {
      await loadList();
      await selectIndex(currentIdx);
      setStatus("ready: file 더블클릭 또는 input 클릭");
    } catch (e) {
      setStatus(String(e));
    }
  })();
</script>
</body>
</html>
"""


class AnchorHandler(BaseHTTPRequestHandler):
    state: BrowserState | None = None

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, code: int, html: str):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_html(200, HTML_PAGE)
            return
        if self.state is None:
            self._send_json(500, {"ok": False, "message": "server state not initialized"})
            return
        if path == "/api/health":
            self._send_json(200, {"ok": True})
            return
        if path == "/api/list":
            payload = self.state.list_payload()
            payload["ok"] = True
            self._send_json(200, payload)
            return
        if path == "/api/select":
            q = parse_qs(parsed.query)
            idx = int(q.get("idx", [self.state.current_idx])[0])
            try:
                data = self.state.select_index(idx)
                self._send_json(200, data)
            except Exception as exc:
                self._send_json(500, {"ok": False, "message": str(exc)})
            return
        self._send_json(404, {"ok": False, "message": f"not found: {path}"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if self.state is None:
            self._send_json(500, {"ok": False, "message": "server state not initialized"})
            return
        if parsed.path != "/api/click":
            self._send_json(404, {"ok": False, "message": f"not found: {parsed.path}"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            payload = json.loads(raw.decode("utf-8"))
            x = float(payload.get("x", 0.0))
            y = float(payload.get("y", 0.0))
            data = self.state.click_anchor(x, y)
            self._send_json(200, data)
        except Exception as exc:
            self._send_json(500, {"ok": False, "message": str(exc)})

    def log_message(self, fmt: str, *args):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} | {self.address_string()} | {fmt % args}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Browser anchor picker server")
    p.add_argument("--acdc_root", type=str, default="/mnt/d/ACDC")
    p.add_argument("--output_dir", type=str, default="exp/dino_segformer_layer_match/browser_picker")
    p.add_argument("--condition", type=str, default="night", choices=CONDITIONS)
    ref_group = p.add_mutually_exclusive_group()
    ref_group.add_argument("--include_ref", dest="include_ref", action="store_true")
    ref_group.add_argument("--exclude_ref", dest="include_ref", action="store_false")
    p.set_defaults(include_ref=False)
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
    p.add_argument("--gpu_retry", type=int, default=8)
    p.add_argument("--gpu_retry_sleep_sec", type=float, default=2.0)
    p.add_argument("--dry_run", action="store_true", help="load first image, one center click, then exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_gpu_or_exit(retry=int(args.gpu_retry), retry_sleep_sec=float(args.gpu_retry_sleep_sec))

    state = BrowserState(args)
    print(f"[server] condition={args.condition}, include_ref={args.include_ref}, images={len(state.items)}")
    print(f"[server] output_dir={state.output_dir}")
    print(f"[server] selected={state.current_item.path if state.current_item else '-'}")

    if args.dry_run:
        assert state.current_image_uint8 is not None
        h, w = state.current_image_uint8.shape[:2]
        state.click_anchor(w * 0.5, h * 0.5)
        print("[server] dry_run done: generated latest_input_anchors.png + latest_panel.png")
        return 0

    AnchorHandler.state = state
    server = ThreadingHTTPServer((args.host, args.port), AnchorHandler)
    print(f"[server] URL: http://{args.host}:{args.port}")
    print("[server] browser에서 file 더블클릭으로 이미지 선택, input 클릭으로 앵커 생성")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
