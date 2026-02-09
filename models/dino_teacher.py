"""Frozen DINOv3 teacher (patch tokens only)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DinoTeacher(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"):
        super().__init__()
        self.dino = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        cfg = self.dino.config
        self.patch_size = getattr(cfg, "patch_size", 16)
        self.embed_dim = getattr(cfg, "hidden_size", 1024)
        self.num_layers = getattr(cfg, "num_hidden_layers", 24)
        num_register_tokens = getattr(cfg, "num_register_tokens", 0)
        self.num_prefix_tokens = 1 + num_register_tokens

    def train(self, mode: bool = True):
        return super().train(False)

    def _align_to_patch(self, x: torch.Tensor, mode: str = "resize"):
        _, _, h, w = x.shape
        if mode == "resize":
            new_h = max((h // self.patch_size) * self.patch_size, self.patch_size)
            new_w = max((w // self.patch_size) * self.patch_size, self.patch_size)
            if new_h != h or new_w != w:
                x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            return x, new_h, new_w
        if mode == "pad":
            new_h = int(math.ceil(h / self.patch_size) * self.patch_size)
            new_w = int(math.ceil(w / self.patch_size) * self.patch_size)
            if new_h != h or new_w != w:
                pad_h = new_h - h
                pad_w = new_w - w
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            return x, new_h, new_w
        raise ValueError(f"align mode must be resize or pad, got {mode}")

    @torch.no_grad()
    def get_patch_tokens(self, x: torch.Tensor, align_mode: str = "resize"):
        x, new_h, new_w = self._align_to_patch(x, align_mode)
        h, w = new_h // self.patch_size, new_w // self.patch_size
        outputs = self.dino(pixel_values=x, output_hidden_states=False, return_dict=True)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        seq_len = last_hidden.shape[1]
        expected = h * w
        prefix = max(seq_len - expected, 0)


        if prefix < 0:
            raise RuntimeError(
                f"[DinoTeacher] seq_len smaller than expected patches: "
                f"seq_len={seq_len}, expected={expected}, (h,w)=({h},{w}), patch_size={self.patch_size}"
            )

        if prefix != self.num_prefix_tokens:
            raise RuntimeError(
                f"[DinoTeacher] prefix token mismatch: "
                f"seq_len={seq_len}, expected={expected}, prefix={prefix}, "
                f"cfg_num_prefix_tokens={self.num_prefix_tokens}, (h,w)=({h},{w})"
            )


        tokens = last_hidden[:, prefix:, :]  # [B, h*w, D]
        return tokens, (h, w)
