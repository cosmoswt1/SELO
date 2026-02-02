"""SELO v0 model: stage1 tiny adapter + projector + frozen DINO teacher."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_backbone import SegFormerBackbone
from .dino_teacher import DinoTeacher


class TinyResidualAdapter2d(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 0.25, scale_init: float = 0.1):
        super().__init__()
        hidden = max(1, int(channels * hidden_ratio))
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(scale_init))

        # near-identity init
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.conv2(self.act(self.conv1(x)))


class SeloV0Model(nn.Module):
    def __init__(
        self,
        segformer_model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        dino_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_classes: int = 19,
        adapter_hidden_ratio: float = 0.25,
        adapter_scale: float = 0.1,
    ):
        super().__init__()
        self.backbone = SegFormerBackbone(model_name=segformer_model, num_classes=num_classes)
        ch1 = self.backbone.out_channels[0]
        self.stage1_adapter = TinyResidualAdapter2d(
            channels=ch1,
            hidden_ratio=adapter_hidden_ratio,
            scale_init=adapter_scale,
        )
        self.dino_teacher = DinoTeacher(model_name=dino_model)
        for p in self.dino_teacher.parameters():
            p.requires_grad = False
        self.dino_teacher.eval()

        dino_dim = self.dino_teacher.embed_dim
        self.stage1_proj = nn.Conv2d(ch1, dino_dim, kernel_size=1, bias=True)
        nn.init.normal_(self.stage1_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.stage1_proj.bias)

    def freeze_backbone(self):
        for p in self.backbone.segformer.segformer.parameters():
            p.requires_grad = False
        for p in self.backbone.segformer.decode_head.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, use_dino: bool = True) -> dict:
        feats = self.backbone.forward_encoder(x)
        f1, f2, f3, f4 = feats
        f1_raw = f1
        f1_adapt = self.stage1_adapter(f1)
        logits = self.backbone.forward_decoder([f1_adapt, f2, f3, f4])

        out = {
            "logits": logits,
            "features": {"f1": f1_adapt, "f2": f2, "f3": f3, "f4": f4},
            "stage1_raw": f1_raw,
            "stage1_adapt": f1_adapt,
        }

        if use_dino:
            with torch.no_grad():
                tokens, (h, w) = self.dino_teacher.get_patch_tokens(x, align_mode="resize")
            b, p, d = tokens.shape
            feat = tokens.permute(0, 2, 1).reshape(b, d, h, w)
            out["dino_feat"] = feat
            out["dino_grid"] = (h, w)

        return out
