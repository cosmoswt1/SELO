"""SELO v0 model: stage3 tiny adapter + projector + frozen DINO teacher."""

from __future__ import annotations

import torch
import torch.nn as nn

from .segformer_backbone import SegFormerBackbone
from .dino_teacher import DinoTeacher


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dim for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(int(channels), eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H, W, C] for nn.LayerNorm(C)
        y = x.permute(0, 2, 3, 1)
        y = self.ln(y)
        return y.permute(0, 3, 1, 2).contiguous()


class Projector2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        proj_type: str = "conv",
        mlp_hidden: int = 256,
        ln_eps: float = 1e-6,
    ):
        super().__init__()
        proj_type = str(proj_type).lower().strip()
        self.proj_type = proj_type

        if proj_type in ("conv", "linear", "1x1"):
            self.proj = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=True)
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.proj.bias)
            return

        if proj_type in ("mlp", "mlp2", "mlp2ln"):
            hidden = int(mlp_hidden)
            if hidden <= 0:
                raise ValueError(f"mlp_hidden must be > 0 (got {mlp_hidden})")
            self.proj = nn.Sequential(
                nn.Conv2d(int(in_channels), hidden, kernel_size=1, bias=True),
                nn.GELU(),
                LayerNorm2d(hidden, eps=float(ln_eps)),
                nn.Conv2d(hidden, int(out_channels), kernel_size=1, bias=True),
            )
            nn.init.normal_(self.proj[0].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.proj[0].bias)
            nn.init.normal_(self.proj[3].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.proj[3].bias)
            return

        raise ValueError(f"Unknown proj_type: {proj_type}. Use 'conv' or 'mlp'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TinyResidualAdapter2d(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 1, scale_init: float = 0.1):
        super().__init__()
        hidden = max(1, int(channels * hidden_ratio))
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        # Spatial mixing (depthwise 3x3) for boundary/structure signal with minimal overhead.
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=False,
        )
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        # Channel-wise residual scale (start near-identity).
        # Shape: [C], broadcast to [1,C,1,1] in forward.
        self.scale = nn.Parameter(torch.full((int(channels),), float(scale_init)))

        # near-identity init
        nn.init.dirac_(self.dwconv.weight)
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.dwconv(h)
        h = self.conv2(h)
        return x + self.scale.view(1, -1, 1, 1) * h


class SeloV0Model(nn.Module):
    def __init__(
        self,
        segformer_model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        dino_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        num_classes: int = 19,
        adapter_hidden_ratio: float = 0.25,
        adapter_scale: float = 0.1,
        proj_type: str = "conv",
        proj_mlp_hidden: int = 256,
    ):
        super().__init__()
        self.backbone = SegFormerBackbone(model_name=segformer_model, num_classes=num_classes)
        ch3 = self.backbone.out_channels[2]
        self.stage3_adapter = TinyResidualAdapter2d(
            channels=ch3,
            hidden_ratio=adapter_hidden_ratio,
            scale_init=adapter_scale,
        )
        self.dino_teacher = DinoTeacher(model_name=dino_model)
        for p in self.dino_teacher.parameters():
            p.requires_grad = False
        self.dino_teacher.eval()

        dino_dim = self.dino_teacher.embed_dim
        self.stage3_proj = Projector2d(
            ch3,
            dino_dim,
            proj_type=proj_type,
            mlp_hidden=int(proj_mlp_hidden),
        )

    def freeze_backbone(self):
        for p in self.backbone.segformer.segformer.parameters():
            p.requires_grad = False
        for p in self.backbone.segformer.decode_head.parameters():
            p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        use_dino: bool = True,
        compute_logits: bool = True,
    ) -> dict:
        f1, f2, f3_raw = self.backbone.forward_stage3(x)
        f3_adapt = self.stage3_adapter(f3_raw)

        out = {
            "logits": None,
            "features": {"f1": f1, "f2": f2, "f3": f3_adapt},
            "stage3_raw": f3_raw,
            "stage3_adapt": f3_adapt,
        }

        if compute_logits:
            f4 = self.backbone.forward_from_stage3(f3_adapt)
            logits = self.backbone.forward_decoder([f1, f2, f3_adapt, f4])
            out["logits"] = logits
            out["features"].update({"f4": f4})

        if use_dino:
            with torch.no_grad():
                tokens, (h, w) = self.dino_teacher.get_patch_tokens(x, align_mode="resize")
            b, p, d = tokens.shape
            feat = tokens.permute(0, 2, 1).reshape(b, d, h, w)
            out["dino_feat"] = feat
            out["dino_grid"] = (h, w)

        return out
