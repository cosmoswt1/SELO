"""Stage3 adapter model for SFDA CKA training."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.dino_teacher import DinoTeacher
from models.segformer_backbone import SegFormerBackbone


class BottleneckDWResidualAdapter(nn.Module):
    """Bottleneck + depthwise conv adapter producing delta feature."""

    def __init__(self, in_channels: int = 320, bottleneck: int = 128):
        super().__init__()
        self.reduce = nn.Conv2d(int(in_channels), int(bottleneck), kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            int(bottleneck),
            int(bottleneck),
            kernel_size=3,
            padding=1,
            groups=int(bottleneck),
            bias=True,
        )
        self.act = nn.GELU()
        self.expand = nn.Conv2d(int(bottleneck), int(in_channels), kernel_size=1, bias=True)

        # Stable start: near-zero residual update.
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(x)
        h = self.dwconv(h)
        h = self.act(h)
        h = self.expand(h)
        return h


class SpatialGate(nn.Module):
    """Spatial gate g in shape [B,1,H,W]."""

    def __init__(self, in_channels: int = 320, bias_init: float = -4.0):
        super().__init__()
        c = int(in_channels)
        self.dwconv = nn.Conv2d(
            c,
            c,
            kernel_size=3,
            padding=1,
            groups=c,
            bias=True,
        )
        gn_groups = min(32, c)
        while c % gn_groups != 0 and gn_groups > 1:
            gn_groups //= 2
        self.norm = nn.GroupNorm(num_groups=gn_groups, num_channels=c, eps=1e-5, affine=True)
        self.proj = nn.Conv2d(c, 1, kernel_size=1, bias=True)

        # Keep gate nearly closed, but preserve gradient path for DWConv.
        with torch.no_grad():
            self.dwconv.weight.zero_()
            self.dwconv.weight[:, 0, 1, 1] = 1.0  # identity-like depthwise kernel
            if self.dwconv.bias is not None:
                self.dwconv.bias.zero_()
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.proj.bias, float(bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.proj(h)
        return torch.sigmoid(h)


class Stage3CKAModel(nn.Module):
    """SegFormer backbone + stage3 adapter + optional frozen DINO teacher."""

    def __init__(
        self,
        *,
        segformer_model: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        dino_model: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        dino_layer: int = 24,
        num_classes: int = 19,
        adapter_bottleneck: int = 128,
        gate_bias_init: float = -4.0,
        enable_dino: bool = True,
        force_gate_one: bool = False,
    ):
        super().__init__()
        self.backbone = SegFormerBackbone(model_name=segformer_model, num_classes=num_classes)
        ch3 = int(self.backbone.out_channels[2])
        self.stage3_adapter = BottleneckDWResidualAdapter(
            in_channels=ch3,
            bottleneck=int(adapter_bottleneck),
        )
        self.stage3_gate = SpatialGate(in_channels=ch3, bias_init=float(gate_bias_init))
        self.force_gate_one = bool(force_gate_one)

        self.enable_dino = bool(enable_dino)
        self.dino_layer = int(dino_layer)
        self.dino_teacher: DinoTeacher | None = None
        if self.enable_dino:
            self.dino_teacher = DinoTeacher(model_name=dino_model)
            for p in self.dino_teacher.parameters():
                p.requires_grad = False
            self.dino_teacher.eval()

    def freeze_backbone(self) -> None:
        for p in self.backbone.segformer.segformer.parameters():
            p.requires_grad = False
        for p in self.backbone.segformer.decode_head.parameters():
            p.requires_grad = False

    def freeze_dino(self) -> None:
        if self.dino_teacher is None:
            return
        for p in self.dino_teacher.parameters():
            p.requires_grad = False
        self.dino_teacher.eval()

    @torch.no_grad()
    def _extract_dino_layer_feat(
        self,
        x: torch.Tensor,
        *,
        strict_same_resolution: bool = True,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.dino_teacher is None:
            raise RuntimeError("DINO teacher is disabled (enable_dino=False).")

        teacher = self.dino_teacher
        x_aligned, new_h, new_w = teacher._align_to_patch(x, mode="resize")
        if strict_same_resolution and (new_h != x.shape[-2] or new_w != x.shape[-1]):
            raise RuntimeError(
                "DINO alignment changed input resolution. "
                f"input={tuple(x.shape[-2:])}, aligned={(new_h, new_w)}"
            )

        ph, pw = new_h // teacher.patch_size, new_w // teacher.patch_size
        expected = ph * pw
        outs = teacher.dino(pixel_values=x_aligned, output_hidden_states=True, return_dict=True)
        hidden_states = outs.hidden_states
        if self.dino_layer <= 0 or self.dino_layer >= len(hidden_states):
            raise RuntimeError(
                f"Invalid dino_layer={self.dino_layer}. "
                f"valid layer index range is [1, {len(hidden_states)-1}]"
            )

        # For the final transformer block, use post-LN tokens (`last_hidden_state`).
        # Earlier layers keep pre-LN hidden states for layer-wise probing consistency.
        if self.dino_layer == (len(hidden_states) - 1):
            hs = outs.last_hidden_state  # [B, seq, D]
        else:
            hs = hidden_states[self.dino_layer]  # [B, seq, D]
        seq_len = int(hs.shape[1])
        prefix = seq_len - int(expected)
        if prefix < 0:
            raise RuntimeError(f"DINO token mismatch: seq_len={seq_len}, expected={expected}")
        if prefix != teacher.num_prefix_tokens:
            raise RuntimeError(
                "DINO prefix token mismatch: "
                f"seq_len={seq_len}, expected={expected}, prefix={prefix}, "
                f"cfg_prefix={teacher.num_prefix_tokens}"
            )
        tokens = hs[:, prefix:, :]  # [B, ph*pw, D]
        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], ph, pw)
        return feat, (ph, pw)

    def forward(
        self,
        x: torch.Tensor,
        *,
        adapter_enabled: bool = True,
        return_intermediates: bool = False,
        gate_detach_for_align: bool = True,
        update_mask: torch.Tensor | None = None,
        use_dino: bool = False,
        compute_logits: bool = True,
        need_stage4_anchor: bool = False,
        strict_dino_resolution: bool = True,
    ) -> dict:
        f1, f2, f3_raw = self.backbone.forward_stage3(x)
        if adapter_enabled:
            delta = self.stage3_adapter(f3_raw)
            if self.force_gate_one:
                gate = torch.ones(
                    (f3_raw.shape[0], 1, f3_raw.shape[2], f3_raw.shape[3]),
                    device=f3_raw.device,
                    dtype=f3_raw.dtype,
                )
            else:
                gate = self.stage3_gate(f3_raw)
            if update_mask is None:
                upd_mask = torch.ones(
                    (f3_raw.shape[0], 1, f3_raw.shape[2], f3_raw.shape[3]),
                    device=f3_raw.device,
                    dtype=f3_raw.dtype,
                )
            else:
                upd_mask = update_mask
                if upd_mask.dim() == 3:
                    upd_mask = upd_mask.unsqueeze(1)
                if upd_mask.shape[-2:] != f3_raw.shape[-2:]:
                    upd_mask = torch.nn.functional.interpolate(
                        upd_mask.float(),
                        size=f3_raw.shape[-2:],
                        mode="nearest",
                    )
                upd_mask = upd_mask.to(device=f3_raw.device, dtype=f3_raw.dtype).clamp(0.0, 1.0)
            update = upd_mask * gate * delta
            gate_for_align = gate.detach() if bool(gate_detach_for_align) else gate
            f3_align = f3_raw + upd_mask * gate_for_align * delta
            f3_pred = f3_raw + update
        else:
            delta = torch.zeros_like(f3_raw)
            gate = torch.zeros(
                (f3_raw.shape[0], 1, f3_raw.shape[2], f3_raw.shape[3]),
                device=f3_raw.device,
                dtype=f3_raw.dtype,
            )
            update = torch.zeros_like(f3_raw)
            upd_mask = torch.zeros(
                (f3_raw.shape[0], 1, f3_raw.shape[2], f3_raw.shape[3]),
                device=f3_raw.device,
                dtype=f3_raw.dtype,
            )
            gate_for_align = gate
            f3_align = f3_raw
            f3_pred = f3_raw

        out = {"logits": None}
        if return_intermediates or adapter_enabled:
            out.update(
                {
                    "stage3_raw": f3_raw,
                    "delta": delta,
                    "gate": gate,
                    "gate_align": gate_for_align,
                    "update_mask": upd_mask,
                    "update": update,
                    "stage3_adapt": f3_pred,
                    "stage3_align": f3_align,
                    "stage4_adapt": None,
                    "stage4_base": None,
                }
            )

        if compute_logits or need_stage4_anchor:
            f4_pred = self.backbone.forward_from_stage3(f3_pred)
            if return_intermediates or adapter_enabled:
                out["stage4_adapt"] = f4_pred
            if compute_logits:
                logits = self.backbone.forward_decoder([f1, f2, f3_pred, f4_pred])
                out["logits"] = logits

        if need_stage4_anchor and (return_intermediates or adapter_enabled):
            with torch.no_grad():
                f4_base = self.backbone.forward_from_stage3(f3_raw)
            out["stage4_base"] = f4_base

        if use_dino:
            if not self.enable_dino:
                raise RuntimeError("use_dino=True but model was initialized with enable_dino=False.")
            with torch.no_grad():
                dino_feat, dino_grid = self._extract_dino_layer_feat(
                    x,
                    strict_same_resolution=bool(strict_dino_resolution),
                )
            out["dino_feat"] = dino_feat
            out["dino_grid"] = dino_grid

        return out
