"""SegFormer backbone wrapper (HF) for encoder features + decoder logits."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class SegFormerBackbone(nn.Module):
    CHANNELS = {
        "b0": [32, 64, 160, 256],
        "b1": [64, 128, 320, 512],
        "b2": [64, 128, 320, 512],
        "b3": [64, 128, 320, 512],
        "b4": [64, 128, 320, 512],
        "b5": [64, 128, 320, 512],
    }

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        num_classes: int = 19,
    ):
        super().__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            output_hidden_states=True,
            return_dict=True,
        )
        self.segformer.config.output_hidden_states = True
        self.segformer.config.return_dict = True

        variant = "b5"
        for v in ["b0", "b1", "b2", "b3", "b4", "b5"]:
            if v in model_name.lower():
                variant = v
                break
        self.out_channels = self.CHANNELS[variant]
        self.stage_strides = (4, 8, 16, 32)

    def _run_encoder_stage(self, hidden_states: torch.Tensor, stage_idx: int) -> torch.Tensor:
        encoder = self.segformer.segformer.encoder
        embedding_layer = encoder.patch_embeddings[stage_idx]
        block_layer = encoder.block[stage_idx]
        norm_layer = encoder.layer_norm[stage_idx]

        hidden_states, height, width = embedding_layer(hidden_states)
        for blk in block_layer:
            hidden_states = blk(hidden_states, height, width, output_attentions=False)[0]
        hidden_states = norm_layer(hidden_states)

        if stage_idx != len(encoder.patch_embeddings) - 1 or self.segformer.config.reshape_last_stage:
            hidden_states = (
                hidden_states.reshape(hidden_states.shape[0], height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        return hidden_states

    def forward_stage1(self, x: torch.Tensor) -> torch.Tensor:
        return self._run_encoder_stage(x, stage_idx=0)

    def forward_stage3(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder stages 0..2 and return (f1, f2, f3)."""
        f1 = self.forward_stage1(x)
        f2 = self._run_encoder_stage(f1, stage_idx=1)
        f3 = self._run_encoder_stage(f2, stage_idx=2)
        return f1, f2, f3

    def forward_from_stage3(self, f3: torch.Tensor) -> torch.Tensor:
        """Run encoder stage 3 only, given stage3 feature map."""
        return self._run_encoder_stage(f3, stage_idx=3)

    def forward_from_stage1(self, f1: torch.Tensor) -> list[torch.Tensor]:
        f2 = self._run_encoder_stage(f1, stage_idx=1)
        f3 = self._run_encoder_stage(f2, stage_idx=2)
        f4 = self._run_encoder_stage(f3, stage_idx=3)
        return [f2, f3, f4]

    def forward_encoder(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.forward_stage1(x)
        f2, f3, f4 = self.forward_from_stage1(f1)
        return [f1, f2, f3, f4]

    def forward_decoder(self, features: list[torch.Tensor]) -> torch.Tensor:
        encoder_hidden_states = tuple(features)
        logits = self.segformer.decode_head(encoder_hidden_states)
        return logits

    def forward(self, x: torch.Tensor):
        feats = self.forward_encoder(x)
        logits = self.forward_decoder(feats)
        return feats, logits
