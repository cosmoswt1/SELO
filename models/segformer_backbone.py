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

    def forward_encoder(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.segformer.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            hidden_states = outputs.encoder_hidden_states
        if hidden_states is None:
            raise RuntimeError("SegFormer did not return encoder hidden states.")
        return list(hidden_states[-4:])

    def forward_decoder(self, features: list[torch.Tensor]) -> torch.Tensor:
        encoder_hidden_states = tuple(features)
        logits = self.segformer.decode_head(encoder_hidden_states)
        return logits

    def forward(self, x: torch.Tensor):
        feats = self.forward_encoder(x)
        logits = self.forward_decoder(feats)
        return feats, logits
