"""Local affinity KL distillation loss (kxk only, no NxN)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAffinityKLLoss(nn.Module):
    def __init__(self, k: int = 5, tau: float = 0.1):
        super().__init__()
        self.k = int(k)
        self.tau = float(tau)

    def _local_sim(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W]
        b, c, h, w = feat.shape
        feat = F.normalize(feat, dim=1)
        patches = F.unfold(feat, kernel_size=self.k, padding=self.k // 2)
        patches = patches.view(b, c, self.k * self.k, h * w)
        center = feat.view(b, c, 1, h * w)
        sim = (center * patches).sum(dim=1)  # [B, K, H*W]
        return sim

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        if student.shape[-2:] != teacher.shape[-2:]:
            teacher = F.interpolate(teacher, size=student.shape[-2:], mode="bilinear", align_corners=False)

        sim_s = self._local_sim(student) / self.tau
        with torch.no_grad():
            sim_t = self._local_sim(teacher) / self.tau

        p_s = F.log_softmax(sim_s, dim=1)
        p_t = F.softmax(sim_t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="batchmean")
        return loss
