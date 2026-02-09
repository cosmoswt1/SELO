"""Local affinity KL distillation loss (kxk only, anchor sampling)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAffinityKLLoss(nn.Module):
    def __init__(
        self,
        k: int = 7,
        tau: float = 0.1,
        anchors: int = 512,
        candidates: int | None = None,
        candidate_chunk: int = 512,
        kcenter_top_m: int = 0,
        per_image: bool = True,
    ):
        super().__init__()
        self.k = int(k)
        self.tau = float(tau)
        self.anchors = int(anchors)
        self.candidates = int(self.anchors if candidates is None else candidates)
        self.candidate_chunk = int(candidate_chunk)
        self.kcenter_top_m = int(kcenter_top_m)
        self.per_image = bool(per_image)

        radius = self.k // 2
        self._offsets: list[tuple[int, int]] = [
            (dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
        ]

    def _kcenter_farthest_point(
        self,
        pts_yx: torch.Tensor,
        first_idx: int,
        k: int,
    ) -> torch.Tensor:
        # pts_yx: [P,2] float32/float16, select indices (long) of size k
        p = int(pts_yx.shape[0])
        k = int(k)
        if k >= p:
            return torch.arange(p, device=pts_yx.device, dtype=torch.long)

        selected = torch.empty((k,), device=pts_yx.device, dtype=torch.long)
        selected[0] = int(first_idx)

        diff = pts_yx - pts_yx[selected[0]].unsqueeze(0)  # [P,2]
        dist = (diff * diff).sum(dim=1)  # [P]
        dist[selected[0]] = -1.0

        for i in range(1, k):
            far = torch.argmax(dist)
            selected[i] = far
            diff = pts_yx - pts_yx[far].unsqueeze(0)
            d = (diff * diff).sum(dim=1)
            dist = torch.minimum(dist, d)
            dist[far] = -1.0

        return selected

    def _sample_anchors(self, h: int, w: int, device: torch.device, count: int) -> torch.Tensor:
        radius = self.k // 2
        if h <= 2 * radius or w <= 2 * radius:
            raise RuntimeError(f"Feature map too small for k={self.k}: ({h},{w})")
        low_h, high_h = radius, h - radius - 1
        low_w, high_w = radius, w - radius - 1
        grid_h = high_h - low_h + 1
        grid_w = high_w - low_w + 1
        total = grid_h * grid_w
        count = min(int(count), total)
        idx = torch.randperm(total, device=device)[:count]
        ys = (idx // grid_w) + low_h
        xs = (idx % grid_w) + low_w
        return torch.stack([ys, xs], dim=1)  # [M, 2], unique

    def _local_sim_norm(self, feat_norm: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        # feat_norm: [B, C, H, W] already normalized, anchors: [M, 2] (y, x)
        b, _c, _h, _w = feat_norm.shape
        m = anchors.shape[0]
        k2 = len(self._offsets)

        ys = anchors[:, 0]
        xs = anchors[:, 1]
        center = feat_norm[:, :, ys, xs]  # [B, C, M]

        sims = torch.empty((b, m, k2), device=feat_norm.device, dtype=torch.float32)
        for oi, (dy, dx) in enumerate(self._offsets):
            neigh = feat_norm[:, :, ys + dy, xs + dx]  # [B, C, M]
            sims[:, :, oi] = (center * neigh).sum(dim=1, dtype=torch.float32)
        return sims  # [B, M, K]

    def _local_sim_norm_per_image(self, feat_norm: torch.Tensor, anchors_yx: torch.Tensor) -> torch.Tensor:
        # feat_norm: [B, C, H, W] already normalized
        # anchors_yx: [B, M, 2] (y, x) per image
        b, c, h, w = feat_norm.shape
        m = int(anchors_yx.shape[1])
        k2 = len(self._offsets)

        # Flatten spatial dims and gather using linear indices for each sample.
        feat_flat = feat_norm.reshape(b, c, h * w)
        ys = anchors_yx[:, :, 0]
        xs = anchors_yx[:, :, 1]
        lin = ys * w + xs  # [B, M]
        lin_exp = lin.unsqueeze(1).expand(b, c, m)
        center = torch.gather(feat_flat, dim=2, index=lin_exp)  # [B, C, M]

        sims = torch.empty((b, m, k2), device=feat_norm.device, dtype=torch.float32)
        for oi, (dy, dx) in enumerate(self._offsets):
            ys2 = ys + dy
            xs2 = xs + dx
            lin2 = ys2 * w + xs2
            lin2_exp = lin2.unsqueeze(1).expand(b, c, m)
            neigh = torch.gather(feat_flat, dim=2, index=lin2_exp)  # [B, C, M]
            sims[:, :, oi] = (center * neigh).sum(dim=1, dtype=torch.float32)
        return sims  # [B, M, K]

    def _compute_candidate_entropy(self, teacher_norm: torch.Tensor, candidate_anchors: torch.Tensor) -> torch.Tensor:
        # teacher_norm: [B, C, H, W] normalized, candidate_anchors: [M,2]
        b, _c, _h, _w = teacher_norm.shape
        m = int(candidate_anchors.shape[0])
        k2 = len(self._offsets)
        chunk = max(1, int(self.candidate_chunk))
        ent_all = torch.empty((b, m), device=teacher_norm.device, dtype=torch.float32)

        for start in range(0, m, chunk):
            end = min(m, start + chunk)
            anchors = candidate_anchors[start:end]
            ys = anchors[:, 0]
            xs = anchors[:, 1]
            center = teacher_norm[:, :, ys, xs]  # [B, C, M]

            sims = torch.empty((b, end - start, k2), device=teacher_norm.device, dtype=torch.float32)
            for oi, (dy, dx) in enumerate(self._offsets):
                neigh = teacher_norm[:, :, ys + dy, xs + dx]  # [B, C, M]
                sims[:, :, oi] = (center * neigh).sum(dim=1, dtype=torch.float32) / self.tau

            p = torch.softmax(sims, dim=2)
            ent = -(p * (p + 1e-6).log()).sum(dim=2)  # [B, M]
            ent_all[:, start:end] = ent

        return ent_all

    def _select_indices_from_entropy_mean(self, ent_mean: torch.Tensor, select_count: int, candidate_anchors: torch.Tensor) -> torch.Tensor:
        # ent_mean: [M]
        m = int(candidate_anchors.shape[0])
        k = min(int(select_count), m)
        if self.kcenter_top_m <= 0:
            return torch.topk(ent_mean, k=k, largest=False).indices

        score = -ent_mean
        pool_m = min(m, max(k, int(self.kcenter_top_m)))
        pool_idx = torch.topk(score, k=pool_m, largest=True).indices  # [pool_m]
        pool_anchors = candidate_anchors[pool_idx]  # [pool_m,2]

        # Greedy k-center (farthest point): first = best score, then farthest from current set.
        first_local = 0  # topk is sorted: best score first
        sel_local = self._kcenter_farthest_point(pool_anchors.float(), first_idx=first_local, k=k)
        return pool_idx[sel_local]

    def _select_indices_per_image(self, ent_bm: torch.Tensor, select_count: int, candidate_anchors: torch.Tensor) -> torch.Tensor:
        # ent_bm: [B, M]
        b, m = ent_bm.shape
        k = min(int(select_count), int(m))
        if self.kcenter_top_m <= 0:
            return torch.topk(ent_bm, k=k, largest=False, dim=1).indices  # [B,k]

        score = -ent_bm  # [B,M]
        pool_m = min(int(m), max(k, int(self.kcenter_top_m)))
        pool_idx = torch.topk(score, k=pool_m, largest=True, dim=1).indices  # [B,pool_m]
        out_idx = torch.empty((b, k), device=ent_bm.device, dtype=torch.long)

        # Greedy k-center independently per image. (B is small; loop is acceptable.)
        for bi in range(int(b)):
            pool_anchors = candidate_anchors[pool_idx[bi]].float()  # [pool_m,2]
            sel_local = self._kcenter_farthest_point(pool_anchors, first_idx=0, k=k)
            out_idx[bi] = pool_idx[bi, sel_local]

        return out_idx  # [B,k]

    def _select_anchors_by_teacher_entropy(
        self,
        teacher_norm: torch.Tensor,
        candidate_anchors: torch.Tensor,
        select_count: int,
    ) -> torch.Tensor:
        # Backward-compat helper (kept for external call sites): batch-mean selection.
        m = int(candidate_anchors.shape[0])
        if m <= select_count:
            return candidate_anchors
        ent_bm = self._compute_candidate_entropy(teacher_norm=teacher_norm, candidate_anchors=candidate_anchors)
        ent_mean = ent_bm.mean(dim=0)
        idx = self._select_indices_from_entropy_mean(ent_mean=ent_mean, select_count=select_count, candidate_anchors=candidate_anchors)
        return candidate_anchors[idx]

    def forward(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        return_stats: bool = False,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if student.shape[-2:] != teacher.shape[-2:]:
            teacher = F.interpolate(teacher, size=student.shape[-2:], mode="bilinear", align_corners=False)

        bsz, _, h, w = student.shape
        candidate_count = max(1, int(self.candidates))
        final_count = max(1, int(self.anchors))
        candidate_anchors = self._sample_anchors(h, w, student.device, count=candidate_count)

        cand_ent = None
        sel_idx = None
        with torch.no_grad():
            teacher_norm = F.normalize(teacher, dim=1)
            need_entropy = return_debug or (candidate_anchors.shape[0] > final_count)
            if need_entropy:
                cand_ent = self._compute_candidate_entropy(teacher_norm=teacher_norm, candidate_anchors=candidate_anchors)  # [B,M]

            if candidate_anchors.shape[0] > final_count:
                assert cand_ent is not None
                if self.per_image:
                    sel_idx = self._select_indices_per_image(
                        ent_bm=cand_ent,
                        select_count=final_count,
                        candidate_anchors=candidate_anchors,
                    )  # [B,K]
                    anchors = candidate_anchors[sel_idx]  # [B,K,2]
                else:
                    ent_mean = cand_ent.mean(dim=0)
                    sel_idx = self._select_indices_from_entropy_mean(
                        ent_mean=ent_mean,
                        select_count=final_count,
                        candidate_anchors=candidate_anchors,
                    )  # [K]
                    anchors = candidate_anchors[sel_idx]  # [K,2]
            else:
                anchors = candidate_anchors

        student_norm = F.normalize(student, dim=1)
        if anchors.ndim == 3:
            sim_s = self._local_sim_norm_per_image(student_norm, anchors) / self.tau
        else:
            sim_s = self._local_sim_norm(student_norm, anchors) / self.tau
        with torch.no_grad():
            if anchors.ndim == 3:
                sim_t = self._local_sim_norm_per_image(teacher_norm, anchors) / self.tau
            else:
                sim_t = self._local_sim_norm(teacher_norm, anchors) / self.tau

        p_s = F.log_softmax(sim_s, dim=2)
        p_t = F.softmax(sim_t, dim=2)
        loss = F.kl_div(p_s, p_t, reduction="none").sum(dim=2)  # [B, M]
        out = loss.mean()
        if (not return_stats) and (not return_debug):
            return out

        # logging-only stats (avoid autograd overhead)
        with torch.no_grad():
            p_s_prob = p_s.exp()
            p_t_prob = p_t
            k2 = int(self.k * self.k)
            anchors_count = int(anchors.shape[1]) if anchors.ndim == 3 else int(anchors.shape[0])
            stats = {
                "tau": float(self.tau),
                "candidates": int(candidate_anchors.shape[0]),
                "anchors": anchors_count,
                "k2": k2,
                "p_t_max": float(p_t_prob.max(dim=2).values.mean().item()),
                "p_t_ent": float(-(p_t_prob * (p_t_prob + 1e-6).log()).sum(dim=2).mean().item()),
                "p_s_max": float(p_s_prob.max(dim=2).values.mean().item()),
                "p_s_ent": float(-(p_s_prob * p_s).sum(dim=2).mean().item()),
            }
        if not return_debug:
            return out, stats

        # return_debug: provide anchor coordinates and entropy for quick diagnostics/visualization
        with torch.no_grad():
            if cand_ent is None:
                cand_ent = self._compute_candidate_entropy(teacher_norm=teacher_norm, candidate_anchors=candidate_anchors)

            # Unify shapes for consumers: selected_anchors always [B, M, 2], selected_ent always [B, M]
            if anchors.ndim == 2:
                anchors_b = anchors.unsqueeze(0).expand(int(bsz), -1, -1)
            else:
                anchors_b = anchors

            if sel_idx is None:
                # No selection happened (candidates <= anchors): treat all candidates as selected.
                sel_ent = cand_ent
            else:
                if sel_idx.ndim == 1:
                    sel_ent = cand_ent[:, sel_idx]  # [B,K]
                else:
                    sel_ent = cand_ent.gather(dim=1, index=sel_idx)  # [B,K]

            debug = {
                "candidate_anchors": candidate_anchors.detach().cpu(),
                "candidate_ent": cand_ent.detach().cpu(),  # [B,M]
                "selected_anchors": anchors_b.detach().cpu(),  # [B,K,2]
                "selected_ent": sel_ent.detach().cpu(),  # [B,K]
                "hw": (int(h), int(w)),
                "per_image": bool(self.per_image),
            }
        return out, stats, debug
