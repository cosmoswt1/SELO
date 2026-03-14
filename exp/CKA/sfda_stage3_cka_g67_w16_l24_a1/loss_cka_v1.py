"""Local gram-space CKA loss (centering-only) for stage3 alignment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def center_token_set(tokens: torch.Tensor) -> torch.Tensor:
    # CKA centering on token axis: [B,N,C] -> [B,N,C]
    return tokens - tokens.mean(dim=1, keepdim=True)


def _l2_normalize_channelwise(feat: torch.Tensor, eps: float) -> torch.Tensor:
    denom = feat.square().sum(dim=1, keepdim=True).sqrt().clamp_min(float(eps))
    return feat / denom


def _build_ssm_divergence_map(student_feat: torch.Tensor, teacher_feat: torch.Tensor, eps: float) -> torch.Tensor:
    # cheap proxy of teacher-student structural mismatch using 4-neighbor cosine differences.
    s = _l2_normalize_channelwise(student_feat.detach().float(), eps=eps)
    t = _l2_normalize_channelwise(teacher_feat.detach().float(), eps=eps)

    cosx_s = (s[:, :, :, :-1] * s[:, :, :, 1:]).sum(dim=1)  # [B,H,W-1]
    cosx_t = (t[:, :, :, :-1] * t[:, :, :, 1:]).sum(dim=1)  # [B,H,W-1]
    cosy_s = (s[:, :, :-1, :] * s[:, :, 1:, :]).sum(dim=1)  # [B,H-1,W]
    cosy_t = (t[:, :, :-1, :] * t[:, :, 1:, :]).sum(dim=1)  # [B,H-1,W]

    cosx_s = F.pad(cosx_s, (0, 1, 0, 0), mode="constant", value=0.0)
    cosx_t = F.pad(cosx_t, (0, 1, 0, 0), mode="constant", value=0.0)
    cosy_s = F.pad(cosy_s, (0, 0, 0, 1), mode="constant", value=0.0)
    cosy_t = F.pad(cosy_t, (0, 0, 0, 1), mode="constant", value=0.0)
    return (cosx_t - cosx_s).abs() + (cosy_t - cosy_s).abs()


def _window_score_map_from_divergence(div_map: torch.Tensor, ws: int) -> torch.Tensor:
    # [B,H,W] -> [B,H-ws+1,W-ws+1]
    kernel = torch.ones((1, 1, int(ws), int(ws)), device=div_map.device, dtype=div_map.dtype)
    return F.conv2d(div_map.unsqueeze(1), kernel, stride=1, padding=0).squeeze(1)


def _lin_to_box(lin_idx: int, w_win: int, ws: int) -> tuple[int, int, int, int]:
    y0 = int(lin_idx) // int(w_win)
    x0 = int(lin_idx) % int(w_win)
    return y0, x0, y0 + int(ws), x0 + int(ws)


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int], eps: float) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0 = max(ay0, by0)
    ix0 = max(ax0, bx0)
    iy1 = min(ay1, by1)
    ix1 = min(ax1, bx1)
    ih = max(0, iy1 - iy0)
    iw = max(0, ix1 - ix0)
    inter = float(ih * iw)
    area_a = float(max(0, ay1 - ay0) * max(0, ax1 - ax0))
    area_b = float(max(0, by1 - by0) * max(0, bx1 - bx0))
    union = max(float(eps), area_a + area_b - inter)
    return inter / union


def _sample_windows_ssm_nms(
    score_map: torch.Tensor,
    *,
    ws: int,
    k: int,
    iou_thr: float,
    topm: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # score_map: [B,h_win,w_win]
    bsz, _h_win, w_win = score_map.shape
    num_all = int(score_map.shape[1] * score_map.shape[2])
    if num_all < int(k):
        raise RuntimeError(f"Not enough candidates for NMS sampling: num_all={num_all}, k={k}")

    topm = int(max(int(k), min(int(topm), num_all)))
    thr = float(iou_thr)

    all_boxes: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    for b in range(bsz):
        flat = score_map[b].reshape(-1)
        order_full = torch.argsort(flat, descending=True)
        order_topm = order_full[:topm].tolist()
        order_full_list = order_full.tolist()

        selected_idx: list[int] = []
        selected_boxes: list[tuple[int, int, int, int]] = []
        selected_set: set[int] = set()

        for lin in order_topm:
            if lin in selected_set:
                continue
            cand = _lin_to_box(lin, w_win=w_win, ws=ws)
            keep = True
            for sel in selected_boxes:
                if _box_iou(cand, sel, eps=eps) > thr:
                    keep = False
                    break
            if not keep:
                continue
            selected_idx.append(int(lin))
            selected_boxes.append(cand)
            selected_set.add(int(lin))
            if len(selected_idx) >= int(k):
                break

        if len(selected_idx) < int(k):
            for lin in order_full_list:
                if lin in selected_set:
                    continue
                cand = _lin_to_box(lin, w_win=w_win, ws=ws)
                keep = True
                for sel in selected_boxes:
                    if _box_iou(cand, sel, eps=eps) > thr:
                        keep = False
                        break
                if not keep:
                    continue
                selected_idx.append(int(lin))
                selected_boxes.append(cand)
                selected_set.add(int(lin))
                if len(selected_idx) >= int(k):
                    break

        if len(selected_idx) != int(k):
            raise RuntimeError(
                "SSM-NMS sampler failed to return k windows under strict IoU threshold. "
                f"got={len(selected_idx)}, k={k}, iou_thr={thr}"
            )

        idx_t = torch.tensor(selected_idx, device=score_map.device, dtype=torch.long)
        boxes_t = torch.tensor(selected_boxes, device=score_map.device, dtype=torch.long)
        scores_t = torch.gather(flat, dim=0, index=idx_t)

        all_boxes.append(boxes_t)
        all_indices.append(idx_t)
        all_scores.append(scores_t)

    return (
        torch.stack(all_boxes, dim=0),
        torch.stack(all_indices, dim=0),
        torch.stack(all_scores, dim=0),
    )


def _window_geometry_stats(
    windows: torch.Tensor,
    *,
    h: int,
    w: int,
    eps: float,
) -> tuple[float, float, float]:
    # windows: [S,4] in (y0,x0,y1,x1)
    win_list = windows.detach().long().cpu().tolist()
    s = len(win_list)
    if s <= 1:
        iou_mean = 0.0
        iou_max = 0.0
    else:
        ious = []
        for i in range(s):
            for j in range(i + 1, s):
                ious.append(_box_iou(tuple(win_list[i]), tuple(win_list[j]), eps=eps))
        if ious:
            iou_mean = float(sum(ious) / len(ious))
            iou_max = float(max(ious))
        else:
            iou_mean = 0.0
            iou_max = 0.0

    cov = torch.zeros((int(h), int(w)), dtype=torch.bool, device=windows.device)
    for y0, x0, y1, x1 in win_list:
        cov[int(y0):int(y1), int(x0):int(x1)] = True
    coverage = float(cov.float().mean().item())
    return iou_mean, iou_max, coverage


def _unfold_select_windows(feat: torch.Tensor, window_size: int, selected_lin: torch.Tensor) -> torch.Tensor:
    # feat: [B,C,H,W], selected_lin: [B,S] -> [B,S,window_size^2,C]
    bsz, ch, _h, _w = feat.shape
    ws2 = int(window_size * window_size)
    uf = F.unfold(feat, kernel_size=window_size, stride=1)  # [B, C*ws2, L]
    idx_exp = selected_lin.unsqueeze(1).expand(-1, uf.shape[1], -1)
    picked = torch.gather(uf, dim=2, index=idx_exp)  # [B, C*ws2, S]
    picked = picked.transpose(1, 2).reshape(bsz, -1, ch, ws2).transpose(2, 3).contiguous()
    return picked


def _local_cka_gram_with_stats(tokens_x: torch.Tensor, tokens_y: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # tokens_*: [B,S,N,C]
    bsz, n_win, n_tok, ch_x = tokens_x.shape
    ch_y = tokens_y.shape[-1]
    x = tokens_x.reshape(bsz * n_win, n_tok, ch_x)
    y = tokens_y.reshape(bsz * n_win, n_tok, ch_y)

    # Keep numerics stable under AMP: perform Gram/denominator math in FP32.
    x = center_token_set(x).float()
    y = center_token_set(y).float()

    gx = x @ x.transpose(1, 2)  # [BS,N,N]
    gy = y @ y.transpose(1, 2)  # [BS,N,N]

    hsic = (gx * gy).sum(dim=(1, 2)).double()
    nx_sq = (gx * gx).sum(dim=(1, 2)).double()
    ny_sq = (gy * gy).sum(dim=(1, 2)).double()

    nx = torch.sqrt(nx_sq.clamp_min(float(eps)))
    ny = torch.sqrt(ny_sq.clamp_min(float(eps)))
    den = (nx * ny).clamp_min(float(eps))
    cka = (hsic / den).float().clamp(min=-1.0, max=1.0)
    return (
        cka.reshape(bsz, n_win),
        nx.reshape(bsz, n_win),
        ny.reshape(bsz, n_win),
        den.reshape(bsz, n_win),
    )


class Stage3CKALoss(nn.Module):
    def __init__(
        self,
        *,
        local_window_size: int = 16,
        local_windows_total: int = 10,
        local_windows_per_step: int = 10,
        boundary_ratio_local: float = 0.6,
        overfit_fixed_sampling: bool = False,
        nms_iou_thr: float = 0.2,
        nms_topm: int = 1024,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.local_window_size = int(local_window_size)
        self.local_windows_total = int(local_windows_total)
        self.local_windows_per_step = int(local_windows_per_step)
        self.boundary_ratio_local = float(boundary_ratio_local)  # kept for backward-compatible args.
        self.overfit_fixed_sampling = bool(overfit_fixed_sampling)
        self.nms_iou_thr = float(nms_iou_thr)
        self.nms_topm = int(nms_topm)
        self.eps = float(eps)
        self._fixed_windows: torch.Tensor | None = None
        self._fixed_use_win_idx: torch.Tensor | None = None

        if self.local_windows_total != 10 or self.local_windows_per_step != 10:
            raise ValueError(
                "This local-only trainer requires local_windows_total=10 and local_windows_per_step=10. "
                f"got total={self.local_windows_total}, per_step={self.local_windows_per_step}"
            )

    def forward(self, stage3_feat: torch.Tensor, dino_feat: torch.Tensor, *, global_step: int | None = None) -> dict[str, torch.Tensor]:
        del global_step  # kept only for call-site compatibility

        if stage3_feat.shape[-2:] != dino_feat.shape[-2:]:
            dino_feat = F.interpolate(dino_feat, size=stage3_feat.shape[-2:], mode="bilinear", align_corners=False)

        bsz, _c, h, w = stage3_feat.shape
        ws = int(self.local_window_size)
        if h < ws or w < ws:
            raise RuntimeError(f"Feature map too small for local_window_size={ws}: got {(h, w)}")

        h_win = h - ws + 1
        w_win = w - ws + 1
        num_all_windows = int(h_win * w_win)
        required = int(self.local_windows_total)
        if num_all_windows < required:
            raise RuntimeError(
                f"Need at least {required} sliding windows but got {num_all_windows}. "
                f"feature={(h, w)}, window_size={ws}"
            )

        with torch.no_grad():
            div_map = _build_ssm_divergence_map(stage3_feat, dino_feat, eps=self.eps)  # [B,H,W]
            score_map = _window_score_map_from_divergence(div_map, ws=ws)  # [B,h_win,w_win]

            if (
                self.overfit_fixed_sampling
                and self._fixed_windows is not None
                and self._fixed_use_win_idx is not None
                and self._fixed_windows.shape == (bsz, required, 4)
                and self._fixed_use_win_idx.shape == (bsz, required)
            ):
                windows = self._fixed_windows.to(device=stage3_feat.device)
                use_win_idx = self._fixed_use_win_idx.to(device=stage3_feat.device)
                flat_scores = score_map.reshape(bsz, -1)
                win_scores = torch.gather(flat_scores, dim=1, index=use_win_idx)
            else:
                windows, use_win_idx, win_scores = _sample_windows_ssm_nms(
                    score_map,
                    ws=ws,
                    k=required,
                    iou_thr=self.nms_iou_thr,
                    topm=self.nms_topm,
                    eps=self.eps,
                )
                if self.overfit_fixed_sampling:
                    self._fixed_windows = windows.detach().cpu()
                    self._fixed_use_win_idx = use_win_idx.detach().cpu()

        xw = _unfold_select_windows(stage3_feat, ws, use_win_idx)  # [B,10,ws2,Cx]
        yw = _unfold_select_windows(dino_feat, ws, use_win_idx)  # [B,10,ws2,Cy]

        cka_local_each, nx_each, ny_each, den_each = _local_cka_gram_with_stats(xw, yw, eps=self.eps)
        loss_local = (1.0 - cka_local_each).mean()

        with torch.no_grad():
            iou_mean_list = []
            iou_max_list = []
            coverage_list = []
            score_min_list = []
            score_median_list = []
            score_max_list = []
            for b in range(bsz):
                iou_mean_b, iou_max_b, coverage_b = _window_geometry_stats(windows[b], h=h, w=w, eps=self.eps)
                if iou_max_b > (self.nms_iou_thr + 1e-6):
                    raise RuntimeError(
                        f"Window IoU constraint violated: max_iou={iou_max_b:.6f}, "
                        f"threshold={self.nms_iou_thr:.6f}"
                    )
                score_b = win_scores[b].float()
                iou_mean_list.append(iou_mean_b)
                iou_max_list.append(iou_max_b)
                coverage_list.append(coverage_b)
                score_min_list.append(float(score_b.min().item()))
                score_median_list.append(float(torch.quantile(score_b, q=0.5).item()))
                score_max_list.append(float(score_b.max().item()))

            win_iou_mean = torch.tensor(iou_mean_list, device=stage3_feat.device, dtype=torch.float32).mean()
            win_iou_max = torch.tensor(iou_max_list, device=stage3_feat.device, dtype=torch.float32).mean()
            win_coverage_ratio = torch.tensor(coverage_list, device=stage3_feat.device, dtype=torch.float32).mean()
            win_score_min = torch.tensor(score_min_list, device=stage3_feat.device, dtype=torch.float32).mean()
            win_score_median = torch.tensor(score_median_list, device=stage3_feat.device, dtype=torch.float32).mean()
            win_score_max = torch.tensor(score_max_list, device=stage3_feat.device, dtype=torch.float32).mean()

        return {
            "loss_local": loss_local,
            "total_loss": loss_local,
            "cka_local": cka_local_each.mean().detach(),
            "div_map": div_map.detach(),
            "div_map_mean": div_map.detach().mean(),
            "num_local_windows": torch.tensor(float(use_win_idx.shape[1]), device=stage3_feat.device),
            # Diagnostics (non-detached cka_local_each is required for per-window grad conflict checks).
            "cka_local_each": cka_local_each,
            "nx_each": nx_each.detach(),
            "ny_each": ny_each.detach(),
            "den_each": den_each.detach(),
            "use_win_idx": use_win_idx.detach(),
            "h_win": torch.tensor(int(h_win), device=stage3_feat.device),
            "w_win": torch.tensor(int(w_win), device=stage3_feat.device),
            "windows_xyxy": windows.detach(),
            "window_scores_each": win_scores.detach(),
            "win_iou_mean": win_iou_mean.detach(),
            "win_iou_max": win_iou_max.detach(),
            "win_coverage_ratio": win_coverage_ratio.detach(),
            "win_score_min": win_score_min.detach(),
            "win_score_median": win_score_median.detach(),
            "win_score_max": win_score_max.detach(),
        }
