#!/usr/bin/env python3
# SFDA on ACDC (NO ACDC GT). Minimal: SegFormer(student) + DINO(teacher) + (local+global) structural distill.

import os, argparse, random, math, glob
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------- data -----------------

def imread_rgb(path: str) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    x = np.array(Image.open(path).convert('RGB'), dtype='float32') / 255.0
    return torch.from_numpy(x).permute(2, 0, 1)  # [3,H,W]


def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    m = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    s = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - m) / s


class ACDCUnlabeled(Dataset):
    # root/rgb_anon_trainvaltest/rgb_anon/{cond}/{split}/**/*_rgb_anon.png
    def __init__(self, root: str, split: str, conds: List[str], crop: int = 896, resize: int = 1024):
        self.crop, self.resize = crop, resize
        base = os.path.join(root, 'rgb_anon_trainvaltest', 'rgb_anon')
        ps = []
        for c in conds:
            ps += glob.glob(os.path.join(base, c, split, '**', '*_rgb_anon.*'), recursive=True)
        self.paths = sorted([p for p in ps if os.path.isfile(p)])
        if not self.paths:
            raise RuntimeError(f'No images found under {base} for split={split}, conds={conds}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = imread_rgb(self.paths[i])
        # resize (keep aspect) then random crop (reflect pad)
        _, H, W = x.shape
        s = self.resize / max(H, W)
        nh, nw = int(round(H * s)), int(round(W * s))
        x = F.interpolate(x[None], (nh, nw), mode='bilinear', align_corners=False)[0]
        _, H, W = x.shape
        if H < self.crop or W < self.crop:
            x = F.pad(x, (0, max(0, self.crop - W), 0, max(0, self.crop - H)), mode='reflect')
            _, H, W = x.shape
        top = random.randint(0, H - self.crop)
        left = random.randint(0, W - self.crop)
        x = x[:, top:top + self.crop, left:left + self.crop]
        return x


# ----------------- teacher (timm DINOv3) -----------------

class DinoTeacher(nn.Module):
    """DINOv3 teacher via timm (works without transformers>=4.56).

    Default model name examples (HF timm weights):
      - vit_small_patch16_dinov3.lvd1689m
      - vit_base_patch16_dinov3.lvd1689m
      - vit_large_patch16_dinov3_qkvb.lvd1689m

    Returns dense patch features as [B, Ct, Ht, Wt].
    """

    def __init__(self, name: str):
        super().__init__()
        import timm
        self.name = name
        self.m = timm.create_model(name, pretrained=True)
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

        # patch size (ViT-style)
        ps = None
        if hasattr(self.m, 'patch_embed') and hasattr(self.m.patch_embed, 'patch_size'):
            ps = self.m.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            ps = ps[0]
        self.patch = int(ps or 16)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] ImageNet-norm
        B, _, H, W = x.shape

        # timm ViT forward_features usually returns tokens [B, 1+N, C] (cls + patch tokens)
        y = self.m.forward_features(x)
        if isinstance(y, dict):
            # common keys across some timm models
            y = y.get('x', None) or y.get('tokens', None) or y.get('last_hidden_state', None)
            if y is None:
                raise RuntimeError(f"Unexpected dict output keys from timm model: {list(y.keys())}")

        if y.dim() == 3:
            tok = y
        elif y.dim() == 2:
            # global embedding only -> not usable for dense distill
            raise RuntimeError('Teacher returned [B,C] global embedding only; choose a ViT DINOv3 timm model.')
        else:
            raise RuntimeError(f'Unexpected teacher output shape: {tuple(y.shape)}')

        # drop prefix tokens (CLS + optional register tokens)
        offset = getattr(self.m, 'num_prefix_tokens', 1)
        if tok.shape[1] > offset:
            tok = tok[:, offset:, :]
        else:
            raise RuntimeError(f'Unexpected token length: {tok.shape[1]} (offset={offset})')

        ht, wt = H // self.patch, W // self.patch
        n = tok.shape[1]
        if ht * wt != n:
            # best-effort square-ish reshape
            ht = int(round(math.sqrt(n)))
            wt = max(1, n // max(1, ht))
            ht = max(1, n // max(1, wt))
            tok = tok[:, :ht * wt, :]

        return tok.view(B, ht, wt, tok.shape[-1]).permute(0, 3, 1, 2).contiguous()


# ----------------- losses -----------------

def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,H,W] -> [B,HW] in [0,1]
    x = x.mean(1, keepdim=True)
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    m = (gx * gx + gy * gy).sqrt().squeeze(1)
    mn, mx = m.amin((1, 2), True), m.amax((1, 2), True)
    m = (m - mn) / (mx - mn + 1e-6)
    return m.flatten(1)


class LocalGatedKL(nn.Module):
    def __init__(self, Cs: int, Ct: int, d: int = 64, k: int = 5, tau: float = 0.1, alpha: float = 2.0, beta: float = 1.0):
        super().__init__()
        assert k % 2 == 1
        self.k, self.pad, self.tau = k, k // 2, tau
        self.alpha, self.beta = alpha, beta
        self.ps = nn.Conv2d(Cs, d, 1, bias=False)
        self.pt = nn.Conv2d(Ct, d, 1, bias=False)
        center = (k * k) // 2
        self.register_buffer('mask', torch.tensor([i for i in range(k * k) if i != center], dtype=torch.long), persistent=False)
        self.maxH = math.log(k * k - 1)

    def _logp(self, f: torch.Tensor) -> torch.Tensor:
        B, d, H, W = f.shape
        f = F.normalize(f, dim=1)
        u = F.unfold(F.pad(f, (self.pad, self.pad, self.pad, self.pad), mode='reflect'), kernel_size=self.k)
        neigh = u.view(B, d, self.k * self.k, H * W).permute(0, 3, 2, 1)  # [B,HW,K2,d]
        cen = f.view(B, d, H * W).permute(0, 2, 1).unsqueeze(2)            # [B,HW,1,d]
        aff = (cen @ neigh.transpose(-1, -2)).squeeze(2)[:, :, self.mask]  # [B,HW,K2-1]
        return F.log_softmax(aff / self.tau, dim=-1)

    def forward(self, fs3: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        H, W = fs3.shape[-2:]
        ft = F.interpolate(ft, (H, W), mode='bilinear', align_corners=False)
        s, t = self.ps(fs3), self.pt(ft)
        t_logp, s_logp = self._logp(t), self._logp(s)
        with torch.no_grad():
            t_p = t_logp.exp()
            ent = -(t_p * t_logp).sum(-1)  # [B,HW]
            w_conf = (1.0 - ent / (self.maxH + 1e-12)).clamp(0, 1)
            w_edge = sobel_mag(t)          # [B,HW]
            w = (w_conf ** self.alpha) * (1.0 + self.beta * w_edge)
            # drop padding border (unfold uses reflect but still best to ignore)
            p = self.pad
            m = torch.ones((fs3.shape[0], H, W), device=fs3.device, dtype=fs3.dtype)
            m[:, :p, :] = 0; m[:, -p:, :] = 0; m[:, :, :p] = 0; m[:, :, -p:] = 0
            w = w * m.flatten(1)
        kl = (t_p * (t_logp - s_logp)).sum(-1)
        return (w * kl).sum() / (w.sum() + 1e-6)


class GlobalProtoEMA(nn.Module):
    def __init__(self, Cs: int, Ct: int, d: int = 128, M: int = 16, tau: float = 0.07, ema: float = 0.99):
        super().__init__()
        self.M, self.tau, self.ema = M, tau, ema
        self.ps = nn.Conv2d(Cs, d, 1, bias=False)
        self.pt = nn.Conv2d(Ct, d, 1, bias=False)
        C = torch.randn(M, d)
        C = C / (C.norm(dim=1, keepdim=True) + 1e-12)
        self.register_buffer('C', C, persistent=True)

    def _assign(self, z: torch.Tensor) -> torch.Tensor:
        return F.softmax((z @ self.C.t()) / self.tau, dim=-1)

    @torch.no_grad()
    def _ema_update(self, zt: torch.Tensor, qt: torch.Tensor):
        BN, d = zt.shape
        denom = qt.sum(0) + 1e-6
        mu = (qt.t() @ zt) / denom.unsqueeze(1)
        mu = mu / (mu.norm(dim=1, keepdim=True) + 1e-12)
        self.C.mul_(self.ema).add_((1 - self.ema) * mu)
        self.C.div_(self.C.norm(dim=1, keepdim=True) + 1e-12)

    def forward(self, fs4: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        H, W = fs4.shape[-2:]
        ft = F.interpolate(ft, (H, W), mode='bilinear', align_corners=False)
        s, t = self.ps(fs4), self.pt(ft)
        B, d, H, W = s.shape
        zs = F.normalize(s.flatten(2).transpose(1, 2), dim=-1)  # [B,N,d]
        zt = F.normalize(t.flatten(2).transpose(1, 2), dim=-1)
        with torch.no_grad():
            qt = self._assign(zt.reshape(-1, d)).view(B, -1, self.M)
            if self.training:
                self._ema_update(zt.reshape(-1, d), qt.reshape(-1, self.M))
            ht = qt.mean(1)
        qs = self._assign(zs.reshape(-1, d)).view(B, -1, self.M)
        hs = qs.mean(1)
        return F.kl_div((hs + 1e-12).log(), ht, reduction='batchmean')


def pseudo_ce(logits: torch.Tensor, thr: float = 0.9) -> torch.Tensor:
    p = logits.softmax(1)
    conf, y = p.max(1)
    m = conf > thr
    if m.sum() < 1:
        return logits.new_tensor(0.0)
    return (F.cross_entropy(logits, y, reduction='none') * m.float()).sum() / (m.float().sum() + 1e-6)


# ----------------- mmseg student -----------------

def build_student(cfg_path: str, ckpt: str, device: str):
    from mmengine.config import Config
    from mmengine.runner.checkpoint import load_checkpoint
    from mmseg.registry import MODELS

    cfg = Config.fromfile(cfg_path)
    # This script normalizes input manually (imagenet_norm), so we do not need
    # mmseg's SegDataPreProcessor. Some mmseg/mmengine version combos may not
    # register it, causing a KeyError. Use the mmengine default instead.
    if hasattr(cfg, 'model') and isinstance(cfg.model, dict):
        cfg.model.pop('data_preprocessor', None)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt, map_location='cpu', revise_keys=[(r'^module\.', '')])
    if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device).train()
    return model


def set_trainable(model: nn.Module, mode: str):
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == 'all':
        for p in model.parameters():
            p.requires_grad_(True)
        return
    for p in model.decode_head.parameters():
        p.requires_grad_(True)
    if mode == 'norm_head':
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm)):
                for p in m.parameters(recurse=False):
                    p.requires_grad_(True)


def forward_logits_feats(model: nn.Module, x: torch.Tensor):
    feats = model.backbone(x)
    if getattr(model, 'with_neck', False) and model.neck is not None:
        feats = model.neck(feats)
    logits = model.decode_head(feats)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    logits = F.interpolate(logits, x.shape[-2:], mode='bilinear', align_corners=False)
    return logits, feats


# ----------------- train -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='/home/kevinlee01/miniconda3/envs/mmseg/lib/python3.8/site-packages/mmseg/.mim/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py')
    ap.add_argument('--checkpoint', default='segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth', help='Source-pretrained SegFormer checkpoint (.pth)')
    ap.add_argument('--acdc_root', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--conds', nargs='*', default=['fog', 'night', 'rain', 'snow'])

    ap.add_argument('--teacher', default='vit_small_patch16_dinov3.lvd1689m')

    ap.add_argument('--crop', type=int, default=896)
    ap.add_argument('--resize', type=int, default=1024)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--amp', action='store_true')

    ap.add_argument('--trainable', choices=['all', 'head', 'norm_head'], default='norm_head')

    ap.add_argument('--lam_loc', type=float, default=1.0)
    ap.add_argument('--lam_glob', type=float, default=0.1)
    ap.add_argument('--lam_pseudo', type=float, default=1.0)
    ap.add_argument('--pseudo_thr', type=float, default=0.9)

    ap.add_argument('--out', default='./work_dirs/sfda')
    ap.add_argument('--save_every', type=int, default=2000)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    dev = args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'

    ds = ACDCUnlabeled(args.acdc_root, args.split, args.conds, crop=args.crop, resize=args.resize)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    it = iter(dl)

    teacher = DinoTeacher(args.teacher).to(dev)
    student = build_student(args.config, args.checkpoint, dev)
    set_trainable(student, args.trainable)

    # infer channels
    with torch.no_grad():
        x0 = next(it).to(dev)
        x0n = imagenet_norm(x0)
        _, feats = forward_logits_feats(student, x0n)
        Cs3, Cs4 = feats[2].shape[1], feats[3].shape[1]
        Ct = teacher(x0n).shape[1]

    loc = LocalGatedKL(Cs3, Ct).to(dev)
    globm = GlobalProtoEMA(Cs4, Ct).to(dev)

    params = [p for p in student.parameters() if p.requires_grad] + list(loc.parameters()) + list(globm.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and str(dev).startswith('cuda'))
    base_lr = args.lr

    it = iter(dl)
    run = {'loc': 0.0, 'glob': 0.0, 'ps': 0.0, 'tot': 0.0}

    for step in range(1, args.iters + 1):
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        x = x.to(dev, non_blocking=True)
        x = imagenet_norm(x)

        # PolyLR (simple, scheduler-free)
        lr = base_lr * (1.0 - step / args.iters) ** 0.9
        for pg in opt.param_groups:
            pg['lr'] = lr

        with torch.cuda.amp.autocast(enabled=args.amp and str(dev).startswith('cuda')):
            with torch.no_grad():
                t = teacher(x)
            logits, feats = forward_logits_feats(student, x)
            # Local window KL can be numerically sensitive in fp16; compute in fp32.
            with torch.cuda.amp.autocast(enabled=False):
                l_loc = loc(feats[2].float(), t.float()) * args.lam_loc
            l_glob = globm(feats[3], t) * args.lam_glob
            l_ps = pseudo_ce(logits, args.pseudo_thr) * args.lam_pseudo
            loss = l_loc + l_glob + l_ps

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        run['loc'] += float(l_loc.detach())
        run['glob'] += float(l_glob.detach())
        run['ps'] += float(l_ps.detach())
        run['tot'] += float(loss.detach())

        if step % args.log_every == 0:
            n = args.log_every
            print(f"[{step:06d}/{args.iters}] tot={run['tot']/n:.4f} loc={run['loc']/n:.4f} glob={run['glob']/n:.4f} ps={run['ps']/n:.4f}")
            for k in run:
                run[k] = 0.0

        if step % args.save_every == 0:
            p = os.path.join(args.out, f'adapt_{step:06d}.pth')
            torch.save({'student': student.state_dict(), 'loc': loc.state_dict(), 'glob': globm.state_dict(), 'step': step, 'args': vars(args)}, p)
            print('[save]', p)

    p = os.path.join(args.out, 'adapt_final.pth')
    torch.save({'student': student.state_dict(), 'loc': loc.state_dict(), 'glob': globm.state_dict(), 'step': args.iters, 'args': vars(args)}, p)
    print('[save]', p)


if __name__ == '__main__':
    main()