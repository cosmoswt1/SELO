"""Train SELO v0: stage3 adapter + DINO local affinity distillation (flow-free)."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ACDCDataset
from losses import LocalAffinityKLLoss
from models import SeloV0Model


def setup_logger(log_path: Path):
    logger = logging.getLogger("selo_v0")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args():
    p = argparse.ArgumentParser(description="SELO v0 training")
    p.add_argument("--acdc_root", type=str, required=True)
    p.add_argument("--conditions", nargs="+", default=["fog", "night", "rain", "snow"])
    p.add_argument("--resize", type=int, default=540)
    p.add_argument("--crop_size", type=int, default=512)

    p.add_argument("--segformer_model", type=str,
                   default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--dino_model", type=str,
                   default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--adapter_hidden_ratio", type=float, default=0.25)
    p.add_argument("--adapter_scale", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_proj", type=float, default=1e-4)
    p.add_argument("--proj_warmup_steps", type=int, default=0,
                   help="처음 N step 동안 proj만 학습하고 이후 proj를 고정한 채 adapter만 학습합니다. (0이면 비활성)")
    p.add_argument(
        "--lr_proj_after_warmup",
        type=float,
        default=None,
        help="proj_warmup_steps 이후에도 proj를 아주 작은 lr로 계속 학습합니다. (None/<=0이면 proj 완전 고정)",
    )
    p.add_argument("--proj_type", type=str, default="mlp", choices=["conv", "mlp"])
    p.add_argument("--proj_mlp_hidden", type=int, default=256,
                   help="proj_type=mlp일 때 hidden dim (1x1 conv 기반 2-layer MLP)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--weight_decay_proj", type=float, default=None,
                   help="proj 파라미터에만 적용할 weight_decay (None이면 weight_decay와 동일)")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_grad_norm_proj", type=float, default=None,
                   help="proj에만 적용할 grad clip (None이면 max_grad_norm와 동일)")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=1)

    p.add_argument("--affinity_k", type=int, default=5)
    p.add_argument("--affinity_tau", type=float, default=0.1)
    p.add_argument("--lambda_aff", type=float, default=1.0)
    p.add_argument("--affinity_anchors", type=int, default=512)
    p.add_argument("--affinity_candidates", type=int, default=0,
                   help="2-stage 앵커 샘플링 후보 개수(0이면 anchors와 동일). 예: 4096")
    p.add_argument("--affinity_kcenter_top_m", type=int, default=0,
                   help="entropy score 상위 M 후보로 제한한 뒤 greedy k-center(farthest point)로 anchors 선택. (0이면 비활성)")
    p.add_argument("--affinity_per_image", type=int, default=1,
                   help="1이면 이미지별로 앵커를 따로 선택합니다(권장). 0이면 배치 평균 entropy로 공통 앵커를 선택합니다.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="SELO")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--diag_anchor_every", type=int, default=0,
                   help="학습 중 anchor 오버레이 이미지를 저장하는 주기(global_step 기준). 0이면 비활성.")
    p.add_argument("--diag_anchor_dir", type=str, default="",
                   help="anchor 오버레이 저장 폴더(비우면 output_dir/diag_train_anchors).")
    p.add_argument("--diag_anchor_max", type=int, default=0,
                   help="저장할 anchor 오버레이 총 이미지 수 상한(0이면 무제한).")
    p.add_argument("--diag_anchor_per_call", type=int, default=1,
                   help="diag 트리거 1회당 배치에서 저장할 이미지 개수(1 권장).")
    return p.parse_args()


def _grad_l2_norm(params) -> float:
    total = None
    for p in params:
        if p.grad is None:
            continue
        g2 = p.grad.detach().float().pow(2).sum()
        total = g2 if total is None else (total + g2)
    if total is None:
        return 0.0
    return float(total.sqrt().item())


def _fmt_lr(v: float) -> str:
    if v == 0.0:
        return "0.0"
    return f"{v:.6g}"

def _denorm_image(x: torch.Tensor) -> np.ndarray:
    # x: [3,H,W], ImageNet norm
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    y = (x * std + mean).clamp(0, 1)
    y = (y * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return y


def _overlay_points(img_rgb: np.ndarray, pts_yx: np.ndarray, color=(255, 0, 0), r: int = 2) -> np.ndarray:
    im = Image.fromarray(img_rgb)
    dr = ImageDraw.Draw(im)
    for y, x in pts_yx:
        x0, y0 = int(x) - r, int(y) - r
        x1, y1 = int(x) + r, int(y) + r
        dr.ellipse([x0, y0, x1, y1], outline=color, width=2)
    return np.array(im)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("[GPU 체크 실패] torch.cuda.is_available() == False", file=sys.stderr)
        print("확인 커맨드:", file=sys.stderr)
        print("  nvidia-smi", file=sys.stderr)
        print("  python -c \"import torch; print(torch.cuda.is_available())\"", file=sys.stderr)
        raise SystemExit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "train.log")
    logger.info("Args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    device = torch.device("cuda")
    anchor_dir = None
    diag_every = max(0, int(args.diag_anchor_every))
    diag_max = max(0, int(args.diag_anchor_max))
    diag_per_call = max(0, int(args.diag_anchor_per_call))
    if diag_every > 0 and diag_per_call > 0:
        anchor_dir = Path(args.diag_anchor_dir) if args.diag_anchor_dir else (output_dir / "diag_train_anchors")
        anchor_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[diag] save train anchors every={diag_every} per_call={diag_per_call} "
            f"max={diag_max if diag_max > 0 else 'inf'} dir={anchor_dir}"
        )
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit("wandb가 설치되어 있지 않습니다. requirements.txt에 추가 후 설치하세요.") from exc
        init_kwargs = {"project": args.wandb_project}
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        if args.wandb_group:
            init_kwargs["group"] = args.wandb_group
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        wandb_run = wandb.init(config=vars(args), **init_kwargs)

    dataset = ACDCDataset(
        root=args.acdc_root,
        split="train",
        conditions=args.conditions,
        resize=args.resize,
        crop_size=(args.crop_size, args.crop_size),
    )

    model = SeloV0Model(
        segformer_model=args.segformer_model,
        dino_model=args.dino_model,
        num_classes=args.num_classes,
        adapter_hidden_ratio=args.adapter_hidden_ratio,
        adapter_scale=args.adapter_scale,
        proj_type=args.proj_type,
        proj_mlp_hidden=int(args.proj_mlp_hidden),
    ).to(device)
    model.freeze_backbone()

    for p in model.parameters():
        p.requires_grad = False

    loss_fn = LocalAffinityKLLoss(
        k=args.affinity_k,
        tau=args.affinity_tau,
        anchors=args.affinity_anchors,
        candidates=(args.affinity_candidates if int(args.affinity_candidates) > 0 else args.affinity_anchors),
        kcenter_top_m=int(args.affinity_kcenter_top_m),
        per_image=(int(args.affinity_per_image) > 0),
    )
    scaler = GradScaler(device="cuda", enabled=args.amp)

    def build_loader(bs: int):
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

    effective_bs = args.batch_size * max(1, args.grad_accum_steps)
    logger.info(f"Using batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}, "
                f"effective_batch={effective_bs}")
    loader = build_loader(args.batch_size)
    lr_adapter_base = float(args.lr)
    lr_proj_base = float(args.lr if args.lr_proj is None else args.lr_proj)
    wd_adapter = float(args.weight_decay)
    wd_proj = float(args.weight_decay if args.weight_decay_proj is None else args.weight_decay_proj)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.stage3_adapter.parameters(), "lr": lr_adapter_base, "weight_decay": wd_adapter},
            {"params": model.stage3_proj.parameters(), "lr": lr_proj_base, "weight_decay": wd_proj},
        ],
    )

    warmup_steps = max(0, int(args.proj_warmup_steps))
    lr_proj_after_warmup = args.lr_proj_after_warmup
    if lr_proj_after_warmup is not None:
        lr_proj_after_warmup = float(lr_proj_after_warmup)
        if lr_proj_after_warmup <= 0.0:
            lr_proj_after_warmup = None
    accum = max(1, int(args.grad_accum_steps))
    warmup_steps_eff = warmup_steps
    if warmup_steps_eff > 0 and (warmup_steps_eff % accum) != 0:
        rounded = ((warmup_steps_eff + accum - 1) // accum) * accum
        logger.info(
            f"[sched] proj_warmup_steps={warmup_steps_eff} 는 grad_accum_steps={accum} 의 배수가 아니어서 "
            f"{rounded} 로 올림(phase 전환 시 누적 grad 혼합 방지)"
        )
        warmup_steps_eff = rounded

    lr_adapter_eff = lr_adapter_base
    lr_proj_eff = lr_proj_base
    phase_name = "joint"

    def set_phase(name: str):
        nonlocal lr_adapter_eff, lr_proj_eff, phase_name

        if name == "warmup_proj":
            # proj only
            for p in model.stage3_adapter.parameters():
                p.requires_grad = False
            for p in model.stage3_proj.parameters():
                p.requires_grad = True
            lr_adapter_eff = 0.0
            lr_proj_eff = lr_proj_base
        elif name == "train_adapter":
            # adapter training + (optional) small proj lr
            for p in model.stage3_adapter.parameters():
                p.requires_grad = True
            if lr_proj_after_warmup is not None:
                for p in model.stage3_proj.parameters():
                    p.requires_grad = True
            else:
                for p in model.stage3_proj.parameters():
                    p.requires_grad = False
            lr_adapter_eff = lr_adapter_base
            lr_proj_eff = 0.0 if lr_proj_after_warmup is None else float(lr_proj_after_warmup)
        elif name == "joint":
            # backward-compatible behavior
            for p in model.stage3_adapter.parameters():
                p.requires_grad = True
            for p in model.stage3_proj.parameters():
                p.requires_grad = True
            lr_adapter_eff = lr_adapter_base
            lr_proj_eff = lr_proj_base
        else:
            raise ValueError(f"unknown phase: {name}")

        optimizer.param_groups[0]["lr"] = lr_adapter_eff
        optimizer.param_groups[1]["lr"] = lr_proj_eff
        optimizer.zero_grad(set_to_none=True)

        phase_name = name
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(
            f"[phase] {phase_name} | lr_adapter={_fmt_lr(lr_adapter_eff)} lr_proj={_fmt_lr(lr_proj_eff)} "
            f"| trainable={', '.join(trainable)}"
        )

    if warmup_steps_eff > 0:
        set_phase("warmup_proj")
        if lr_proj_after_warmup is None:
            logger.info(f"[sched] warmup_steps={warmup_steps} (effective={warmup_steps_eff}) 이후 proj 고정 + adapter 학습")
        else:
            logger.info(
                f"[sched] warmup_steps={warmup_steps} (effective={warmup_steps_eff}) 이후 adapter 학습 + "
                f"proj lr={_fmt_lr(float(lr_proj_after_warmup))}"
            )
    else:
        set_phase("joint")

    start = time.time()
    model.train()
    global_step = 0
    best_train_aff_kl = None
    saved_anchors = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}", file=sys.stdout)
        for step, batch in enumerate(pbar):
            if warmup_steps_eff > 0:
                want = "warmup_proj" if global_step < warmup_steps_eff else "train_adapter"
                if want != phase_name:
                    set_phase(want)
            do_log = (step % args.log_every == 0)
            do_anchor = False
            if anchor_dir is not None and diag_every > 0 and (global_step % diag_every == 0):
                if diag_max <= 0 or saved_anchors < diag_max:
                    do_anchor = True
            images = batch["image"].to(device)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                # Training objective only needs stage3 + DINO; computing downstream stages/logits
                # builds a huge autograd graph and will blow up VRAM.
                out = model(images, use_dino=True, compute_logits=False)
                f3 = out["stage3_adapt"]
                proj = model.stage3_proj(f3)
                dino_feat = out["dino_feat"]
                aff_stats = None
                aff_debug = None
                if do_anchor:
                    aff_loss, aff_stats, aff_debug = loss_fn(proj, dino_feat, return_stats=True, return_debug=True)
                elif do_log:
                    aff_loss, aff_stats = loss_fn(proj, dino_feat, return_stats=True)
                else:
                    aff_loss = loss_fn(proj, dino_feat, return_stats=False)
                raw_loss = aff_loss * args.lambda_aff
                loss = raw_loss

            loss = loss / max(1, args.grad_accum_steps)
            scaler.scale(loss).backward()

            if do_anchor and aff_debug is not None and anchor_dir is not None and diag_per_call > 0:
                # Map selected anchors (y,x) on 67x67 grid -> input pixels for visualization.
                inp_h, inp_w = images.shape[-2:]
                ah, aw = aff_debug["hw"]
                sel = aff_debug["selected_anchors"].numpy().astype(np.int64)  # [B,M,2]

                paths = batch.get("path", ["unknown"] * int(images.shape[0]))
                conds = batch.get("condition", ["unknown"] * int(images.shape[0]))
                save_n = min(int(images.shape[0]), int(diag_per_call))
                if diag_max > 0:
                    save_n = min(save_n, max(0, diag_max - saved_anchors))

                for j in range(save_n):
                    sel_j = sel[j]
                    ys = (sel_j[:, 0] + 0.5) * (inp_h / ah)
                    xs = (sel_j[:, 1] + 0.5) * (inp_w / aw)
                    pts = np.stack([ys, xs], axis=1)
                    cond = conds[j] if isinstance(conds, (list, tuple)) else str(conds)
                    cond_dir = anchor_dir / str(cond)
                    cond_dir.mkdir(parents=True, exist_ok=True)
                    name = Path(paths[j]).name if j < len(paths) else f"idx{j}"
                    img_rgb = _denorm_image(images[j].detach())
                    over = _overlay_points(img_rgb, pts_yx=pts, color=(255, 0, 0), r=2)
                    out_path = cond_dir / f"gs{global_step:07d}_e{epoch:02d}_s{step:05d}_{name}"
                    Image.fromarray(over).save(out_path)
                    saved_anchors += 1

            if epoch == 0 and step == 0:
                inp_h, inp_w = images.shape[-2:]
                dino_h, dino_w = out["dino_grid"]
                logger.info(
                    "Stage3 shape: " + str(tuple(out["stage3_raw"].shape))
                )
                logger.info(
                    f"Alignment: input={inp_h}x{inp_w}, stage3={f3.shape[-2:]} "
                    f"dino={dino_h}x{dino_w}"
                )

            if do_log:
                pbar.set_postfix(loss=f"{raw_loss.item():.4f}")
                with torch.no_grad():
                    f_raw = out["stage3_raw"].detach()
                    f_adapt = out["stage3_adapt"].detach()
                    base_rms = float(f_raw.float().pow(2).mean().sqrt().item())
                    delta_rms = float((f_adapt - f_raw).float().pow(2).mean().sqrt().item())
                    delta_over_base = float(delta_rms / (base_rms + 1e-12))

                scale = float(scaler.get_scale()) if args.amp else 1.0
                g_adapter = _grad_l2_norm(model.stage3_adapter.parameters()) / max(scale, 1e-12)
                g_proj = _grad_l2_norm(model.stage3_proj.parameters()) / max(scale, 1e-12)
                with torch.no_grad():
                    s = model.stage3_adapter.scale.detach().float()
                    adapter_scale_mean = float(s.mean().item())
                    adapter_scale_max = float(s.abs().max().item())

                logger.info(f"STEP {epoch} {step} LOSS {raw_loss.item():.6f}")
                logger.info(
                    f"DBG {epoch} {step} "
                    f"base_rms={base_rms:.6f} delta_rms={delta_rms:.6f} delta/base={delta_over_base:.6f} "
                    f"g_adapter={g_adapter:.6f} g_proj={g_proj:.6f} "
                    f"adapter_scale(mean)={adapter_scale_mean:.6f} adapter_scale(|max|)={adapter_scale_max:.6f} "
                    f"lr_adapter={_fmt_lr(lr_adapter_eff)} lr_proj={_fmt_lr(lr_proj_eff)}"
                )
                logger.info(
                    f"AFF {epoch} {step} "
                    f"tau={aff_stats['tau']:g} candidates={aff_stats['candidates']} anchors={aff_stats['anchors']} "
                    f"(req a={aff_stats.get('anchors_req','?')} c={aff_stats.get('candidates_req','?')} grid={aff_stats.get('grid_total','?')}) "
                    f"k2={aff_stats['k2']} "
                    f"sim_s(mean/std/max)={aff_stats.get('sim_s_mean', float('nan')):.4f}/"
                    f"{aff_stats.get('sim_s_std', float('nan')):.4f}/"
                    f"{aff_stats.get('sim_s_max', float('nan')):.4f} "
                    f"sim_t(mean/std/max)={aff_stats.get('sim_t_mean', float('nan')):.4f}/"
                    f"{aff_stats.get('sim_t_std', float('nan')):.4f}/"
                    f"{aff_stats.get('sim_t_max', float('nan')):.4f} "
                    f"p_t_max={aff_stats['p_t_max']:.4f} p_t_ent={aff_stats['p_t_ent']:.4f} "
                    f"p_s_max={aff_stats['p_s_max']:.4f} p_s_ent={aff_stats['p_s_ent']:.4f}"
                )
                if best_train_aff_kl is None or raw_loss.item() < best_train_aff_kl:
                    best_train_aff_kl = float(raw_loss.item())
                if wandb_run is not None:
                    wandb.log(
                        {
                            "loss": float(raw_loss.item()),
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "phase": phase_name,
                            "lr_adapter": lr_adapter_eff,
                            "lr_proj": lr_proj_eff,
                            "lambda_aff": args.lambda_aff,
                            "best_train_aff_kl": float(best_train_aff_kl)
                            if best_train_aff_kl is not None
                            else float(raw_loss.item()),
                            "dbg/base_rms": base_rms,
                            "dbg/delta_rms": delta_rms,
                            "dbg/delta_over_base": delta_over_base,
                            "dbg/g_adapter": g_adapter,
                            "dbg/g_proj": g_proj,
                            "dbg/adapter_scale": adapter_scale,
                            "aff/p_t_max": float(aff_stats["p_t_max"]),
                            "aff/p_t_ent": float(aff_stats["p_t_ent"]),
                            "aff/p_s_max": float(aff_stats["p_s_max"]),
                            "aff/p_s_ent": float(aff_stats["p_s_ent"]),
                        },
                        step=global_step,
                    )

            if (step + 1) % max(1, args.grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                # Clip adapter/proj separately (proj may have different threshold).
                torch.nn.utils.clip_grad_norm_(model.stage3_adapter.parameters(), args.max_grad_norm)
                max_proj = args.max_grad_norm if args.max_grad_norm_proj is None else float(args.max_grad_norm_proj)
                torch.nn.utils.clip_grad_norm_(model.stage3_proj.parameters(), max_proj)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1

    logger.info(f"Training done. elapsed_sec={time.time() - start:.1f}")
    adapter_ckpt = {
        "stage3_adapter": model.stage3_adapter.state_dict(),
        "stage3_proj": model.stage3_proj.state_dict(),
        "args": vars(args),
    }
    torch.save(adapter_ckpt, output_dir / "adapter.pth")
    logger.info(f"Saved checkpoint: {output_dir / 'adapter.pth'}")
    if wandb_run is not None:
        if best_train_aff_kl is not None:
            wandb_run.summary["best_train_aff_kl"] = float(best_train_aff_kl)
        wandb_run.finish()


if __name__ == "__main__":
    main()
