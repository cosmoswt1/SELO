"""Train SELO v0: stage1 adapter + DINO local affinity distillation (flow-free)."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--auto_batch", action="store_true")
    p.add_argument("--auto_batch_candidates", type=str, default="8,6,4,2,1")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--affinity_k", type=int, default=5)
    p.add_argument("--affinity_tau", type=float, default=0.1)
    p.add_argument("--lambda_aff", type=float, default=1.0)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def _parse_candidates(text: str):
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            continue
    return [v for v in vals if v > 0]


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
    ).to(device)
    model.freeze_backbone()

    for p in model.parameters():
        p.requires_grad = False
    for p in model.stage1_adapter.parameters():
        p.requires_grad = True
    for p in model.stage1_proj.parameters():
        p.requires_grad = True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info("Trainable params: " + ", ".join(trainable))

    loss_fn = LocalAffinityKLLoss(k=args.affinity_k, tau=args.affinity_tau)
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

    chosen_bs = args.batch_size
    if args.auto_batch:
        candidates = _parse_candidates(args.auto_batch_candidates) or [args.batch_size]
        logger.info(f"Auto-batch candidates: {candidates}")
        for bs in candidates:
            try:
                torch.cuda.empty_cache()
                loader_try = build_loader(bs)
                batch = next(iter(loader_try))
                images = batch["image"].to(device)
                with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                    out = model(images, use_dino=True)
                    f1 = out["stage1_adapt"]
                    f1_pool = F.avg_pool2d(f1, kernel_size=4, stride=4)
                    proj = model.stage1_proj(f1_pool)
                    dino_feat = out["dino_feat"]
                    loss = loss_fn(proj, dino_feat) * args.lambda_aff
                scaler.scale(loss).backward()
                for p in model.parameters():
                    p.grad = None
                chosen_bs = bs
                logger.info(f"Auto-batch selected: {chosen_bs}")
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.info(f"Auto-batch OOM at bs={bs}, trying smaller...")
                    torch.cuda.empty_cache()
                    continue
                raise
    logger.info(f"Using batch_size={chosen_bs}")

    loader = build_loader(chosen_bs)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # hook to capture stage1 shape from encoder hidden states
    stage1_shape = {}

    def hook_fn(_module, _inp, output):
        hs = getattr(output, "hidden_states", None) or getattr(output, "encoder_hidden_states", None)
        if hs is None:
            return
        stage1 = hs[-4]
        stage1_shape["shape"] = tuple(stage1.shape)

    handle = model.backbone.segformer.segformer.encoder.register_forward_hook(hook_fn)

    start = time.time()
    model.train()
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch}", file=sys.stdout)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device)

            with autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                out = model(images, use_dino=True)
                f1 = out["stage1_adapt"]
                f1_pool = F.avg_pool2d(f1, kernel_size=4, stride=4)
                proj = model.stage1_proj(f1_pool)
                dino_feat = out["dino_feat"]
                loss = loss_fn(proj, dino_feat) * args.lambda_aff

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if epoch == 0 and step == 0:
                handle.remove()
                inp_h, inp_w = images.shape[-2:]
                dino_h, dino_w = out["dino_grid"]
                logger.info(
                    "Stage1 shape (hook): " + str(stage1_shape.get("shape", "NA"))
                )
                logger.info(
                    f"Alignment: input={inp_h}x{inp_w}, stage1={f1.shape[-2:]} "
                    f"pooled={f1_pool.shape[-2:]}, dino={dino_h}x{dino_w}"
                )

            if step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                logger.info(f"STEP\t{epoch}\t{step}\tLOSS\t{loss.item():.6f}")

    logger.info(f"Training done. elapsed_sec={time.time() - start:.1f}")
    adapter_ckpt = {
        "stage1_adapter": model.stage1_adapter.state_dict(),
        "stage1_proj": model.stage1_proj.state_dict(),
        "args": vars(args),
    }
    torch.save(adapter_ckpt, output_dir / "adapter.pth")
    logger.info(f"Saved checkpoint: {output_dir / 'adapter.pth'}")


if __name__ == "__main__":
    main()
