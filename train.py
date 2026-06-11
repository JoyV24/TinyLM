"""
Pretraining loop -- the same recipe used to pretrain GPT/Claude-class models,
scaled down: next-token prediction over a token stream with AdamW, linear
warmup + cosine decay, gradient accumulation, gradient clipping, mixed
precision, periodic eval, and checkpointing.

Usage (after prepare_data.py):
  python train.py --preset nano --max_iters 2000
  python train.py --preset tiny --batch_size 16 --grad_accum 4   # on a GPU
  python train.py --resume out/ckpt.pt                            # continue
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from configs import PRESETS
from model import GPT, ModelConfig


def get_batch(data, block_size, batch_size, device):
    if len(data) < block_size + 2:
        # split shorter than the context length (tiny corpus): tile it
        data = np.tile(data, (block_size + 2) // len(data) + 1)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(
        data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(
        data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, datasets, args, ctx, device):
    model.eval()
    out = {}
    for split, data in datasets.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, model.cfg.block_size, args.batch_size, device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / args.warmup_iters
    if it >= args.max_iters:
        return args.min_lr
    ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


def main():
    ap = argparse.ArgumentParser()
    # data / io
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    # model
    ap.add_argument("--preset", default="nano", choices=list(PRESETS))
    ap.add_argument("--block_size", type=int, default=None,
                    help="override the preset's context length")
    ap.add_argument("--dropout", type=float, default=0.0)
    # optimization
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="gradient accumulation steps (simulates bigger batches)")
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # eval / logging
    ap.add_argument("--eval_interval", type=int, default=250)
    ap.add_argument("--eval_iters", type=int, default=20)
    ap.add_argument("--log_interval", type=int, default=10)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (Linux + GPU recommended)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 on modern GPUs, fp16 + grad scaler on older ones, fp32 on CPU
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    elif device == "cuda":
        dtype = "float16"
    else:
        dtype = "float32"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtype]
    ctx = (nullcontext() if device == "cpu"
           else torch.autocast(device_type=device, dtype=ptdtype))
    print(f"device={device} dtype={dtype}")

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    np_dtype = np.dtype(meta["dtype"])
    datasets = {
        "train": np.memmap(os.path.join(args.data_dir, "train.bin"),
                           dtype=np_dtype, mode="r"),
        "val": np.memmap(os.path.join(args.data_dir, "val.bin"),
                         dtype=np_dtype, mode="r"),
    }

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    iter_num, best_val = 0, float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = ModelConfig(**ckpt["model_config"])
        model = GPT(cfg)
        model.load_state_dict(ckpt["model"])
        iter_num = ckpt["iter_num"]
        best_val = ckpt["best_val"]
    else:
        preset = dict(PRESETS[args.preset])
        if args.block_size:
            preset["block_size"] = args.block_size
        cfg = ModelConfig(vocab_size=meta["vocab_size"], dropout=args.dropout,
                          **preset)
        model = GPT(cfg)
    model.to(device)
    print(f"model: preset={args.preset} {model.num_params() / 1e6:.2f}M params, "
          f"context={cfg.block_size}")

    tokens_per_iter = args.batch_size * args.grad_accum * cfg.block_size
    print(f"tokens per iteration: {tokens_per_iter:,} "
          f"(dataset has {meta['train_tokens']:,} train tokens "
          f"-> ~{args.max_iters * tokens_per_iter / max(meta['train_tokens'], 1):.1f} "
          f"epochs at max_iters={args.max_iters})")

    optimizer = model.configure_optimizer(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device)
    if args.resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    scaler = torch.amp.GradScaler(device, enabled=(dtype == "float16"))

    if args.compile:
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    raw_model = model._orig_mod if args.compile else model
    x, y = get_batch(datasets["train"], cfg.block_size, args.batch_size, device)
    t0 = time.time()

    while iter_num <= args.max_iters:
        lr = get_lr(iter_num, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, datasets, args, ctx, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                if iter_num > 0:
                    ckpt_out = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": asdict(raw_model.cfg),
                        "iter_num": iter_num,
                        "best_val": best_val,
                    }
                    path = os.path.join(args.out_dir, "ckpt.pt")
                    torch.save(ckpt_out, path)
                    print(f"  saved checkpoint to {path} (val {best_val:.4f})")

        for micro_step in range(args.grad_accum):
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            # prefetch next batch while the forward/backward runs
            x, y = get_batch(datasets["train"], cfg.block_size,
                             args.batch_size, device)
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tps = tokens_per_iter * args.log_interval / dt if iter_num > 0 else 0
            print(f"iter {iter_num}: loss {loss.item() * args.grad_accum:.4f}, "
                  f"lr {lr:.2e}, {dt * 1000 / max(args.log_interval, 1):.0f} ms/iter, "
                  f"{tps:,.0f} tok/s")
        iter_num += 1

    print(f"\nTraining done. Best val loss: {best_val:.4f}. "
          f"Checkpoint: {os.path.join(args.out_dir, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
