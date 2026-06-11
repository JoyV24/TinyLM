"""
Supervised fine-tuning (SFT) -- the stage that turns a pretrained base model
into a chat assistant (the "instruction tuning" step of the GPT/Claude
training pipeline, minus RLHF).

Data format: one JSON object per line in data/sft/*.jsonl:
  {"messages": [{"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}
("system" role is also supported as the first message.)

Each conversation is rendered with the chat template and the loss is masked
so the model is only trained on the assistant's tokens.

Usage (after pretraining):
  python sft.py --init_from out/ckpt.pt --epochs 3
"""

import argparse
import glob
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch

from model import GPT, ModelConfig
from tokenizer import BPETokenizer, encode_chat


def load_examples(sft_dir, tokenizer, block_size):
    examples = []
    skipped = 0
    files = sorted(glob.glob(os.path.join(sft_dir, "*.jsonl")))
    if not files:
        raise SystemExit(f"No .jsonl files found in {sft_dir}")
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)["messages"]
                ids, labels = encode_chat(tokenizer, messages)
                if len(ids) > block_size:
                    skipped += 1
                    continue
                examples.append((ids, labels))
    print(f"Loaded {len(examples)} conversations "
          f"({skipped} skipped as longer than block_size={block_size})")
    return examples


def make_batch(examples, idxs, pad_id, device):
    max_len = max(len(examples[i][0]) for i in idxs)
    B = len(idxs)
    x = torch.full((B, max_len), pad_id, dtype=torch.long)
    y = torch.full((B, max_len), -1, dtype=torch.long)
    for row, i in enumerate(idxs):
        ids, labels = examples[i]
        x[row, :len(ids)] = torch.tensor(ids)
        y[row, :len(labels)] = torch.tensor(labels)
    # next-token shift: predict y[t+1] from x[<=t]
    return x[:, :-1].to(device), y[:, 1:].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", default="out/ckpt.pt",
                    help="pretrained checkpoint to fine-tune")
    ap.add_argument("--sft_dir", default="data/sft")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" and torch.cuda.is_bf16_supported()
           else nullcontext())

    tokenizer = BPETokenizer.load(args.tokenizer)

    print(f"Loading pretrained model from {args.init_from}")
    ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.train()
    print(f"model: {model.num_params() / 1e6:.2f}M params")

    examples = load_examples(args.sft_dir, tokenizer, cfg.block_size)
    steps_per_epoch = math.ceil(len(examples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    optimizer = model.configure_optimizer(
        args.weight_decay, args.learning_rate, (0.9, 0.95), device)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        for b in range(steps_per_epoch):
            idxs = order[b * args.batch_size:(b + 1) * args.batch_size]
            x, y = make_batch(examples, idxs, tokenizer.pad_id, device)

            # linear warmup + cosine decay
            if step < warmup_steps:
                lr = args.learning_rate * (step + 1) / warmup_steps
            else:
                ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = args.learning_rate * 0.5 * (1 + math.cos(math.pi * ratio))
            for group in optimizer.param_groups:
                group["lr"] = lr

            with ctx:
                _, loss = model(x, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 10 == 0:
                dt = time.time() - t0
                t0 = time.time()
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps}: "
                      f"loss {loss.item():.4f}, lr {lr:.2e}, "
                      f"{dt:.1f}s/10 steps")
            step += 1

    out_path = os.path.join(args.out_dir, "ckpt_sft.pt")
    torch.save({
        "model": model.state_dict(),
        "model_config": asdict(cfg),
        "iter_num": step,
        "best_val": float("nan"),
    }, out_path)
    print(f"\nSFT done. Saved chat model to {out_path}")
    print("Try it:  python chat.py --ckpt out/ckpt_sft.pt")


if __name__ == "__main__":
    main()
