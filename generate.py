"""
Text completion from a trained checkpoint (base model style).

Usage:
  python generate.py --ckpt out/ckpt.pt --prompt "Once upon a time"
  python generate.py --ckpt out/ckpt.pt --prompt "..." --temperature 0.7 --top_p 0.9
"""

import argparse

import torch

from model import GPT, ModelConfig
from tokenizer import BPETokenizer


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)

    ids = [tokenizer.bos_id] + tokenizer.encode(args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for s in range(args.num_samples):
        out = model.generate(x, args.max_new_tokens,
                             temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p,
                             eos_id=tokenizer.eos_id)
        text = tokenizer.decode(out[0, 1:].tolist())  # drop <|bos|>
        print(f"--- sample {s + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
