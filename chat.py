"""
Interactive chat REPL for an SFT'd checkpoint -- multi-turn conversation
using the chat template, just like talking to a (very small) assistant.

Usage:
  python chat.py --ckpt out/ckpt_sft.pt
  python chat.py --ckpt out/ckpt_sft.pt --system "You are a pirate."

Commands inside the REPL:  /reset clears history, /quit exits.
"""

import argparse

import torch

from generate import load_model
from tokenizer import BPETokenizer, encode_chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt_sft.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BPETokenizer.load(args.tokenizer)
    model = load_model(args.ckpt, device)
    print(f"Loaded {model.num_params() / 1e6:.2f}M-param model on {device}. "
          f"/reset clears history, /quit exits.\n")

    def fresh_history():
        return ([{"role": "system", "content": args.system}]
                if args.system else [])

    messages = fresh_history()
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            messages = fresh_history()
            print("(history cleared)")
            continue

        messages.append({"role": "user", "content": user})
        ids, _ = encode_chat(tokenizer, messages, add_generation_prompt=True)
        if len(ids) >= model.cfg.block_size:
            print("(context full -- clearing oldest turns)")
            while len(ids) >= model.cfg.block_size and len(messages) > 2:
                messages.pop(0)
                ids, _ = encode_chat(tokenizer, messages,
                                     add_generation_prompt=True)

        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, args.max_new_tokens,
                             temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p,
                             eos_id=tokenizer.eos_id)
        new_ids = out[0, len(ids):].tolist()
        if new_ids and new_ids[-1] == tokenizer.eos_id:
            new_ids = new_ids[:-1]
        reply = tokenizer.decode(new_ids).strip()
        print(f"bot> {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
