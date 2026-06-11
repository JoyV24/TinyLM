"""
Data preparation pipeline for pretraining.

  1. Reads every .txt file in --input_dir (this is YOUR data -- drop any
     plain-text files in there).
  2. Trains the BPE tokenizer on it (unless one already exists).
  3. Tokenizes every document, joining documents with <|eos|> so the model
     learns document boundaries.
  4. Splits into train/val and writes flat binary token files (train.bin,
     val.bin) that train.py memory-maps -- the same trick nanoGPT and most
     pretraining pipelines use.

Usage:
  python prepare_data.py --input_dir data/raw --vocab_size 4096
"""

import argparse
import glob
import json
import os

import numpy as np

from tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/raw",
                    help="directory of .txt files to train on")
    ap.add_argument("--out_dir", default="data",
                    help="where train.bin / val.bin / tokenizer.json go")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    ap.add_argument("--retrain_tokenizer", action="store_true",
                    help="retrain tokenizer even if tokenizer.json exists")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.txt"),
                             recursive=True))
    if not files:
        raise SystemExit(
            f"No .txt files found in {args.input_dir}. "
            f"Put your training text there first.")

    docs = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read().strip()
        if text:
            docs.append(text)
    full_text = "\n\n".join(docs)
    print(f"Read {len(docs)} document(s), {len(full_text):,} characters total "
          f"from {len(files)} file(s).")

    # ------------------------------------------------------------------
    # tokenizer
    # ------------------------------------------------------------------
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(tok_path) and not args.retrain_tokenizer:
        print(f"Loading existing tokenizer from {tok_path}")
        tok = BPETokenizer.load(tok_path)
    else:
        print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) ...")
        tok = BPETokenizer()
        tok.train(full_text, vocab_size=args.vocab_size)
        tok.save(tok_path)
        print(f"Saved tokenizer to {tok_path}")
    print(f"Actual vocab size: {tok.vocab_size}")

    # ------------------------------------------------------------------
    # tokenize documents, join with <|eos|>
    # ------------------------------------------------------------------
    all_ids = []
    for i, doc in enumerate(docs):
        all_ids.extend(tok.encode(doc))
        all_ids.append(tok.eos_id)
        print(f"  tokenized doc {i + 1}/{len(docs)} "
              f"(running total {len(all_ids):,} tokens)")

    n = len(all_ids)
    n_val = max(int(n * args.val_fraction), 1)
    train_ids = all_ids[:-n_val]
    val_ids = all_ids[-n_val:]

    dtype = np.uint16 if tok.vocab_size < 65536 else np.uint32
    np.array(train_ids, dtype=dtype).tofile(os.path.join(args.out_dir, "train.bin"))
    np.array(val_ids, dtype=dtype).tofile(os.path.join(args.out_dir, "val.bin"))

    meta = {
        "vocab_size": tok.vocab_size,
        "dtype": str(np.dtype(dtype)),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    chars_per_token = len(full_text) / max(n, 1)
    print(f"\nDone. train: {len(train_ids):,} tokens | val: {len(val_ids):,} tokens "
          f"| ~{chars_per_token:.2f} chars/token")
    print(f"Wrote train.bin, val.bin, meta.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
