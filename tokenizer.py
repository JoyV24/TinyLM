"""
Byte-level BPE tokenizer, written from scratch (no external dependencies).

This is the same algorithm family used by GPT-2/3/4 and similar models:
  1. Text is split into "words" with a regex (so merges never cross word
     boundaries, which keeps the vocab clean).
  2. Each word is converted to raw UTF-8 bytes (so ANY text is representable,
     no <unk> token ever needed).
  3. Training repeatedly merges the most frequent adjacent pair of tokens
     until the target vocab size is reached.

Special tokens (<|bos|>, <|eos|>, role tokens for chat) live at the top of
the vocab and are never produced by the BPE merges themselves -- pipeline
code inserts them explicitly by id.
"""

import json
import re
import os
from collections import Counter
from functools import lru_cache

# Word-splitting pattern, in the spirit of GPT-2's tokenizer regex.
# Python's `re` treats \w as unicode-aware, which is good enough here.
SPLIT_PATTERN = re.compile(r"'\w+| ?\w+| ?[^\w\s]+|\s+")

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


class BPETokenizer:
    def __init__(self):
        self.merges = {}          # (id, id) -> merged id, in creation order
        self.vocab = {}           # id -> bytes
        self.special_tokens = {}  # str -> id

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(self, text, vocab_size, verbose=True):
        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert num_merges > 0, f"vocab_size must be > {256 + len(SPECIAL_TOKENS)}"

        # Count unique words; BPE training then only touches unique words,
        # which makes training fast even on multi-MB corpora.
        word_freqs = Counter(SPLIT_PATTERN.findall(text))
        # each word is a tuple of token ids, starting as raw bytes
        words = {w: tuple(w.encode("utf-8")) for w in word_freqs}

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            # count adjacent pairs across all words, weighted by word frequency
            stats = Counter()
            for w, freq in word_freqs.items():
                ids = words[w]
                for pair in zip(ids, ids[1:]):
                    stats[pair] += freq
            if not stats:
                break  # nothing left to merge (tiny corpus)

            pair = max(stats, key=stats.get)
            new_id = 256 + i
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

            # apply the merge to every word that contains the pair
            for w, ids in words.items():
                if pair[0] in ids:
                    words[w] = tuple(self._merge(list(ids), pair, new_id))

            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}, count {stats[pair]})")

        self.merges = merges
        self.vocab = vocab
        base = 256 + len(merges)
        self.special_tokens = {t: base + j for j, t in enumerate(SPECIAL_TOKENS)}
        self._encode_word_cached.cache_clear()

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def pad_id(self):
        return self.special_tokens["<|pad|>"]

    @property
    def bos_id(self):
        return self.special_tokens["<|bos|>"]

    @property
    def eos_id(self):
        return self.special_tokens["<|eos|>"]

    @lru_cache(maxsize=65536)
    def _encode_word_cached(self, word):
        ids = list(word.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            # apply the earliest-created merge present (lowest merged id)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return tuple(ids)

    def encode(self, text):
        ids = []
        for word in SPLIT_PATTERN.findall(text):
            ids.extend(self._encode_word_cached(word))
        return ids

    def decode(self, ids):
        inv_special = {v: k for k, v in self.special_tokens.items()}
        parts = []
        for i in ids:
            if i in inv_special:
                parts.append(inv_special[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path):
        data = {
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.merges = {(a, b): idx for a, b, idx in data["merges"]}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        tok.special_tokens = data["special_tokens"]
        return tok


# ----------------------------------------------------------------------
# Chat template (used by sft.py and chat.py)
#
# Rendered format:
#   <|bos|><|system|>...<|eos|><|user|>...<|eos|><|assistant|>...<|eos|>...
#
# Returns (ids, labels) where labels are -1 (ignored by the loss) everywhere
# except on assistant content + its <|eos|> -- so SFT only teaches the model
# to produce assistant turns, exactly like real instruction-tuning pipelines.
# ----------------------------------------------------------------------
def encode_chat(tokenizer, messages, add_generation_prompt=False):
    ids = [tokenizer.bos_id]
    labels = [-1]
    for m in messages:
        role_token = f"<|{m['role']}|>"
        assert role_token in tokenizer.special_tokens, f"unknown role {m['role']}"
        content_ids = tokenizer.encode(m["content"])
        ids += [tokenizer.special_tokens[role_token]] + content_ids + [tokenizer.eos_id]
        if m["role"] == "assistant":
            labels += [-1] + content_ids + [tokenizer.eos_id]
        else:
            labels += [-1] * (len(content_ids) + 2)
    if add_generation_prompt:
        ids.append(tokenizer.special_tokens["<|assistant|>"])
        labels.append(-1)
    return ids, labels


if __name__ == "__main__":
    # quick self-test
    tok = BPETokenizer()
    sample = "Hello world! This is a quick tokenizer self-test. Hello hello hello."
    tok.train(sample * 20, vocab_size=300, verbose=False)
    enc = tok.encode("Hello world! Unseen텍스트 also works.")
    assert tok.decode(enc) == "Hello world! Unseen텍스트 also works."
    print(f"self-test OK, vocab_size={tok.vocab_size}, sample ids={enc[:10]}")
