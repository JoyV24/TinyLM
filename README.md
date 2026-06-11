# tinylm — a GPT/Claude-class LLM from scratch in PyTorch

A complete, minimal-but-real LLM training stack. Same architecture family and
pipeline as GPT / Llama / Claude-class models, scaled down so it trains on
**your own data** on a single machine:

- **Tokenizer** — byte-level BPE, trained from scratch on your corpus
  ([tokenizer.py](tokenizer.py))
- **Architecture** — decoder-only transformer: causal self-attention with
  grouped-query attention (GQA), rotary position embeddings (RoPE), RMSNorm,
  SwiGLU MLPs, weight tying, FlashAttention via
  `scaled_dot_product_attention` ([model.py](model.py))
- **Pretraining** — next-token prediction with AdamW, warmup + cosine LR,
  gradient accumulation/clipping, mixed precision, checkpointing
  ([train.py](train.py))
- **SFT** — instruction tuning with a chat template and assistant-only loss
  masking, turning the base model into a chat model ([sft.py](sft.py))
- **Inference** — KV-cached sampling with temperature / top-k / top-p
  ([generate.py](generate.py)), interactive multi-turn chat ([chat.py](chat.py))

> Note: Claude's exact architecture is not public. This implements the
> modern published decoder-only transformer stack that GPT-2/3, Llama, and
> Claude-class models all belong to — the pipeline stages (tokenizer →
> pretrain → SFT → sample) are the real ones, minus RLHF and minus a few
> thousand GPUs.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Put YOUR training text in data/raw/ as .txt files, then:
python prepare_data.py --input_dir data/raw --vocab_size 4096

# 2. Pretrain a base model (preset: nano/micro/tiny/small)
python train.py --preset nano --max_iters 2000

# 3. Try base-model text completion
python generate.py --ckpt out/ckpt.pt --prompt "Once upon a time"

# 4. Put chat data in data/sft/*.jsonl (see format below), then fine-tune
python sft.py --init_from out/ckpt.pt --epochs 3

# 5. Chat with it
python chat.py --ckpt out/ckpt_sft.pt
```

## Using your own data

**Pretraining**: drop any plain-text `.txt` files into `data/raw/`
(subfolders fine) and rerun `prepare_data.py --retrain_tokenizer`. More text
is better — a few MB minimum for coherent output, and pick `--vocab_size`
roughly proportional to corpus size (4k for small corpora, 8–32k for bigger).

**SFT**: one JSON object per line in `data/sft/*.jsonl`:

```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

`system` is supported as an optional first message. Multi-turn conversations
work — just include more messages. Loss is computed only on assistant turns.

## Model presets ([configs.py](configs.py))

| preset | layers | width | heads (q/kv) | context | ~params |
|--------|--------|-------|--------------|---------|---------|
| nano   | 4      | 128   | 4 / 2        | 256     | ~1M     |
| micro  | 6      | 288   | 6 / 3        | 512     | ~7M     |
| tiny   | 8      | 512   | 8 / 4        | 1024    | ~28M    |
| small  | 12     | 768   | 12 / 4       | 1024    | ~95M    |

CPU: stick to `nano`/`micro`. GPU is auto-detected (bf16/fp16 autocast,
fused AdamW, FlashAttention all kick in automatically); `tiny`/`small` want
a GPU. Resume training any time with `python train.py --resume out/ckpt.pt`.

## Expectations

Loss starts near `ln(vocab_size)` (~8.3 for 4k vocab) and should fall fast.
With only kilobytes of data a model memorizes rather than generalizes —
that's expected; the pipeline is identical to the big leagues, the magic
ingredient there is simply 10^12 more tokens.
