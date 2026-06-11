"""
A modern decoder-only transformer LLM, from scratch in PyTorch.

This is the same architecture family as GPT / Llama / Claude-class models:
  - token embeddings (weight-tied with the output head)
  - N transformer blocks, each:
      RMSNorm -> causal self-attention (multi-head, grouped-query, RoPE)
      RMSNorm -> SwiGLU MLP
    with residual connections around both (pre-norm)
  - final RMSNorm -> linear head over the vocabulary
  - trained with next-token-prediction cross-entropy

Attention uses torch.scaled_dot_product_attention, which dispatches to
FlashAttention kernels on GPU automatically. Generation uses a KV cache,
just like production inference engines.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    n_layer: int = 8        # number of transformer blocks
    n_head: int = 8         # query heads
    n_kv_head: int = 4      # key/value heads (GQA; set == n_head for full MHA)
    n_embd: int = 512       # model width
    block_size: int = 512   # max context length
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (no mean subtraction, no bias)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms) * self.weight


def apply_rope(x, cos, sin):
    """Rotary position embedding. x: (B, n_head, T, head_dim)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # grouped-query attention: each kv head serves n_head/n_kv_head q heads
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # During incremental decoding (T == 1) the query may attend to the
        # whole cache, so no causal mask is needed; otherwise mask causally.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(T > 1),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), new_cache


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward, as in Llama/PaLM: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(2 * (4 * cfg.n_embd) / 3)
        hidden = 64 * ((hidden + 63) // 64)  # round up to multiple of 64
        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # precompute RoPE tables for the full context
        head_dim = cfg.n_embd // cfg.n_head
        inv_freq = 1.0 / (cfg.rope_theta ** (
            torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(cfg.block_size).float()
        freqs = torch.outer(t, inv_freq)              # (block_size, head_dim/2)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-stream output projections
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok_emb.weight.numel()  # don't double-count tied head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size"
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x, _ = block(x, cos, sin)
        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss
        # inference-time micro-optimization: only project the last position
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def _forward_with_cache(self, idx, caches, start_pos):
        """One incremental step (or the prefill) using the KV cache."""
        T = idx.shape[1]
        cos = self.rope_cos[start_pos:start_pos + T].to(idx.device)
        sin = self.rope_sin[start_pos:start_pos + T].to(idx.device)
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x, caches[i] = block(x, cos, sin, caches[i])
        x = self.final_norm(x)
        return self.lm_head(x[:, [-1], :])

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, eos_id=None):
        """Autoregressive sampling with a KV cache.

        idx: (B, T) prompt token ids. Returns prompt + generated ids.
        """
        self.eval()
        idx = idx[:, -self.cfg.block_size:]
        caches = [(None, None)] * self.cfg.n_layer
        logits = self._forward_with_cache(idx, caches, start_pos=0)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]
            if temperature <= 0:  # greedy
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits[logits < kth] = -float("inf")
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[mask] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter_(
                        1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
            if idx.shape[1] >= self.cfg.block_size:
                break  # context full
            logits = self._forward_with_cache(next_id, caches,
                                              start_pos=idx.shape[1] - 1)
        return idx

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with weight decay applied only to matrices (not norms/embeddings'
        1-D params), the standard LLM recipe."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                 fused=use_fused)


if __name__ == "__main__":
    # quick self-test
    cfg = ModelConfig(vocab_size=512, n_layer=2, n_head=4, n_kv_head=2,
                      n_embd=64, block_size=128)
    model = GPT(cfg)
    print(f"params: {model.num_params():,}")
    x = torch.randint(0, 512, (2, 16))
    logits, loss = model(x, targets=x)
    print(f"loss at init: {loss.item():.3f} "
          f"(expected ~{math.log(cfg.vocab_size):.3f} for random init)")
    out = model.generate(x[:, :4], max_new_tokens=8, temperature=0.8, top_k=50)
    print(f"generate self-test OK, output shape {tuple(out.shape)}")
