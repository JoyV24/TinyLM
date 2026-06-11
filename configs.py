"""Model size presets. Pick with --preset in train.py.

Rough parameter counts assume a ~8k vocab; they scale a bit with vocab size.
"""

PRESETS = {
    # good for CPU experiments / sanity checks
    "nano": dict(n_layer=4, n_head=4, n_kv_head=2, n_embd=128, block_size=256),    # ~1M
    # trains in minutes-to-hours on CPU, small GPU instantly
    "micro": dict(n_layer=6, n_head=6, n_kv_head=3, n_embd=288, block_size=512),   # ~7M
    # needs a GPU to be pleasant
    "tiny": dict(n_layer=8, n_head=8, n_kv_head=4, n_embd=512, block_size=1024),   # ~28M
    # GPT-2-small scale
    "small": dict(n_layer=12, n_head=12, n_kv_head=4, n_embd=768, block_size=1024),  # ~95M
}
