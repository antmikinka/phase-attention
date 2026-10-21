"""Data for phase_model: TinyStories bytes (train split only for training) and a synthetic
phase-aligned associative-recall task."""
from __future__ import annotations
import os
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tinystories")


def tinystories(split: str) -> torch.Tensor:
    """uint8 byte stream. train = first 40 MB of TinyStoriesV2-GPT4-train.txt; val = TinyStoriesV2-GPT4-valid.txt."""
    f = os.path.join(ROOT, "train_head.txt" if split == "train" else "valid.txt")
    return torch.from_numpy(np.fromfile(f, dtype=np.uint8).astype(np.int64))


def lm_batch(data: torch.Tensor, B: int, T: int, g: torch.Generator) -> torch.Tensor:
    i = torch.randint(0, data.numel() - T - 1, (B,), generator=g)
    return torch.stack([data[j:j + T + 1] for j in i.tolist()])


def val_windows(data: torch.Tensor, n: int, T: int) -> list[torch.Tensor]:
    step = (data.numel() - T - 1) // n
    return [data[k * step:k * step + T + 1] for k in range(n)]


# ----------------------------------------------------------------------------- associative recall
FILLER = (32, 128)          # printable ascii filler
KEYS = (128, 160)           # 32 keys
VALS = (160, 192)           # 32 values
QUERY = 255
LAG_BLOCKS = (1, 2, 3, 4, 7, 8, 15, 16)    # odd = stored in the OTHER phase than the query (P=4 blocks)


def recall_batch(B: int, T: int = 128, P: int = 4, n_pairs: int = 10, g: torch.Generator | None = None,
                 lag_blocks=LAG_BLOCKS):
    """Sequence of T tokens = T/P blocks. A store block is [f, key, value, f]; a query block (lag L blocks later)
    is [QUERY, key, value*, f] and the model must predict value* at the key position.
    With phase(t) = floor(t/P) mod 2, odd L puts the stored pair in the phase opposite to the query
    (= the 'suppressed' phase for kv phase attention), even L in the same phase.
    Returns x (B,T+1 incl. next-token), target mask positions (B,n_pairs), lags in tokens (B,n_pairs)."""
    nb = T // P
    x = torch.randint(*FILLER, (B, T + 1), generator=g)
    pos = torch.zeros(B, n_pairs, dtype=torch.long); lag = torch.zeros(B, n_pairs, dtype=torch.long)
    for b in range(B):
        used = set()
        keys = (torch.randperm(KEYS[1] - KEYS[0], generator=g)[:n_pairs] + KEYS[0]).tolist()
        j = 0; tries = 0
        while j < n_pairs:
            tries += 1
            if tries > 2000:                       # dead end: restart this sequence's placement
                used.clear(); j = 0; tries = 0; x[b] = torch.randint(*FILLER, (T + 1,), generator=g)
            L = lag_blocks[int(torch.randint(0, len(lag_blocks), (1,), generator=g))]
            s = int(torch.randint(0, nb - L, (1,), generator=g)); q = s + L
            if s in used or q in used:
                continue
            used.update((s, q))
            v = int(torch.randint(*VALS, (1,), generator=g))
            x[b, s * P + 1], x[b, s * P + 2] = keys[j], v
            x[b, q * P], x[b, q * P + 1], x[b, q * P + 2] = QUERY, keys[j], v
            pos[b, j] = q * P + 1; lag[b, j] = L * P
            j += 1
    return x, pos, lag
