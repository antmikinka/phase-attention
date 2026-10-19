"""Datasets: synthetic binocular-rivalry task and byte-level TinyStories."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RivalryConfig:
    n_symbols: int = 16   # symbols per eye
    seq_len: int = 48
    delay: int = 8        # suppressed eye shows noise for the last `delay` steps before the query
    recall: str = "last"  # "last": recall last clean suppressed symbol;
                          # "extrap": report where the suppressed progression IS now (needs tracking through the noise)
    @property
    def vocab_size(self):
        return self.n_symbols + 3
    @property
    def CUE_L(self):
        return self.n_symbols
    @property
    def CUE_R(self):
        return self.n_symbols + 1
    @property
    def QUERY(self):
        return self.n_symbols + 2


def make_rivalry_batch(rc: RivalryConfig, B: int, g: torch.Generator):
    """Each position carries two symbols (left eye, right eye).

    * pos 0: cue token in both eyes -> which eye is dominant (CUE_L / CUE_R)
    * pos 1..T-2: each eye shows an arithmetic progression mod V
      (random start, random non-zero step); for the final `delay` positions
      before the query the SUPPRESSED eye shows random noise.
    * pos T-1: QUERY in both eyes.
    Targets:
    * report: at pos t in [0, T-3], the dominant eye's symbol at t+1
    * recall: at pos T-1, the suppressed eye's LAST CLEAN symbol (delayed recall
      of content that was never reported)
    Returns ids (B,T,2), targets (B,T) with -100 where unscored, plus labels for probes.
    """
    V, T, D = rc.n_symbols, rc.seq_len, rc.delay
    n = T - 2
    start = torch.randint(0, V, (B, 2), generator=g)
    step = torch.randint(1, V, (B, 2), generator=g)
    t = torch.arange(1, n + 1)
    streams = (start[:, :, None] + step[:, :, None] * t[None, None, :]) % V  # B,2,n
    dom = torch.randint(0, 2, (B,), generator=g)
    sup = 1 - dom
    eyes = streams.clone()
    noise = torch.randint(0, V, (B, D), generator=g)
    bi = torch.arange(B)
    eyes[bi, sup, n - D:] = noise
    ids = torch.empty(B, T, 2, dtype=torch.long)
    cue = torch.where(dom == 0, rc.CUE_L, rc.CUE_R)
    ids[:, 0, :] = cue[:, None]
    ids[:, 1:T - 1, :] = eyes.permute(0, 2, 1)
    ids[:, T - 1, :] = rc.QUERY
    dom_stream = streams[bi, dom]   # B,n  (clean)
    sup_stream = streams[bi, sup]   # B,n  (clean, ground truth)
    tgt = torch.full((B, T), -100, dtype=torch.long)
    tgt[:, 0:n] = dom_stream          # predict dominant symbol at next position
    last_clean = sup_stream[:, n - D - 1]
    if rc.recall == "extrap":
        s_sup = start[bi, sup]; st_sup = step[bi, sup]
        last_clean = (s_sup + st_sup * (n + 1)) % V   # value the suppressed eye would show at the query position
    tgt[:, T - 1] = last_clean
    labels = {"dom": dom_stream, "sup": sup_stream, "sup_seen": eyes[bi, sup], "dom_eye": dom,
              "recall": last_clean}
    return ids, tgt, labels


class ByteText:
    """Byte-level corpus (vocab 256). Splits the file 95/5 into train/val."""

    def __init__(self, path: str, val_frac: float = 0.05):
        raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        cut = int(len(raw) * (1 - val_frac))
        self.train, self.val = raw[:cut], raw[cut:]

    def batch(self, split: str, B: int, T: int, g: np.random.Generator):
        arr = self.train if split == "train" else self.val
        ix = g.integers(0, len(arr) - T - 1, size=B)
        x = np.stack([arr[i:i + T + 1] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x)
        return x[:, :-1], x[:, 1:]

    def fixed_val(self, n: int, T: int, seed: int = 1234):
        g = np.random.default_rng(seed)
        return self.batch("val", n, T, g)
