"""Local time-reversal utilities (inspired by the 20 ms locally-reversed
speech demo): reverse the order *inside* successive chunks of length C while
keeping the chunks themselves in order."""
from __future__ import annotations

import torch


def chunk_reverse(x: torch.Tensor, chunk: int, dim: int = 1) -> torch.Tensor:
    """Reverse elements within consecutive chunks of size `chunk` along `dim`.
    A trailing partial chunk is reversed as well. chunk<=1 is the identity."""
    if chunk <= 1:
        return x
    T = x.shape[dim]
    idx = torch.arange(T, device=x.device)
    start = (idx // chunk) * chunk
    end = torch.clamp(start + chunk, max=T) - 1
    src = start + (end - idx)
    return x.index_select(dim, src)


def chunk_positions(T: int, chunk: int) -> torch.Tensor:
    """Coarse position ids: token t -> t // chunk."""
    return torch.arange(T) // max(1, chunk)


def chunk_end_mask(T: int, chunk: int) -> torch.Tensor:
    """True at positions t whose visible context [0..t] is a whole number of
    chunks (t % chunk == chunk-1). The target at those positions is the first
    token of the next chunk, so scoring there compares intact vs locally
    reversed context fairly."""
    t = torch.arange(T)
    return (t % chunk) == (chunk - 1)
