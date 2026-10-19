"""Pluggable attention interface.

Every attention module takes (x, rope) and returns a tensor shaped like x.
To "update the attention later", implement a new subclass of `Attention`,
register it with `@register_attention("name")` and set
`ModelConfig.attention = "name"`. Nothing else in the model changes.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_REGISTRY: Dict[str, Type["Attention"]] = {}


def register_attention(name: str) -> Callable[[Type["Attention"]], Type["Attention"]]:
    def deco(cls: Type["Attention"]) -> Type["Attention"]:
        ATTENTION_REGISTRY[name] = cls
        return cls
    return deco


class RotaryEmbedding(nn.Module):
    """RoPE with an optional coarse (chunk-level) position mode.

    chunk=1   -> standard RoPE (position t)
    chunk=C>1 -> every token in chunk c=t//C gets position c, so the
                 positional signal only encodes order *between* chunks
                 (within-chunk order is visible only through the causal mask).
    """

    def __init__(self, head_dim: int, base: float = 10000.0, chunk: int = 1):
        super().__init__()
        assert head_dim % 2 == 0
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.chunk = int(chunk)

    def angles(self, T: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(T, device=device or self.inv_freq.device)
        if self.chunk > 1:
            pos = torch.div(pos, self.chunk, rounding_mode="floor")
        f = torch.outer(pos.float(), self.inv_freq)  # (T, hd/2)
        return f.cos(), f.sin()

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, hd); rotate-half convention
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        c = torch.cat([cos, cos], -1)
        s = torch.cat([sin, sin], -1)
        rot = torch.cat([-x2, x1], -1)
        return x * c + rot * s


class Attention(nn.Module):
    """Interface. Subclasses must implement forward(x, rope) -> same shape."""

    def __init__(self, d_model: int, n_heads: int, **kwargs):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


@register_attention("causal")
class CausalSelfAttention(Attention):
    """Standard multi-head causal self-attention with RoPE (baseline)."""

    def __init__(self, d_model: int, n_heads: int, **kwargs):
        super().__init__(d_model, n_heads)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = rope.angles(T, x.device)
        q, k = rope.rotate(q, cos, sin), rope.rotate(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


@register_attention("phase_mask")
class PhaseMaskAttention(CausalSelfAttention):
    """Cheap ablation (option C in the plan): single stream, but a query in
    phase k may only attend to keys whose phase is k (phase = (t//P) % K).
    Same params/FLOPs as the baseline. Not used in the v0 smoke runs."""

    def __init__(self, d_model: int, n_heads: int, period: int = 4, n_phases: int = 2, **kwargs):
        super().__init__(d_model, n_heads)
        self.period, self.n_phases = period, n_phases

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = rope.angles(T, x.device)
        q, k = rope.rotate(q, cos, sin), rope.rotate(k, cos, sin)
        t = torch.arange(T, device=x.device)
        ph = (t // self.period) % self.n_phases
        causal = t[None, :] <= t[:, None]
        same = ph[None, :] == ph[:, None]
        mask = causal & (same | (t[None, :] == t[:, None]))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, D))
