"""Baseline TinyLM and the PhaseLM (phase-alternating multi-stream) prototype."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ATTENTION_REGISTRY, RotaryEmbedding


@dataclass
class ModelConfig:
    vocab_size: int = 256
    n_channels: int = 1          # input ids per position (2 = left/right eye for rivalry)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    mlp_mult: float = 8 / 3
    attention: str = "causal"    # key in ATTENTION_REGISTRY -> the swappable part
    attention_kwargs: dict = field(default_factory=dict)
    rope_chunk: int = 1          # >1 = CoarseRoPE (chunk-level positions)
    # --- PhaseLM only ---
    arch: str = "baseline"       # "baseline" | "phase"
    n_streams: int = 2           # K competing content streams
    period: int = 4              # tokens per phase micro-epoch
    schedule: str = "fixed"      # fixed | random | all | learned
    inhibition: bool = True      # mutual inhibition between streams
    tie_embeddings: bool = True

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float):
        super().__init__()
        h = int(round(d * mult / 8)) * 8
        self.gate_up = nn.Linear(d, 2 * h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    """Pre-norm block: x + Attn(norm(x)); x + MLP(norm(x)). Attention is pluggable."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        attn_cls = ATTENTION_REGISTRY[cfg.attention]
        self.attn = attn_cls(cfg.d_model, cfg.n_heads, **cfg.attention_kwargs)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_mult)

    def forward(self, x, rope):
        x = x + self.attn(self.n1(x), rope)
        return x + self.mlp(self.n2(x))


class MultiEmbedding(nn.Module):
    """Sum of per-channel embeddings (channel = eye in the rivalry task)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(cfg.vocab_size, cfg.d_model) for _ in range(cfg.n_channels))
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

    def forward(self, ids):
        if ids.dim() == 2:
            ids = ids.unsqueeze(-1)
        return sum(tab(ids[..., c]) for c, tab in enumerate(self.tables))


class _Base(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = MultiEmbedding(cfg)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, chunk=cfg.rope_chunk)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.emb.tables[0].weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TinyLM(_Base):
    """Baseline: standard causal transformer (RoPE, SwiGLU, RMSNorm)."""

    def forward(self, ids, return_hidden: bool = False):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x, self.rope)
        h = self.norm(x)
        logits = self.head(h)
        return (logits, {"final": h}) if return_hidden else logits


def phase_schedule(kind: str, B: int, K: int, T: int, period: int, device=None,
                   generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Access weights a[b,k,t] (sum over k = 1).
    fixed : stream (t//P) % K has access (deterministic alternation)
    random: each length-P micro-epoch is given to a uniformly random stream
            (same duty cycle and epoch length, no regular rhythm)
    all   : every stream writes with weight 1/K (no-phase control)"""
    t = torch.arange(T, device=device)
    if kind == "fixed":
        ph = ((t // period) % K).expand(B, T)
    elif kind == "random":
        n_ep = (T + period - 1) // period
        ep = torch.randint(0, K, (B, n_ep), generator=generator).to(device)
        ph = ep.repeat_interleave(period, dim=1)[:, :T]
    elif kind == "all":
        return torch.full((B, K, T), 1.0 / K, device=device)
    else:
        raise ValueError(kind)
    return F.one_hot(ph, K).permute(0, 2, 1).float()


class PhaseLM(_Base):
    """Phase-alternating multi-stream prototype (PhaseAttention v0).

    * K content streams h^k share ALL block weights (same "population").
    * Each layer, every stream is updated by the same block (causal attention
      over that stream's own history); mutual inhibition pushes streams apart.
    * Only the stream that currently has *access* (phase schedule a[k,t])
      writes its layer update into the workspace stream z, with a learned
      access gain. z is continuous across switches (the "drone") and is fed
      back to all streams through a learned scalar per layer.
    * The LM head reads only z. Suppressed streams are still computed and can
      be decoded with a probe (see `return_hidden`).
    Extra params vs TinyLM: K*D stream embeddings + D gain + 2 scalars/layer
    (+ D*K gate if schedule == "learned").
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        K, D, L = cfg.n_streams, cfg.d_model, cfg.n_layers
        self.stream_emb = nn.Parameter(torch.randn(K, D) * 0.02)
        self.access_gain = nn.Parameter(torch.ones(D))
        self.inhib = nn.Parameter(torch.full((L,), 0.05), requires_grad=cfg.inhibition)
        if not cfg.inhibition:
            self.inhib.data.zero_()
        self.feedback = nn.Parameter(torch.zeros(L))
        self.gate = nn.Linear(D, K, bias=True) if cfg.schedule == "learned" else None
        self.fixed_schedule: Optional[torch.Tensor] = None  # optional override (eval)

    def access(self, x0: torch.Tensor) -> torch.Tensor:
        B, T, _ = x0.shape
        if self.fixed_schedule is not None:
            return self.fixed_schedule.to(x0.device)
        if self.cfg.schedule == "learned":
            # per-micro-epoch soft gate from the input embedding; straight-through hard choice
            soft = self.gate(x0).softmax(-1).permute(0, 2, 1)  # B,K,T
            hard = F.one_hot(soft.argmax(1), self.cfg.n_streams).permute(0, 2, 1).float()
            return hard + soft - soft.detach()
        return phase_schedule(self.cfg.schedule, B, self.cfg.n_streams, T, self.cfg.period, x0.device)

    def forward(self, ids, return_hidden: bool = False):
        cfg = self.cfg
        K = cfg.n_streams
        x0 = self.emb(ids)                                  # B,T,D
        B, T, D = x0.shape
        a = self.access(x0)                                 # B,K,T
        h = x0.unsqueeze(1) + self.stream_emb[None, :, None, :]   # B,K,T,D
        z = x0
        for i, blk in enumerate(self.blocks):
            hk = h.reshape(B * K, T, D)
            new = blk(hk, self.rope).view(B, K, T, D)
            delta = new - h
            h = new
            if K > 1:
                others = (h.sum(1, keepdim=True) - h) / (K - 1)
                h = h - self.inhib[i] * others
            z = z + self.access_gain * (a.unsqueeze(-1) * delta).sum(1)
            h = h + self.feedback[i] * z.unsqueeze(1)
        zf = self.norm(z)
        logits = self.head(zf)
        if return_hidden:
            return logits, {"final": zf, "streams": self.norm(h), "access": a}
        return logits


def build_model(cfg: ModelConfig) -> nn.Module:
    return PhaseLM(cfg) if cfg.arch == "phase" else TinyLM(cfg)
