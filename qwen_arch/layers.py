"""Built-in layer types.

attention: "qwen3" (stock), "phase" (PhaseAttention, modes "kv" and "head")
position : "rope" (stock), "chunk_rope" (positions shared inside chunks of C)
extra    : "tracking" (continuous slow state that content couples to)
Every new type has an identity setting that reproduces stock Qwen3 numerically.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register


# ============================================================ positions
@register("position", "rope")
class RotaryPositions(nn.Module):
    """Stock Qwen3 RoPE (same op order as HF so logits match)."""

    def __init__(self, c, **kw):
        super().__init__()
        d = c.head_dim
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        self.register_buffer("inv_freq", inv, persistent=False)

    def position_ids(self, pos: torch.Tensor) -> torch.Tensor:
        return pos

    def forward(self, pos, device):
        p = self.position_ids(pos).float().to(device)
        freqs = (self.inv_freq[:, None] @ p[None, :]).transpose(0, 1)  # T, d/2
        emb = torch.cat([freqs, freqs], -1)
        return emb.cos(), emb.sin()


@register("position", "chunk_rope")
class ChunkRoPE(RotaryPositions):
    """Every token in chunk c = t // C gets RoPE position c*C (scale="keep") or c
    (scale="compress"). Relative positions then only encode order BETWEEN chunks;
    within a chunk, order is visible only through the causal mask.
    C=1 == stock RoPE."""

    def __init__(self, c, chunk: int = 4, scale: str = "keep", **kw):
        super().__init__(c)
        self.chunk, self.scale = int(chunk), scale

    def position_ids(self, pos):
        cidx = torch.div(pos, self.chunk, rounding_mode="floor")
        return cidx * self.chunk if self.scale == "keep" else cidx


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


# ============================================================ attention
class _RMS(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        dt = x.dtype; x = x.float()
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dt)


@register("attention", "qwen3")
class Qwen3Attention(nn.Module):
    """Stock Qwen3 GQA attention with per-head q/k RMSNorm."""

    def __init__(self, c, idx, **kw):
        super().__init__()
        self.H, self.KV, self.hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.H * self.hd, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.KV * self.hd, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, self.KV * self.hd, bias=False)
        self.o_proj = nn.Linear(self.H * self.hd, c.hidden_size, bias=False)
        self.q_norm = _RMS(self.hd, c.rms_norm_eps)
        self.k_norm = _RMS(self.hd, c.rms_norm_eps)
        self.scale = self.hd ** -0.5
        self.record = False      # store side buffers for probing
        self.buffer = {}

    def qkv(self, x, cos, sin, pos, cache):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.H, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.KV, self.hd)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.KV, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        kpos = pos
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
                kpos = torch.cat([cache["kpos"], pos])
            cache["k"], cache["v"], cache["kpos"] = k, v, kpos
        rep = self.H // self.KV
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        return q, k, v, kpos

    def attend(self, q, k, v, pos, kpos):
        T, S = q.shape[2], k.shape[2]
        if T == S:
            return F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, scale=self.scale)
        mask = kpos[None, :] <= pos[:, None]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

    def forward(self, x, cos, sin, pos, cache=None):
        B, T, _ = x.shape
        q, k, v, kpos = self.qkv(x, cos, sin, pos, cache)
        y = self.attend(q, k, v, pos, kpos)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


def phase_of(pos, period, offset=0):
    return (torch.div(pos + offset, period, rounding_mode="floor")) % 2


@register("attention", "phase")
class PhaseAttention(Qwen3Attention):
    """Drop-in Qwen3 attention reusing the pretrained q/k/v/o weights. The same heads
    time-multiplex between two content streams on a fixed period P.

    mode="kv"  : streams = key/value phases. Key s belongs to phase floor(s/P)%2.
                 A query at time t (phase floor(t/P)%2) treats in-phase keys as the
                 ACCESS stream and out-of-phase keys as the SUPPRESSED stream:
                     y = softmax(scores + log(gain)·[access] + log(supp)·[suppressed]) V
                 gain=supp=1 is exactly stock attention; supp=0 = only the access
                 stream writes; gain>1 = access gain boost.
    mode="head": streams = head groups (heads [0,H/2) with kv [0,KV/2) vs the rest).
                 Group floor(t/P)%2 has access at time t; its output is scaled by
                 `gain`, the other group's by `supp`. gain=supp=1 is stock.
    In both modes the suppressed stream's output (pre-o_proj) is kept in
    self.buffer["suppressed"] when self.record is True, so it can be decoded.
    out_gain multiplies the whole attention output written to the residual."""

    def __init__(self, c, idx, mode: str = "kv", period: int = 4, gain: float = 1.0,
                 supp: float = 1.0, offset: int = 0, out_gain: float = 1.0, **kw):
        super().__init__(c, idx)
        self.mode, self.period, self.gain, self.supp = mode, int(period), float(gain), float(supp)
        self.offset, self.out_gain = int(offset), float(out_gain)

    def forward(self, x, cos, sin, pos, cache=None):
        B, T, _ = x.shape
        q, k, v, kpos = self.qkv(x, cos, sin, pos, cache)
        qph = phase_of(pos, self.period, self.offset)
        if self.mode == "kv":
            kph = phase_of(kpos, self.period, self.offset)
            causal = kpos[None, :] <= pos[:, None]                    # T,S
            access = (kph[None, :] == qph[:, None]) & causal
            supp = (~access) & causal
            bias = torch.zeros(T, k.shape[2])
            if self.gain != 1.0:
                bias = bias + math.log(self.gain) * access
            if self.supp != 1.0:
                bias = torch.where(supp, torch.tensor(math.log(self.supp) if self.supp > 0 else float("-inf")), bias)
            bias = bias.masked_fill(~causal, float("-inf"))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
            if self.record:
                sm = torch.zeros(T, k.shape[2]).masked_fill(~supp, float("-inf"))
                ys = F.scaled_dot_product_attention(q, k, v, attn_mask=sm, scale=self.scale)
                ys = torch.nan_to_num(ys, nan=0.0) * supp.any(-1)[None, None, :, None]
                self.buffer = {"suppressed": ys.transpose(1, 2).reshape(B, T, -1).detach(),
                               "access_stream": y.transpose(1, 2).reshape(B, T, -1).detach(), "qphase": qph}
        elif self.mode == "head":
            y = self.attend(q, k, v, pos, kpos)                      # B,H,T,hd
            grp = (torch.arange(self.H) >= self.H // 2).long()        # H
            acc = (grp[:, None] == qph[None, :]).float()              # H,T
            scale = acc * self.gain + (1 - acc) * self.supp
            if self.record:
                ys = y * (1 - acc)[None, :, :, None]
                ya = y * acc[None, :, :, None]
                self.buffer = {"suppressed": ys.transpose(1, 2).reshape(B, T, -1).detach(),
                               "access_stream": ya.transpose(1, 2).reshape(B, T, -1).detach(), "qphase": qph}
            y = y * scale[None, :, :, None]
        else:
            raise ValueError(self.mode)
        return self.out_gain * self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


# ============================================================ extra layers
class ExtraLayer(nn.Module):
    def __init__(self, c, idx, **kw):
        super().__init__()
        self.idx = idx


@register("extra", "tracking")
class TrackingLayer(ExtraLayer):
    """Continuous slow 'tracking' state (after Tsao's 'drone' analogy): a causal,
    bias-corrected EMA over positions of the residual stream,
        m_t = sum_{j<t} a^(t-1-j) h_j / sum_{j<t} a^(t-1-j)      (excludes t; skips the sink at pos 0)
    and the content couples to it only when 'in harmony':
        h_t <- h_t + lam * c_t * (m_t - h_t)
        c_t = sigmoid(sharp * (cos(h_t, m_t) - tau))   gate="harmony"
        c_t = 1                                       gate="always"
        c_t = [phase(t) == 0]                         gate="phase" (couple on alternate micro-epochs)
    lam=0 is the identity. State is carried in the KV cache for generation."""

    def __init__(self, c, idx, alpha: float = 0.9, lam: float = 0.05, gate: str = "harmony",
                 tau: float = 0.5, sharp: float = 10.0, period: int = 4, skip_sink: bool = True, **kw):
        super().__init__(c, idx)
        self.alpha, self.lam, self.gate, self.tau, self.sharp = alpha, lam, gate, tau, sharp
        self.period, self.skip_sink = period, skip_sink
        self.record = False
        self.buffer = {}

    def forward(self, h, pos, cache=None):
        if self.lam == 0.0:
            return h
        B, T, D = h.shape
        key = f"track{id(self)}"
        st = cache.get(key) if cache is not None else None
        num0 = st[0] if st is not None else torch.zeros(B, D)
        den0 = st[1] if st is not None else torch.zeros(B)
        w_in = torch.ones(T)
        if self.skip_sink:
            w_in = w_in * (pos != 0).float()
        a = self.alpha
        t = torch.arange(T, dtype=torch.float32)
        # exclusive EMA: state BEFORE adding token t
        expo = (t[:, None] - 1 - t[None, :])
        W = torch.where(expo >= 0, a ** expo.clamp(min=0), torch.zeros(())) * w_in[None, :]   # T,T
        carry = a ** t                                                                          # weight on prior state
        num = torch.einsum("ts,bsd->btd", W, h) + carry[None, :, None] * num0[:, None, :]
        den = W.sum(-1)[None, :] + carry[None, :] * den0[:, None]
        valid = den > 1e-6
        m = num / den.clamp(min=1e-6)[..., None]
        if self.gate == "harmony":
            cgate = torch.sigmoid(self.sharp * (F.cosine_similarity(h, m, dim=-1) - self.tau))
        elif self.gate == "always":
            cgate = torch.ones(B, T)
        elif self.gate == "phase":
            cgate = (phase_of(pos, self.period) == 0).float()[None, :].expand(B, T)
        else:
            raise ValueError(self.gate)
        cgate = cgate * valid.float()
        if self.skip_sink:
            cgate = cgate * (pos != 0).float()[None, :]
        out = h + self.lam * cgate[..., None] * (m - h)
        if cache is not None:  # inclusive state after the last token
            num_T = a * num[:, -1] + w_in[-1] * h[:, -1]
            den_T = a * den[:, -1] + w_in[-1]
            cache[key] = (num_T, den_T)
        if self.record:
            self.buffer = {"state": m.detach(), "gate": cgate.detach()}
        return out
