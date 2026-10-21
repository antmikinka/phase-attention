"""Mixers, positions and extra layers for phase_model. Everything here is trained from scratch."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- helpers
def ema_matrix(T: int, alpha: torch.Tensor) -> torch.Tensor:
    """Causal EMA as a TxT matrix: m_t = sum_{s<=t} (1-a) a^(t-s) x_s (differentiable in alpha)."""
    t = torch.arange(T)
    d = (t[:, None] - t[None, :]).float()
    w = (1 - alpha) * torch.exp(d.clamp(min=0) * torch.log(alpha))
    return w * (d >= 0)


def shift_right(x: torch.Tensor) -> torch.Tensor:      # (B,T,D) -> value at t-1, zeros at t=0
    return F.pad(x, (0, 0, 1, 0))[:, :-1]


class LowRank(nn.Module):
    def __init__(self, d: int, r: int):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)
        nn.init.normal_(self.up.weight, std=0.02 / math.sqrt(r))

    def forward(self, x):
        return self.up(self.down(x))


# ----------------------------------------------------------------------------- positions
class Rope(nn.Module):
    """Rotary positions. chunk_rope: p = (1-b) t + b * (t // C), b = sigmoid(blend_logit) learnable."""
    def __init__(self, dh: int, theta: float, spec: dict):
        super().__init__()
        self.kind = spec.get("type", "rope")
        self.register_buffer("inv", 1.0 / theta ** (torch.arange(0, dh, 2).float() / dh), persistent=False)
        if self.kind == "chunk_rope":
            self.C = int(spec.get("chunk", 4))
            b = float(spec.get("blend", 0.5))
            init = torch.tensor(math.log(b / (1 - b)))
            self.blend_logit = nn.Parameter(init) if spec.get("learn", True) else None
            if self.blend_logit is None:
                self.register_buffer("blend_fixed", init, persistent=False)

    def blend(self):
        return torch.sigmoid(self.blend_logit if self.blend_logit is not None else self.blend_fixed)

    def positions(self, T):
        t = torch.arange(T).float()
        if self.kind == "chunk_rope":
            b = self.blend()
            return (1 - b) * t + b * torch.div(t, self.C, rounding_mode="floor")
        return t

    def forward(self, q, k):                            # (B,H,T,dh)
        ang = self.positions(q.shape[2])[:, None] * self.inv[None]
        cos, sin = ang.cos(), ang.sin()

        def rot(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)
        return rot(q), rot(k)


# ----------------------------------------------------------------------------- phase clock
class PhaseClock(nn.Module):
    """s(t) in [0,1]: ~0 in phase 0, ~1 in phase 1. Soft & learnable:
       s = 1/2 (1 - tanh(k sin(pi (t + 1/2 + phi) / P))),  params phi, log P, log k.
       hard=True: s = floor(t/P) mod 2 exactly (no parameters)."""
    def __init__(self, period=4.0, sharp=4.0, offset=0.0, learn=True, hard=False):
        super().__init__()
        self.hard, self.P0 = hard, int(period)
        if not hard:
            mk = (lambda v: nn.Parameter(torch.tensor(float(v)))) if learn else (lambda v: torch.tensor(float(v)))
            self.offset, self.log_period, self.log_sharp = mk(offset), mk(math.log(period)), mk(math.log(sharp))

    def forward(self, T):
        t = torch.arange(T).float()
        if self.hard:
            return (torch.div(t, self.P0, rounding_mode="floor") % 2)
        P, k = self.log_period.exp(), self.log_sharp.exp()
        return 0.5 * (1 - torch.tanh(k * torch.sin(math.pi * (t + 0.5 + self.offset) / P)))

    def values(self):
        if self.hard:
            return {"period": float(self.P0), "sharp": float("inf"), "offset": 0.0}
        return {"period": float(self.log_period.exp()), "sharp": float(self.log_sharp.exp()), "offset": float(self.offset)}


# ----------------------------------------------------------------------------- mixers
class CausalAttention(nn.Module):
    def __init__(self, c, spec_mixer, spec_pos):
        super().__init__()
        self.H, self.dh = c.n_heads, c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.o = nn.Linear(c.d_model, c.d_model, bias=False)
        self.rope = Rope(self.dh, c.rope_theta, spec_pos)
        self.ysup = None; self.s = None

    def heads(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, k)
        return q, k, v

    def merge(self, y):
        B, H, T, dh = y.shape
        return y.transpose(1, 2).reshape(B, T, H * dh)

    def forward(self, x):
        q, k, v = self.heads(x)
        return self.o(self.merge(F.scaled_dot_product_attention(q, k, v, is_causal=True)))


class PhaseAttention(CausalAttention):
    """Alternating access/suppressed phases.
    kv mode : keys whose phase differs from the query's are down-weighted by supp_h (bias log(supp + (1-supp) A),
              A = soft same-phase indicator). supp=1 -> standard attention exactly; hard -> supp=0 & exact phases.
              Exposes the suppressed stream ysup (attention restricted to the other-phase keys) for feedback.
    head mode: head group g = h % 2 is active in phase g; the inactive group's output is scaled by supp_h.
    skip=True (kv, hard, no feedback): really computes only same-phase attention by splitting the sequence into
              its two phase sub-sequences -> half the attention FLOPs; bitwise-equivalent math to the mask."""
    def __init__(self, c, m, spec_pos):
        super().__init__(c, m, spec_pos)
        self.mode = m.get("mode", "kv")
        self.hard = bool(m.get("hard", False))
        self.clock = PhaseClock(m.get("period", 4), m.get("sharp", 4.0), m.get("offset", 0.0),
                                m.get("learn_phase", True), self.hard)
        supp = float(m.get("supp", 0.05))
        self.log_supp = None if self.hard else nn.Parameter(torch.full((self.H,), math.log(max(supp, 1e-6))))
        self.expose = False; self.skip = bool(m.get("skip", False))

    def supp(self):
        return torch.zeros(self.H) if self.hard else self.log_supp.exp().clamp(max=1.0)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.heads(x)
        s = self.clock(T); self.s = s
        if self.mode == "head":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            g = (torch.arange(self.H) % 2).float()[:, None]                       # H,1
            act = g * s[None] + (1 - g) * (1 - s[None])                          # H,T  1 = active
            w = act + (1 - act) * self.supp()[:, None]
            return self.o(self.merge(y * w[None, :, :, None]))
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        if self.hard and self.skip and not self.expose:
            return self.o(self.merge(self._split_attention(q, k, v, s)))
        A = s[:, None] * s[None, :] + (1 - s[:, None]) * (1 - s[None, :])        # T,T same-phase
        if self.hard:
            bias = torch.zeros(T, T).masked_fill(A < 0.5, float("-inf"))[None]
        else:
            sp = self.supp()[:, None, None]
            bias = torch.log((sp + (1 - sp) * A).clamp(min=1e-9))
        bias = bias.masked_fill(~causal, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None])
        if self.expose:
            sup = (A < 0.5) & causal
            m = torch.zeros(T, T).masked_fill(~sup, float("-inf"))
            ys = torch.nan_to_num(F.scaled_dot_product_attention(q, k, v, attn_mask=m), nan=0.0)
            ys = ys * sup.any(-1)[None, None, :, None]
            self.ysup = self.o(self.merge(ys))
        return self.o(self.merge(y))

    @staticmethod
    def _split_attention(q, k, v, s):
        y = torch.empty_like(q)
        for ph in (0, 1):
            idx = (s == ph).nonzero().squeeze(1)
            y[:, :, idx] = F.scaled_dot_product_attention(q[:, :, idx], k[:, :, idx], v[:, :, idx], is_causal=True)
        return y


MIXERS = {"attn": CausalAttention, "phase": PhaseAttention}


# ----------------------------------------------------------------------------- extra layers
class Gate(nn.Module):
    """harmony: sigmoid(sharp (cos(x_t, EMA(x)_{t-1}) - tau)); boundary: |s_t - s_{t-1}|; either = max; always = 1."""
    def __init__(self, kind="either", alpha=0.9, tau=0.5, sharp=8.0):
        super().__init__()
        self.kind = kind
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha / (1 - alpha))))
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.log_sharp = nn.Parameter(torch.tensor(math.log(sharp)))
        self.last = None

    def forward(self, x, s=None):
        B, T, D = x.shape
        if self.kind == "always":
            g = torch.ones(B, T)
        else:
            harm = boundary = None
            if self.kind in ("harmony", "either"):
                m = shift_right(ema_matrix(T, torch.sigmoid(self.alpha_logit)) @ x)
                c = F.cosine_similarity(x, m, dim=-1, eps=1e-6)
                harm = torch.sigmoid(self.log_sharp.exp() * (c - self.tau))
            if self.kind in ("boundary", "either"):
                boundary = (s - torch.cat([s[:1], s[:-1]])).abs()[None].expand(B, T)
            g = harm if boundary is None else boundary if harm is None else torch.maximum(harm, boundary)
        self.last = g.detach()
        return g


class Feedback(nn.Module):
    """Suppressed-stream buffer -> restore/feedback into the residual, gated:
       h <- h + lam * g_t * U D (ysup_t)   (placed after attention, before the MLP).
       source 'shuffled' replaces ysup_t by ysup_s for a random s <= t (causal control: right statistics, wrong content)."""
    needs_side = True

    def __init__(self, c, spec):
        super().__init__()
        self.proj = LowRank(c.d_model, int(spec.get("rank", 32)))
        self.lam = nn.Parameter(torch.tensor(float(spec.get("lam", 1.0))))
        self.gate = Gate(spec.get("gate", "either"), spec.get("alpha", 0.9), spec.get("tau", 0.5), spec.get("sharp", 8.0))
        self.source = spec.get("source", "suppressed")

    def forward(self, h, mixer):
        ys = mixer.ysup
        if self.source == "shuffled":           # causal shuffle: each t gets the buffer of a random earlier s <= t
            T = ys.shape[1]
            src = (torch.rand(T) * (torch.arange(T) + 1)).long().clamp(max=T - 1)
            ys = ys[:, src]
        g = self.gate(ys, mixer.s)
        return h + self.lam * g[..., None] * self.proj(ys)


class Tracking(nn.Module):
    """Slow EMA state of the residual, fed back when the current state 'harmonises' with it:
       m = EMA_a(h);  h <- h + lam * g_t * U D (m_t),  g = harmony(h_t, m_{t-1})."""
    needs_side = False

    def __init__(self, c, spec):
        super().__init__()
        self.proj = LowRank(c.d_model, int(spec.get("rank", 32)))
        self.lam = nn.Parameter(torch.tensor(float(spec.get("lam", 1.0))))
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(spec.get("alpha", 0.95) / (1 - spec.get("alpha", 0.95)))))
        self.gate = Gate("harmony", spec.get("alpha", 0.95), spec.get("tau", 0.5), spec.get("sharp", 8.0))

    def forward(self, h, mixer=None):
        m = ema_matrix(h.shape[1], torch.sigmoid(self.alpha_logit)) @ h
        g = self.gate(h)
        return h + self.lam * g[..., None] * self.proj(m)


EXTRAS = {"feedback": Feedback, "tracking": Tracking}
