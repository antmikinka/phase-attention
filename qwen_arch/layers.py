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


def _logit(x):
    x = min(max(float(x), 1e-3), 1 - 1e-3)
    return math.log(x / (1 - x))


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

    def __init__(self, c, chunk: int = 4, scale: str = "keep", trainable: bool = False, blend: float = 0.99, **kw):
        super().__init__(c)
        self.chunk, self.scale, self.trainable = int(chunk), scale, bool(trainable)
        if self.trainable:   # learnable blend between stock positions (0) and chunk positions (1)
            self.blend_logit = nn.Parameter(torch.tensor(_logit(blend)))

    def position_ids(self, pos):
        cidx = torch.div(pos, self.chunk, rounding_mode="floor")
        cp = cidx * self.chunk if self.scale == "keep" else cidx
        if self.trainable:
            b = torch.sigmoid(self.blend_logit)
            return (1 - b) * pos.float() + b * cp.float()
        return cp


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
        self.expose = False      # compute self.side (suppressed stream in residual space) for coupling layers
        self.side = None

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
        # ---- trainable set (only created when trainable=True; the float path stays bit-exact) ----
        self.trainable = bool(kw.get("trainable", False))
        self.train_supp = bool(kw.get("train_supp", False))
        self.learn_phase = bool(kw.get("learn_phase", False))
        if self.trainable:
            H = self.H
            self.log_gain = nn.Parameter(torch.full((H,), math.log(self.gain)))          # access gain per head
            self.log_out = nn.Parameter(torch.full((H,), math.log(self.out_gain)))       # write gain per head
            if self.train_supp:
                assert self.supp > 0, "train_supp needs supp > 0 (log-parameterised)"
                self.log_supp = nn.Parameter(torch.full((H,), math.log(self.supp)))
            if self.learn_phase:   # soft square-wave phase with learnable offset, period and sharpness
                self.phase_offset = nn.Parameter(torch.tensor(float(self.offset)))
                self.log_period = nn.Parameter(torch.tensor(math.log(self.period)))
                self.log_psharp = nn.Parameter(torch.tensor(math.log(float(kw.get("phase_sharp", 20.0)))))

    def forward(self, x, cos, sin, pos, cache=None):
        B, T, _ = x.shape
        q, k, v, kpos = self.qkv(x, cos, sin, pos, cache)
        qph = phase_of(pos, self.period, self.offset)
        if self.mode == "kv" and self.trainable:
            y = self._forward_kv_trainable(q, k, v, pos, kpos, B, T)
            return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))
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
            if self.record or self.expose:
                sm = torch.zeros(T, k.shape[2]).masked_fill(~supp, float("-inf"))
                ys = F.scaled_dot_product_attention(q, k, v, attn_mask=sm, scale=self.scale)
                ys = torch.nan_to_num(ys, nan=0.0) * supp.any(-1)[None, None, :, None]
                if self.expose:
                    self.side = self.o_proj(ys.transpose(1, 2).reshape(B, T, -1))
                    # restore signal: what the suppressed stream would add at its STOCK attention share.
                    # stock = w_acc*y_acc + w_sup*y_sup with w = partition masses under unbiased softmax.
                    sc = (q @ k.transpose(-1, -2)) * self.scale                       # B,H,T,S
                    lse_a = sc.masked_fill(~access, float("-inf")).logsumexp(-1)
                    lse_s = sc.masked_fill(~supp, float("-inf")).logsumexp(-1)
                    w_sup = torch.exp(lse_s - torch.logaddexp(lse_a, lse_s)).nan_to_num(0.0)[..., None]
                    am = torch.zeros(T, k.shape[2]).masked_fill(~access, float("-inf"))
                    ya = F.scaled_dot_product_attention(q, k, v, attn_mask=am, scale=self.scale)
                    y_stock = (1 - w_sup) * ya + w_sup * ys
                    self.side_restore = self.o_proj((y_stock - y).transpose(1, 2).reshape(B, T, -1))
            if self.record:
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


    def soft_phase(self, p):
        """~0 for phase 0, ~1 for phase 1; differentiable in offset/period/sharpness."""
        P = self.log_period.exp()
        return 0.5 * (1 - torch.tanh(self.log_psharp.exp() * torch.sin(math.pi * (p.float() + 0.5 + self.phase_offset) / P)))

    def _forward_kv_trainable(self, q, k, v, pos, kpos, B, T):
        S = k.shape[2]
        causal = kpos[None, :] <= pos[:, None]
        gain = self.log_gain.exp()[:, None, None]                                   # H,1,1
        if self.learn_phase:
            pq, pk = self.soft_phase(pos), self.soft_phase(kpos)
            A = pq[:, None] * pk[None, :] + (1 - pq[:, None]) * (1 - pk[None, :])   # soft "same phase" T,S
            access = (A > 0.5) & causal
            sup_w = self.log_supp.exp()[:, None, None] if self.train_supp else self.supp
            bias = torch.log(gain * A + sup_w * (1 - A) + 1e-9)                     # H,T,S
        else:
            qph, kph = phase_of(pos, self.period, self.offset), phase_of(kpos, self.period, self.offset)
            access = (kph[None, :] == qph[:, None]) & causal
            if self.train_supp:
                lsup = self.log_supp[:, None, None]
            else:
                lsup = torch.tensor(math.log(self.supp) if self.supp > 0 else float("-inf"))
            bias = torch.where(access, gain.log().expand(-1, T, S), lsup.expand(self.H, T, S) if lsup.dim() else lsup)
        supp = (~access) & causal
        bias = bias.masked_fill(~causal, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[None], scale=self.scale)
        if self.record or self.expose:
            sm = torch.zeros(T, S).masked_fill(~supp, float("-inf"))
            ys = torch.nan_to_num(F.scaled_dot_product_attention(q, k, v, attn_mask=sm, scale=self.scale), nan=0.0)
            ys = ys * supp.any(-1)[None, None, :, None]
            if self.expose:
                self.side = self.o_proj(ys.transpose(1, 2).reshape(B, T, -1))
                sc = (q @ k.transpose(-1, -2)) * self.scale
                lse_a = sc.masked_fill(~access, float("-inf")).logsumexp(-1)
                lse_s = sc.masked_fill(~supp, float("-inf")).logsumexp(-1)
                w_sup = torch.exp(lse_s - torch.logaddexp(lse_a, lse_s)).nan_to_num(0.0)[..., None]
                am = torch.zeros(T, S).masked_fill(~access, float("-inf"))
                ya = F.scaled_dot_product_attention(q, k, v, attn_mask=am, scale=self.scale)
                self.side_restore = self.o_proj(((1 - w_sup) * ya + w_sup * ys - y).transpose(1, 2).reshape(B, T, -1))
            if self.record:
                self.buffer = {"suppressed": ys.transpose(1, 2).reshape(B, T, -1).detach(),
                               "access_stream": y.transpose(1, 2).reshape(B, T, -1).detach(),
                               "qphase": phase_of(pos, self.period, self.offset)}
        return y * self.log_out.exp()[None, :, None, None]


# ============================================================ extra layers
class ExtraLayer(nn.Module):
    """Base for inserted layers. With trainable=True the scalar knobs listed in
    TRAINABLE become nn.Parameters (lam, mix, tau raw; alpha via sigmoid; sharp via
    exp) and are materialised into the usual attributes at the start of forward,
    so the forward code is shared with the frozen float path."""
    TRAINABLE = ()

    def __init__(self, c, idx, **kw):
        super().__init__()
        self.idx = idx
        self.trainable = bool(kw.get("trainable", False))
        self.last_gate = None

    def _make_trainable(self):
        if not self.trainable:
            return
        for name in self.TRAINABLE:
            v = getattr(self, name)
            if name == "alpha":
                setattr(self, "alpha_logit", nn.Parameter(torch.tensor(_logit(v))))
            elif name == "sharp":
                setattr(self, "log_sharp", nn.Parameter(torch.tensor(math.log(float(v)))))
            else:
                setattr(self, name + "_p", nn.Parameter(torch.tensor(float(v))))

    def _materialize(self):
        if not self.trainable:
            return
        for name in self.TRAINABLE:
            if name == "alpha":
                object.__setattr__(self, "alpha", torch.sigmoid(self.alpha_logit))
            elif name == "sharp":
                object.__setattr__(self, "sharp", self.log_sharp.exp())
            else:
                object.__setattr__(self, name, getattr(self, name + "_p") * 1.0)


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
        super().__init__(c, idx, **kw)
        self.alpha, self.lam, self.gate, self.tau, self.sharp = alpha, lam, gate, tau, sharp
        self.period, self.skip_sink = period, skip_sink
        self.record = False
        self.buffer = {}
        self._make_trainable()

    TRAINABLE = ("lam", "alpha", "tau", "sharp")

    def forward(self, h, pos, cache=None):
        self._materialize()
        if not self.trainable and self.lam == 0.0:
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
        self.last_gate = cgate
        if cache is not None:  # inclusive state after the last token
            num_T = a * num[:, -1] + w_in[-1] * h[:, -1]
            den_T = a * den[:, -1] + w_in[-1]
            cache[key] = (num_T, den_T)
        if self.record:
            self.buffer = {"state": m.detach(), "gate": cgate.detach(), "gate_mean": float(cgate.mean())}
        return out


@register("extra", "suppressed_tracking")
class SuppressedTrackingLayer(ExtraLayer):
    """Feed the hidden (suppressed) stream back through a tracking state.

    PhaseAttention (kv mode) in the SAME decoder layer exposes its suppressed-stream
    output projected to residual space, s_t = o_proj(softmax over out-of-phase keys . V).
    The slow tracking state absorbs it:
        u_j = h_j + mix * (|h_j| / |s_j|) * s_j               (norm-matched mix)
        m_t = sum_{j<=t} a^(t-j) u_j / sum_{j<=t} a^(t-j)      (causal EMA, sink at pos 0 skipped)
    and couples back into the residual only on selected steps:
        h_t <- h_t + lam * c_t * (m_t - h_t)
    gate:
        "harmony"   c_t = sigmoid(sharp * (min(cos(h_t, m_t), cos(s_t, m_t)) - tau))
        "boundary"  c_t = 1 on the first token of each phase micro-epoch (t % P == 0)
        "either"    max(harmony, boundary);  "both": harmony * boundary
        "always"    c_t = 1
        "bernoulli" c_t ~ Bernoulli(p), fixed per position (duty-cycle-matched control)
    source (controls):
        "suppressed" the real side buffer
        "random"     Gaussian noise with the same per-token norm as s_t
        "shuffled"   s_{pi(t)} with pi(t) drawn uniformly from [0, t] (causal, content-mismatched)
        "none"       mix forced to 0 (tracking only, no buffer)
    coupling:
        "pull"   (above) pull h toward the slow residual state that absorbed the buffer
        "restore" side = o_proj(y_stock - y_actual): the suppressed stream's content at its stock attention
                 share (needs the buffer's attention mass). Applied between attention and MLP
                 (stage="post_attn"); h_t <- h_t + lam * c_t * side_t, slow state m_t = EMA(side) only
                 drives the harmony gate cos(side_t, m_t) > tau. lam=1, c=1 recovers stock attention.
        "inject" the slow state tracks the suppressed stream only, m_t = EMA_{j<=t}(mix * s_j),
                 and is added back: h_t <- h_t + lam * c_t * m_t; harmony = cos(s_t, m_t) > tau
                 (current hidden content agrees with the tracked hidden content)
    lam == 0 is the exact identity (the attention does not even compute the buffer then)."""

    def __init__(self, c, idx, alpha: float = 0.7, lam: float = 0.1, mix: float = 1.0, gate: str = "harmony",
                 tau: float = 0.2, sharp: float = 10.0, period: int = 4, source: str = "suppressed",
                 skip_sink: bool = True, seed: int = 0, coupling: str = "pull", p: float = 0.5, **kw):
        super().__init__(c, idx, **kw)
        self.coupling, self.p = coupling, float(p)
        self.alpha, self.lam, self.mix, self.gate, self.tau, self.sharp = alpha, lam, mix, gate, tau, sharp
        self.period, self.source, self.skip_sink, self.seed = int(period), source, skip_sink, seed
        self.needs_side = lam != 0.0 and source != "none" and mix != 0.0
        # "restore" couples between attention and MLP by default (stage="post_attn")
        self.stage = kw.get("stage", "post_attn" if coupling == "restore" else "post_mlp")
        self._make_trainable()
        r = int(kw.get("adapter_rank", 0))
        self.adapter = None
        if r > 0:   # small low-rank adapter on the fed-back buffer; zero-init up-proj => starts as identity
            D = c.hidden_size
            self.adapter = nn.Sequential(nn.Linear(D, r, bias=False), nn.Linear(r, D, bias=False))
            nn.init.normal_(self.adapter[0].weight, std=D ** -0.5)
            nn.init.zeros_(self.adapter[1].weight)

        self.record = False
        self.buffer = {}

    TRAINABLE = ("lam", "mix", "alpha", "tau", "sharp")

    def _source(self, h, side, pos, cache, key):
        B, T, D = h.shape
        if side is None or not self.needs_side:
            return torch.zeros_like(h)
        if self.coupling == "restore":
            side = getattr(self, "_restore", None)
        if self.source == "suppressed":
            return side
        g = torch.Generator().manual_seed(self.seed * 100003 + self.idx)
        if self.source == "random":
            # per-position deterministic noise (same values with or without KV cache)
            noise = torch.stack([torch.randn(D, generator=torch.Generator().manual_seed(int(p) * 7919 + self.idx + self.seed * 104729))
                                 for p in pos])[None].expand(B, T, D)
            return noise / noise.norm(dim=-1, keepdim=True).clamp(min=1e-6) * side.norm(dim=-1, keepdim=True)
        if self.source == "shuffled":
            hist = torch.cat([cache[key + "hist"], side], 1) if cache is not None and key + "hist" in cache else side
            if cache is not None:
                cache[key + "hist"] = hist
            start = hist.shape[1] - T
            idx = torch.tensor([int(torch.randint(0, int(p) + 1, (1,), generator=torch.Generator().manual_seed(int(p) * 31 + self.idx)))
                                for p in pos]) - (int(pos[0]) - start)
            return hist[:, idx.clamp(min=0)]
        raise ValueError(self.source)

    def forward(self, h, pos, cache=None, side=None):
        self._materialize()
        if not self.trainable and self.lam == 0.0:
            return h
        B, T, D = h.shape
        key = f"strack{id(self)}"
        s = self._source(h, side, pos, cache, key)
        if self.adapter is not None:
            s = s + self.adapter(s)
        if self.coupling == "restore":
            u = self.mix * s
        elif self.coupling == "inject":
            u = self.mix * s                      # slow state of the suppressed stream itself (attention-output scale)
        elif self.needs_side:
            scale = h.norm(dim=-1, keepdim=True) / s.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scale = torch.where(s.norm(dim=-1, keepdim=True) > 0, scale, torch.zeros_like(scale))
            u = h + self.mix * scale * s
        else:
            u = h
        st = cache.get(key) if cache is not None else None
        num0 = st[0] if st is not None else torch.zeros(B, D)
        den0 = st[1] if st is not None else torch.zeros(B)
        w_in = (pos != 0).float() if self.skip_sink else torch.ones(T)
        a = self.alpha
        t = torch.arange(T, dtype=torch.float32)
        expo = t[:, None] - t[None, :]
        W = torch.where(expo >= 0, a ** expo.clamp(min=0), torch.zeros(())) * w_in[None, :]
        carry = a ** (t + 1)
        num = torch.einsum("ts,bsd->btd", W, u) + carry[None, :, None] * num0[:, None, :]
        den = W.sum(-1)[None, :] + carry[None, :] * den0[:, None]
        valid = (den > 1e-6).float()
        m = num / den.clamp(min=1e-6)[..., None]
        harm = None
        if self.gate in ("harmony", "either", "both") and self.coupling in ("inject", "restore"):
            # harmony = the current suppressed content agrees with the slow tracked suppressed state
            harm = torch.sigmoid(self.sharp * (F.cosine_similarity(s, m, dim=-1) - self.tau))
        elif self.gate in ("harmony", "either", "both"):
            ch = F.cosine_similarity(h, m, dim=-1)
            cs = F.cosine_similarity(s, m, dim=-1) if self.needs_side else ch
            harm = torch.sigmoid(self.sharp * (torch.minimum(ch, cs) - self.tau))
        bnd = ((pos % self.period) == 0).float()[None, :].expand(B, T)
        if self.gate == "bernoulli":   # duty-cycle-matched random gate (control); deterministic per position
            u01 = torch.tensor([float(torch.rand(1, generator=torch.Generator().manual_seed(int(p) * 7 + self.idx * 1009 + self.seed)))
                                for p in pos])
            harm = (u01 < self.p).float()[None, :].expand(B, T)
        cg = {"harmony": lambda: harm, "boundary": lambda: bnd, "either": lambda: torch.maximum(harm, bnd),
              "both": lambda: harm * bnd, "always": lambda: torch.ones(B, T), "bernoulli": lambda: harm}[self.gate]()
        cg = cg * valid
        if self.skip_sink:
            cg = cg * (pos != 0).float()[None, :]
        if cache is not None:
            cache[key] = (num[:, -1], den[:, -1])
        self.last_gate = cg
        if self.record:
            self.buffer = {"gate_mean": float(cg.mean()), "gate_open": float((cg > 0.5).float().mean()), "state": m.detach()}
        if self.coupling == "restore":
            return h + self.lam * cg[..., None] * u      # inject the CURRENT restore signal; slow state only gates
        if self.coupling == "inject":
            return h + self.lam * cg[..., None] * m
        return h + self.lam * cg[..., None] * (m - h)
