"""Phase attention, Tracking and suppressed-stream feedback grafted onto HF `Qwen3_5ForCausalLM` (Qwen3.8 structure).

Only the 1-in-4 *full-attention* layers get phase attention: the 3-in-4 Gated DeltaNet layers have no score matrix
to mask.  For those an experimental, default-OFF "phase write gate" scales the DeltaNet write strength beta per
value head by phase (PhaseBetaProj).

Formulas (t = absolute token position, s(t) in [0,1] from the reused phase_model.layers.PhaseClock):
  soft clock : s = 1/2 (1 - tanh(k sin(pi (t + 1/2 + phi) / P)))        hard clock: s = floor(t/P) mod 2
  A[q,k]     = s_q s_k + (1-s_q)(1-s_k)                                  (same-phase indicator)
  kv mode    : score += log(1 - (1-sp_h)(1-A))   == log(sp_h + (1-sp_h) A), written so sp_h=1 gives exactly 0
               hard   : score += -inf where A < 1/2 (sp_h = 0)
  head mode  : y_h,t *= 1 - (1-act_h,t)(1-sp_h),  act = g_h s_t + (1-g_h)(1-s_t),  g_h = h mod 2
  feedback   : h <- h + lam g_t U D(ysup_t)  (after attention residual, before the MLP); ysup = attention over the
               *other-phase* keys only, through the same output gate + o_proj; g = max(harmony, boundary)
  tracking   : h <- h + lam g_t U D(EMA_a(h)_t), g = harmony(h_t, EMA_{t-1})
  write gate : beta_h,t *= 1 - (1-act_h,t)(1-sw_h)   (DeltaNet layers, value heads, OFF by default)
Strength 0 (sp = 1, lam = 0, sw = 1) is bitwise identical to the stock eager model (tested).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase_model.layers import PhaseClock, Feedback, Tracking
from transformers.models.qwen3_5 import modeling_qwen3_5 as hfq


@dataclass
class PhaseSpec:
    mode: str | None = "kv"          # "kv" | "head" | None (no phase attention)
    hard: bool = False
    period: float = 4.0
    sharp: float = 4.0
    offset: float = 0.0
    learn_phase: bool = True
    supp: float = 1.0                # initial per-head suppression (1.0 = strength 0 / identity)
    layers: list[int] | None = None  # full-attention layer ids to wrap (None = all of them)
    feedback: dict | None = None     # e.g. {"rank": 8, "lam": 0.0, "gate": "either"}
    tracking: dict | None = None     # e.g. {"rank": 8, "lam": 0.0, "alpha": 0.95}
    delta_write: dict | None = None  # experimental, e.g. {"enabled": True, "supp": 1.0}

    def to_dict(self):
        return asdict(self)


class _Ctx:
    """Shared per-forward state (absolute token positions) for wrappers that do not receive position ids."""
    pos: torch.Tensor | None = None


def clock_at(clock: PhaseClock, pos: torch.Tensor) -> torch.Tensor:
    """PhaseClock evaluated at arbitrary absolute positions (PhaseClock.forward only does arange(T))."""
    t = pos.float()
    if clock.hard:
        return torch.div(t, clock.P0, rounding_mode="floor") % 2
    P, k = clock.log_period.exp(), clock.log_sharp.exp()
    return 0.5 * (1 - torch.tanh(k * torch.sin(math.pi * (t + 0.5 + clock.offset) / P)))


def _make_clock(spec: PhaseSpec) -> PhaseClock:
    return PhaseClock(spec.period, spec.sharp, spec.offset, spec.learn_phase, spec.hard)


def _positions(kwargs, T, ctx: _Ctx):
    pos = kwargs.get("position_ids")
    if pos is None:
        pos = ctx.pos
    if pos is None:
        return torch.arange(T)
    pos = pos.reshape(-1, pos.shape[-1])[0] if pos.dim() > 1 else pos
    return pos


class PhaseQwenAttention(nn.Module):
    """Wraps hfq.Qwen3_5Attention; reuses its projections/norms/RoPE/eager kernel and only changes the additive mask
    (kv mode) or the per-head output scale (head mode)."""

    def __init__(self, orig: hfq.Qwen3_5Attention, spec: PhaseSpec, ctx: _Ctx):
        super().__init__()
        self.orig, self.spec, self.ctx = orig, spec, ctx
        self.mode, self.hard = spec.mode, bool(spec.hard)
        self.H = orig.config.num_attention_heads
        self.clock = _make_clock(spec)
        self.log_supp = None if self.hard else nn.Parameter(torch.full((self.H,), math.log(max(spec.supp, 1e-6))))
        self.active = True       # False -> call the untouched stock module
        self.expose = False      # compute the suppressed stream for feedback
        self.ysup = None
        self.s = None

    def supp(self):
        return torch.zeros(self.H) if self.hard else self.log_supp.exp().clamp(max=1.0)

    def phase_bias(self, s_q, s_k):
        """(H, T, S) additive kv-mode bias."""
        A = s_q[:, None] * s_k[None, :] + (1 - s_q[:, None]) * (1 - s_k[None, :])
        if self.hard:
            return torch.zeros_like(A).masked_fill(A < 0.5, float("-inf"))[None].expand(self.H, -1, -1)
        sp = self.supp()[:, None, None]
        return torch.log((1 - (1 - sp) * (1 - A)[None]).clamp(min=1e-9))

    def head_weight(self, s_q):
        """(H, T) multiplicative head-mode weight."""
        g = (torch.arange(self.H) % 2).float()[:, None]
        act = g * s_q[None] + (1 - g) * (1 - s_q[None])
        return 1 - (1 - act) * (1 - self.supp()[:, None])

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        if not self.active or self.mode is None:
            return self.orig(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        o = self.orig
        input_shape = hidden_states.shape[:-1]
        B, T = input_shape
        hidden_shape = (*input_shape, -1, o.head_dim)
        q, gate = torch.chunk(o.q_proj(hidden_states).view(*input_shape, -1, o.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        q = o.q_norm(q.view(hidden_shape)).transpose(1, 2)
        k = o.k_norm(o.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = o.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = hfq.apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, o.layer_idx)
        S = k.shape[2]
        pos_q = _positions(kwargs, T, self.ctx).to(hidden_states.device)
        pos_k = torch.arange(S, device=pos_q.device) + (pos_q[-1] + 1 - S)   # cache holds a contiguous prefix
        s_q, s_k = clock_at(self.clock, pos_q), clock_at(self.clock, pos_k)
        self.s = s_q
        if attention_mask is None:
            attention_mask = torch.zeros(T, S).masked_fill(
                pos_k[None, :] > pos_q[:, None], torch.finfo(q.dtype).min)[None, None]
        mask = attention_mask
        if self.mode == "kv":
            mask = attention_mask + self.phase_bias(s_q, s_k)[None].to(q.dtype)
        y, _ = hfq.eager_attention_forward(o, q, k, v, mask, scaling=o.scaling, dropout=0.0)   # (B,T,H,dh)
        if self.mode == "head":
            y = y * self.head_weight(s_q).T[None, :, :, None].to(y.dtype)
        if self.expose:
            causal = attention_mask > torch.finfo(q.dtype).min / 2
            A = s_q[:, None] * s_k[None, :] + (1 - s_q[:, None]) * (1 - s_k[None, :])
            sup = causal & (A < 0.5)
            has = sup.any(-1, keepdim=True)                          # (B,1,T,1) query has an other-phase key
            m2 = torch.zeros(sup.shape, dtype=q.dtype).masked_fill(~(sup | ~has), float("-inf"))
            ys, _ = hfq.eager_attention_forward(o, q, k, v, m2, scaling=o.scaling, dropout=0.0)
            ys = ys * has[:, 0, :, :, None].to(ys.dtype)             # such queries -> 0 (finite fwd and grads)
            ys = ys.reshape(*input_shape, -1) * torch.sigmoid(gate)
            self.ysup = o.o_proj(ys)
        y = y.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
        return o.o_proj(y), None


class PhaseDecoderLayer(nn.Module):
    """Wraps hfq.Qwen3_5DecoderLayer; inserts feedback / tracking after the attention residual, before the MLP."""

    def __init__(self, orig: hfq.Qwen3_5DecoderLayer, spec: PhaseSpec, ctx: _Ctx):
        super().__init__()
        self.orig, self.ctx = orig, ctx
        d = orig.hidden_size
        c = type("C", (), {"d_model": d})()
        self.feedback = Feedback(c, spec.feedback) if spec.feedback else None
        self.tracking = Tracking(c, spec.tracking) if spec.tracking else None
        if self.feedback is not None:
            orig.self_attn.expose = True
        self.active = True

    def forward(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                past_key_values=None, **kwargs):
        o = self.orig
        if not self.active or (self.feedback is None and self.tracking is None):
            return o(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                     position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        residual = hidden_states
        hidden_states = o.input_layernorm(hidden_states)
        hidden_states, _ = o.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=past_key_values,
                                       position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states
        if self.feedback is not None:
            hidden_states = self.feedback(hidden_states, o.self_attn)
        if self.tracking is not None:
            hidden_states = self.tracking(hidden_states)
        residual = hidden_states
        hidden_states = o.post_attention_layernorm(hidden_states)
        hidden_states = o.mlp(hidden_states)
        return residual + hidden_states

    def gates(self):
        out = []
        for e in (self.feedback, self.tracking):
            if e is not None and e.gate.last is not None:
                out.append(e.gate.last)
        return out


class PhaseBetaProj(nn.Module):
    """EXPERIMENTAL, default OFF. Wraps GatedDeltaNet.in_proj_b (beta logits) so that
    beta' = sigmoid(b) * w,  w_h,t = 1 - (1-act_h,t)(1-sw_h), act = g_h s_t + (1-g_h)(1-s_t), g_h = h mod 2
    (value heads in HF's grouped order). Returned as a logit so the stock DeltaNet code applies sigmoid."""

    def __init__(self, orig: nn.Linear, n_v_heads: int, spec: PhaseSpec, ctx: _Ctx):
        super().__init__()
        dw = spec.delta_write or {}
        self.orig, self.ctx, self.Hv = orig, ctx, n_v_heads
        self.enabled = bool(dw.get("enabled", False))
        self.clock = _make_clock(spec)
        self.log_wsupp = nn.Parameter(torch.full((n_v_heads,), math.log(max(float(dw.get("supp", 1.0)), 1e-6))))
        self.register_buffer("grp", (torch.arange(n_v_heads) % 2).float(), persistent=False)

    def wsupp(self):
        return self.log_wsupp.exp().clamp(max=1.0)

    def weight(self, T):
        pos = self.ctx.pos if self.ctx.pos is not None else torch.arange(T)
        s = clock_at(self.clock, pos)                                     # (T,)
        act = self.grp[None] * s[:, None] + (1 - self.grp[None]) * (1 - s[:, None])   # (T,Hv)
        return 1 - (1 - act) * (1 - self.wsupp()[None])

    def forward(self, x):
        b = self.orig(x)
        if not self.enabled:
            return b
        beta = (torch.sigmoid(b) * self.weight(x.shape[1])[None].to(b.dtype)).clamp(1e-7, 1 - 1e-7)
        return torch.log(beta) - torch.log1p(-beta)


class PhaseQwen38(nn.Module):
    """Holds a HF Qwen3_5ForCausalLM and wraps its layers in place. `stock()` toggles every wrapper off."""

    def __init__(self, hf_model, spec: PhaseSpec):
        super().__init__()
        self.hf, self.spec, self.ctx = hf_model, spec, _Ctx()
        cfg = hf_model.config
        full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
        self.phase_layers = full if spec.layers is None else [i for i in spec.layers if i in full]
        layers = hf_model.model.layers
        self.attn, self.dec, self.beta = {}, {}, {}
        for i in self.phase_layers:
            layers[i].self_attn = PhaseQwenAttention(layers[i].self_attn, spec, self.ctx)
            self.attn[i] = layers[i].self_attn
            if spec.feedback or spec.tracking:
                layers[i] = PhaseDecoderLayer(layers[i], spec, self.ctx)
                self.dec[i] = layers[i]
        if spec.delta_write:
            for i, t in enumerate(cfg.layer_types):
                if t == "linear_attention":
                    la = layers[i].linear_attn
                    la.in_proj_b = PhaseBetaProj(la.in_proj_b, la.num_v_heads, spec, self.ctx)
                    self.beta[i] = la.in_proj_b

    def set_active(self, on: bool):
        for m in list(self.attn.values()) + list(self.dec.values()):
            m.active = on
        for m in self.beta.values():
            m.enabled = on and bool((self.spec.delta_write or {}).get("enabled", False))

    def forward(self, input_ids, position_offset: int = 0, **kw):
        T = input_ids.shape[1]
        self.ctx.pos = torch.arange(T) + position_offset
        pos = self.ctx.pos[None].expand(input_ids.shape[0], -1)
        try:
            return self.hf(input_ids=input_ids, position_ids=pos, use_cache=False, **kw).logits
        finally:
            self.ctx.pos = None

    def new_parameters(self):
        base = {id(p) for p in self.hf_base_parameters()}
        return [p for p in self.parameters() if id(p) not in base]

    def hf_base_parameters(self):
        out = []
        for n, p in self.hf.named_parameters():
            if not any(k in n for k in (".clock.", "log_supp", "log_wsupp", ".feedback.", ".tracking.")):
                out.append(p)
        return out

    def gate_values(self):
        g = []
        for d in self.dec.values():
            g += d.gates()
        return g

    # -------- export: phase tensors with HF-style names (consumed by the llama.cpp converter) --------
    def phase_state(self):
        st, meta = {}, {"layers": {}}
        for i, a in self.attn.items():
            cl = a.clock.values()
            meta["layers"][i] = {"mode": a.mode, "hard": a.hard, **cl}
            st[f"model.layers.{i}.phase.supp"] = a.supp().detach().float().clone()
        for i, d in self.dec.items():
            for nm, e in (("feedback", d.feedback), ("tracking", d.tracking)):
                if e is not None:
                    st[f"model.layers.{i}.phase.{nm}.down"] = e.proj.down.weight.detach().clone()
                    st[f"model.layers.{i}.phase.{nm}.up"] = e.proj.up.weight.detach().clone()
                    st[f"model.layers.{i}.phase.{nm}.lam"] = e.lam.detach().reshape(1).clone()
        for i, b in self.beta.items():
            meta["layers"].setdefault(i, {}).update({"delta_write": b.enabled, **b.clock.values()})
            st[f"model.layers.{i}.phase.write_supp"] = b.wsupp().detach().float().clone()
            st[f"model.layers.{i}.phase.write_grp"] = b.grp.detach().clone()
        meta["spec"] = self.spec.to_dict()
        return st, meta
