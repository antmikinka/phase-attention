from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig, BlockSpec
from .layers import MIXERS, EXTRAS


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gu = nn.Linear(d, 2 * h, bias=False); self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        g, u = self.gu(x).chunk(2, -1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, c: ModelConfig, spec: BlockSpec):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.mixer = MIXERS[spec.mixer.get("type", "attn")](c, spec.mixer, spec.position)
        self.mlp = SwiGLU(c.d_model, c.mlp_hidden)
        self.pre = nn.ModuleList(); self.post = nn.ModuleList()
        for e in spec.extras:
            layer = EXTRAS[e["type"]](c, e)
            (self.pre if layer.needs_side else self.post).append(layer)
        if len(self.pre):
            assert spec.mixer.get("type") == "phase" and spec.mixer.get("mode", "kv") == "kv", "feedback needs phase kv"
            self.mixer.expose = True

    def forward(self, x):
        h = x + self.mixer(self.n1(x))
        for e in self.pre:
            h = e(h, self.mixer)
        h = h + self.mlp(self.n2(h))
        for e in self.post:
            h = e(h, self.mixer)
        return h


class PhaseLM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        specs = c.blocks or [BlockSpec() for _ in range(c.n_layers)]
        specs = [s if isinstance(s, BlockSpec) else BlockSpec(**s) for s in specs]
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c, s) for s in specs)
        self.norm = RMSNorm(c.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and not getattr(m, "_keep_init", False):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids):
        h = self.emb(ids)
        for b in self.blocks:
            h = b(h)
        return self.norm(h) @ self.emb.weight.T           # tied head

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def phase_report(self):
        """Learned phase/gate quantities per block (to check for collapse)."""
        out = []
        for i, b in enumerate(self.blocks):
            r = {}
            m = b.mixer
            if hasattr(m, "clock"):
                r.update(m.clock.values()); r["supp_mean"] = float(m.supp().mean()); r["mode"] = m.mode
            if hasattr(m.rope, "blend_logit"):
                r["chunk_blend"] = float(m.rope.blend())
            for e in list(b.pre) + list(b.post):
                r[type(e).__name__ + "_lam"] = float(e.lam)
                if e.gate.last is not None:
                    r[type(e).__name__ + "_gate_mean"] = float(e.gate.last.mean())
                    r[type(e).__name__ + "_gate_open"] = float((e.gate.last > 0.5).float().mean())
            if r:
                out.append({"block": i, **r})
        return out
