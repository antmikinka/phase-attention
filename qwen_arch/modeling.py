"""Qwen3 decoder written from scratch (fp32, CPU-friendly) with a per-layer
spec that chooses the attention type, the position type, and extra layers
inserted after the decoder layer. Loads the pretrained HF safetensors."""
from __future__ import annotations

import glob, json, os, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import ATTENTION_TYPES, POSITION_TYPES, EXTRA_LAYER_TYPES


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class LayerSpec:
    attention: Dict[str, Any] = field(default_factory=lambda: {"type": "qwen3"})
    position: Dict[str, Any] = field(default_factory=lambda: {"type": "rope"})
    extra: List[Dict[str, Any]] = field(default_factory=list)


def _select(sel, L):
    """'all' | 'none' | '8-19' | '0,4,8' | 'every4' | 'every4@8-19' -> set of layer ids"""
    if sel in (None, "none", ""):
        return set()
    if sel == "all":
        return set(range(L))
    out = set()
    m = re.fullmatch(r"every(\d+)(?:@(\d+)-(\d+))?", sel)
    if m:
        k = int(m.group(1)); a = int(m.group(2) or 0); b = int(m.group(3) or L - 1)
        return set(range(a, b + 1, k))
    for part in str(sel).split(","):
        if "-" in part:
            a, b = map(int, part.split("-")); out |= set(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def parse_layers(L: int, rules: Optional[List[Dict[str, Any]]] = None) -> List[LayerSpec]:
    """rules: list of {"layers": sel, "attention": {...}} / {"layers": sel, "position": {...}} /
    {"layers": sel, "extra": {...}}; later rules override earlier ones (extra layers append)."""
    specs = [LayerSpec() for _ in range(L)]
    for r in rules or []:
        for i in sorted(_select(r.get("layers", "all"), L)):
            if "attention" in r:
                specs[i].attention = dict(r["attention"])
            if "position" in r:
                specs[i].position = dict(r["position"])
            if "extra" in r:
                specs[i].extra.append(dict(r["extra"]))
    return specs


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


class MLP(nn.Module):
    def __init__(self, c: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, c: Qwen3Config, idx: int, spec: LayerSpec):
        super().__init__()
        self.idx, self.spec = idx, spec
        a = dict(spec.attention); p = dict(spec.position)
        self.self_attn = ATTENTION_TYPES[a.pop("type")](c, idx, **a)
        self.positions = POSITION_TYPES[p.pop("type")](c, **p)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c)
        self.extras = nn.ModuleList()
        for e in spec.extra:
            e = dict(e)
            self.extras.append(EXTRA_LAYER_TYPES[e.pop("type")](c, idx, **e))
        # an extra layer that consumes the attention's suppressed side buffer turns exposure on
        self.self_attn.expose = any(getattr(e, "needs_side", False) for e in self.extras)

    def forward(self, x, pos, cache):
        cos, sin = self.positions(pos, x.device)
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, pos, cache)
        side = getattr(self.self_attn, "side", None) if self.self_attn.expose else None
        for e in self.extras:          # extras that couple between attention and MLP
            if getattr(e, "stage", "post_mlp") == "post_attn":
                e._restore = getattr(self.self_attn, "side_restore", None)
                h = e(h, pos, cache, side=side)
        h = h + self.mlp(self.post_attention_layernorm(h))
        for e in self.extras:
            if getattr(e, "stage", "post_mlp") == "post_attn":
                continue
            h = e(h, pos, cache, side=side) if getattr(e, "needs_side", False) else e(h, pos, cache)
        return h


class LayerCache(dict):
    """Per-layer KV cache / state (keys: 'k', 'v', and extras' own keys)."""


class PhaseQwen3ForCausalLM(nn.Module):
    def __init__(self, c: Qwen3Config, specs: Optional[List[LayerSpec]] = None):
        super().__init__()
        self.config = c
        specs = specs or [LayerSpec() for _ in range(c.num_hidden_layers)]
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(c, i, s) for i, s in enumerate(specs))
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, path_or_id: str = "Qwen/Qwen3-0.6B", rules=None, dtype=torch.float32):
        path = _resolve(path_or_id)
        c = Qwen3Config.from_json(os.path.join(path, "config.json"))
        model = cls(c, parse_layers(c.num_hidden_layers, rules))
        from safetensors.torch import load_file
        sd = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            sd.update(load_file(f))
        own = {}
        for k, v in sd.items():
            k2 = k.removeprefix("model.")
            own[k2] = v.to(dtype)
        missing, unexpected = model.load_state_dict(own, strict=False)
        missing = [m for m in missing if not m.startswith("lm_head") and ".extras." not in m]
        assert not missing, f"missing weights: {missing[:5]}"
        assert not [u for u in unexpected if not u.startswith("lm_head")], unexpected[:5]
        model.rules = rules or []
        return model.to(dtype).eval()

    # ---------------------------------------------------------------- forward
    def forward(self, ids: torch.Tensor, caches: Optional[List[LayerCache]] = None, start: int = 0,
                return_hidden: bool = False):
        T = ids.shape[1]
        pos = torch.arange(start, start + T, device=ids.device)
        x = self.embed_tokens(ids)
        hidden = []
        for i, layer in enumerate(self.layers):
            x = layer(x, pos, caches[i] if caches is not None else None)
            if return_hidden:
                hidden.append(x)
        logits = self.lm_head(self.norm(x))
        return (logits, hidden) if return_hidden else logits

    def new_caches(self):
        return [LayerCache() for _ in self.layers]

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int, eos: Optional[int] = None):
        caches = self.new_caches()
        logits = self(ids, caches, 0)
        out = []
        n = ids.shape[1]
        for _ in range(max_new_tokens):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt)
            if eos is not None and int(nxt) == eos:
                break
            logits = self(nxt, caches, n)
            n += 1
        return torch.cat(out, 1)


def _resolve(p):
    if os.path.isdir(p):
        return p
    root = os.path.expanduser(f"~/.cache/huggingface/hub/models--{p.replace('/', '--')}/snapshots")
    snaps = sorted(glob.glob(os.path.join(root, "*")))
    assert snaps, f"model {p} not in HF cache"
    return snaps[-1]
