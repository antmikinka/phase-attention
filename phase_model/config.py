from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class BlockSpec:
    """One block. mixer: {"type": "attn"} | {"type": "phase", "mode": "kv"|"head", ...}
    position: {"type": "rope"} | {"type": "chunk_rope", "chunk": 4, "blend": 0.5, "learn": True}
    extras: list of {"type": "feedback", ...} | {"type": "tracking", ...}"""
    mixer: dict = field(default_factory=lambda: {"type": "attn"})
    position: dict = field(default_factory=lambda: {"type": "rope"})
    extras: list = field(default_factory=list)


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    mlp_hidden: int = 688
    rope_theta: float = 10000.0
    blocks: list = field(default_factory=list)      # list[BlockSpec]; empty -> all standard

    def to_dict(self):
        return asdict(self)


def _select(sel: str, n: int) -> list[int]:
    """'all' | '2-5' | '0,3,7' | 'every2' | 'every2@1' (every 2nd starting at 1)."""
    if sel == "all":
        return list(range(n))
    if sel.startswith("every"):
        step, _, start = sel[5:].partition("@")
        return list(range(int(start or 0), n, int(step)))
    if "-" in sel:
        a, b = sel.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in sel.split(",")]


def layer_specs(n_layers: int, rules: list[dict]) -> list[BlockSpec]:
    """rules: [{"layers": "all", "mixer": {...}} , {"layers": "every2", "extra": {...}}, ...]; later rules win."""
    specs = [BlockSpec() for _ in range(n_layers)]
    for r in rules:
        for i in _select(r.get("layers", "all"), n_layers):
            if "mixer" in r:
                specs[i].mixer = dict(r["mixer"])
            if "position" in r:
                specs[i].position = dict(r["position"])
            if "extra" in r:
                specs[i].extras = specs[i].extras + [dict(r["extra"])]
    return specs
