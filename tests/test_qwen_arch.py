"""Tests for the Qwen3 surgery model. Uses a randomly initialised 2-layer
mini-config (fast, no weights needed) plus an optional real-weights check."""
import os, sys
import pytest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwen_arch import PhaseQwen3ForCausalLM, Qwen3Config, parse_layers, register, ATTENTION_TYPES
from qwen_arch.layers import Qwen3Attention
from qwen_arch.modeling import _select

C = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16)


NEW_KEYS = ("log_gain", "log_out", "log_supp", "phase_offset", "log_period", "log_psharp", "lam_p", "mix_p",
            "alpha_logit", "tau_p", "log_sharp", "adapter", "blend_logit")


def is_new(name):
    return any(k in name for k in NEW_KEYS)


def build(rules=None, seed=0):
    torch.manual_seed(seed)
    m = PhaseQwen3ForCausalLM(C, parse_layers(C.num_hidden_layers, rules))
    for n, p in m.named_parameters():
        if not is_new(n):
            torch.nn.init.normal_(p, std=0.2)
    return m.eval()


def same_weights(a, b):
    b.load_state_dict({k: v for k, v in a.state_dict().items() if not is_new(k)}, strict=False)
    return b


IDS = torch.randint(0, 64, (2, 23), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("rules", [
    [{"layers": "all", "attention": {"type": "phase", "mode": "kv", "period": 4, "gain": 1.0, "supp": 1.0}}],
    [{"layers": "all", "attention": {"type": "phase", "mode": "head", "period": 3, "gain": 1.0, "supp": 1.0}}],
    [{"layers": "all", "position": {"type": "chunk_rope", "chunk": 1}}],
    [{"layers": "all", "extra": {"type": "tracking", "lam": 0.0}}],
])
def test_identity_settings_match_stock(rules):
    base = build()
    m = same_weights(base, build(rules))
    with torch.no_grad():
        assert torch.allclose(base(IDS), m(IDS), atol=1e-4)


@pytest.mark.parametrize("rules", [
    None,
    [{"layers": "1-2", "attention": {"type": "phase", "mode": "kv", "period": 3, "gain": 2.0, "supp": 0.0}}],
    [{"layers": "all", "attention": {"type": "phase", "mode": "head", "period": 2, "gain": 2.0, "supp": 0.0}}],
    [{"layers": "all", "position": {"type": "chunk_rope", "chunk": 4}}],
    [{"layers": "0,2", "extra": {"type": "tracking", "lam": 0.3, "gate": "harmony", "tau": 0.0}}],
    [{"layers": "1", "extra": {"type": "tracking", "lam": 0.3, "gate": "phase", "period": 2}}],
])
def test_kv_cache_matches_full_forward(rules):
    m = build(rules)
    with torch.no_grad():
        full = m(IDS)
        caches = m.new_caches()
        a = m(IDS[:, :10], caches, 0)
        steps = [a]
        for t in range(10, 23):
            steps.append(m(IDS[:, t:t + 1], caches, t))
        inc = torch.cat(steps, 1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_modifications_change_outputs_and_are_causal():
    base = build()
    for rules in ([{"layers": "all", "attention": {"type": "phase", "mode": "kv", "period": 4, "supp": 0.0}}],
                  [{"layers": "all", "position": {"type": "chunk_rope", "chunk": 4}}],
                  [{"layers": "all", "extra": {"type": "tracking", "lam": 0.5, "gate": "always"}}]):
        m = same_weights(base, build(rules))
        with torch.no_grad():
            y = m(IDS)
            assert not torch.allclose(base(IDS), y, atol=1e-4)
            x2 = IDS.clone(); x2[:, 15:] = (x2[:, 15:] + 1) % 64
            assert torch.allclose(y[:, :15], m(x2)[:, :15], atol=1e-5)


def test_suppressed_buffer_recorded():
    m = build([{"layers": "1", "attention": {"type": "phase", "mode": "kv", "period": 4, "supp": 0.0}}])
    m.layers[1].self_attn.record = True
    with torch.no_grad():
        m(IDS)
    buf = m.layers[1].self_attn.buffer
    assert buf["suppressed"].shape == (2, 23, 4 * 16)
    assert buf["suppressed"][:, :4].abs().sum() == 0      # no out-of-phase keys yet
    assert buf["suppressed"][:, 4:].abs().sum() > 0


def test_registry_and_selector():
    @register("attention", "zero_attn")
    class ZeroAttn(Qwen3Attention):
        def forward(self, x, cos, sin, pos, cache=None):
            return torch.zeros_like(x)
    m = build([{"layers": "every2", "attention": {"type": "zero_attn"}}])
    assert isinstance(m.layers[0].self_attn, ZeroAttn) and not isinstance(m.layers[1].self_attn, ZeroAttn)
    assert _select("8-11", 28) == {8, 9, 10, 11}
    assert _select("every4@8-19", 28) == {8, 12, 16}
    assert _select("0,3", 28) == {0, 3}


@pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")),
                    reason="Qwen3-0.6B not cached")
def test_real_qwen3_bitwise_equal_to_hf():
    from transformers import AutoModelForCausalLM
    ids = torch.tensor([[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]])
    m = PhaseQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").eval()
    with torch.no_grad():
        assert torch.equal(m(ids), hf(ids).logits)


ST = lambda **k: {"type": "suppressed_tracking", "alpha": 0.6, "lam": 0.3, "mix": 1.0, "period": 3, **k}
KVH = {"type": "phase", "mode": "kv", "period": 3, "gain": 1.0, "supp": 0.0}


def test_suppressed_tracking_identity_when_lam_zero():
    base = build()
    m = same_weights(base, build([{"layers": "all", "attention": {"type": "phase", "mode": "kv", "period": 3}},
                                  {"layers": "all", "extra": ST(lam=0.0)}]))
    assert not m.layers[0].self_attn.expose
    with torch.no_grad():
        assert torch.equal(base(IDS), m(IDS))


@pytest.mark.parametrize("src,gate", [("suppressed", "inject_harmony"), ("suppressed", "restore_harmony"), ("shuffled", "restore_boundary"), ("random", "inject_always"), ("suppressed", "harmony"), ("suppressed", "boundary"), ("suppressed", "either"),
                                      ("random", "always"), ("shuffled", "always"), ("none", "always")])
def test_suppressed_tracking_cache_and_causality(src, gate):
    coup = gate.split("_")[0] if "_" in gate else "pull"
    gate = gate.split("_")[-1]
    rules = [{"layers": "1-2", "attention": KVH}, {"layers": "1-2", "extra": ST(source=src, gate=gate, tau=-1.0, coupling=coup)}]
    m = build(rules)
    with torch.no_grad():
        full = m(IDS)
        caches = m.new_caches()
        steps = [m(IDS[:, :10], caches, 0)] + [m(IDS[:, t:t + 1], caches, t) for t in range(10, 23)]
        assert torch.allclose(full, torch.cat(steps, 1), atol=1e-4)
        x2 = IDS.clone(); x2[:, 15:] = (x2[:, 15:] + 1) % 64
        assert torch.allclose(full[:, :15], m(x2)[:, :15], atol=1e-5)


def test_suppressed_buffer_actually_feeds_back():
    a = build([{"layers": "1", "attention": KVH}, {"layers": "1", "extra": ST(source="suppressed", gate="always")}])
    b = same_weights(a, build([{"layers": "1", "attention": KVH}, {"layers": "1", "extra": ST(source="none", gate="always")}]))
    assert a.layers[1].self_attn.expose and not b.layers[1].self_attn.expose
    with torch.no_grad():
        ya, yb = a(IDS), b(IDS)
    assert torch.allclose(ya[:, :3], yb[:, :3], atol=1e-5)   # no suppressed keys yet in the first micro-epoch
    assert not torch.allclose(ya[:, 4:], yb[:, 4:], atol=1e-4)


@pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")),
                    reason="Qwen3-0.6B not cached")
def test_real_qwen3_bitwise_with_zero_strength_feedback():
    from transformers import AutoModelForCausalLM
    ids = torch.tensor([[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]])
    m = PhaseQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", rules=[
        {"layers": "8-19", "attention": {"type": "phase", "mode": "kv", "period": 4, "gain": 1.0, "supp": 1.0}},
        {"layers": "8-19", "extra": {"type": "suppressed_tracking", "lam": 0.0, "mix": 1.0}},
        {"layers": "every4@8-19", "extra": {"type": "tracking", "lam": 0.0}}])
    hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").eval()
    with torch.no_grad():
        assert torch.equal(m(ids), hf(ids).logits)


def test_restore_with_full_gate_recovers_stock_attention():
    """Hard kv suppression + restore coupling with lam=1 on every step == stock model."""
    base = build()
    m = same_weights(base, build([{"layers": "all", "attention": KVH},
                                  {"layers": "all", "extra": ST(coupling="restore", lam=1.0, gate="always", skip_sink=False)}]))
    hard = same_weights(base, build([{"layers": "all", "attention": KVH}]))
    with torch.no_grad():
        y, yb, yh = m(IDS), base(IDS), hard(IDS)
    assert torch.allclose(y, yb, atol=1e-4), (y - yb).abs().max()
    assert not torch.allclose(yh, yb, atol=1e-3)


TRULES = [
    {"layers": "1-2", "attention": {"type": "phase", "mode": "kv", "period": 3, "gain": 1.0, "supp": 0.0, "trainable": True}},
    {"layers": "1-2", "extra": {"type": "suppressed_tracking", "coupling": "restore", "gate": "either", "tau": 0.5,
                               "alpha": 0.7, "lam": 1.0, "period": 3, "trainable": True, "adapter_rank": 4}},
    {"layers": "0", "position": {"type": "chunk_rope", "chunk": 4, "trainable": True}},
    {"layers": "0", "extra": {"type": "tracking", "lam": 0.1, "gate": "harmony", "trainable": True}},
]


def test_trainable_layers_get_gradients_and_match_frozen_at_init():
    m = build(TRULES)
    names = [n for n, p in m.named_parameters() if any(k in n for k in ("log_gain", "log_out", "lam_p", "mix_p", "alpha_logit",
                                                                        "tau_p", "log_sharp", "adapter", "blend_logit"))]
    assert len(names) >= 10
    for p in m.parameters():
        p.requires_grad_(False)
    for n, p in m.named_parameters():
        if n in names:
            p.requires_grad_(True)
    y = m(IDS)
    y.logsumexp(-1).mean().backward()
    got = {n for n, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any("lam_p" in n for n in got) and any("log_gain" in n or "log_out" in n for n in got) and any("blend_logit" in n for n in got)
    # same model, frozen float path (blend=1 exactly vs 0.99 differs -> compare without chunk/tracking extras)
    frozen_rules = [dict(r) for r in TRULES[:2]]
    frozen_rules[0] = {"layers": "1-2", "attention": {k: v for k, v in TRULES[0]["attention"].items() if k != "trainable"}}
    frozen_rules[1] = {"layers": "1-2", "extra": {k: v for k, v in TRULES[1]["extra"].items() if k not in ("trainable", "adapter_rank")}}
    t = same_weights(m, build(TRULES[:2]))
    f = same_weights(m, build(frozen_rules))
    with torch.no_grad():
        assert torch.allclose(t(IDS), f(IDS), atol=1e-4)


def test_trainable_strength_zero_is_stock():
    base = build()
    m = same_weights(base, build([
        {"layers": "all", "attention": {"type": "phase", "mode": "kv", "period": 3, "gain": 1.0, "supp": 1.0, "trainable": True, "train_supp": True}},
        {"layers": "all", "extra": {"type": "tracking", "lam": 0.0, "trainable": True}},
        {"layers": "all", "extra": {"type": "suppressed_tracking", "coupling": "restore", "lam": 0.0, "trainable": True}}]))
    with torch.no_grad():
        assert torch.allclose(base(IDS), m(IDS), atol=1e-4)


def test_trainable_cache_matches_full():
    m = build(TRULES)
    with torch.no_grad():
        full = m(IDS)
        caches = m.new_caches()
        steps = [m(IDS[:, :10], caches, 0)] + [m(IDS[:, t:t + 1], caches, t) for t in range(10, 23)]
    assert torch.allclose(full, torch.cat(steps, 1), atol=1e-4)


def test_learned_soft_phase_close_to_hard():
    m = build([{"layers": "1", "attention": {"type": "phase", "mode": "kv", "period": 4, "supp": 0.0, "trainable": True,
                                             "learn_phase": True, "phase_sharp": 50.0}}])
    ph = m.layers[1].self_attn.soft_phase(torch.arange(16))
    assert torch.allclose(ph, ((torch.arange(16) // 4) % 2).float(), atol=1e-3)
