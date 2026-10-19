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


def build(rules=None, seed=0):
    torch.manual_seed(seed)
    m = PhaseQwen3ForCausalLM(C, parse_layers(C.num_hidden_layers, rules))
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return m.eval()


def same_weights(a, b):
    b.load_state_dict(a.state_dict(), strict=False)
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
