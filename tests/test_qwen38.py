"""qwen38/: phase attention on a tiny random-init copy of the Qwen3.8-27B structure (HF qwen3_5 classes).
Run with the transformers venv:  /home/box/venvs/phase-hf/bin/python -m pytest tests/test_qwen38.py -q"""
import math
import pytest
import torch

transformers = pytest.importorskip("transformers")
from qwen38 import make_tiny_model, PhaseSpec, PhaseQwen38
from qwen38.phase import clock_at

T = 24


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(123)
    return torch.randint(0, 248320, (2, T))


def stock_logits(ids, seed=0):
    m = make_tiny_model(seed)
    with torch.no_grad():
        return m(input_ids=ids, use_cache=False).logits


def test_tiny_structure():
    c = make_tiny_model(0).config
    assert c.layer_types == ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]
    assert c.num_attention_heads // c.num_key_value_heads == 24 // 4          # GQA ratio of the real 27B
    assert c.linear_num_value_heads // c.linear_num_key_heads == 48 // 16    # DeltaNet V:QK ratio
    assert c.attn_output_gate and c.rope_parameters["partial_rotary_factor"] == 0.25
    assert sum(c.rope_parameters["mrope_section"]) == int(c.head_dim * 0.25) // 2


@pytest.mark.parametrize("spec", [
    PhaseSpec(mode="kv", supp=1.0),
    PhaseSpec(mode="head", supp=1.0),
    PhaseSpec(mode="kv", supp=1.0, feedback={"rank": 8, "lam": 0.0, "gate": "either"},
              tracking={"rank": 8, "lam": 0.0}),
    PhaseSpec(mode="kv", supp=1.0, delta_write={"enabled": False, "supp": 0.2}),
])
def test_strength0_bitwise(ids, spec):
    ref = stock_logits(ids)
    pm = PhaseQwen38(make_tiny_model(0), spec)
    with torch.no_grad():
        out = pm(ids)
    assert torch.equal(out, ref)


def test_inactive_is_stock(ids):
    ref = stock_logits(ids)
    pm = PhaseQwen38(make_tiny_model(0), PhaseSpec(mode="kv", supp=0.01, feedback={"rank": 8, "lam": 1.0}))
    pm.set_active(False)
    with torch.no_grad():
        assert torch.equal(pm(ids), ref)
    pm.set_active(True)
    with torch.no_grad():
        assert not torch.allclose(pm(ids), ref)


def test_hard_mask_correct(ids):
    pm = PhaseQwen38(make_tiny_model(0), PhaseSpec(mode="kv", hard=True, period=4))
    a = next(iter(pm.attn.values()))
    s = clock_at(a.clock, torch.arange(T))
    assert s.tolist() == [float((t // 4) % 2) for t in range(T)]
    bias = a.phase_bias(s, s)
    same = (torch.arange(T)[:, None] // 4 % 2) == (torch.arange(T)[None, :] // 4 % 2)
    assert torch.equal(torch.isinf(bias[0]), ~same)
    assert torch.all(bias[:, same] == 0)


def test_hard_mask_equals_explicit_split(ids):
    """Hard phase-kv == running the stock attention separately on each phase's sub-sequence of keys."""
    pm = PhaseQwen38(make_tiny_model(0), PhaseSpec(mode="kv", hard=True, period=4))
    a = next(iter(pm.attn.values()))
    o = a.orig
    h = torch.randn(1, T, o.config.hidden_size)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hfq
    rot = pm.hf.model.rotary_emb
    pos = torch.arange(T)[None]
    pe = rot(h, pos[None].expand(3, -1, -1))
    causal = torch.full((T, T), torch.finfo(h.dtype).min).triu(1)[None, None]
    with torch.no_grad():
        y_phase, _ = a(h, pe, causal, position_ids=pos)
        ph = (torch.arange(T) // 4) % 2
        same = ph[:, None] == ph[None, :]
        y_ref, _ = o(h, pe, causal.masked_fill(~same, torch.finfo(h.dtype).min))
    assert torch.allclose(y_phase, y_ref, atol=1e-6)


def test_soft_bias_formula():
    pm = PhaseQwen38(make_tiny_model(0), PhaseSpec(mode="kv", supp=0.05, sharp=3.0, offset=0.3))
    a = next(iter(pm.attn.values()))
    s = clock_at(a.clock, torch.arange(T))
    A = s[:, None] * s[None] + (1 - s[:, None]) * (1 - s[None])
    b = a.phase_bias(s, s)
    assert torch.allclose(b, torch.log(0.05 + 0.95 * A)[None].expand_as(b), atol=1e-6)
    assert b.max() <= 1e-6 and b.min() >= math.log(0.05) - 1e-5


def test_head_mode_weights():
    pm = PhaseQwen38(make_tiny_model(0), PhaseSpec(mode="head", hard=True, period=4))
    a = next(iter(pm.attn.values()))
    s = clock_at(a.clock, torch.arange(T))
    w = a.head_weight(s)
    for h in range(a.H):
        assert torch.equal(w[h], (s if h % 2 else 1 - s))    # hard: active group on, other group off


def test_delta_write_gate_changes_output(ids):
    ref = stock_logits(ids)
    pm = PhaseQwen38(make_tiny_model(0), PhaseSpec(mode=None, delta_write={"enabled": True, "supp": 1.0}))
    with torch.no_grad():
        assert torch.allclose(pm(ids), ref, atol=1e-4)          # sw = 1: identity up to the logit round trip
    pm = PhaseQwen38(make_tiny_model(0), PhaseSpec(mode=None, delta_write={"enabled": True, "supp": 0.1}))
    with torch.no_grad():
        assert (pm(ids) - ref).abs().max() > 1e-3


def test_frozen_base_grads_only_new_params(ids):
    spec = PhaseSpec(mode="kv", supp=0.5, feedback={"rank": 8, "lam": 0.1, "gate": "either"},
                     tracking={"rank": 8, "lam": 0.1}, delta_write={"enabled": True, "supp": 0.5})
    pm = PhaseQwen38(make_tiny_model(0), spec)
    for p in pm.hf_base_parameters():
        p.requires_grad_(False)
    new = pm.new_parameters()
    assert len(new) > 0 and all(p.requires_grad for p in new)
    loss = pm(ids[:, :16]).float().logsumexp(-1).mean()
    loss.backward()
    assert all(p.grad is None for p in pm.hf_base_parameters())
    assert all(torch.isfinite(p.grad).all() for p in new if p.grad is not None)
    got = [n for n, p in pm.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert any("log_supp" in n for n in got) and any("feedback" in n for n in got)
    assert any("tracking" in n for n in got) and any("log_wsupp" in n for n in got)
    n_new = sum(p.numel() for p in new)
    n_base = sum(p.numel() for p in pm.hf_base_parameters())
    assert n_new < 0.01 * n_base
