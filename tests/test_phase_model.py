"""Tests for the from-scratch phase_model package (no pretrained weights)."""
import copy, os, sys
import pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase_model import ModelConfig, PhaseLM, layer_specs
from phase_model.data import recall_batch, QUERY

ARMS = {
    "baseline": [],
    "phase_kv_fb": [{"mixer": {"type": "phase", "mode": "kv"}}, {"extra": {"type": "feedback"}}],
    "phase_head": [{"mixer": {"type": "phase", "mode": "head"}}],
    "phase_track": [{"mixer": {"type": "phase"}}, {"layers": "every2", "extra": {"type": "tracking"}}],
    "chunkrope": [{"position": {"type": "chunk_rope", "chunk": 4}}],
    "shuffled": [{"mixer": {"type": "phase"}}, {"extra": {"type": "feedback", "source": "shuffled"}}],
    "hard_skip": [{"mixer": {"type": "phase", "hard": True, "skip": True}}],
}


def small(rules, **kw):
    torch.manual_seed(0)
    return PhaseLM(ModelConfig(d_model=64, n_layers=4, n_heads=4, mlp_hidden=96, blocks=layer_specs(4, rules), **kw))


def copy_base(dst, src):
    sd = {k: v for k, v in src.state_dict().items()}
    missing = dst.load_state_dict(sd, strict=False)
    return missing


@pytest.mark.parametrize("arm", list(ARMS))
def test_causal_and_grad(arm):
    m = small(ARMS[arm]); x = torch.randint(0, 256, (2, 40))
    torch.manual_seed(1); y1 = m(x)
    x2 = x.clone(); x2[:, 30:] = torch.randint(0, 256, (2, 10))
    torch.manual_seed(1); y2 = m(x2)
    assert torch.allclose(y1[:, :30], y2[:, :30], atol=1e-5), "future tokens leaked"
    y1.logsumexp(-1).mean().backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if p.requires_grad), arm


@pytest.mark.parametrize("mode", ["kv", "head"])
def test_phase_with_supp_one_is_standard_attention(mode):
    base = small([])
    ph = small([{"mixer": {"type": "phase", "mode": mode, "supp": 1.0}}])
    copy_base(ph, base)
    x = torch.randint(0, 256, (2, 33))
    assert torch.allclose(base(x), ph(x), atol=1e-5)


def test_hard_skip_equals_hard_mask():
    a = small([{"mixer": {"type": "phase", "hard": True, "skip": False}}])
    b = small([{"mixer": {"type": "phase", "hard": True, "skip": True}}])
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (2, 64))
    assert torch.allclose(a(x), b(x), atol=1e-5)


def test_feedback_lam_zero_equals_phase_kv():
    a = small([{"mixer": {"type": "phase"}}])
    b = small([{"mixer": {"type": "phase"}}, {"extra": {"type": "feedback", "lam": 0.0}}])
    copy_base(b, a)
    x = torch.randint(0, 256, (2, 33))
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_chunkrope_blend_zero_is_rope():
    a = small([])
    b = small([{"position": {"type": "chunk_rope", "chunk": 4, "blend": 1e-7, "learn": False}}])
    copy_base(b, a)
    x = torch.randint(0, 256, (2, 33))
    assert torch.allclose(a(x), b(x), atol=1e-4)


def test_soft_clock_sharp_matches_hard():
    from phase_model.layers import PhaseClock
    s = PhaseClock(4, sharp=200.0)(32); h = PhaseClock(4, hard=True)(32)
    assert torch.allclose(s, h, atol=1e-3)


def test_param_matching():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import scratch_train as st
    for arm in st.ARMS:
        m, target = st.build(arm, 0)
        assert abs(m.n_params() / target - 1) < 0.005, arm


def test_recall_layout():
    x, pos, lag = recall_batch(8, g=torch.Generator().manual_seed(0))
    for b in range(8):
        for j in range(pos.shape[1]):
            p, L = int(pos[b, j]), int(lag[b, j])
            assert x[b, p - 1] == QUERY and x[b, p] == x[b, p - L] and x[b, p + 1] == x[b, p - L + 1]
            assert ((p // 4) - ((p - L) // 4)) == L // 4
