import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase_attention import ModelConfig, build_model, chunk_reverse, ATTENTION_REGISTRY, register_attention
from phase_attention.attention import CausalSelfAttention, RotaryEmbedding
from phase_attention.model import phase_schedule, PhaseLM
from phase_attention.chunks import chunk_end_mask
from phase_attention.data import RivalryConfig, make_rivalry_batch, ByteText
from phase_attention.probe import ridge_probe


def small(**kw):
    c = ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_baseline_shapes_and_causality():
    torch.manual_seed(0)
    m = build_model(small()).eval()
    x = torch.randint(0, 32, (2, 10))
    y = m(x)
    assert y.shape == (2, 10, 32)
    x2 = x.clone(); x2[:, 7:] = (x2[:, 7:] + 1) % 32
    y2 = m(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-5)


def test_phase_causality_and_shapes():
    torch.manual_seed(0)
    for sched in ["fixed", "all", "learned"]:
        m = build_model(small(arch="phase", schedule=sched)).eval()
        x = torch.randint(0, 32, (2, 12))
        y, hid = m(x, return_hidden=True)
        assert y.shape == (2, 12, 32) and hid["streams"].shape == (2, 2, 12, 32)
        x2 = x.clone(); x2[:, 9:] = (x2[:, 9:] + 3) % 32
        assert torch.allclose(y[:, :9], m(x2)[:, :9], atol=1e-5)


def test_phase_params_close_to_baseline():
    b = build_model(small()).n_params()
    p = build_model(small(arch="phase")).n_params()
    assert 0 < p - b < 0.02 * b + 200


def test_schedules():
    a = phase_schedule("fixed", 1, 2, 8, 2)
    assert a[0, 0].tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    assert torch.allclose(a.sum(1), torch.ones(1, 8))
    r = phase_schedule("random", 3, 2, 10, 2)
    assert torch.allclose(r.sum(1), torch.ones(3, 10))
    assert (r[:, :, 0::2] == r[:, :, 1::2]).all()   # constant within micro-epochs
    al = phase_schedule("all", 1, 2, 5, 2)
    assert torch.allclose(al, torch.full_like(al, 0.5))


def test_only_access_stream_reaches_head():
    """With a fixed schedule and zero feedback, perturbing the suppressed
    stream's embedding must not change logits at positions where the other
    stream has access... except through inhibition; disable it."""
    torch.manual_seed(0)
    m = PhaseLM(small(arch="phase", schedule="fixed", period=3, inhibition=False)).eval()
    x = torch.randint(0, 32, (1, 9))
    y0 = m(x)
    with torch.no_grad():
        m.stream_emb[1] += 1.0   # stream 1 has access at t in 3..5
    y1 = m(x)
    same = torch.isclose(y0, y1, atol=1e-5).all(-1)[0]
    assert same[:3].all() and same[6:].all() and not same[3:6].all()


def test_chunk_reverse():
    x = torch.arange(10).view(1, 10)
    assert chunk_reverse(x, 4).tolist() == [[3, 2, 1, 0, 7, 6, 5, 4, 9, 8]]
    assert torch.equal(chunk_reverse(chunk_reverse(x, 3), 3), x)
    assert torch.equal(chunk_reverse(x, 1), x)
    assert chunk_end_mask(8, 4).tolist() == [False, False, False, True] * 2


def test_coarse_rope_invariance():
    r = RotaryEmbedding(8, chunk=4)
    c, s = r.angles(8)
    assert torch.allclose(c[0], c[3]) and not torch.allclose(c[3], c[4])


def test_attention_registry_swap():
    @register_attention("test_identity")
    class Ident(CausalSelfAttention):
        def forward(self, x, rope):
            return torch.zeros_like(x)
    m = build_model(small(attention="test_identity"))
    assert isinstance(m.blocks[0].attn, Ident)
    assert m(torch.randint(0, 32, (1, 4))).shape == (1, 4, 32)
    m2 = build_model(small(attention="phase_mask", attention_kwargs={"period": 2}))
    assert m2(torch.randint(0, 32, (1, 6))).shape == (1, 6, 32)


def test_rivalry_batch():
    rc = RivalryConfig(n_symbols=8, seq_len=20, delay=4)
    ids, tgt, lab = make_rivalry_batch(rc, 5, torch.Generator().manual_seed(0))
    assert ids.shape == (5, 20, 2)
    n = 18
    for b in range(5):
        d = lab["dom_eye"][b].item()
        assert ids[b, 0, 0].item() == (rc.CUE_L if d == 0 else rc.CUE_R)
        assert ids[b, 1:n + 1, d].tolist() == lab["dom"][b].tolist()       # dominant eye clean
        assert ids[b, 1:n - 3, 1 - d].tolist() == lab["sup"][b, : n - 4].tolist()
        assert tgt[b, -1].item() == lab["sup"][b, n - 5].item()
        assert tgt[b, 3].item() == ids[b, 4, d].item()


def test_ridge_probe_separable():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 4, (400,), generator=g)
    X = torch.nn.functional.one_hot(y, 4).float() * 3 + 0.1 * torch.randn(400, 4, generator=g)
    assert ridge_probe(X[:200], y[:200], X[200:], y[200:], 4) > 0.95


def test_rivalry_extrap_target():
    rc = RivalryConfig(n_symbols=8, seq_len=20, delay=4, recall="extrap")
    ids, tgt, lab = make_rivalry_batch(rc, 6, torch.Generator().manual_seed(1))
    n = 18
    for b in range(6):
        s = lab["sup"][b]
        step = (s[1] - s[0]) % 8
        assert tgt[b, -1].item() == (s[-1] + step) % 8
