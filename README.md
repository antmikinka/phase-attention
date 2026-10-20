# phase-attention (private)

**Main track: zero-training architecture surgery on pretrained Qwen3-0.6B.** `qwen_arch/` is a from-scratch Qwen3 model definition that loads the HF weights. A per-layer spec and a registry decide which layers get which new layer type:

| Layer type | What it does |
|---|---|
| **PhaseAttention** | Reuses the pretrained q/k/v/o weights. The same heads alternate every P tokens between an *access* stream, which writes to the residual and can get a gain boost, and a *suppressed* stream, which is kept in a side buffer and can be decoded. Modes: `kv` (key phases) and `head` (head-group alternation). |
| **ChunkRoPE** | Tokens inside a chunk of C share a RoPE position, so order only matters between chunks. |
| **Tracking** | A continuous slow EMA state that the residual couples to, gated by harmony or by phase. |

With every new layer disabled, the logits are **bitwise identical** to HF `Qwen3ForCausalLM`.

* Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* Measurements: [RESULTS_SURGERY.md](RESULTS_SURGERY.md)
* Reproduce: `python scripts/run_surgery.py --windows 64` (WikiText-2, first 32,768 test tokens)

```python
from qwen_arch import PhaseQwen3ForCausalLM
m = PhaseQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", rules=[
    {"layers": "8-19", "attention": {"type": "phase", "mode": "kv", "period": 4, "gain": 2.0, "supp": 1.0}}])
```

---

## Trained new layers (frozen base) — see `RESULTS_TRAINED.md`

Only the new-layer parameters were trained (Qwen3-0.6B frozen; KL to stock + LM + gate penalty; WikiText-2 train).
Every arm beats stock ppl on WikiText-2 test. Matched identity controls with the same knobs and **no** suppression
do as well or better. Training rescues hard phase only by switching it off (sharpness → 0.6, gain ≈ supp,
tracking λ → 0). The shuffled-buffer control is much worse than the real suppressed stream.

## Retention reward — see `RESULTS_RETENTION.md`

A jointly trained readout rewarded keeping suppressed-phase tokens decodable at lags 4–60. No phase arm retained
more than matched identity / stock controls. The alternation collapsed anyway (gate fully open, sharpness ≈ 0.8).

## Earlier tiny-model prototype (paused)

This is a new tiny language-model architecture that trains on a CPU. It has a **clean, swappable attention interface**, and it already includes a first prototype of the attention we plan to swap in later: **PhaseAttention v0**, which is phase-alternating and multi-stream.

Two recent observations inspired it, and they are used **only as engineering metaphors** (see [docs/SOURCES.md](docs/SOURCES.md)):

1. **Hesse, …, Engel & Tsao (bioRxiv, 2026-10-16), *Rapid alternation of conscious and suppressed stimulus representations in the primate brain*.** During binocular flash suppression, the same IT and prefrontal populations alternate about 15 times a second (roughly every 60 ms) between representing the perceived face and the suppressed face. Both faces can be reconstructed, and the conscious epochs show higher gain. Tsao explicitly says this does **not** show that consciousness has a "frame rate".
2. **Locally time-reversed speech** (Oruk Labs demo; Saberi & Perrott, Nature 1999). Speech stays intelligible when it is reversed inside short windows (about 20–50 ms) as long as the windows stay in order.

## What is here

| Piece | File |
|---|---|
| Attention interface, registry, RoPE / CoarseRoPE, `causal` baseline, `phase_mask` ablation | `phase_attention/attention.py` |
| `TinyLM` baseline (RMSNorm, SwiGLU, RoPE) and `PhaseLM` (K shared-weight streams, phase-gated workspace, inhibition, feedback) | `phase_attention/model.py` |
| `chunk_reverse`, chunk-end masks | `phase_attention/chunks.py` |
| Synthetic binocular-rivalry task, byte-level TinyStories | `phase_attention/data.py` |
| Ridge probes for decoding suppressed content | `phase_attention/probe.py` |
| Training and evaluation for each arm | `scripts/train.py` |
| Smoke runs | `scripts/run_smoke.sh`, `scripts/run_smoke_extrap.sh` |
| Design, plan, and hypotheses | `docs/ARCHITECTURE.md`, `docs/CLEAR_THOUGHT_PLAN.md`, `PLAN.md` |

## Swapping the attention later

```python
from phase_attention import Attention, register_attention, ModelConfig, build_model

@register_attention("my_phase_attn")
class MyPhaseAttention(Attention):
    def __init__(self, d_model, n_heads, **kw): ...
    def forward(self, x, rope): ...          # (B,T,D) -> (B,T,D)

model = build_model(ModelConfig(attention="my_phase_attn"))
```

## Quick start

```bash
pip install -r requirements.txt
curl -L -o data/tinystories_valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
pytest -q
python scripts/train.py --task rivalry --arm phase_fixed --seq 48 --steps 3000
python scripts/train.py --task lm --arm baseline --steps 1200
```

Arms:

| Group | Arms |
|---|---|
| Baselines | `baseline`, `baseline_2x` (2× depth, matches FLOPs) |
| Phase variants | `phase_fixed`, `phase_all` (no-phase control), `phase_random` (random-phase control), `phase_fixed_noinhib`, `phase_learned` |
| Other | `coarse4` (CoarseRoPE with C=4), `phase_mask` |

Results go to `results/*.json`, and the smoke summary is in `RESULTS_SMOKE.md`.
