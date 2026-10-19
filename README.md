# phase-attention (private)

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
