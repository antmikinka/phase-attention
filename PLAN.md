# Plan

| Stage | What | Status |
|---|---|---|
| 0 | Verify sources (docs/SOURCES.md) | done |
| 1 | Clear Thought design (docs/CLEAR_THOUGHT_PLAN.md) | done |
| 2 | Baseline TinyLM + swappable Attention interface + tests | done |
| 3 | PhaseLM v0 (multi-stream, phase-gated workspace) + chunk-reversal utils | done |
| 4 | Smoke: rivalry (6 arms) + byte LM (4 arms), 1 seed, 2 threads | running / see RESULTS_SMOKE.md |
| 5 | Full: 3 seeds, longer, harder rivalry (longer delay, more symbols), P sweep {1,2,4,8} | next |
| 6 | "Update the attention": learned gating, head-split phase (option B), phase_mask, oscillator-driven phase | later |
| 7 | CoarseRoPE + reversal-augmentation sweep (C in {2,4,8}) | later |

Success criteria for each hypothesis are in docs/CLEAR_THOUGHT_PLAN.md. The key rule is that a phase gain only counts if it beats **both** `phase_all` (no phase) and `phase_random` (random phase) at the same parameter count, and if it holds across seeds.
