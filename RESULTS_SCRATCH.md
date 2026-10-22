# RESULTS_SCRATCH — phase layers in a model trained FROM SCRATCH (nothing from Qwen)

> **INTERIM: seed 0 only.** Seed 1 is running and this file will be updated when it finishes. Design is in
> `docs/MODEL_FROM_SCRATCH.md`. Raw per-run results: `results/scratch/<task>_<arm>_s<seed>.json`.

Every model is a ~6.39M-param byte-level decoder trained from random init. Params are matched within 0.04%.
Tokens, steps, LR schedule and data order are identical across arms. Eval tok/s in the tables was measured while
two runs shared the CPU, so it is only indicative. Section (c) will hold a clean benchmark.

## (a) TinyStories byte-level LM (val = 64×256 windows of the valid split)

| arm | params | seeds | val bpb (mean ± sd) | byte ppl | eval tok/s | vs baseline: Δbpb, windows better /64, sign p / Wilcoxon p (seed-mean windows) | per-seed Δbpb |
|---|---|---|---|---|---|---|---|
| baseline | 6,394,112 | 2 | 1.1864 ± 0.0071 | 2.276 | 16,742 | - | - |
| phase_kv_fb | 6,396,280 | 1 | 1.2082 | 2.311 | 9,663 | +0.0168, 18/64, 0.00062 / 0.00018 | s0: +0.0168 |
| CTRL_shuffled_fb | 6,396,280 | 1 | 1.2230 | 2.334 | 9,083 | +0.0315, 13/64, 1.9e-06 / 6.9e-08 | s0: +0.0315 |
| phase_head | 6,394,200 | 1 | 1.2491 | 2.377 | 17,392 | +0.0576, 4/64, 7.4e-14 / 1e-11 | s0: +0.0576 |
| phase_track | 6,392,172 | 1 | 1.2159 | 2.323 | 11,614 | +0.0244, 15/64, 2.4e-05 / 5.5e-06 | s0: +0.0244 |
| chunkrope | 6,394,120 | 1 | 1.1937 | 2.287 | 19,325 | +0.0022, 29/64, 0.53 / 0.73 | s0: +0.0022 |
| phase_kv | 6,394,200 | 1 | 1.2025 | 2.301 | 15,069 | +0.0110, 23/64, 0.033 / 0.039 | s0: +0.0110 |

## (b) Associative recall (10 pairs / 128 tokens, P=4 blocks; chance ≈ 1/32 = 3.1%)

Lags in tokens. **Odd number of blocks (4, 12, 28, 60) = pair stored in the phase OPPOSITE the query** (suppressed for kv phase); 8, 16, 32, 64 = same phase.

| arm | params | seeds | acc@4 | acc@8 | acc@12 | acc@16 | acc@28 | acc@32 | acc@60 | acc@64 | cross-phase | same-phase | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 6,394,112 | 1 | 52.7 | 57.1 | 49.3 | 52.8 | 57.2 | 61.4 | 77.3 | 79.7 | 59.2 | 62.8 | 24,281 |
| phase_kv_fb | 6,396,280 | 1 | 50.3 | 58.1 | 48.9 | 51.6 | 51.8 | 60.1 | 73.5 | 79.7 | 56.3 | 62.4 | 20,836 |
| CTRL_shuffled_fb | 6,396,280 | 1 | 47.1 | 58.7 | 50.2 | 52.0 | 54.1 | 61.1 | 75.7 | 79.6 | 56.9 | 62.9 | 18,063 |
| phase_head | 6,394,200 | 1 | 52.2 | 57.4 | 43.2 | 45.2 | 48.0 | 50.7 | 71.1 | 74.3 | 53.7 | 57.0 | 24,415 |
| phase_track | 6,392,172 | 1 | 55.6 | 55.4 | 49.8 | 52.9 | 58.7 | 54.5 | 79.6 | 77.8 | 61.0 | 60.2 | 25,371 |
| chunkrope | 6,394,120 | 1 | 64.0 | 46.0 | 41.7 | 38.3 | 50.8 | 52.2 | 76.7 | 74.2 | 58.4 | 52.7 | 29,023 |
| phase_kv | 6,394,200 | 1 | 52.9 | 57.1 | 48.3 | 49.9 | 55.5 | 62.1 | 74.5 | 78.6 | 57.9 | 62.0 | 23,964 |

Per-seed cross-phase / same-phase accuracy:

- baseline: s0 59.2 / 62.8
- phase_kv_fb: s0 56.3 / 62.4
- CTRL_shuffled_fb: s0 56.9 / 62.9
- phase_head: s0 53.7 / 57.0
- phase_track: s0 61.0 / 60.2
- chunkrope: s0 58.4 / 52.7
- phase_kv: s0 57.9 / 62.0

Paired tests vs baseline over the 32 held-out eval batches (seed-mean accuracy; long = lags 28–64):

| arm | metric | Δ acc (pp) | batches better/worse | sign p / Wilcoxon p |
|---|---|---|---|---|
| phase_kv_fb | cross-phase (4,12,28,60) | -3.0 | 4/28 | 1.9e-05 / 6.3e-06 |
| phase_kv_fb | same-phase (8,16,32,64) | -0.4 | 15/17 | 0.86 / 0.57 |
| phase_kv_fb | long cross (28,60) | -4.5 | 3/29 | 2.6e-06 / 2.1e-06 |
| CTRL_shuffled_fb | cross-phase (4,12,28,60) | -2.3 | 7/25 | 0.0021 / 0.00015 |
| CTRL_shuffled_fb | same-phase (8,16,32,64) | +0.1 | 17/15 | 0.86 / 0.83 |
| CTRL_shuffled_fb | long cross (28,60) | -2.3 | 8/24 | 0.007 / 0.002 |
| phase_head | cross-phase (4,12,28,60) | -5.5 | 1/31 | 1.5e-08 / 9.2e-07 |
| phase_head | same-phase (8,16,32,64) | -5.8 | 1/31 | 1.5e-08 / 9.2e-07 |
| phase_head | long cross (28,60) | -7.7 | 1/31 | 1.5e-08 / 1.1e-06 |
| phase_track | cross-phase (4,12,28,60) | +1.8 | 27/5 | 0.00011 / 0.0006 |
| phase_track | same-phase (8,16,32,64) | -2.6 | 4/28 | 1.9e-05 / 9.7e-05 |
| phase_track | long cross (28,60) | +2.0 | 20/11 | 0.15 / 0.0028 |
| chunkrope | cross-phase (4,12,28,60) | -0.8 | 10/22 | 0.05 / 0.051 |
| chunkrope | same-phase (8,16,32,64) | -10.1 | 0/32 | 4.7e-10 / 8.3e-07 |
| chunkrope | long cross (28,60) | -3.5 | 3/29 | 2.6e-06 / 2.6e-06 |
| phase_kv | cross-phase (4,12,28,60) | -1.4 | 9/23 | 0.02 / 0.00084 |
| phase_kv | same-phase (8,16,32,64) | -0.9 | 14/18 | 0.6 / 0.065 |
| phase_kv | long cross (28,60) | -2.3 | 8/24 | 0.007 / 0.00039 |

## Learned phase / gate parameters (mean over blocks; per seed)

- **lm/phase_kv_fb/s0**: Feedback_gate_mean=0.961, Feedback_gate_open=0.981, Feedback_lam=0.944, offset=-0.0854, period=4.21, sharp=2.08, supp_mean=0.0754
- **lm/CTRL_shuffled_fb/s0**: Feedback_gate_mean=0.839, Feedback_gate_open=0.916, Feedback_lam=0.757, offset=-0.0389, period=4, sharp=2.05, supp_mean=0.0769
- **lm/phase_head/s0**: offset=0.103, period=3.96, sharp=2.01, supp_mean=0.0795
- **lm/phase_track/s0**: Tracking_gate_mean=0.928, Tracking_gate_open=0.958, Tracking_lam=1.08, offset=-0.0778, period=4.07, sharp=2.05, supp_mean=0.0766
- **lm/chunkrope/s0**: chunk_blend=0.469
- **lm/phase_kv/s0**: offset=-0.0674, period=4.07, sharp=2.05, supp_mean=0.0766
- **recall/phase_kv_fb/s0**: Feedback_gate_mean=0.829, Feedback_gate_open=0.925, Feedback_lam=0.939, offset=0.0313, period=3.95, sharp=3.58, supp_mean=0.0521
- **recall/CTRL_shuffled_fb/s0**: Feedback_gate_mean=0.718, Feedback_gate_open=0.803, Feedback_lam=0.868, offset=0.0146, period=3.98, sharp=3.64, supp_mean=0.0516
- **recall/phase_head/s0**: offset=-0.0679, period=4.05, sharp=2.95, supp_mean=0.0599
- **recall/phase_track/s0**: Tracking_gate_mean=0.798, Tracking_gate_open=0.9, Tracking_lam=0.933, offset=0.0547, period=3.96, sharp=3.56, supp_mean=0.0509
- **recall/chunkrope/s0**: chunk_blend=0.474
- **recall/phase_kv/s0**: offset=0.0322, period=3.98, sharp=3.64, supp_mean=0.052

## Interim reading (seed 0)

- **LM (TinyStories bpb):** the baseline is best (1.1915).
  - ChunkRoPE is tied with it (+0.002, p = 0.5).
  - Every phase arm is worse: phase_kv +0.011 (p ≈ 0.03), phase_kv + feedback +0.017, phase + tracking +0.024,
    shuffled feedback +0.032, phase_head +0.058 (p < 1e-10).
  - Feedback with the real suppressed stream beats the shuffled-buffer control (1.208 vs 1.223). So the
    suppressed-stream content is used, but it does not make up for the cost of suppression.
- **Recall:** the alternation gets a real job here (the pair is stored in the suppressed phase).
  - phase_kv and phase_kv + feedback **lose** cross-phase recall relative to the baseline (−1.4 and −3.0 pp, p < 0.02)
    and keep same-phase recall.
  - phase + tracking is the one arm that **gains** cross-phase recall: +1.8 pp, 27 of 32 batches, p ≈ 1e-4. It
    loses 2.6 pp on same-phase pairs, so overall it is roughly neutral.
  - ChunkRoPE is much better at lag 4 (64 vs 53%) but loses 10 pp on same-phase lags.
  - phase_head is worse everywhere.
- **Collapse? No — unlike the frozen-base experiments.** Trained from scratch, the models keep strong suppression
  (supp ≈ 0.05–0.08, vs init 0.05) and keep the period at ≈ 4. Sharpness softens from 4 to ≈ 2 (LM) / ≈ 3.6
  (recall), and the feedback gate stays mostly open (0.83–0.96).
