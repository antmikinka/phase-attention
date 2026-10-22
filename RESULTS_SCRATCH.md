# RESULTS_SCRATCH — phase layers in a model trained FROM SCRATCH (nothing from Qwen)

Every model is a ~6.39M-param **byte-level** decoder trained from random init: no pretrained weights, no Qwen
tokenizer, no surgery. Parameter counts are matched within 0.04%, and tokens, steps, LR schedule and data order are
identical across arms. **2 seeds per arm and task** (seeds 0, 1): 28 training runs. Design is in
`docs/MODEL_FROM_SCRATCH.md`. Raw per-run results (window losses, per-batch accuracies, phase reports):
`results/scratch/<task>_<arm>_s<seed>.json`. Training curves: `logs/scratch/*.jsonl`.

- **LM:** TinyStories V2, trained on the first 40 MB of the train file. 1500 steps × 16 × 256 bytes ≈ 6.1M bytes.
  Val bpb on 64 × 256 windows of the official valid file.
- **Recall:** 128-token sequences with 10 key–value pairs each. 1000 steps × 64 sequences. 2048 held-out eval
  sequences.
- **Paired tests:** for LM, over the 64 val windows (seed-averaged loss per window); for recall, over the 32
  held-out eval batches (seed-averaged accuracy). Exact sign test / Wilcoxon signed-rank.
- **Eval tok/s in (a)/(b):** measured while two training runs shared the CPU, so it is only indicative. See (c).

## (a) TinyStories byte-level LM (val = 64×256 windows of the valid split)

| arm | params | seeds | val bpb (mean ± sd) | byte ppl | eval tok/s | vs baseline: Δbpb, windows better /64, sign p / Wilcoxon p (seed-mean windows) | per-seed Δbpb |
|---|---|---|---|---|---|---|---|
| baseline | 6,394,112 | 2 | 1.1864 ± 0.0071 | 2.276 | 16,742 | - | - |
| phase_kv_fb | 6,396,280 | 2 | 1.2087 ± 0.0006 | 2.311 | 9,915 | +0.0222, 11/64, 1e-07 / 2.5e-08 | s0: +0.0168, s1: +0.0277 |
| CTRL_shuffled_fb | 6,396,280 | 2 | 1.2145 ± 0.0120 | 2.321 | 8,729 | +0.0281, 7/64, 7.6e-11 / 1.3e-09 | s0: +0.0315, s1: +0.0246 |
| phase_head | 6,394,200 | 2 | 1.2400 ± 0.0129 | 2.362 | 17,623 | +0.0535, 3/64, 4.7e-15 / 6.4e-12 | s0: +0.0576, s1: +0.0494 |
| phase_track | 6,392,172 | 2 | 1.2074 ± 0.0120 | 2.309 | 11,746 | +0.0209, 9/64, 3.5e-09 / 1.6e-07 | s0: +0.0244, s1: +0.0175 |
| chunkrope | 6,394,120 | 2 | 1.1969 ± 0.0046 | 2.293 | 18,512 | +0.0105, 21/64, 0.0081 / 0.0092 | s0: +0.0022, s1: +0.0188 |
| phase_kv | 6,394,200 | 2 | 1.2043 ± 0.0026 | 2.304 | 12,383 | +0.0179, 19/64, 0.0016 / 9.8e-05 | s0: +0.0110, s1: +0.0247 |

## (b) Associative recall (10 pairs / 128 tokens, P=4 blocks; chance ≈ 1/32 = 3.1%)

Lags in tokens. **Odd number of blocks (4, 12, 28, 60) = pair stored in the phase OPPOSITE the query** (suppressed for kv phase); 8, 16, 32, 64 = same phase.

| arm | params | seeds | acc@4 | acc@8 | acc@12 | acc@16 | acc@28 | acc@32 | acc@60 | acc@64 | cross-phase | same-phase | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 6,394,112 | 2 | 51.3 | 58.4 | 49.0 | 53.2 | 58.3 | 62.6 | 77.7 | 80.1 | 59.2 | 63.6 | 26,366 |
| phase_kv_fb | 6,396,280 | 2 | 51.1 | 57.1 | 46.3 | 51.5 | 53.9 | 60.8 | 75.1 | 79.7 | 56.7 | 62.4 | 20,874 |
| CTRL_shuffled_fb | 6,396,280 | 2 | 48.7 | 59.2 | 47.1 | 52.0 | 56.2 | 61.7 | 76.0 | 79.7 | 57.1 | 63.2 | 18,796 |
| phase_head | 6,394,200 | 2 | 50.5 | 59.5 | 41.9 | 43.8 | 49.8 | 52.7 | 73.3 | 75.6 | 54.0 | 58.0 | 26,369 |
| phase_track | 6,392,172 | 2 | 52.4 | 54.4 | 51.7 | 52.5 | 56.6 | 59.3 | 77.4 | 79.2 | 59.6 | 61.4 | 25,617 |
| chunkrope | 6,394,120 | 2 | 64.0 | 46.3 | 41.1 | 37.4 | 51.1 | 52.0 | 76.7 | 73.9 | 58.3 | 52.5 | 29,911 |
| phase_kv | 6,394,200 | 2 | 52.9 | 56.9 | 45.5 | 49.6 | 55.9 | 59.5 | 76.4 | 79.4 | 57.8 | 61.4 | 26,464 |

Per-seed cross-phase / same-phase accuracy:

- baseline: s0 59.2 / 62.8; s1 59.1 / 64.4
- phase_kv_fb: s0 56.3 / 62.4; s1 57.1 / 62.3
- CTRL_shuffled_fb: s0 56.9 / 62.9; s1 57.2 / 63.5
- phase_head: s0 53.7 / 57.0; s1 54.3 / 59.1
- phase_track: s0 61.0 / 60.2; s1 58.2 / 62.6
- chunkrope: s0 58.4 / 52.7; s1 58.3 / 52.2
- phase_kv: s0 57.9 / 62.0; s1 57.7 / 60.9

Paired tests vs baseline over the 32 held-out eval batches (seed-mean accuracy; long = lags 28–64):

| arm | metric | Δ acc (pp) | batches better/worse | sign p / Wilcoxon p |
|---|---|---|---|---|
| phase_kv_fb | cross-phase (4,12,28,60) | -2.5 | 2/30 | 2.5e-07 / 1.8e-06 |
| phase_kv_fb | same-phase (8,16,32,64) | -1.3 | 10/22 | 0.05 / 0.0055 |
| phase_kv_fb | long cross (28,60) | -3.5 | 2/30 | 2.5e-07 / 1.5e-06 |
| CTRL_shuffled_fb | cross-phase (4,12,28,60) | -2.1 | 4/28 | 1.9e-05 / 2.8e-06 |
| CTRL_shuffled_fb | same-phase (8,16,32,64) | -0.4 | 12/20 | 0.22 / 0.23 |
| CTRL_shuffled_fb | long cross (28,60) | -1.9 | 7/25 | 0.0021 / 0.00084 |
| phase_head | cross-phase (4,12,28,60) | -5.2 | 0/32 | 4.7e-10 / 8.3e-07 |
| phase_head | same-phase (8,16,32,64) | -5.6 | 0/32 | 4.7e-10 / 8.3e-07 |
| phase_head | long cross (28,60) | -6.5 | 0/32 | 4.7e-10 / 8.3e-07 |
| phase_track | cross-phase (4,12,28,60) | +0.5 | 22/10 | 0.05 / 0.063 |
| phase_track | same-phase (8,16,32,64) | -2.2 | 4/28 | 1.9e-05 / 4.4e-06 |
| phase_track | long cross (28,60) | -1.0 | 8/23 | 0.011 / 0.029 |
| chunkrope | cross-phase (4,12,28,60) | -0.8 | 12/20 | 0.22 / 0.029 |
| chunkrope | same-phase (8,16,32,64) | -11.1 | 0/32 | 4.7e-10 / 8.3e-07 |
| chunkrope | long cross (28,60) | -4.1 | 1/31 | 1.5e-08 / 1.1e-06 |
| phase_kv | cross-phase (4,12,28,60) | -1.4 | 5/27 | 0.00011 / 4.4e-05 |
| phase_kv | same-phase (8,16,32,64) | -2.2 | 2/30 | 2.5e-07 / 4e-06 |
| phase_kv | long cross (28,60) | -1.9 | 5/27 | 0.00011 / 6.5e-05 |

## Learned phase / gate parameters (mean over blocks; per seed)

- **lm/phase_kv_fb/s0**: Feedback_gate_mean=0.961, Feedback_gate_open=0.981, Feedback_lam=0.944, offset=-0.0854, period=4.21, sharp=2.08, supp_mean=0.0754
- **lm/phase_kv_fb/s1**: Feedback_gate_mean=0.962, Feedback_gate_open=0.982, Feedback_lam=0.948, offset=-0.0998, period=4.17, sharp=2.09, supp_mean=0.0752
- **lm/CTRL_shuffled_fb/s0**: Feedback_gate_mean=0.839, Feedback_gate_open=0.916, Feedback_lam=0.757, offset=-0.0389, period=4, sharp=2.05, supp_mean=0.0769
- **lm/CTRL_shuffled_fb/s1**: Feedback_gate_mean=0.842, Feedback_gate_open=0.915, Feedback_lam=0.762, offset=-0.0887, period=4.1, sharp=2.06, supp_mean=0.0765
- **lm/phase_head/s0**: offset=0.103, period=3.96, sharp=2.01, supp_mean=0.0795
- **lm/phase_head/s1**: offset=-0.0931, period=4.01, sharp=2.02, supp_mean=0.0784
- **lm/phase_track/s0**: Tracking_gate_mean=0.928, Tracking_gate_open=0.958, Tracking_lam=1.08, offset=-0.0778, period=4.07, sharp=2.05, supp_mean=0.0766
- **lm/phase_track/s1**: Tracking_gate_mean=0.921, Tracking_gate_open=0.94, Tracking_lam=1.09, offset=-0.0417, period=4.02, sharp=2.06, supp_mean=0.0766
- **lm/chunkrope/s0**: chunk_blend=0.469
- **lm/chunkrope/s1**: chunk_blend=0.466
- **lm/phase_kv/s0**: offset=-0.0674, period=4.07, sharp=2.05, supp_mean=0.0766
- **lm/phase_kv/s1**: offset=-0.0622, period=4.04, sharp=2.04, supp_mean=0.0771
- **recall/phase_kv_fb/s0**: Feedback_gate_mean=0.829, Feedback_gate_open=0.925, Feedback_lam=0.939, offset=0.0313, period=3.95, sharp=3.58, supp_mean=0.0521
- **recall/phase_kv_fb/s1**: Feedback_gate_mean=0.882, Feedback_gate_open=0.952, Feedback_lam=0.941, offset=0.0359, period=3.96, sharp=3.66, supp_mean=0.0505
- **recall/CTRL_shuffled_fb/s0**: Feedback_gate_mean=0.718, Feedback_gate_open=0.803, Feedback_lam=0.868, offset=0.0146, period=3.98, sharp=3.64, supp_mean=0.0516
- **recall/CTRL_shuffled_fb/s1**: Feedback_gate_mean=0.608, Feedback_gate_open=0.687, Feedback_lam=0.883, offset=0.0309, period=3.97, sharp=3.65, supp_mean=0.0506
- **recall/phase_head/s0**: offset=-0.0679, period=4.05, sharp=2.95, supp_mean=0.0599
- **recall/phase_head/s1**: offset=-0.0746, period=4.05, sharp=2.89, supp_mean=0.0591
- **recall/phase_track/s0**: Tracking_gate_mean=0.798, Tracking_gate_open=0.9, Tracking_lam=0.933, offset=0.0547, period=3.96, sharp=3.56, supp_mean=0.0509
- **recall/phase_track/s1**: Tracking_gate_mean=0.702, Tracking_gate_open=0.791, Tracking_lam=0.877, offset=0.0423, period=3.95, sharp=3.73, supp_mean=0.0498
- **recall/chunkrope/s0**: chunk_blend=0.474
- **recall/chunkrope/s1**: chunk_blend=0.48
- **recall/phase_kv/s0**: offset=0.0322, period=3.98, sharp=3.64, supp_mean=0.052
- **recall/phase_kv/s1**: offset=0.0215, period=3.99, sharp=3.72, supp_mean=0.0519

## (c) Throughput and attention work actually computed (forward only, untrained weights, 8 threads; median of runs)

| variant | QK entries computed (rel. baseline causal kernel) | tok/s T=256 | T=1024 | T=2048 |
|---|---|---|---|---|
| baseline | 1.0 | 32,556 | 30,124 | 23,790 |
| hard_kv_mask | 2.0 | 33,331 | 23,231 | 10,868 |
| hard_kv_split(skip) | 0.5 | 33,289 | 26,903 | 22,642 |
| soft_kv | 2.0 | 32,032 | 11,815 | 3,721 |
| soft_kv_feedback | 4.0 | 36,048 | 7,987 | 2,306 |
| phase_head | 1.0 | 52,385 | 44,868 | 33,214 |

How to read (c):
- The baseline uses the fused causal kernel, which already skips the upper triangle.
- `hard_kv_split(skip)` really computes only same-phase score entries (half of what causal attention computes). It is
  about 2× faster than the same hard phase implemented as a dense mask at T = 2048 (22.6k vs 10.9k tok/s). It is
  **not** faster than the baseline (23.8k): gather/scatter overhead eats the saved FLOPs at this size.
- The soft (learnable) phase needs a dense additive mask, and feedback adds a second partition pass. Both are
  2–10× slower at long T. **None of the trained arms skip any compute.**
- The feedback gate is open 70–98% of the time (below). Even a sparse implementation could skip at most the closed
  fraction of the suppressed-stream pass.
- `phase_head` appeared faster than the baseline in this script. An interleaved re-check at T = 1024 gave 26–29k
  tok/s for both (baseline 26.2k / 28.3k / 29.4k, head 28.4k / 27.8k), so that difference is a measurement artifact.
  Head mode computes the same causal attention as the baseline.

## Do the learned phase parameters collapse?

**No**, unlike the frozen-base Qwen experiments (RESULTS_TRAINED / RESULTS_RETENTION), where training switched the
phase off. Here, with every weight trained from scratch:
- the period stays at 3.95–4.21 (init 4);
- the offset stays within ±0.1;
- supp stays at 0.05–0.08 (init 0.05), so other-phase keys are still down-weighted about 13–20×;
- sharpness falls from 4 to ≈ 2.0 (LM) and ≈ 3.6 (recall), so the phase boundary softens but remains.

Caveat: these scalars share lr 1e-3 with the weights. log supp could have moved by more than its observed +0.4
(sharpness moved by −0.7 in log space), but a longer run or a higher scalar LR might still let the phase decay. The
rest of the network adapts around the suppression rather than removing it.

The feedback gate stays mostly open (LM 0.96; recall 0.83–0.88). The shuffled-buffer control learns to close its
gate more (LM 0.84; recall 0.61–0.72) and to shrink λ (0.76 vs 0.94). It recognises that the shuffled content is
less useful.

## Verdict

1. **LM:** the standard transformer is best (1.186 bpb).
   - Every new layer costs bpb at matched params and tokens, significantly on both seeds:
     ChunkRoPE +0.011 (p ≈ 0.008), phase_kv +0.018, phase + tracking +0.021, phase_kv + feedback +0.022,
     shuffled feedback +0.028, phase_head +0.054 (p < 1e-11).
   - Feedback from the real suppressed stream is slightly better than the shuffled-buffer control (1.209 vs 1.215;
     the per-seed sign is mixed).
2. **Recall (the alternation has a real job):** no phase arm beats the baseline on suppressed-phase (cross-phase)
   pairs.
   - phase_kv −1.4 pp; phase_kv + feedback −2.5 pp (30 of 32 batches worse, p ≈ 2e-7); shuffled −2.1 pp;
     phase_head −5.2 pp.
   - phase + tracking is the only arm level with the baseline on cross-phase pairs (+0.5 pp, p ≈ 0.05; +1.8 on seed
     0, −0.9 on seed 1). It pays −2.2 pp on same-phase pairs.
   - The feedback path does **not** recover suppressed-phase memory: it is no better than the shuffled buffer on
     recall (56.7 vs 57.1).
   - ChunkRoPE trades a large lag-4 gain (64 vs 51%) for −11 pp on same-phase lags.
3. **Compute:** real attention skipping exists only for the hard, fixed phase without feedback. Even that halves
   the score FLOPs without beating a fused causal kernel in wall-clock time at T ≤ 2048. The learnable soft versions
   cost more compute, not less.

**Bottom line:** at 6.4M params and about 6M training bytes, built from scratch, the alternating-suppression layers
are trainable and stable: they don't collapse. But they give no accuracy, memory or speed advantage over a plain
transformer of the same size. The most interesting signal is phase + tracking holding cross-phase recall level with
the baseline while plain phase arms lose it, which suggests the slow tracking state does carry information across
phases. It is small, seed-dependent and not a net win.

Not run (time): a third seed, larger models or longer training, and a sparse kernel that would make gate-closed
positions actually skip work.

## Tests

`pytest -q tests/test_phase_model.py tests/test_model.py` → **26 passed**. They check:
- causality of every arm (future tokens can't change past logits), including the causal shuffled control;
- gradient flow;
- supp = 1 equals standard attention (kv and head modes);
- the hard split ("skip") path equals the hard mask;
- feedback with λ = 0 equals phase kv;
- ChunkRoPE with blend → 0 equals RoPE;
- the sharp soft clock matches the hard clock;
- param matching within 0.5%;
- the recall-task layout.

`tests/test_qwen_arch.py` belongs to the historical `qwen_arch/` package and needs transformers plus the Qwen
weights, so it is not part of this work.
