# phase_model — a decoder-only LM built from scratch with the new layers

**Nothing from Qwen.** There are no pretrained weights, no Qwen tokenizer and no surgery. `phase_model/` is a fresh
PyTorch implementation trained from random init on CPU. `qwen_arch/` stays in the repo only as history (the
earlier surgery / frozen-base experiments) and is not imported by anything here.

## Package layout

| file | contents |
|---|---|
| `phase_model/config.py` | `ModelConfig` (vocab, d_model, n_layers, n_heads, mlp_hidden, rope_theta, blocks). `BlockSpec(mixer, position, extras)`. `layer_specs(n_layers, rules)` with selectors `all`, `2-5`, `0,3,7`, `every2`, `every2@1`; later rules win |
| `phase_model/tokenizer.py` | `ByteTokenizer`: UTF-8 bytes, vocab 256, so loss/ln2 is bits-per-byte |
| `phase_model/layers.py` | mixers, positions, phase clock, gate, feedback, tracking (below) |
| `phase_model/model.py` | `PhaseLM`: embedding → N pre-norm blocks → RMSNorm → tied head. `phase_report()` |
| `phase_model/data.py` | TinyStories bytes (train/val) and the phase-aligned associative-recall generator |
| `phase_model/stats.py` | paired sign test and Wilcoxon signed-rank test |

Block: `h = x + Mixer(norm(x))`; then pre-MLP extras (feedback); `h = h + SwiGLU(norm(h))`; then post-MLP extras (tracking).

## Selectable per-block components

**Mixers** (`BlockSpec.mixer["type"]`)
- `attn`: standard causal multi-head attention (fused causal kernel).
- `phase`: PhaseAttention. Every phase variant has a **phase clock**
  `s(t) = ½(1 − tanh(k·sin(π(t + ½ + φ)/P)))` with learnable offset φ, period P (init 4) and sharpness k (init 4).
  `hard: true` replaces it with exact ⌊t/P⌋ mod 2 and no parameters.
  - `mode: kv`: keys whose phase differs from the query's are down-weighted. A(t, s) is the soft same-phase
    indicator and the attention bias is `log(supp_h + (1 − supp_h)·A)`, with a learnable per-head `supp`
    (init 0.05). supp = 1 is exactly standard attention (tested). Hard mode gives supp = 0. With a feedback extra,
    the mixer also exposes the **suppressed stream** `ysup`: attention restricted to other-phase keys, projected by
    the same output matrix.
  - `mode: head`: head group h % 2 is active in phase h % 2. The inactive group's output is scaled by `supp_h`.
  - `skip: true` (hard kv, no feedback): the sequence is split into its two phase sub-sequences and each runs
    through a causal kernel. Same-phase pairs are the only scores actually computed (about ¼ T² vs ½ T² for causal
    attention). The result equals the masked computation (tested).

**Positions** (`BlockSpec.position`)
- `rope`: standard RoPE.
- `chunk_rope`: positions `(1 − β)·t + β·⌊t/C⌋`, with a learnable blend β = σ(blend_logit) (init 0.5, C = 4).

**Extras** (`BlockSpec.extras`)
- `feedback`: suppressed-stream buffer → restore. `h ← h + λ·g_t·U·D(ysup_t)` (rank 32, λ learnable).
  - **Gate g:** `harmony` = σ(k_g·(cos(ysup_t, EMA(ysup)_{t−1}) − τ)); `boundary` = |s_t − s_{t−1}|;
    `either` = max of the two; or `always`. α, τ, k_g are learnable.
  - **`source: shuffled`:** a causal control. Each t receives ysup from a random earlier s ≤ t, so the statistics
    are right but the content is wrong.
- `tracking`: slow state m = EMA_α(h) (causal, computed as an EMA matrix, α learnable, init 0.95).
  `h ← h + λ·g_t·U·D(m_t)` with a harmony gate between h_t and m_{t−1}.

**Parameter matching:** `scripts/scratch_train.py: build()` adjusts the shared MLP hidden size so every arm is within
0.5% of the baseline's parameter count (tested).

## Experiment configuration (`scripts/scratch_train.py`, queue `scripts/run_scratch.sh`)

- **Model:** d = 256, 8 layers, 8 heads, SwiGLU (hidden ≈ 688, adjusted per arm), tied embeddings. About 6.39M
  params for every arm.
- **Arms:**
  - `baseline`
  - `phase_kv_fb`: phase kv in all layers + feedback with the either gate
  - `phase_head`
  - `phase_track`: phase kv + tracking every 2nd block
  - `chunkrope`
  - `CTRL_shuffled_fb`
  - `phase_kv`: no feedback, which isolates the feedback's effect
- **(a) LM:** TinyStories V2 (GPT-4) bytes. Training uses only the first 40 MB of the official **train** file; the
  validation file is used for eval only (64 × 256 windows). 1500 steps × 16 × 256 bytes ≈ 6.1M training bytes.
- **(b) Associative recall:** 128 tokens = 32 blocks of P = 4. Each sequence holds 10 pairs.
  - A store block is `[f, key, value, f]`. The query block, L blocks later, is `[QUERY, key, value*, f]`.
  - Loss is taken only at the answer. Accuracy uses 2048 held-out sequences generated from a separate seed.
  - L ∈ {1, 2, 3, 4, 7, 8, 15, 16} blocks (4 to 64 tokens). Odd L stores the pair in the phase **opposite** the
    query, which is the suppressed phase for kv phase attention, so the alternation has a real job. Even L is the
    same-phase contrast.
  - 1000 steps × 64 sequences.
- **Training:** AdamW (β 0.9/0.95, wd 0.1 on matrices), lr 1e-3, 100-step warmup, cosine to 10%, grad clip 1.0.
  Identical tokens/steps/schedule for every arm. Seeds 0 and 1. 4 threads per run, two runs at a time. Resumable
  checkpoints in `ckpt/scratch/`. Each run writes `results/scratch/<task>_<arm>_s<seed>.json` and is committed
  as soon as it finishes.
- **(c) Compute:** `scripts/bench_skip.py` measures forward tok/s at T = 256/1024/2048 and the attention score
  entries each variant actually computes.

Results are in `RESULTS_SCRATCH.md`.
