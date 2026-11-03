# Phase attention in Qwen3.8-27B: PyTorch + llama.cpp (`qwen35_phase`), tested on a tiny copy

Status (2026-11-02): code for both paths is built and tested **only on a tiny random-init model with the Qwen3.8-27B
structure**. The real 27B weights were not downloaded, nothing was trained at scale, and no GPU or ROCm was used.

## 1. Target architecture (from Qwen/Qwen3.8-27B config.json)
- 64 layers = 16 × (3 × Gated DeltaNet + 1 × gated full attention), hidden 5120, FFN 17408, vocab 248320
- full attention: 24 Q / 4 KV heads, head_dim 256, sigmoid output gate (`attn_output_gate`), partial RoPE 0.25,
  interleaved MRoPE sections [11, 11, 10], theta 1e7
- DeltaNet: 16 QK / 48 V heads, head dim 128, conv kernel 4

**Phase attention only applies to the 16 full-attention layers.** The 48 DeltaNet layers have no score matrix. For them
there is an experimental, default-OFF *phase write gate* (scales DeltaNet's write strength β by phase).

## 2. Tiny copy (`qwen38/tiny.py`)
HF `transformers==5.22.1` `Qwen3_5ForCausalLM` (text only; real qwen3_5 classes, not our own modeling), eager attention.
8 layers = 2 × (3 lin + 1 full), hidden 64, FFN 128, **6 Q / 1 KV heads** (kept the real 6:1 GQA ratio instead of the
4:1 in the brief), head_dim 32, rotary dim 8 → MRoPE sections [2, 1, 1], DeltaNet 2 QK / 6 V heads of dim 16 (real 1:3),
vocab kept at 248320 so the real Qwen3.8 tokenizer works (only tokenizer files were downloaded, ~23 MB).
32.2M params (almost all embeddings), MTP off (`mtp_num_hidden_layers=0`, converted with `--no-mtp`).

## 3. Design
Notation: t = absolute position, clock s(t) ∈ [0,1] (reused `phase_model.layers.PhaseClock` parameters):
soft s = ½(1 − tanh(k·sin(π(t + ½ + φ)/P))), hard s = ⌊t/P⌋ mod 2. A[q,k] = s_q s_k + (1−s_q)(1−s_k).

| piece | PyTorch (`qwen38/phase.py`) | llama.cpp (`qwen35_phase`) |
|---|---|---|
| kv mode | score += log(1 − (1−sp_h)(1−A)) per head (sp_h=1 → exactly 0); hard: −inf where A<½ | `kq_b` bias from the new F32 position inputs (`phase_q_pos` = ubatch positions, `phase_kv_pos` = KV cell positions), passed to the existing `build_attn(..., kq_b, ...)` (non-flash path; a non-null `kq_b` turns flash attention off for that layer) |
| head mode | y_h,t ×= 1 − (1−act)(1−sp_h), act = g_h s_t + (1−g_h)(1−s_t), g_h = h mod 2 | same multiply on the attention output before the sigmoid gate; g_h is stored as tensor `phase_hgrp`; works with flash attn |
| feedback (restore, harmony-or-boundary gate) | reused `phase_model.layers.Feedback`: h += λ·g·U D(ysup), ysup = other-phase-only attention → output gate → o_proj | **TODO**: loader refuses `feedback_rank > 0`; converter only accepts λ = 0 |
| tracking | reused `phase_model.layers.Tracking` (EMA of the residual, harmony gate) | **TODO** (needs per-sequence EMA state across ubatches) |
| DeltaNet write gate (experimental, OFF) | wraps `in_proj_b`: β' = σ(b)·(1 − (1−act)(1−sw_h)) | multiply after `beta_sigmoid`; tensors `phase_wsupp`, `phase_wgrp` (V heads reordered grouped→tiled like `ssm_beta`) |

Gradients through the suppressed-stream pass are kept finite: queries with no other-phase key use a dummy mask and are
zeroed afterwards. Without this, `nan_to_num` hid NaN grads; a test now catches this.

### GGUF metadata (arch `qwen35_phase`, all per-layer arrays of length block_count)
`qwen35_phase.phase.mode` (0 off / 1 kv / 2 head), `.phase.hard`, `.phase.period`, `.phase.sharpness`, `.phase.offset`,
`.phase.delta_write`, plus reserved u32 `.phase.feedback_rank` / `.phase.tracking_rank` (must be 0).
Tensors: `blk.N.phase_supp` [n_head], `blk.N.phase_hgrp` [n_head], `blk.N.phase_wsupp` / `blk.N.phase_wgrp` [n_v_heads].

### llama.cpp change set (`llamacpp/qwen35_phase.patch`, 15 files, +428/−8, base ggml-org master `2cbca389f`)
- new arch enum/name + added to every `QWEN35` list (hybrid memory, rs rollback, IMROPE, MTP wiring); same model class as qwen35
- `llama_hparams`: phase arrays; `llama_layer`: 4 phase tensors; KV keys + tensor names/infos
- `llm_graph_input_phase_pos` + `llama_kv_cache::set_input_kv_pos` (single stream only, asserts otherwise)
- `models/qwen35.cpp`: `build_phase_clock`, `build_phase_kq_bias`, `build_phase_head_weight` built from stock ggml ops
  (scale_bias, sin, tanh, floor, mul/add, repeat, clamp, log). All of them also exist in the CUDA backend, which the HIP
  build compiles, so ROCm should need no new kernels. **Not tested.**
- `gguf-py` constants and `conversion/qwen35_phase.py` (`Qwen3_5PhaseForCausalLM`; reads `phase_config` from config.json
  and `phase.safetensors`)
The fork lives only at /workspace/llama.cpp-phase, branch `qwen35-phase`. It was not pushed and no PR was opened.

## 4. What was tested (all numbers from runs on 2026-11-02)
PyTorch, `tests/test_qwen38.py` (12 passed). The older suites `tests/test_model.py` and `tests/test_phase_model.py` still pass (26 passed).
- strength 0 is **bitwise identical** (`torch.equal`) to the stock eager tiny model for kv, head, kv+feedback+tracking (λ=0), and kv+write-gate(off)
- inactive wrappers are bitwise stock; active with supp 0.01 differs
- hard mask: clock = ⌊t/4⌋ mod 2 exactly; −inf exactly on cross-phase pairs; hard-kv output == stock attention with an explicit same-phase mask (atol 1e-6)
- soft bias == log(sp + (1−sp)A); head-mode weights hard → 0/1 per head group
- write gate: sw=1 ≈ identity (max 2.2e-5, from the logit round trip); sw=0.1 changes logits (max diff 6.1)
- frozen base: base params get no grad, every new param group gets a finite grad, new params < 1% of base

Frozen-base training smoke (`scripts/qwen38_smoke_train.py` → `results/qwen38_smoke_train.json`): 30 steps, batch 2×64
TinyStories tokens (real Qwen tokenizer), AdamW lr 1e-2 on 4,132 new params (base 32.2M frozen, verified unchanged),
loss = KL + 0.2·LM + 0.01·gate. Runs in 9.7 s on CPU; KL 6.0e-5 → 8.5e-3, gate open 0.31 → 0.48, mean supp 0.90 → 0.96.
The base is random, so this only shows that the training loop runs. It says nothing about quality.

llama.cpp vs PyTorch (`scripts/qwen38_gguf_check.py` → `results/qwen38_gguf_check.json`). 64 TinyStories tokens,
F32 GGUFs (133 MB each). Each GGUF was decoded as one ubatch, as 7-token chunks, and token by token (chunk 1, through the
KV cache and recurrent state). Max |logit diff| over all 64 × 248320 logits (reference |logit| max = 4.47):

| model | torch: phase vs stock | llama.cpp: phase vs stock | f32 KV, no FA | f32 KV, FA on | f16 KV (default), any |
|---|---|---|---|---|---|
| stock `qwen35` | 0 (reference) | n/a | 9.5e-05 | 3.0e-03 | 3.0e-03 |
| `kv_soft_strength0` | 0.000 | 0.000 | 9.5e-05 | 9.5e-05 | 1.7e-03 |
| `kv_soft` | 2.485 | 2.485 | 9.8e-05 | 9.8e-05 | 4.1e-03 |
| `kv_hard` | 3.218 | 3.218 | 1.0e-04 | 1.0e-04 | 2.7e-03 |
| `head_soft` | 3.013 | 3.013 | 8.6e-05 | 3.0e-03 | 3.0e-03 |
| `kv_soft_delta_write` | 6.631 | 6.631 | 8.0e-05 | 8.0e-05 | 1.8e-03 |

- phase-on changes llama.cpp's outputs by the same amount as PyTorch's (columns 2 and 3 agree to 3 decimals), and argmax
  agrees on 100% of positions for every variant. **Hard-mask equivalence holds:** kv_hard matches PyTorch to 1.0e-4.
- kv variants give the same number with "FA on" because a non-null `kq_b` makes llama.cpp use the non-flash path for
  those layers. head mode keeps flash attention, so it shows the same f16-cast error as stock.
- The ~1e-4 floor is the same for stock `qwen35`, so it is not caused by the phase code. The 1–4e-3 errors come from the
  f16 KV cache or flash attention's f16 cast.

## 5. TODO / known limits
- tracking + feedback in llama.cpp (needs recurrent EMA state in the hybrid memory and a second other-phase attention pass)
- soft kv mode materialises a [n_kv, n_tokens, n_head] bias and disables flash attention on those layers. For long
  contexts this needs a fused flash-attention variant (per-head bias, or the clock evaluated in-kernel).
- multi-stream / multi-sequence KV (`set_input_kv_pos` asserts n_stream == 1); MTP head has no phase
- PyTorch kv mode uses eager attention (O(T²) memory per full layer); KV-cache decoding in the wrapper assumes a
  contiguous cached prefix (training path uses `use_cache=False`)
- tiny model only. Loading the real `Qwen3_5ForConditionalGeneration` checkpoint into `Qwen3_5ForCausalLM` (skipping vision) is untested

## 6. ROCm / HIP build (not run here: no AMD GPU and no ROCm on the box)
`cmake -DGGML_HIP=ON` fails here as expected: "Failed to find HIP root directory". On a machine with ROCm (6.x/7.x, which is a multi-GB install):
```bash
bash llamacpp/build_cpu.sh            # clone + apply patch (or reuse the branch)
cd /workspace/llama.cpp-phase
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \   # gfx1100 = RX 7900; gfx942 = MI300
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-hip -j --target llama
# then compile llamacpp/phase_logits.cpp against build-hip/bin and rerun scripts/qwen38_gguf_check.py with -ngl 99
```
What to check on the GPU: `test-backend-ops` for SIN/FLOOR/LOG/CLAMP/SCALE(bias) on HIP, then the same logit table with
layers offloaded.

## 7. Real-27B recipe for one 24–48 GB GPU (planned, NOT run)
1. Weights: BF16 is 55.6 GB, so it does not fit. Load the frozen base in 4-bit (bitsandbytes NF4, roughly 15–17 GB) on CUDA,
   or in bf16 on ≥ 64 GB. Install `flash-linear-attention` + `causal-conv1d`, otherwise DeltaNet falls back to slow torch code.
2. `PhaseQwen38(model, PhaseSpec(mode="kv", supp=0.9, feedback={"rank":32,"lam":0.0}, tracking={"rank":32,"lam":0.0}))`
   wraps the 16 full-attention layers. New params ≈ 16 × (24 + 2·2·5120·32 + gate scalars) ≈ 10.5M (<0.04% of 27B).
3. Train only the new params with KL(stock‖phase) + 0.2·LM + gate penalty. Use `set_active(False)` for the teacher pass
   (same weights, no second copy). Sequences of 1–2k tokens with gradient checkpointing. **Chunk the KL/LM over tokens**,
   because 248k-vocab logits are about 1 GB per 1k tokens in fp32.
4. Keep the controls from the earlier work: matched-parameter identity, shuffled buffer, random gate, plus paired tests
   over seeds. Earlier frozen-Qwen3-0.6B results were negative, so treat 27B as a test, not an expected win.
5. Export: `qwen38.export.save_phase` (only works for kv/head/write gate; λ must be 0 until llama.cpp supports
   tracking/feedback) → `convert_hf_to_gguf.py --no-mtp` → `llama-quantize` (phase tensors are 1-D and stay F32).

## Reproduce
```bash
V=/home/box/venvs/phase-hf/bin/python      # transformers 5.22.1, torch 2.14.1+cpu, cmake/ninja/ziglang via pip
$V -m pytest tests/test_qwen38.py -q
$V scripts/qwen38_smoke_train.py
CC=/home/box/zigcc/zcc CXX=/home/box/zigcc/zc++ PATH=/home/box/venvs/phase-hf/bin:$PATH bash llamacpp/build_cpu.sh
$V scripts/qwen38_gguf_check.py           # writes tiny GGUFs to /workspace/qwen38-tiny (≈0.8 GB, not committed)
```
