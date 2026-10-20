# Zero-training architecture surgery on Qwen3-0.6B

Setup: Qwen/Qwen3-0.6B, fp32, CPU with 3 torch threads. **No training of any kind.** Perplexity is measured on WikiText-2 raw test, using the first 32,768 tokens as 64 windows of 512. Generations are greedy, chat template with `enable_thinking=False`, 40 new tokens.

**TPS is noisy.** The CPU was shared with a 6-core job from another project, so treat speed differences under about 20% as noise. Every modified layer here costs at most a few percent more FLOPs than stock. `kv` mode builds an explicit T×S bias mask, and `head` mode does one extra scaling.

## Exactness

With every new layer disabled (`stock`), and with PhaseAttention at its identity setting `gain=supp=1` on all 28 layers, the logits are **bitwise equal** to HF `Qwen3ForCausalLM`:

* `torch.equal` is True and the max absolute difference is 0.0.
* Perplexity is 28.912, the same as the earlier krystal-qwen runs.

## Verdicts (zero training)

* **Most promising, near-free modifications** (within 0–3% of stock and the text reads normally):
  * `track_phase_mid` +0.5%
  * `track_harmony_mid` +1.0%
  * `track_always_mid` +1.3%
  * `phase_kv_soft_mid` +2.1%
  * `combo_kvsoft_track` +3.1%

  The pretrained model tolerates a soft access-gain boost (2× attention mass on in-phase keys) and a weak slow-state coupling.
* **Hard time-multiplexing breaks the pretrained model.** Letting only the access stream write gives about 4× perplexity (kv mode 112–116; head mode 100), and the generations become empty or degenerate. The pretrained heads were never trained to see half their context at a time. This configuration needs training (or at least distillation) to be usable.
* **The suppressed stream can be decoded, and it carries different content from the access stream.** With hard kv suppression at P=4, the suppressed buffer at layer 14 identifies the token at the most recent suppressed-phase position 28.5% of the time. The access stream manages 21.2%, and the majority baseline is 9.7%. For the most recent access-phase token the pattern reverses: 10.9% from the suppressed buffer against 39.3% from the access stream. That is the double dissociation we wanted. Head-group alternation shows no such dissociation (16–19% for both), because both head groups attend to the same keys.
* **ChunkRoPE:**
  * Applied to all layers, it hurts in proportion to C: C=2 gives +51% and C=4 gives +155%.
  * Applied only to the upper layers 14–27 with C=4, it costs +9.9% and the text stays fluent.
  * It makes the model less sensitive to within-chunk reversal (62× damage for stock, 13–23× for ChunkRoPE), but reversed context is still unusable (perplexity above 690). Pretrained token LMs are not robust the way speech perception is.

| variant | layers / settings | WikiText-2 ppl | Δ vs stock | prefill tok/s | decode tok/s | greedy reply to 'why is the sky blue' (first 90 chars) |
|---|---|---|---|---|---|---|
| stock | – | 28.91 | +0.0% | 348 | 12.9 | The sky appears blue because of the way sunlight interacts with the Earth's atmosphere, sc |
| phase_kv_identity_all | all: type=phase, mode=kv, period=4, gain=1.0, supp=1.0 | 28.91 | +0.0% | 350 | 12.8 | The sky appears blue because of the way sunlight interacts with the Earth's atmosphere, sc |
| phase_kv_soft_mid | 8-19: type=phase, mode=kv, period=4, gain=2.0, supp=1.0 | 29.53 | +2.1% | 310 | 12.7 | The sky appears blue because of the scattering of sunlight by the atmosphere, which causes |
| phase_kv_hard_mid | 8-19: type=phase, mode=kv, period=4, gain=1.0, supp=0.0 | 116.48 | +302.9% | 356 | 12.3 |  |
| phase_kv_hard_mid_P16 | 8-19: type=phase, mode=kv, period=16, gain=1.0, supp=0.0 | 112.62 | +289.5% | 320 | 4.8 |  |
| phase_kv_hard_all | all: type=phase, mode=kv, period=4, gain=1.0, supp=0.0 | 111.66 | +286.2% | 286 | 9.9 | The blue </think>  The blue is a color blue. |
| phase_head_hard_mid | 8-19: type=phase, mode=head, period=4, gain=2.0, supp=0.0 | 99.61 | +244.5% | 182 | 9.1 | Okay, let's break this down. The user is asking about the sky. So, the first part is about |
| phase_head_soft_mid | 8-19: type=phase, mode=head, period=4, gain=1.5, supp=0.5 | 35.47 | +22.7% | 278 | 9.8 | The sky is blue because of the way light interacts with the atmosphere, where shorter wave |
| chunk2_all | all: type=chunk_rope, chunk=2 | 43.76 | +51.3% | 285 | 11.7 | The sky appears blue because of the scattering of sunlight by the Earth's atmosphere, whic |
| chunk4_all | all: type=chunk_rope, chunk=4 | 73.60 | +154.6% | 289 | 12.0 | The sky is blue because of the Earth's atmosphere. |
| chunk8_all | all: type=chunk_rope, chunk=8 | 101.07 | +249.6% | 277 | 11.8 | <think> </think>  The sky appears blue because of the blue color is due to the way of the  |
| chunk4_upper | 14-27: type=chunk_rope, chunk=4 | 31.76 | +9.9% | 286 | 11.6 | The sky appears blue because of the way light interacts with the Earth's atmosphere, where |
| track_harmony_mid | every4@8-19: type=tracking, alpha=0.9, lam=0.05, gate=harmony, tau=0.5 | 29.22 | +1.0% | 286 | 10.0 | The sky appears blue because of the scattering of sunlight by the Earth's atmosphere, whic |
| track_always_mid | every4@8-19: type=tracking, alpha=0.9, lam=0.05, gate=always | 29.29 | +1.3% | 272 | 10.0 | The sky appears blue because of the scattering of sunlight by the Earth's atmosphere, whic |
| track_phase_mid | every4@8-19: type=tracking, alpha=0.9, lam=0.05, gate=phase, period=4 | 29.06 | +0.5% | 279 | 10.1 | The sky appears blue because of the way sunlight interacts with the Earth's atmosphere, sc |
| combo_kvsoft_track | 8-19: type=phase, mode=kv, period=4, gain=2.0, supp=1.0; every4@8-19: type=tracking, alpha=0.9, lam=0.05, gate=harmony, tau=0.5 | 29.82 | +3.1% | 270 | 10.6 | The sky appears blue because of the scattering of sunlight by the atmosphere, which causes |

**Chunk-reversal robustness** (ppl on chunk-end targets, first 8 windows; context reversed inside chunks of C)

| variant | C | intact | reversed | ratio |
|---|---|---|---|---|
| stock | 4 | 29.8 | 1858.5 | 62.3× |
| chunk2_all | 2 | 44.1 | 693.4 | 15.7× |
| chunk4_all | 4 | 77.9 | 1776.1 | 22.8× |
| chunk8_all | 8 | 153.9 | 1984.2 | 12.9× |
| chunk4_upper | 4 | 31.9 | 1709.6 | 53.6× |

**Suppressed-stream decoding** (ridge probe at layer 14, 8 windows, top-40 token classes, 60/40 split)

| variant | target | majority | from suppressed buffer | from access stream |
|---|---|---|---|---|
| phase_kv_soft_mid | token_at_last_suppressed_pos | 9.7% | 26.6% | 22.2% |
| phase_kv_soft_mid | token_at_last_access_pos | 9.7% | 18.0% | 36.7% |
| phase_kv_hard_mid | token_at_last_suppressed_pos | 9.7% | 28.5% | 21.2% |
| phase_kv_hard_mid | token_at_last_access_pos | 9.7% | 10.9% | 39.3% |
| phase_kv_hard_mid_P16 | token_at_last_suppressed_pos | 5.5% | 12.0% | 8.4% |
| phase_kv_hard_mid_P16 | token_at_last_access_pos | 9.5% | 11.9% | 39.1% |
| phase_head_hard_mid | token_at_last_suppressed_pos | 9.7% | 15.7% | 17.1% |
| phase_head_hard_mid | token_at_last_access_pos | 9.7% | 19.0% | 18.7% |

stock vs HF Qwen3ForCausalLM (64 tokens): {'max_abs_diff': 0.0, 'bitwise_equal': True, 'argmax_agree': 1.0}

phase_kv_identity_all vs HF Qwen3ForCausalLM (64 tokens): {'max_abs_diff': 0.0, 'bitwise_equal': True, 'argmax_agree': 1.0}

---

# Feeding the suppressed stream back through tracking (`suppressed_tracking`), zero training

The setup is the same as above: 32,768 WikiText-2 tokens, fp32, 3 threads on a shared CPU. Stock perplexity is **28.91**. Hard phase (`kv`, P=4, supp=0, layers 8–19) is **116.48**. Soft phase (gain=2, supp=1, layers 8–19) is **29.53**. Every feedback layer sits on layers 8–19 unless noted. "gate" is the fraction of steps coupled, averaged over layers in the last window.

**Screening (2,048 tokens, first 4 windows, before the full run):**
* The literal design, where the tracking state absorbs the buffer and the residual is pulled toward it (`pull`), only made hard phase worse: 137 to 165,000 depending on λ and α.
* Adding back the raw suppressed output (`inject`) also failed: 116 at best, and 1,700+ at λ=1.
* Both double-count. The suppressed content has to come back at its **stock attention share**, which is the `restore` coupling. That uses the buffer plus its partition mass, applied between attention and MLP.

| variant | ppl | Δ vs stock | gate | prefill / decode tok/s | sample (sky / fraction / lighthouse) |
|---|---|---|---|---|---|
| hard phase alone (from above) | 116.48 | +303% | – | 356 / 12.3 | empty / garbled / "quiet quiet quiet" |
| **hard + restore, always** (upper bound) | **28.91** | 0.0% | 1.00 | 188 / 9.2 | identical to stock |
| **hard + restore, harmony τ=0.5** | **30.36** | +5.0% | 0.64 | 190 / 7.4 | fluent; the fraction answer starts slightly differently |
| hard + restore, harmony τ=0.7 | 31.26 | +8.1% | 0.51 | 189 / 10.3 | fluent; a different story opening |
| **hard + restore, harmony OR boundary τ=0.5** | **29.21** | **+1.0%** | 0.75 | 184 / 9.5 | ≈ stock |
| hard + restore, phase boundary only | 58.16 | +101% | 0.25 | 191 / 9.8 | empty sky answer, wrong fraction |
| CTRL: hard + restore, Bernoulli p=0.62 (duty matched to harmony τ=0.5) | 34.86 | +20.6% | 0.62 | 187 / 9.5 | OK but repetitive |
| CTRL: hard + restore, Bernoulli p=0.25 (duty matched to boundary) | 56.56 | +95.6% | 0.26 | 194 / 9.2 | empty sky answer |
| CTRL: hard + restore, **shuffled** buffer, harmony | 255.27 | +783% | 0.58 | 185 / 9.4 | garbage |
| CTRL: hard + restore, **random** norm-matched buffer, harmony | 290.79 | +906% | 0.63 | 181 / 9.7 | garbage |
| hard + pull (tracking absorbs buffer), λ=0.1, α=0 | 129.69 | +349% | 1.00 | 214 / 8.0 | broken |
| CTRL: hard + pull without buffer, λ=0.1, α=0 | 116.48 | +303% | 1.00 | 287 / 10.2 | same as hard. *This is degenerate: with α=0 the inclusive EMA equals h, so the layer is the identity.* |
| **soft + restore, harmony τ=0.5** | **28.93** | **+0.06%** | 0.76 | 181 / 9.2 | ≈ stock |
| soft + pull with buffer (mix 0.25), every 4th layer of 8–19, λ=0.05, α=0.9, harmony | 29.70 | +2.7% | 0.51 | 189 / 7.9 | fluent |
| CTRL: soft + pull without buffer, same settings | 29.79 | +3.0% | 0.91 | 256 / 8.9 | fluent |

**Probe at layer 14 (hard + restore harmony τ=0.5; same probe as above):**
* Token at the last suppressed-phase position: suppressed buffer **25.9%**, access stream 24.0%, majority 9.7%.
* Token at the last access-phase position: suppressed buffer 16.9%, access stream **39.9%**.
* So the double dissociation survives the feedback. It is a little weaker than hard phase alone (28.5 / 21.2) because the restored content now also reaches the access stream on coupled steps.

## Verdict

* **(a) Feedback rescues hard phase suppression, but only in `restore` form.** Hard phase goes from 116.5 to **30.4** coupling on 64% of steps by harmony, and to **29.2 (+1%)** with harmony-or-boundary at 75%.
  * The **harmony gate is better than chance.** At the same duty cycle, random Bernoulli gating gives 34.9, and harmony gives 30.4.
  * The **phase-boundary gate on its own is no better than random** (58.2 against 56.6 at about 25%).
  * The **content matters.** Shuffled or random buffers fed through the same coupling give 255–291, which is far worse than no feedback at all.
  * The `pull` and `inject` couplings, where the tracking state absorbs the buffer and the residual is pulled toward it, do **not** rescue anything (130 and above).
* **Caveat.** `restore` uses the suppressed keys' stock attention mass, and λ=1 on every step is exactly stock attention (a tested identity). So the rescue measures *how few steps need the hidden stream's content*, about 64–75% of steps. It is not free context. In this implementation every step still computes both streams, so there is **no speed gain yet**. Prefill drops from about 350 to about 185 tok/s, because the restore path adds an explicit score matrix and two extra attention passes per layer. TPS is noisy on the shared CPU.
* **(b) Soft phase stays within 3%.** Soft + restore harmony costs **+0.06%**. Soft + pull with the buffer costs +2.7%, against +3.0% for the same tracking without the buffer. The buffer helps only marginally there.

## Trainable set for a frozen-base, new-layers-only follow-up

The base Qwen3 weights stay frozen. These would become `nn.Parameter`s:

| Layer | Trainable parameters |
|---|---|
| PhaseAttention | `gain` and `supp` per layer (and per head); phase `offset`; a soft period, e.g. a learned sinusoidal phase; `out_gain` |
| suppressed_tracking | `lam`, `mix`, `alpha` (sigmoid-parameterized), `tau`, `sharp` per layer; optionally a small low-rank adapter on the side buffer (D→r→D) instead of the fixed o_proj restore |
| tracking | `lam`, `alpha`, `tau`, `sharp` |
| ChunkRoPE | a learnable interpolation between t and ⌊t/C⌋·C |

That is a few hundred scalars per model, or about 0.5–2M parameters with low-rank side adapters. The objective would be distillation (KL to stock logits) plus LM loss, with a sparsity or compute penalty on the gate rate.
