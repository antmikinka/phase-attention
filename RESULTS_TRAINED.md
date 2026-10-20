# RESULTS_TRAINED — training ONLY the new layers (Qwen3-0.6B frozen)

**Setup.** All Qwen3-0.6B weights are frozen and shared with the stock teacher. Only the new-layer parameters train
(see `docs/ARCHITECTURE.md` §5). Loss = KL(stock ‖ student) + 0.2·CE + gate_w·mean(gate). Data: WikiText-2 **train**
split only. 200 steps × 2 × 256 tokens = 102,400 train tokens per arm. Adam: lr 0.02 for scalars, 1e-3 for the
rank-8 adapter. CPU, 8 threads, about 16 min per arm. Eval: the first 32,768 WikiText-2 **test** tokens (64 × 512
windows), the same windows as `RESULTS_SURGERY.md`. Paired tests are per window: exact sign test and Wilcoxon
signed-rank. Script: `scripts/train_new_layers.py`, queue `scripts/run_trained.sh`, table `scripts/summarize_trained.py`.
Raw data: `results/trained.json`, `results/trained_vs_controls.json`. Learned params: `results/trained_params/*.pt`
(≤0.8 MB each). Training curves: `logs/train_<arm>.jsonl`.

All arms use layers 8–19 and period P = 4. "hard" = suppressed keys masked (supp = 0). "restore_either" = the
suppressed stream is fed back through suppressed_tracking (restore coupling, gate = harmony OR boundary, rank-8 adapter).

**Two identity controls were added because the arms beat stock.** Training on WikiText-2 train with a CE term adapts
the new knobs to the domain. Per-head output gains and the adapter alone can lower WikiText ppl. Each control has the
same trainable knobs as the arms, but its phase starts as the identity (gain = supp = 1, no suppression at init):
- `CTRL_identity_trainable`: 197k params, matched to the restore arms.
- `CTRL_identity_small`: 576 params, matched to hard_alone / soft_track.

## Main table

Stock Qwen3-0.6B (same 64x512 windows): ppl 28.91

| arm | trainable params | untrained ppl | trained ppl | Δ vs stock | gate mean / open (trained) | prefill / decode TPS | sign p, Wilcoxon p vs stock | vs untrained | mean Δnll vs stock (wins/64) |
|---|---|---|---|---|---|---|---|---|---|
| hard_restore_either | 197,052 | 29.21 | **25.69** | -11.2% | 0.997 / 0.999 | 387 / 13.9 | 1.08e-19, 3.61e-12 | 1.08e-19, 3.61e-12 | -0.118 (64) |
| CTRL_hard_restore_either_shuffled | 197,052 | 274.05 | **32.66** | +13.0% | 0.541 / 0.584 | 327 / 17.0 | 7.05e-18, 4.79e-12 | 1.08e-19, 3.61e-12 | 0.122 (1) |
| hard_restore_either_gw1.0 | 197,052 | 29.21 | **28.24** | -2.3% | 0.298 / 0.249 | 394 / 13.8 | 0.00156, 1.28e-05 | 5.56e-10, 1.52e-09 | -0.0235 (45) |
| hard_restore_either_gw0.2 | 197,052 | 29.21 | **26.01** | -10.1% | 0.620 / 0.683 | 266 / 15.6 | 1.08e-19, 3.61e-12 | 1.08e-19, 3.61e-12 | -0.106 (64) |
| hard_restore_either_gw0.05 | 197,052 | 29.21 | **25.66** | -11.2% | 0.873 / 0.977 | 459 / 15.3 | 1.08e-19, 3.61e-12 | 1.08e-19, 3.61e-12 | -0.119 (64) |
| hard_alone | 420 | 116.45 | **27.56** | -4.7% | - | 623 / 16.9 | 1.08e-19, 3.61e-12 | 1.08e-19, 3.61e-12 | -0.0481 (64) |
| soft_track | 588 | 29.82 | **27.54** | -4.8% | 0.088 / 0.000 | 682 / 21.7 | 1.08e-19, 3.61e-12 | 1.08e-19, 3.61e-12 | -0.0487 (64) |
| chunk4 | 28 | 72.49 | **45.19** | +56.3% | - | 681 / 20.1 | 1.08e-19, 3.61e-12 | 1.08e-19, 3.61e-12 | 0.447 (0) |
| CTRL_identity_trainable | 197,244 | 28.91 | **25.15** | -13.0% | 0.997 / 0.999 | 520 / 18.0 | 1.08e-19, 3.61e-12 | 1.08e-19, 3.61e-12 | -0.14 (64) |
| CTRL_identity_small | 576 | 28.91 | **27.54** | -4.8% | - | 703 / 23.1 | 1.08e-19, 3.61e-12 | 1.08e-19, 3.61e-12 | -0.0487 (64) |

TPS is a single noisy CPU measurement: the same architecture ranges from 266 to 459 prefill tok/s. Stock in
RESULTS_SURGERY is 348 / 12.9. The implementation computes every partition (access, suppressed, stock-share), so
**no arm currently saves compute**. The gate rate is only a proxy for what a lazy implementation could skip.

## Versus the matched identity controls (the real test)

Mean Δnll is per token (negative = arm better). "wins" counts the windows (out of 64) where the arm is better.

| arm | vs `CTRL_identity_small` (27.54) | vs `CTRL_identity_trainable` (25.15) |
|---|---|---|
| hard_restore_either (25.69) | — | +0.021, 3/64 wins, sign p 5e-15 → **worse** |
| hard_restore_either_gw0.05 (25.66) | — | +0.020, 4/64, p 7e-14 → **worse** |
| hard_restore_either_gw0.2 (26.01) | — | +0.034, 3/64, p 5e-15 → worse |
| hard_restore_either_gw1.0 (28.24) | — | +0.116, 0/64 → worse |
| hard_alone (27.56) | +0.0007, 24/64, sign p 0.06, Wilcoxon p 0.16 → **indistinguishable** | — |
| soft_track (27.54) | +0.0001, 33/64, p 0.9 / 0.29 → **indistinguishable** | — |
| CTRL shuffled buffer (32.66) | — | +0.26, 0/64 → much worse |
| chunk4 (45.19) | +0.50, 0/64 → much worse | — |

## Gate penalty: ppl vs gate rate (restore arms)

| gate_w | gate mean | gate open frac | trained ppl | Δ vs stock 28.91 |
|---|---|---|---|---|
| untrained | 0.759 | 0.766 | 29.21 | +1.0% |
| 0 | 0.997 | 0.999 | 25.69 | −11.2% |
| 0.05 | 0.873 | 0.977 | 25.66 | −11.2% |
| 0.2 | 0.620 | 0.683 | 26.01 | −10.1% |
| 1.0 | 0.298 | 0.249 | 28.24 | −2.3% |

The "either" gate always opens on boundary steps (t % 4 == 0), so the open rate can't go below about 0.25. At
gw = 1.0 the gate is at that floor and only boundary positions are restored. That configuration still beats stock
(28.24, p = 1e-5), but it is worse than the same knobs without suppression (vs CTRL_identity_trainable +0.116 nats).
Its "below stock" therefore comes from domain adaptation. On the knob-matched comparison, every extra closed-gate
position costs ppl.

## What the training learned (from `values` in trained.json)

- **hard_alone** (learnable soft phase, init sharpness 20): sharpness collapsed from e^3.0 = 20 to about e^−0.5 ≈ 0.6
  in every layer, and the phase offset went to −0.5. The soft phase now ranges only over about 0.23–0.77. Cross-phase
  keys get a bias of about −1.0 against −0.4 for same-phase keys, so the masking became a mild ~0.6-nat preference.
  **When allowed, training removed the hard suppression.** The period stayed near 4 (e^1.07–1.35 ≈ 2.9–3.9).
  ppl went 116 → 27.6, matching the 576-param control.
- **soft_track**: log_gain ≈ +0.33 and log_supp ≈ +0.36 in each layer, so the access and suppressed weights are almost
  equal and the phase effect is gone. Tracking strength lam → 0.000. Training switched both new mechanisms off.
- **restore arms with gw = 0**: the gate opened to 99.9%. That is full restoration, i.e. stock attention plus the
  adapter and output gains.
- **shuffled-buffer control**: went from 274 to 32.7 by closing its gate to about 54% and learning to ignore the bad
  signal. It is still clearly worse than the real buffer (p = 1e-19 vs hard_restore_either), so the real
  suppressed-stream content matters.
- **chunk4** (ChunkRoPE blend, init β = 0.99): β fell to ~0.2–0.5 in most layers in only 200 steps. Seven layers stayed
  high (0.82–0.99: layers 2, 4, 15, 17, 20, 23, 26). The run had not converged (KL still 0.88) and ppl is 45 against
  72 untrained. Training pushes the positions back toward ordinary RoPE.

## Layer-14 probe (hard_restore_either, trained; 8 test windows; ridge decoder)

| target | majority | from suppressed buffer | from access stream |
|---|---|---|---|
| token at last suppressed-phase position | 9.7% | 28.5% | 26.4% (untrained run: 21.2%) |
| token at last access-phase position | 9.7% | 18.2% | 41.2% |

After training, the access stream carries much more of the suppressed-phase token (21.2% → 26.4%). The restore
coupling learned to put the suppressed information back. The buffer still decodes it best.
`hard_restore_either` and `gw0.05` are tied for the best phase arm (25.69 vs 25.66), so the probe was run only on the former.

## Generations (trained, greedy)

All arms except chunk4 stay fluent and on-topic. chunk4 mis-reads the arithmetic prompt ("3 + 1/8 + 1/4") and
makes a word error ("spirem"). Full text is in `results/trained.json` → `<arm>.trained.generations`.


- **hard_restore_either**: "The sky appears blue because of the scattering of sunlight by the Earth's atmosphere, which causes shorter wavelengths of light (blue) to be" / 'To add the fractions $ \\frac{3}{4} + \\frac{1}{8} $, we need a common denominator. The least common denominator of 4 and 8 is ' / 'The lighthouse keeper stood at the edge of the cliff, watching the horizon as the sun dipped be
- **CTRL_hard_restore_either_shuffled**: 'The sky is blue because of the way the sun emits a blue light, which is why the sky appears blue.' / 'The sum of $ \\frac{3}{4} + \\frac{1}{8} $ is:\n\n$$\n\\frac{3}{4} + \\frac{1}{8} =' / "**The Lighthouse Keeper's Story**  \n\nIn the heart of the coastal village of Eldermore, a lighthouse keeper’s life was defined by the steady "
- **hard_restore_either_gw1.0**: 'The sky appears blue because of the scattering of sunlight by the atmosphere. This scattering causes the sky to appear blue, making the sky ' / 'The sum of the fractions is:\n\n$$\n\\frac{3}{4} + \\frac{1}{8} = \\frac{6}{8} + \\frac{1}{8}' / 'The sea was calm, and the lighthouse stood tall on the cliffside, its beacon casting long shadows. The wind howled through 
- **hard_restore_either_gw0.2**: 'The sky appears blue because it contains a mixture of nitrogen and oxygen molecules, which scatter light in the visible spectrum. This scatt' / 'To add the fractions $ \\frac{3}{4} + \\frac{1}{8} $, we need a common denominator. The least common denominator (LCD) of 4 and ' / 'The lighthouse keeper stood at the edge of the cliff, watching the horizon as the sun di
- **hard_restore_either_gw0.05**: 'The sky appears blue because it contains a mixture of nitrogen and oxygen molecules, which scatter light in the atmosphere, making the sky a' / 'To add the fractions $ \\frac{3}{4} + \\frac{1}{8} $, we need a common denominator. The least common denominator (LCD) of 4 and ' / 'The lighthouse keeper stood at the edge of the cliff, watching the horizon as the sun d
- **hard_alone**: 'The sky appears blue because it contains a mixture of nitrogen and oxygen molecules, which scatter light in the atmosphere. This scattering ' / 'To add the fractions $ \\frac{3}{4} + \\frac{1}{8} $, we need a common denominator. The least common denominator (LCD) of 4 and ' / 'The lighthouse keeper stood at the edge of the cliff, his coat stiff with the wind, his eyes fixed on t
- **soft_track**: "The sky appears blue because of the way sunlight interacts with the Earth's atmosphere, scattering shorter wavelengths of light (blue) more " / 'To add the fractions $ \\frac{3}{4} + \\frac{1}{8} $, we need a common denominator. The least common denominator of 4 and 8 is ' / 'The lighthouse keeper stood at the edge of the cliff, his coat stiff with the wind, his eyes fixed on th
- **chunk4**: 'The sky is blue because it contains a mixture of nitrogen and oxygen, which are gases that make up the atmosphere.' / "We are given the expression:\n\n$$\n3 + \\frac{1}{8} + \\frac{1}{4}\n$$\n\nLet's compute it step by step:\n\n1. First, add" / 'The lighthouse stood at the edge of the sea, its tall spirem casting long shadows across the shore, and the keeper, a man with a worn face 
- **CTRL_identity_trainable**: "The sky appears blue because of the scattering of sunlight by the Earth's atmosphere, which disperses shorter wavelengths (blue) more than l" / 'To add the fractions $ \\frac{3}{4} + \\frac{1}{8} $, we need a common denominator. The least common denominator of 4 and 8 is ' / 'The lighthouse keeper stood at the edge of the cliff, staring at the horizon where the sea 
- **CTRL_identity_small**: "The sky appears blue because of the way sunlight interacts with the Earth's atmosphere, scattering shorter wavelengths of light (blue) more " / 'To add the fractions $ \\frac{3}{4} + \\frac{1}{8} $, we need a common denominator. The least common denominator of 4 and 8 is ' / 'The lighthouse keeper stood at the edge of the cliff, his coat stiff with the wind, his eyes fi


## Verdict

1. **Does training rescue hard phase? Only by undoing it.** Every phase arm lands at or below stock ppl, but only
   because of domain adaptation from the knobs. Against controls with the same knobs and no suppression, each phase
   arm is either significantly worse (all restore arms, chunk4) or statistically identical (hard_alone, soft_track).
   In the identical cases, training had learned to turn the phase off: sharpness → 0.6, gain ≈ supp, tracking lam → 0.
   Under a KL + LM objective, the frozen model has no use for ~15 Hz-style alternating suppression.
2. **Gate penalty curve:** ppl rises smoothly as the gate closes (25.7 @ 0.999 → 26.0 @ 0.68 → 28.2 @ 0.25 floor).
   The trade-off is real and roughly monotone, but each closed position is a cost, never a gain. Compute is not
   saved yet in this implementation.
3. **Controls:** the shuffled buffer is much worse than the real one (32.7 vs 25.7). The suppressed stream carries
   useful content and the restore coupling uses it. That is consistent with the probe and the untrained findings.
4. **ChunkRoPE** wants to go back to plain RoPE (β 0.99 → mostly 0.2–0.5) and is still the worst arm.
5. **Strength 0 / identity is preserved:** strength-0 configs and every trainable layer at init are bitwise
   identical to stock and to the untrained versions (tests below).

**Next steps, if any:** hard alternation brings no LM-loss benefit with a frozen base. A test that would be
convincing must give the alternation something to do: an objective that rewards the dissociation
(suppressed-but-decodable), or long-context/compute budgets where gated-off positions are actually skipped.

## Tests

`pytest -q tests` → **43 passed** (not skipped). That includes the real-model bitwise test against HF Qwen3
(`test_real_qwen3_bitwise_equal_to_hf`), zero-strength feedback bitwise equal to stock, trainable strength-0 equal to
stock, trainable init equal to the frozen layer, cache equivalence, and gradient flow.
