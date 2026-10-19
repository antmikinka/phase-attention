# Smoke results (v0): 1 seed, 2 torch threads, shared CPU

**These are smoke tests, not evidence.** Each arm has one seed and a few minutes of training. The 2-stream arms use 2× the FLOPs of `baseline`. `baseline_2x` matches FLOPs but has 2× the parameters.

* Rivalry task: d=128, 4 layers, T=48, 16 symbols, delay of 8, 3000 steps at batch 32. Probes are closed-form ridge, fit on 256 sequences and tested on 256 others.
* Byte-level LM: TinyStories (valid split, cut 95/5), T=128, 1200 steps at batch 32.

## Verdicts

* **H1 (phase-gated access keeps suppressed content more decodable): not supported.**
  * Every arm solves the easy "recall the last clean suppressed symbol" query (97–100%). The task is too easy to separate arms on that metric.
  * The suppressed-next probe ranks the arms the *opposite* way to H1. `phase_fixed` scores 18% on z and 32% on the best stream. That is below `phase_all` (33/39), `phase_random` (69/67) and `noinhib` (55/56). The single-stream `baseline` scores about 10%, near chance at 6.25%.
  * The probe score tracks *under-training*. The arms with high suppressed decodability are the ones whose dominant report is still incomplete: random 87%, noinhib 83%, 2x 90%. The converged arms (`baseline`, `phase_fixed`) have discarded the unreported stream. So the probe needs to be compared at matched report accuracy, or after full convergence.
  * Removing inhibition slowed learning: the report was 36% at step 1000 and 83% at step 3000, against 100% for `phase_fixed`.
* **H2 (phase gating costs ≤3% bpb against `phase_all`): supported in the smoke test.** `phase_fixed` scored 1.515 bpb, `phase_all` 1.517 and `baseline` 1.511, all within noise. The 2-stream model doubles FLOPs and buys nothing on LM loss at this scale.
* **H3 (byte LMs are robust to local reversal): refuted** for models trained without augmentation.
  * Reversing context inside chunks of just 2 bytes takes bpb from about 1.5 to about 8, which is worse than uniform. Byte text is not like speech, where 20–50 ms windows survive.
  * CoarseRoPE(4) shrinks the damage slightly (5.1× against 5.6×) but costs 6% clean bpb (1.607 against 1.511).
  * Getting any robustness will need reversal augmentation in training.

### rivalry (final eval, 512 held-out sequences; chance = 6.25%)

| arm | params | steps | dominant report acc | suppressed query acc | probe: suppressed next (z / best stream) | probe: suppressed during noise (z / best stream) | probe: dominant next (z) |
|---|---|---|---|---|---|---|---|
| baseline | 796,544 | 3000 | 100.0 | 100.0 | 9.5 / – | 9.9 / – | 100.0 |
| phase_fixed | 796,936 | 3000 | 100.0 | 100.0 | 18.4 / 31.5 | 8.1 / 9.1 | 99.4 |
| phase_all | 796,936 | 3000 | 98.6 | 100.0 | 32.6 / 38.5 | 8.0 / 7.5 | 99.1 |
| phase_random | 796,936 | 3000 | 87.3 | 100.0 | 69.3 / 67.4 | 11.2 / 13.4 | 84.1 |
| phase_fixed_noinhib | 796,936 | 3000 | 82.8 | 96.9 | 55.3 / 56.2 | 10.4 / 11.0 | 73.1 |
| baseline_2x | 1,588,096 | 3000 | 89.7 | 100.0 | 37.9 / – | 13.2 / – | 87.0 |

### rivalry_extrap (final eval, 512 held-out sequences; chance = 6.25%)

| arm | params | steps | dominant report acc | suppressed query acc | probe: suppressed next (z / best stream) | probe: suppressed during noise (z / best stream) | probe: dominant next (z) |
|---|---|---|---|---|---|---|---|
| baseline | 796,544 | 3000 | 96.2 | 100.0 | 26.4 / – | 6.3 / – | 96.3 |
| phase_fixed | 796,936 | 3000 | 97.7 | 100.0 | 13.5 / 17.2 | 6.6 / 6.5 | 97.5 |
| phase_all | 796,936 | 2000 | 49.0 | 51.0 | 18.3 / 21.6 | 7.1 / 7.7 | 51.0 |

### byte-level TinyStories LM (val bpb, 64×128 bytes; reversal = context reversed inside chunks of C bytes, scored at chunk ends)

| arm | params | steps | val bpb | C=2 intact→reversed (ratio) | C=4 | C=8 |
|---|---|---|---|---|---|---|
| baseline | 824,448 | 1200 | 1.511 | 1.486→8.256 (5.56×) | 1.421→9.114 (6.41×) | 1.347→9.368 (6.96×) |
| phase_fixed | 824,840 | 1200 | 1.515 | 1.499→8.059 (5.38×) | 1.419→9.083 (6.40×) | 1.355→9.502 (7.01×) |
| phase_all | 824,840 | 1200 | 1.517 | 1.495→8.195 (5.48×) | 1.406→9.144 (6.50×) | 1.340→9.384 (7.00×) |
| coarse4 | 824,448 | 1200 | 1.607 | 1.604→8.177 (5.10×) | 1.527→9.096 (5.96×) | 1.491→9.410 (6.31×) |
