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
