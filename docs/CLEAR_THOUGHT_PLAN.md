# Clear Thought planning record

This plan was made with the Clear Thought MCP tools: `sequentialthinking` (5 steps), then `decisionframework` (multi-criteria), then `scientificmethod` (hypotheses and experiment design). The tool calls only record structured reasoning. The content below is the reasoning that went into them.

## Sequential thinking

1. **Framing.** The preprint (see SOURCES.md) found that the same IT/PFC neurons alternate about every 60 ms between the conscious and suppressed face. Conscious epochs show higher gain, and both faces can be reconstructed. Tsao's follow-up says this is not a frame rate. She proposes a continuous tracking "drone" that couples to content only at harmonious phases, and compares the effect to a bistable attractor (Jezek 2011). The speech result is that order inside about 20–50 ms windows is largely dispensable. **ML translation:**
   - (i) shared weights time-multiplex K competing hypotheses;
   - (ii) only the stream with *access* writes to a continuous broadcast or workspace stream, with higher gain;
   - (iii) the suppressed streams keep being computed and stay decodable;
   - (iv) order matters between chunks, not within them.
2. **PhaseAttention v0.** There are K content streams `h^k`. All of them run the same block weights, with causal attention over each stream's own history. Mutual inhibition `h^k -= β·mean_{j≠k} h^j` pushes the streams apart. A workspace `z` accumulates `gain ⊙ Σ_k a_k(t)·Δh^k`, where `a(t)` is a phase schedule. The streams read `z` back through a learned scalar for each layer. The LM head reads only `z`.
3. **Tasks.**
   - Synthetic binocular-rivalry task: two eyes, each showing an arithmetic progression. A cue marks which eye is dominant. The model reports the dominant eye's next symbol, and at a final query it recalls the last clean symbol of the suppressed eye, whose eye showed noise for 8 steps before the query.
   - Probes decode the suppressed stream's next symbol, and its hidden continuation during the noise.
   - Byte-level TinyStories LM, measured in bits per byte (bpb).
   - Local-reversal evaluation (C = 2/4/8 bytes) and a CoarseRoPE(C) variant.
4. **Confounds and controls.** With K=2 the model uses about 2× the FLOPs at equal parameters. Controls:
   - (C1) baseline at equal parameters;
   - (C2) baseline at 2× depth, which matches FLOPs;
   - (C3) 2-stream all-access, i.e. no phase;
   - (C4) random-phase with the same duty cycle and epoch length;
   - (C5) no inhibition.

   A claim about phase needs to beat C3 and C4, not only C1. The main risk is stream collapse, so we monitor the stream-specific probes.
5. **Plan.** Build v0 with the pluggable interface and tests. Run smoke tests at 2 threads, then go to 3 seeds and add learned gating and a CoarseRoPE sweep.

## Decision framework (multi-criteria)

| Option | Fidelity (0.35) | Controls (0.25) | CPU cost (0.20) | Swappable (0.20) | Score |
|---|---|---|---|---|---|
| **A. Multi-stream shared weights + phase-gated workspace** | 0.9 | 0.85 | 0.5 | 0.7 | **0.755** |
| B. Head-split phase gating | 0.4 | 0.7 | 0.95 | 0.9 | 0.69 |
| C. Phase-alternating attention mask | 0.5 | 0.6 | 0.95 | 0.9 | 0.69 |

**Decision:** A becomes PhaseAttention v0. C is implemented as the `phase_mask` attention class, a cheap ablation behind the same interface. B is planned for v1.

## Scientific method: hypotheses (status: proposed, prior confidence about 0.35)

* **H1:** On the rivalry task, `phase_fixed` beats the baseline, `phase_all` and `phase_random` by at least 5 points on suppressed recall or suppressed-probe accuracy across 3 seeds, and loses no more than 2 points of dominant-report accuracy.
  - *Falsified if* `phase_all` or `phase_random` match it. Any gain would then come from multi-stream compute, not phase.
* **H2:** On the byte LM, phase gating costs no more than 3% bpb compared with `phase_all`.
  - *If it costs more*, phase gating stays an optional module only.
* **H3:** Byte LMs are fairly robust to within-chunk reversal at C=2 (bpb up less than 10%) but not at C≥8. CoarseRoPE(C) improves that robustness at no more than 2% clean bpb cost.
* **Controlled:** parameter count (all 2-stream arms are identical), data, steps, seed, threads.
* **Confound:** FLOPs, handled by C2 and C3.
