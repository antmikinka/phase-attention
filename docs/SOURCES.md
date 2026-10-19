# Sources: what was verified and what wasn't

Checked on 2026-10-19 using the public X API and bioRxiv/web search. Times are in PDT (UTC-7).

## 1. Tsao lab preprint: VERIFIED

* **Title:** *Rapid alternation of conscious and suppressed stimulus representations in the primate brain*
* **Authors:** Janis K. Hesse, Frank F. Lanfranchi, Conor McGrory, Yuelin Shi, Tatiana A. Engel, Doris Y. Tsao. Lanfranchi and McGrory contributed equally. Hesse and Tsao are the corresponding authors.
* **bioRxiv:** posted 2026-10-16, DOI [10.64898/2026.10.15.759299](https://doi.org/10.64898/2026.10.15.759299), page <https://www.biorxiv.org/content/10.64898/2026.10.15.759299v1>
* **Claims taken from the abstract** (verified wording):
  * Neuropixels recordings were made across several face patches in inferotemporal (IT) and prefrontal cortex during **binocular flash suppression**.
  * While the stimulus and the percept stay constant, activity alternates **"about 15 times per second"** between *conscious* and *suppressed* micro-epochs. That is about 60–67 ms per epoch.
  * In conscious micro-epochs, the conscious face can be reconstructed. In suppressed micro-epochs, the **suppressed** face can be reconstructed from the same population.
  * Single neurons are selective for the conscious stimulus in one epoch and the suppressed stimulus in the other.
  * The two epochs can still be told apart: **feature tuning is amplified (higher gain) during conscious micro-epochs**.
  * Prefrontal cortex looks qualitatively similar. The authors conclude that conscious access requires *temporal selection beyond localization*.
* **Thread by @doristsao**, starting at <https://x.com/doristsao/status/2111316522098733078> (2026-10-16 21:40). It says "thousands of neurons", "alternates every ~60 ms", "conscious perception does not reside in a specific brain region, but rather a specific temporal phase", and "cells concertedly switched their tuning every 60 ms".
* **Follow-up clarification by Tsao**, <https://x.com/doristsao/status/2111859732806074812> (2026-10-18 09:39). VERIFIED:
  1. "We have not shown that 'consciousness has a frame rate.'" The results come from only a few nodes, and only during binocular rivalry. IT may not alternate at all outside this artificial setting.
  2. She proposes a continuous **tracking process** in posterior parietal cortex, which she compares to a steady "drone note". IT content couples to it only at moments of "harmony", so experience stays continuous.
  3. She compares the result to Jezek et al. 2011, where hippocampal place-cell maps flicker in sync with theta (~125 ms), and suggests a **bistable attractor** in IT.
  4. She asks for comments on bioRxiv.
  * In a separate reply (2026-10-17 11:59) she says they looked for coupling to LFP bands and found "nothing compelling so far".

## 2. Locally time-reversed speech demo: post VERIFIED, audio NOT independently checked

* **Nathan Roll (@nrol_ling), Oruk Labs:** <https://x.com/nrol_ling/status/2111853310533849244> (2026-10-18 09:14). The text says "Speech can survive being flipped backward every twenty milliseconds", and the demo plays windows of 200, 100, 50 and 20 ms before the original. There is a 20.4 s video, which I did **not** listen to or analyze. The "intelligible at 20 ms" claim is the post's, not mine.
* **Carlos E. Perez (@IntuitMachine):** <https://x.com/IntuitMachine/status/2112182269309829490> (2026-10-19 07:01). It quotes the post above with "BREAKING PROOF!! Continuity we perceive in consciousness is an illusion! …". **This is hype, not evidence.** Tsao's follow-up explicitly argues against the "frame rate / discrete consciousness" reading, and the speech demo concerns auditory robustness, not consciousness.
* **Prior literature (verified):** Saberi & Perrott, *Cognitive restoration of reversed speech*, Nature 398:760 (1999), [doi:10.1038/19652](https://doi.org/10.1038/19652). Speech reversed within local segments is rated perfectly intelligible up to about 50 ms segments, and intelligibility falls to about 50% near 130 ms. The authors attribute this to slow 3–8 Hz envelope cues. The Oruk demo is a modern restatement of this known effect.

## What this project does and doesn't take from the sources

* We borrow **mechanism metaphors** as engineering hypotheses:
  * the same population time-multiplexes competing contents
  * only one phase gets access or broadcast, with higher gain
  * a continuous tracking stream preserves continuity
  * order matters between short chunks more than within them
* We claim **nothing** about consciousness, frame rates, or biological plausibility. Any result here is only a statement about small language models.
