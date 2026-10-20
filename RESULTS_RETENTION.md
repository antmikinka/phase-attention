# RESULTS_RETENTION — rewarding readable suppressed-phase information

**Question.** If the frozen-base phase arms get a job (keep suppressed-phase tokens decodable at long lags), do
they retain that information better than matched controls? The bar is: at matched readout capacity and matched
trainable params, beat (a) the identity-trainable control and (b) the shuffled-buffer control, keep ppl within 3%
of stock, and keep the alternation from collapsing.

## Design (`scripts/train_retention.py`, queue `scripts/run_retention.sh`, table `scripts/summarize_retention.py`)

- **Frozen Qwen3-0.6B.** Only the new layers (layers 8–19, P = 4) and the readout train. Data: WikiText-2 **train**
  split only. 150 steps × 2 × 256 tokens, 8 threads.
- **Loss:** KL(stock ‖ student) + 0.2·CE + aux_w · mean over lags k of CE(readout_k(h19[t]) → token[t−k]).
- **Readout input:** h19, the residual stream after layer 19 (end of the modified block), RMS-normalised.
- **Lags k = 4, 12, 28, 60.** Each is an odd number of 4-token blocks back, so token[t−k] is always in the
  *suppressed* phase relative to query t. In the hard-phase layers 8–19, t cannot attend to it directly.
- **Readout, the same for every arm:** per lag, Linear(1024, 64) → Linear(64, 2049 + bias). The classes are the
  2048 most frequent TRAIN tokens plus "other": 794,884 params. Accuracy = argmax over the 2048 real tokens;
  "other" targets count as wrong.
- **Detach arms:** the readout trains, but the reward never reaches the new layers. This separates "the reward
  changed the layers" from "a readout was fitted".
- **Fresh probe (held out, independent of the trained readout):** ridge with intercept (λ = 0.1·n, top-200 target
  classes). Fit on 24 random **train** windows, scored on the test windows. Lags 1, 4, 8, 12, 28, 32, 60, where 8 and 32
  are same-phase contrasts.
- **Eval:** first 32,768 WikiText-2 **test** tokens (64 × 512), positions 64, 71, …, 505 per window. Paired
  sign/Wilcoxon tests over the 64 windows. The readout never saw test tokens.
- **Arms:** hard + restore (either gate) and the identity control at aux_w 0.03 / 0.1 / 0.3; shuffled buffer at 0.1;
  hard-alone (learnable soft phase) and the small identity control at 0.1; detach versions of hard + restore and
  identity; stock + readout only.
- **Params:** hard + restore 197,052 vs identity 197,244 new params; hard_alone 420 vs identity_small 576. Every arm
  has the same readout.

## Main table

Majority baseline (always predict the most frequent TRAIN token ' the'): lag 1 5.6 / lag 4 5.6 / lag 8 5.5 / lag 12 4.8 / lag 28 4.3 / lag 32 5.5 / lag 60 5.6%

Stock ppl 28.91; ppl budget (+3%) 29.78

| arm | new params | readout params | ppl | Δ stock | gate mean/open | trained readout top-1 % @ lag 4 / 12 / 28 / 60 | fresh probe % @ lag 1 / 4 / 8 / 12 / 28 / 32 / 60 |
|---|---|---|---|---|---|---|---|
| hard_restore_aux0.1 | 197,052 | 794,884 | 25.88 | -10.5% | 1.00/1.00 | 9.2 / 4.4 / 4.1 / 2.9 | 29.0 / 6.4 / 3.9 / 3.8 / 2.8 / 2.9 / 2.6 |
| CTRL_identity_aux0.1 | 197,244 | 794,884 | 25.52 | -11.7% | 0.99/1.00 | 9.3 / 4.2 / 3.9 / 2.8 | 28.9 / 6.8 / 3.7 / 3.7 / 2.9 / 2.8 / 2.4 |
| CTRL_shuffled_aux0.1 | 197,052 | 794,884 | 33.07 | +14.4% | 0.53/0.57 | 8.3 / 4.0 / 3.1 / 3.6 | 28.5 / 5.9 / 3.8 / 3.3 / 2.7 / 2.9 / 2.7 |
| STOCK_readout | 0 | 794,884 | 28.91 | -0.0% | - | 9.6 / 4.5 / 4.0 / 3.4 | 29.2 / 6.4 / 4.0 / 3.8 / 2.7 / 2.7 / 2.6 |
| hard_restore_aux0.3 | 197,052 | 794,884 | 26.13 | -9.6% | 1.00/1.00 | 9.7 / 4.4 / 4.0 / 3.0 | 28.8 / 6.5 / 3.9 / 3.8 / 2.8 / 2.9 / 2.6 |
| CTRL_identity_aux0.3 | 197,244 | 794,884 | 25.88 | -10.5% | 1.00/1.00 | 9.7 / 4.3 / 4.0 / 2.9 | 29.1 / 6.9 / 3.9 / 3.7 / 2.8 / 3.1 / 2.7 |
| hard_restore_aux0.03 | 197,052 | 794,884 | 25.86 | -10.6% | 1.00/1.00 | 9.3 / 4.3 / 4.2 / 2.8 | 28.8 / 6.5 / 3.8 / 3.9 / 2.8 / 2.8 / 2.5 |
| CTRL_identity_aux0.03 | 197,244 | 794,884 | 25.54 | -11.7% | 1.00/1.00 | 9.2 / 4.1 / 3.9 / 2.8 | 29.1 / 6.7 / 3.7 / 3.7 / 2.7 / 2.8 / 2.3 |
| hard_alone_aux0.1 | 420 | 794,884 | 27.64 | -4.4% | - | 9.1 / 4.2 / 4.2 / 2.8 | 29.2 / 6.6 / 4.0 / 3.8 / 2.6 / 2.8 / 2.4 |
| CTRL_identity_small_aux0.1 | 576 | 794,884 | 27.59 | -4.6% | - | 9.9 / 4.4 / 4.2 / 3.0 | 29.0 / 6.7 / 3.9 / 3.9 / 2.8 / 2.9 / 2.4 |
| hard_restore_detach | 197,052 | 794,884 | 25.86 | -10.6% | 1.00/1.00 | 9.2 / 4.3 / 4.1 / 2.8 | 28.9 / 6.6 / 3.7 / 3.8 / 2.8 / 2.8 / 2.4 |
| CTRL_identity_detach | 197,244 | 794,884 | 25.53 | -11.7% | 1.00/1.00 | 9.2 / 4.2 / 3.8 / 2.7 | 29.1 / 6.7 / 3.8 / 3.6 / 2.7 / 2.8 / 2.5 |

Retention is measured at long lags in the residual stream. **Beyond lag 4, every arm, stock included, is at or
below the majority baseline** (always "the": 4.3–5.6%). Above-chance retention exists only at lags 1 and 4.

### Paired tests (64 test windows): retention accuracy, arm − control (positive = arm retains more)

| arm vs control | metric | lag 1 | lag 4 | lag 8 | lag 12 | lag 28 | lag 32 | lag 60 |
|---|---|---|---|---|---|---|---|---|
| hard_restore_aux0.1 vs CTRL_identity_aux0.1 | readout |  | -0.1 (16/64, p=0.74/0.58) |  | +0.2 (16/64, p=0.15/0.12) | +0.3 (11/64, p=0.12/0.034) |  | +0.1 (10/64, p=0.63/0.3) |
| hard_restore_aux0.1 vs CTRL_identity_aux0.1 | fresh | +0.0 (13/64, p=1/0.72) | -0.4 (9/64, p=0.087/0.061) | +0.2 (17/64, p=0.17/0.11) | +0.1 (9/64, p=0.61/0.22) | -0.0 (6/64, p=0.61/0.66) | +0.1 (10/64, p=0.81/0.42) | +0.2 (13/64, p=0.38/0.18) |
| hard_restore_aux0.1 vs CTRL_shuffled_aux0.1 | readout |  | +1.0 (33/64, p=0.07/0.0081) |  | +0.4 (26/64, p=0.22/0.3) | +1.0 (35/64, p=0.0011/0.00042) |  | -0.7 (15/64, p=0.026/0.021) |
| hard_restore_aux0.1 vs CTRL_shuffled_aux0.1 | fresh | +0.4 (25/64, p=0.68/0.53) | +0.5 (23/64, p=0.64/0.066) | +0.1 (25/64, p=0.55/0.79) | +0.5 (29/64, p=0.1/0.054) | +0.1 (22/64, p=1/0.7) | +0.0 (24/64, p=1/0.88) | -0.1 (26/64, p=1/0.7) |
| hard_restore_aux0.1 vs STOCK_readout | readout |  | -0.4 (20/64, p=0.31/0.4) |  | -0.1 (20/64, p=0.76/0.7) | +0.1 (21/64, p=1/0.71) |  | -0.5 (20/64, p=0.2/0.18) |
| hard_restore_aux0.1 vs STOCK_readout | fresh | -0.2 (11/64, p=0.34/0.17) | -0.0 (15/64, p=1/0.9) | -0.1 (14/64, p=1/0.74) | +0.0 (13/64, p=1/0.85) | +0.1 (9/64, p=1/0.42) | +0.2 (12/64, p=0.14/0.078) | +0.0 (11/64, p=1/0.88) |
| hard_restore_aux0.1 vs hard_restore_detach | readout |  | +0.0 (11/64, p=1/1) |  | +0.0 (11/64, p=1/0.7) | +0.0 (6/64, p=1/0.83) |  | +0.1 (9/64, p=0.8/0.51) |
| hard_restore_aux0.1 vs hard_restore_detach | fresh | +0.0 (12/64, p=1/0.71) | -0.2 (5/64, p=0.096/0.092) | +0.2 (11/64, p=0.057/0.032) | +0.0 (6/64, p=1/1) | +0.0 (7/64, p=1/0.81) | +0.0 (6/64, p=0.75/0.57) | +0.1 (11/64, p=0.21/0.26) |
| hard_restore_aux0.3 vs STOCK_readout | readout |  | +0.1 (22/64, p=1/0.94) |  | -0.0 (21/64, p=1/0.72) | +0.0 (25/64, p=1/0.91) |  | -0.4 (23/64, p=0.49/0.19) |
| hard_restore_aux0.3 vs STOCK_readout | fresh | -0.3 (14/64, p=0.11/0.14) | +0.1 (21/64, p=0.63/0.57) | -0.1 (14/64, p=0.72/0.62) | -0.0 (14/64, p=1/0.88) | +0.1 (13/64, p=0.52/0.62) | +0.2 (16/64, p=0.33/0.15) | +0.0 (12/64, p=0.85/0.79) |
| hard_restore_aux0.3 vs CTRL_identity_aux0.3 | readout |  | -0.0 (15/64, p=1/0.92) |  | +0.1 (16/64, p=1/0.64) | +0.0 (10/64, p=1/0.85) |  | +0.0 (14/64, p=0.69/0.76) |
| hard_restore_aux0.3 vs CTRL_identity_aux0.3 | fresh | -0.2 (17/64, p=0.63/0.29) | -0.4 (16/64, p=0.27/0.1) | +0.0 (16/64, p=0.57/0.7) | +0.1 (15/64, p=0.56/0.68) | -0.0 (11/64, p=1/0.71) | -0.2 (10/64, p=0.099/0.26) | -0.1 (10/64, p=0.33/0.43) |
| hard_restore_aux0.3 vs hard_restore_detach | readout |  | +0.5 (21/64, p=0.043/0.024) |  | +0.1 (14/64, p=1/0.62) | -0.1 (14/64, p=0.6/0.47) |  | +0.1 (13/64, p=0.84/0.52) |
| hard_restore_aux0.3 vs hard_restore_detach | fresh | -0.1 (16/64, p=0.74/0.72) | -0.1 (20/64, p=0.5/0.59) | +0.2 (18/64, p=0.12/0.12) | -0.0 (11/64, p=0.84/0.73) | -0.0 (14/64, p=0.72/0.78) | +0.1 (15/64, p=0.42/0.52) | +0.2 (14/64, p=0.54/0.43) |
| hard_restore_aux0.03 vs STOCK_readout | readout |  | -0.3 (18/64, p=0.11/0.34) |  | -0.2 (21/64, p=0.88/0.38) | +0.2 (21/64, p=1/0.62) |  | -0.6 (19/64, p=0.12/0.11) |
| hard_restore_aux0.03 vs STOCK_readout | fresh | -0.3 (8/64, p=0.076/0.028) | +0.1 (18/64, p=0.47/0.59) | -0.2 (11/64, p=0.44/0.23) | +0.1 (14/64, p=0.69/0.56) | +0.1 (11/64, p=1/0.57) | +0.1 (9/64, p=0.27/0.18) | -0.1 (9/64, p=0.52/0.68) |
| hard_restore_aux0.03 vs CTRL_identity_aux0.03 | readout |  | +0.0 (15/64, p=1/0.89) |  | +0.1 (13/64, p=0.26/0.25) | +0.2 (10/64, p=0.45/0.089) |  | +0.0 (11/64, p=1/0.7) |
| hard_restore_aux0.03 vs CTRL_identity_aux0.03 | fresh | -0.2 (9/64, p=0.23/0.18) | -0.2 (12/64, p=0.22/0.24) | +0.0 (13/64, p=0.84/0.73) | +0.2 (13/64, p=0.38/0.14) | +0.1 (10/64, p=0.63/0.55) | +0.0 (9/64, p=1/1) | +0.2 (16/64, p=0.33/0.29) |
| hard_restore_aux0.03 vs hard_restore_detach | readout |  | +0.0 (4/64, p=0.69/0.48) |  | -0.1 (4/64, p=0.55/0.39) | +0.0 (2/64, p=0.5/0.35) |  | +0.0 (4/64, p=1/1) |
| hard_restore_aux0.03 vs hard_restore_detach | fresh | -0.1 (4/64, p=0.55/0.39) | -0.1 (2/64, p=0.45/0.24) | +0.1 (5/64, p=0.22/0.13) | +0.0 (3/64, p=0.62/0.42) | +0.0 (1/64, p=1/1) | -0.0 (1/64, p=1/0.77) | +0.0 (4/64, p=0.38/0.57) |
| hard_alone_aux0.1 vs STOCK_readout | readout |  | -0.5 (16/64, p=0.13/0.15) |  | -0.3 (13/64, p=0.23/0.2) | +0.2 (20/64, p=0.3/0.33) |  | -0.6 (10/64, p=0.0034/0.0068) |
| hard_alone_aux0.1 vs STOCK_readout | fresh | +0.0 (18/64, p=0.73/1) | +0.2 (20/64, p=0.5/0.31) | +0.0 (11/64, p=1/0.82) | -0.0 (12/64, p=1/0.87) | -0.1 (9/64, p=0.82/0.55) | +0.1 (15/64, p=0.42/0.31) | -0.1 (13/64, p=0.58/0.39) |
| hard_alone_aux0.1 vs CTRL_identity_small_aux0.1 | readout |  | -0.8 (16/64, p=0.072/0.025) |  | -0.3 (14/64, p=0.19/0.32) | -0.0 (19/64, p=1/0.91) |  | -0.2 (18/64, p=0.64/0.37) |
| hard_alone_aux0.1 vs CTRL_identity_small_aux0.1 | fresh | +0.2 (20/64, p=0.15/0.32) | -0.0 (15/64, p=0.86/0.78) | +0.1 (18/64, p=0.26/0.42) | -0.1 (12/64, p=0.7/0.63) | -0.2 (6/64, p=0.17/0.11) | -0.0 (13/64, p=1/0.87) | +0.0 (12/64, p=1/1) |
| hard_restore_detach vs CTRL_identity_detach | readout |  | -0.0 (11/64, p=0.69/0.92) |  | +0.1 (16/64, p=0.23/0.22) | +0.3 (15/64, p=0.21/0.036) |  | +0.2 (13/64, p=0.17/0.11) |
| hard_restore_detach vs CTRL_identity_detach | fresh | -0.1 (9/64, p=0.17/0.26) | -0.0 (15/64, p=1/0.88) | -0.1 (13/64, p=1/0.73) | +0.2 (13/64, p=0.26/0.12) | +0.1 (11/64, p=0.48/0.29) | +0.0 (9/64, p=0.8/0.64) | -0.1 (10/64, p=0.68/0.59) |

### Learned phase / gate params (layers 8, 14, 19)

- **hard_restore_aux0.1**: `layers.8.extras.0.lam_p`=0.964; `layers.8.extras.0.tau_p`=-0.282; `layers.14.extras.0.lam_p`=0.972; `layers.14.extras.0.tau_p`=-0.155; `layers.19.extras.0.lam_p`=0.987; `layers.19.extras.0.tau_p`=-0.192
- **CTRL_identity_aux0.1**: `layers.8.self_attn.log_supp`=[-0.02, -0.4, 0.05, 0.3]; `layers.8.extras.0.lam_p`=0.972; `layers.8.extras.0.tau_p`=0.043; `layers.14.self_attn.log_supp`=[0.03, 0.01, -0.03, -0.11]; `layers.14.extras.0.lam_p`=0.908; `layers.14.extras.0.tau_p`=0.305; `layers.19.self_attn.log_supp`=[-0.11, -0.18, 0.79, 0.05]; `layers.19.extras.0.lam_p`=0.958; `layers.19.extras.0.tau_p`=0.047
- **CTRL_shuffled_aux0.1**: `layers.8.extras.0.lam_p`=0.447; `layers.8.extras.0.tau_p`=0.753; `layers.14.extras.0.lam_p`=0.559; `layers.14.extras.0.tau_p`=0.625; `layers.19.extras.0.lam_p`=0.431; `layers.19.extras.0.tau_p`=0.977
- **hard_restore_aux0.3**: `layers.8.extras.0.lam_p`=0.954; `layers.8.extras.0.tau_p`=-0.256; `layers.14.extras.0.lam_p`=0.952; `layers.14.extras.0.tau_p`=-0.183; `layers.19.extras.0.lam_p`=0.981; `layers.19.extras.0.tau_p`=-0.218
- **CTRL_identity_aux0.3**: `layers.8.self_attn.log_supp`=[-0.03, 0.18, 0.25, 0.08]; `layers.8.extras.0.lam_p`=0.919; `layers.8.extras.0.tau_p`=0.066; `layers.14.self_attn.log_supp`=[0.31, -0.45, 0.13, 0.1]; `layers.14.extras.0.lam_p`=0.987; `layers.14.extras.0.tau_p`=0.133; `layers.19.self_attn.log_supp`=[-0.26, 0.05, 0.02, -0.34]; `layers.19.extras.0.lam_p`=0.984; `layers.19.extras.0.tau_p`=0.203
- **hard_restore_aux0.03**: `layers.8.extras.0.lam_p`=0.976; `layers.8.extras.0.tau_p`=-0.289; `layers.14.extras.0.lam_p`=0.979; `layers.14.extras.0.tau_p`=-0.14; `layers.19.extras.0.lam_p`=0.997; `layers.19.extras.0.tau_p`=-0.191
- **CTRL_identity_aux0.03**: `layers.8.self_attn.log_supp`=[-0.04, -0.41, 0.04, 0.35]; `layers.8.extras.0.lam_p`=0.979; `layers.8.extras.0.tau_p`=0.106; `layers.14.self_attn.log_supp`=[-0.04, -0.14, -0.09, -0.06]; `layers.14.extras.0.lam_p`=0.937; `layers.14.extras.0.tau_p`=0.266; `layers.19.self_attn.log_supp`=[-0.16, -0.27, 0.82, 0.09]; `layers.19.extras.0.lam_p`=0.984; `layers.19.extras.0.tau_p`=-0.007
- **hard_alone_aux0.1**: `layers.8.self_attn.phase_offset`=-0.587; `layers.8.self_attn.log_period`=1.26; `layers.8.self_attn.log_psharp`=-0.061; `layers.14.self_attn.phase_offset`=-0.501; `layers.14.self_attn.log_period`=1.436; `layers.14.self_attn.log_psharp`=-0.367; `layers.19.self_attn.phase_offset`=-0.501; `layers.19.self_attn.log_period`=0.699; `layers.19.self_attn.log_psharp`=-0.156
- **CTRL_identity_small_aux0.1**: `layers.8.self_attn.log_supp`=[0.0, -0.02, -0.01, -0.01]; `layers.14.self_attn.log_supp`=[0.0, 0.0, -0.0, 0.01]; `layers.19.self_attn.log_supp`=[0.01, -0.04, 0.01, -0.05]
- **hard_restore_detach**: `layers.8.extras.0.lam_p`=0.979; `layers.8.extras.0.tau_p`=-0.291; `layers.14.extras.0.lam_p`=0.984; `layers.14.extras.0.tau_p`=-0.133; `layers.19.extras.0.lam_p`=1.0; `layers.19.extras.0.tau_p`=-0.19
- **CTRL_identity_detach**: `layers.8.self_attn.log_supp`=[-0.01, -0.37, 0.03, 0.35]; `layers.8.extras.0.lam_p`=0.986; `layers.8.extras.0.tau_p`=0.148; `layers.14.self_attn.log_supp`=[0.02, -0.15, -0.14, -0.03]; `layers.14.extras.0.lam_p`=0.958; `layers.14.extras.0.tau_p`=0.248; `layers.19.self_attn.log_supp`=[-0.16, -0.3, 0.8, 0.07]; `layers.19.extras.0.lam_p`=0.996; `layers.19.extras.0.tau_p`=-0.024

(Cell = mean accuracy difference in percentage points; then (windows where the arm is strictly better / 64, sign p / Wilcoxon p). Many windows tie.)

## Did the alternation survive the reward?

- **Restore gate:** it opened fully (mean 0.995–0.996, open 0.999) at every aux weight. This is the same collapse
  as without the reward (RESULTS_TRAINED), so suppressed positions are restored to stock-like attention.
- **hard_alone:** the phase sharpness fell from 20 to about e^−0.06 to e^−0.37 ≈ 0.7–0.94, which is again a mild
  soft preference, not alternation. The period drifted per layer (≈ 3.5 at L8, 4.2 at L14, 2.0 at L19). Its ppl
  (27.64) and retention match the 576-param identity control.
- **Shuffled control:** it closed its gate to about 0.53 and lowered λ (≈ 0.45–0.56) to ignore the noise. Its ppl is
  33.1 (+14%), so it fails the ppl budget.
- **ppl budget (≤ 29.78):** every arm except shuffled is inside it, and below stock, because of the domain adaptation
  explained in RESULTS_TRAINED.

## Verdict

**Null result. The retention reward gives the alternation no advantage.**
- At matched params and readout, hard + restore vs the identity control shows no consistent difference at any lag
  or aux weight. Differences are ≤ 0.4 pp, with sign tests p ≥ 0.09. The single Wilcoxon p = 0.034 at lag 28
  (aux 0.1, +0.3 pp) does not replicate at aux 0.03 or 0.3, nor in the fresh probe, and is expected by chance among
  ~80 tests.
- Hard + restore beats the shuffled control on the trained readout at lags 4 and 28 (+1.0 pp, p ≤ 0.008), but not
  at lag 60 (−0.7 pp, p = 0.02) and not in the fresh probe.
- No phase arm beats stock + readout.
- Reward vs detach is ≈ 0 everywhere: the gradient barely changed the layers.
- The phase mechanisms collapse toward the identity whether or not the reward is present.
- The reason is structural: h19 holds almost no token identity beyond lag 4 in **any** configuration. A rank-64
  readout cannot pull it out, and the cheap way for the new layers to help, opening the restore gate, is the same
  thing the identity control does.

**What would be needed to test the idea properly:** a readout over a dedicated memory/buffer stream rather than
the residual, or a task where suppressed tokens matter for prediction (copy/recall at a delay). Both would require
more than knob-level trainable capacity on a frozen base.

## Tests

`pytest -q tests` → **43 passed** (not skipped), including the real-model bitwise identity test against HF Qwen3
and strength-0 = stock.

Artifacts: `results/retention.json` (all metrics, per-window values, paired stats vs stock ppl) and
`logs/ret_<arm>.{log,jsonl}`. The arms evaluated before the final metric fix (hard_restore_aux0.1,
CTRL_identity_aux0.1) were re-evaluated with `--reeval`. Learned params, including the readout (~3.2 MB per arm),
are kept on the box in `results/retention_params/` and are not committed (`*.pt` is gitignored).
