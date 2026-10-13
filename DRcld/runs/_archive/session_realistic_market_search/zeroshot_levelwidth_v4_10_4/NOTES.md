# zeroshot_levelwidth_v4_10_4

**Test**: zero-shot, INFERENCE ONLY -- `realistic_market_x0_diversity_v4_wbehind6` checkpoint
(x0-diversity training, home-market 1.77x/100%) evaluated on ALL SIX quantities (D*, G_res,
D_min, D_max, G_min, G_max) perturbed via the level+width scheme at a MILDER intensity than the
project's standard training range: +/-10% level, +/-4% width (half of the usual +/-20%/+/-8%,
chosen by the user as a fairer first probe of a checkpoint that never saw ANY market diversity
during training). x0 FRESH RANDOM per test instance (matching this checkpoint's own convention).

**Results**: mean 1.55x, median 1.50x, min 0.95x, max 2.33x, win-rate 97% (29/30).

**Comparison across perturbation intensity** (same checkpoint, all zero-shot, no retraining):
| perturbation | mean speedup | mean % reduction | win-rate |
|---|---|---|---|
| none (home market) | 1.77x | 43.5% | 100% |
| level+width +/-10%/+/-4% (this test) | 1.55x | 35.5% | 97% (29/30) |
| level+width +/-20%/+/-8% (full training-scale) | 1.42x | 29.6% | 93% (28/30) |
| D*/G_res-only +/-20% | 1.33x | 24.8% | 80% (24/30) |

**Interpretation**: degradation is graceful, not a cliff -- generalization holds up well at milder
perturbation intensity, with only 1/30 instances marginally below break-even (0.95x). Confirms the
x0-diversity checkpoint has some genuine, if imperfect, robustness to market shift even with zero
retraining, and the amount of degradation scales smoothly with how far the test market is pushed
from the one it was trained on.
