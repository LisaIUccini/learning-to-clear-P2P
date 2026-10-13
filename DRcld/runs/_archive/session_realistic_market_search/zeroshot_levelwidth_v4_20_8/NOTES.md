# zeroshot_levelwidth_v4_20_8

**Test**: zero-shot, INFERENCE ONLY -- `realistic_market_x0_diversity_v4_wbehind6` checkpoint
(x0-diversity training, home-market 1.77x/100%) evaluated on ALL SIX quantities (D*, G_res,
D_min, D_max, G_min, G_max) perturbed via the level+width scheme at the project's standard
TRAINING-scale intensity (+/-20% level, +/-8% width -- same as v6/gmin_only_market_v1's own
dataset). x0 FRESH RANDOM per test instance (matching this checkpoint's own convention).

**Results**: mean 1.42x, median 1.37x, min 0.96x, max 2.08x, win-rate 93% (28/30).

See [[zeroshot_levelwidth_v4_10_4]]'s NOTES.md for the full cross-intensity comparison table.

**Interpretation**: at full training-scale perturbation intensity, this x0-diversity-only
checkpoint still delivers a real average improvement (29.6% mean reduction) with zero retraining,
but with more losses (2/30) than the milder +/-10%/+/-4% probe -- degradation scales smoothly with
perturbation size, no cliff.
