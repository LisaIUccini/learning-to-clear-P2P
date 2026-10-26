# official_N10_d0.8_seed0_replicate_fixed

**Official pilot** -- recalibrated market generator (build_calibrated_market.py), density=0.8,
base_seed=0. gamma*=4.0 (plateau-center grid search, seed=0), lambda=1.0
(kept at project default -- a 2026-10-14 lambda sweep found 15-40% gains at lambda>1 for most
densities, deferred to future work per user decision). rho_variant=10 (gnn_rho=0.9914,
unroll=1159).

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4% perturbation,
x0 free (fresh random per instance/epoch).

**Results**: mean 3.84x, median 4.07x,
min 2.40x, max 5.36x, win-rate
100%.
Mean % reduction: 72.5%, median 75.4%.
