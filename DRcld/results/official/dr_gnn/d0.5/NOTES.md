# official_N10_d0.5_seed1_replicate

**Official pilot** -- recalibrated market generator (build_calibrated_market.py), density=0.5,
base_seed=1. gamma*=6.0 (plateau-center grid search, seed=0), lambda=1.0
(kept at project default -- a 2026-10-14 lambda sweep found 15-40% gains at lambda>1 for most
densities, deferred to future work per user decision). rho_variant=10 (gnn_rho=0.9825,
unroll=570).

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4% perturbation,
x0 free (fresh random per instance/epoch).

**Results**: mean 1.96x, median 1.86x,
min 1.15x, max 2.75x, win-rate
100%.
Mean % reduction: 46.8%, median 46.4%.
