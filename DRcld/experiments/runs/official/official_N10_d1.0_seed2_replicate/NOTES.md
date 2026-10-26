# official_N10_d1.0_seed2_replicate

**Official pilot** -- recalibrated market generator (build_calibrated_market.py), density=1.0,
base_seed=2. gamma*=6.0 (plateau-center grid search, seed=0), lambda=1.0
(kept at project default -- a 2026-10-14 lambda sweep found 15-40% gains at lambda>1 for most
densities, deferred to future work per user decision). rho_variant=10 (gnn_rho=0.9926,
unroll=1354).

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4% perturbation,
x0 free (fresh random per instance/epoch).

**Results**: mean 2.02x, median 1.85x,
min 0.72x, max 4.19x, win-rate
73%.
Mean % reduction: 33.1%, median 45.8%.
