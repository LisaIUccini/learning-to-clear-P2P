# official_N10_d0.4_seed3_replicate

**Official pilot** -- recalibrated market generator (build_calibrated_market.py), density=0.4,
base_seed=3. gamma*=4.0 (plateau-center grid search, seed=0), lambda=1.0
(kept at project default -- a 2026-10-14 lambda sweep found 15-40% gains at lambda>1 for most
densities, deferred to future work per user decision). rho_variant=10 (gnn_rho=0.9686,
unroll=318).

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4% perturbation,
x0 free (fresh random per instance/epoch).

**Results**: mean 2.60x, median 2.59x,
min 1.66x, max 3.82x, win-rate
100%.
Mean % reduction: 59.5%, median 61.4%.
