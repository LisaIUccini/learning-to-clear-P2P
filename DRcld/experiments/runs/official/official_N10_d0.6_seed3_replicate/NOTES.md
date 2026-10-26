# official_N10_d0.6_seed3_replicate

**Official pilot** -- recalibrated market generator (build_calibrated_market.py), density=0.6,
base_seed=3. gamma*=4.0 (plateau-center grid search, seed=0), lambda=1.0
(kept at project default -- a 2026-10-14 lambda sweep found 15-40% gains at lambda>1 for most
densities, deferred to future work per user decision). rho_variant=10 (gnn_rho=0.9688,
unroll=321).

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4% perturbation,
x0 free (fresh random per instance/epoch).

**Results**: mean 1.88x, median 1.88x,
min 1.44x, max 2.37x, win-rate
100%.
Mean % reduction: 45.8%, median 46.9%.
