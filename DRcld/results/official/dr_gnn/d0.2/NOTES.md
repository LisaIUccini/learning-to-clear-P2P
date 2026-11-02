# official_N10_d0.2_seed3_replicate

**Official pilot** -- recalibrated market generator (build_calibrated_market.py), density=0.2,
base_seed=3. gamma*=4.0 (plateau-center grid search, seed=0), lambda=1.0
(kept at project default -- a 2026-10-14 lambda sweep found 15-40% gains at lambda>1 for most
densities, deferred to future work per user decision). rho_variant=10 (gnn_rho=0.9765,
unroll=426).

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4% perturbation,
x0 free (fresh random per instance/epoch).

**Results**: mean 6.54x, median 6.02x,
min 3.17x, max 13.89x, win-rate
100%.
Mean % reduction: 83.0%, median 83.4%.
