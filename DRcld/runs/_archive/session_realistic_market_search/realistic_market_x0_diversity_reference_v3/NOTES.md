# cross_density/realistic_market_x0_diversity_reference_v3

**Test**: reproduces `seed0_density1.0_seed0_drslope`'s exact recipe (x0-diversity, one fixed
market, gamma=4.0, unroll=370, GNN_RHO=0.973, GNN_SCALE=1.0, loss=asymmetric,
w_behind=2.0, w_ahead=1.0, omega_mode='dr_slope', n_train=25, epochs=60) on
the REALISTIC market (density=1.0, N0=10, base_seed=0, G>=0 by construction) instead of the old
abstract-unit default market -- so it is directly comparable, on the SAME market, to the
fixed-x0/parameter-diversity axis (v1-v13, best result v5: 1.25x/100%).

**Motivation**: a diagnostic found `seed0_density1.0_seed0_drslope` catastrophically fails
zero-shot on a 6-parameter perturbation of the REALISTIC market (0.56-0.70x, near-0% win-rate) but
scores 1.55x/100% zero-shot on the SAME perturbation of ITS OWN (old) market -- proving the earlier
failure was a market-mismatch artifact, not evidence against generalizing across parameter
perturbation. This run removes that confound entirely.

**Dataset**: ONE fixed market, 25 training "instances" (all literally the same spec), x0
freshly randomized per (instance, epoch) during training (not fixed, unlike v1-v13) and per test
instance at evaluation -- 30 test instances, all the same spec, fresh random x0 each.

**Results (home market, x0-diversity only)**: mean 0.60x, median
0.59x, min 0.37x,
max 0.94x, win-rate
0%
(0/30).

**Question answered**: does x0-diversity training on the SAME realistic market v5 uses match
`seed0_density1.0_seed0_drslope`'s 2.09x/100% home-market result? This checkpoint is also intended
for a follow-up zero-shot test on v5-style 6-parameter perturbations, to get a true apples-to-apples
comparison against v5's dedicated parameter-diversity training (1.25x/100%).
