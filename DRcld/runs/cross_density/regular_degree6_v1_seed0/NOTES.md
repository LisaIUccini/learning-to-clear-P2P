# regular_degree6_v1_seed0

**Test**: DEGREE-HETEROGENEITY ISOLATION -- same combined (level+width + free x0) training recipe
as the density sweep, but the base topology is k-REGULAR (every node has EXACTLY degree=6
neighbors, zero degree heterogeneity) instead of density-driven random (which has meaningful
degree variance even at the same mean). degree=6 is the densest non-complete k-regular
graph possible on N=10 nodes, chosen to match density=0.8's mean degree (7.6) as closely as a
regular graph can, while staying short of the complete graph (density=1.0's own degenerate case).

**Motivation**: isolates whether density=0.8 outperforming density=1.0 (and more generally, the
zigzag shape of % improvement vs density) is driven by degree HETEROGENEITY specifically, as
opposed to average connectivity/edge count alone. If this regular-graph run's % improvement is
close to density=0.8's (~67%), heterogeneity isn't the driver. If it drops toward density=1.0's
level (~38-42%) despite matching 0.8's edge count, heterogeneity is implicated.

**Calibration**: unroll=490, gnn_rho=0.9796, freshly measured on THIS regular topology
(base_seed=0, degree=6, level+width perturbed + fresh x0, n=40).
Note: 2 of 6 candidate seeds for this topology (seeds 0 and 4) showed catastrophic DR
non-convergence (39/40 and 21/40 hitting the iteration cap) -- WORSE than any random-graph density
seed tested so far -- suggesting regular/symmetric topologies may be especially prone to DR limit
cycles. base_seed=0 was chosen as a clean, well-converging seed among the remainder.

**Recipe**: w_ahead=1.6, w_behind=6.0, gamma_mono=2.0, omega_mode=dr_slope,
gnn_scale=1.8, gnn_rho_slow=0.99, gnn_scale_slow=0.4 -- matching
density=0.8's own champion recipe as closely as possible, so the ONLY deliberate difference is
topology regularity.

**Results**: mean 2.72x, median 2.64x,
min 1.56x, max 4.39x, win-rate
100%
(30/30).
Mean % reduction (per-instance average, the correct definition): 61.2%, median 62.1%.

**Compare to**: density=0.8 (combined_density08_v1): 67.7% mean / 69.4% median.
density=1.0 (combined_v17_wahead16_s18): 38.5% mean / 39.8% median.
