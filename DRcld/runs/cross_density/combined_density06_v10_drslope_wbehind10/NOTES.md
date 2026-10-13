# combined_density06_v10_drslope_wbehind10

**Test**: COMBINED diversity design -- market instances perturbed via level+width
(+/-10%/+/-4%) AND x0 left FREE (fresh random per instance/epoch), instead of
fixing one axis while varying the other (v6: fixed x0 + market diversity; v4: fixed market + x0
diversity). Realistic market, density=1.0, base_seed=0.

**Motivation**: a zero-shot test of v4's checkpoint (x0-diversity-only) on this exact perturbed
distribution looked deceptively strong (1.55x/97%) evaluated with fresh random x0, but a controlled
re-test on v6's own protocol (same test set, v6's single fixed x0) showed only 1.08x/70% -- below
v6's own dedicated training (1.25x/100%). This run trains directly on the combined distribution
instead of relying on zero-shot transfer from either single-axis model.

**Calibration**: unroll=321, gnn_rho=0.9688, freshly measured on this exact combined
distribution (level+width perturbed market + fresh random x0, n=40): mean=771.2, cv=0.2426.

**Recipe**: w_ahead=1.6, w_behind=10.0, gamma_mono=2.0, omega_mode=dr_slope,
gnn_scale=1.8, gnn_rho_slow=0.99, gnn_scale_slow=0.4 (slow channel kept
ON, matching v4's own config, since this design extends v4's x0-diversity mechanism).

**Results**: mean 1.26x, median 1.24x,
min 1.02x, max 1.54x, win-rate
100%
(30/30).
Mean % reduction (per-instance average, the correct definition): 19.7%, median 19.3%.

**Density**: 0.6 (base_seed=3).
