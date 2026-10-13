# gmin_only_market_v1

**Test**: fixed-x0/parameter-diversity design (same as v6, the project's best fixed-x0 result,
1.25x/100%), but the base market changes ONLY `Gmin_range`: (-2,0) -> (0,2) (old abstract-market
defaults otherwise, NOT `REALISTIC_MARKET_CONFIG`'s independently-chosen kW-scale ranges for
Gmax/G_res/D_spread). Isolates whether G>=0 alone (the one change the user always understood to be
there, vs `seed0_density1.0_seed0_drslope`'s own base market) explains any of the gap to that
reference's 2.09x/100%, separate from the other three range changes.

**Calibration**: unroll=282, gnn_rho=0.9645, freshly measured on this exact dataset (fixed,
unclipped x0) via the project's standing rule: mean=564.5, cv=0.1006, n=40.

**Recipe**: identical to v6 otherwise -- w_ahead=1.4, w_behind=6.0, gamma_mono=2.0,
omega_mode=dr_slope, gnn_scale=1.0, level+width perturbation +/-20%/+/-8%.

**Results**: mean 1.37x, median 1.36x,
min 1.07x, max 1.62x, win-rate
100%
(30/30).
