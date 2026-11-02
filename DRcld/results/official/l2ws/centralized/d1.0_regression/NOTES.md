# l2ws_N10_d1.0_seed2_regression

**L2WS benchmark** (Sambharya, Hall, Amos & Stellato, JMLR 2024 -- "Learning to Warm-Start
Fixed-Point Optimization Algorithms"), applied to this project's DR p2p market-clearing fixed
point. A feedforward MLP h_w(theta) predicts ONLY the starting point z^0 for plain,
UN-perturbed DR -- the per-iteration DR operator itself is completely unmodified, unlike this
project's own Phi_w (which perturbs every step). Compared against Phi_w results on the SAME
(density=1.0, base_seed=2) combination as the most directly comparable benchmark.

density=1.0, base_seed=2, gamma=6.0 (same as the Phi_w run on this
combination), training unroll k=1354 (same calibrated value used for Phi_w), hidden_dim=
64, n_hidden_layers=1, lr=0.003, loss_mode=regression.

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4%
perturbation, same "old" generator + c<b fix (c_old_range=(0.05, 0.18)) as
train_official_pilot.py.

**Baseline**: cold-start plain DR from a fresh random x0 (this project's and the paper's own
baseline) -- NOT the same-x0 isolation design used for Phi_w, since a warm start's entire point
is choosing a BETTER starting point, not correcting the trajectory from a shared one.

**Results**: mean 2.49x, median 2.19x,
min 0.94x, max 4.38x, win-rate
83%.
Mean % reduction: 45.9%, median 54.3%.
