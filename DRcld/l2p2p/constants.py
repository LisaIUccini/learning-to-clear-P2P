"""Numerical constants shared by every official experiment.

Per-density hyperparameters (gamma, K, rho, loss weights, ...) are not here:
they live in `official.py`, next to the runs they produced.
"""

BASE_MARKET_N = 10        # prosumers in every base market

# Seeds. Train/test/val instance sets use TRAIN_SEED, TRAIN_SEED + 1 and
# TRAIN_SEED + 2; test-time initial points use TEST_SEED * 7919 + i.
TRAIN_SEED = 0
TEST_SEED = 100

# Douglas-Rachford iteration (Algorithm 1)
DR_LAMBDA = 1.0           # relaxation, in (0, 2)
DR_MAX_ITER = 4000        # iteration cap
DR_TOLERANCE = 1e-5       # stop when ||x^{k+1} - x^k|| < tol
DR_BISECT_STEPS = 30      # bisection steps of each local proximal step

# Learned correction Phi_w and its offline training
GNN_HIDDEN_DIM = 32
TRAIN_LR = 3e-3
TRAIN_GRAD_CLIP = 1.0
TRAIN_GAMMA_MONO = 2.0    # weight of the monotonicity penalty (beta in the paper)
TRAIN_W_BEHIND = 2.0      # default asymmetric-loss slopes
TRAIN_W_AHEAD = 1.0
