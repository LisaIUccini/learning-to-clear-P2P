"""
config.py
=========

Every tunable parameter used by the current best/default pipeline, in one
place, grouped by what they control. This is the SOURCE OF TRUTH for the
`fixed_market` recipe (validated: 92% win-rate, 1.40x/1.48x mean/median --
see runs/INDEX.md) -- demo scripts import from here instead of hardcoding
values inline, so changing a number here changes every script that uses
it, and there is exactly one place to look to see what a run actually used.

Each group corresponds to a dataclass constructed elsewhere in the
codebase (market.MarketConfig, algorithm.DRConfig, train.TrainConfig,
gnn.GNNConfig) -- this file only collects the VALUES, it does not
redefine any of those classes.

To run an experiment with different numbers: either edit the values below
(affects every script that imports them) or pass overrides directly to the
dataclass constructor in your own script, e.g.
`train_mod.TrainConfig(..., unroll=400)` -- see demo_fixed_market_longhorizon.py
for an example of a script that intentionally overrides a value from here.
"""

# ======================================================================
# Market generation (market.MarketConfig, market.sample_base)
# ======================================================================
MARKET_DENSITY = 0.3       # graph edge density in [0,1]: 0 = spanning tree
                            # (sparsest connected graph), 1 = complete graph
BASE_MARKET_N = 10          # number of prosumers in the reference market
BASE_MARKET_SEED = 0        # seed that generates the reference market itself
                            # (graph + parameters) -- every experiment so far
                            # has used the SAME reference market for comparability

# --- dataset variation (how much instances differ from the reference) ---
DATASET_PCT = 0.10          # +/- fraction of variation between instances (10%)
N_TRAIN_INSTANCES = 25      # including the base/reference instance itself
N_TEST_INSTANCES = 12

# Seed for evaluation/inference (NOT training): used for x0 generation in
# evaluate(), the recorded residual curves, and save_instance_plots() --
# the SAME seed everywhere so "instance i" means the same random starting
# point x0 in the printed table, the aggregate median-band plot, and the
# per-instance diagnostic images. (These three used to disagree -- the
# curves were recorded with a different seed base than the table/plots --
# now fixed to all reference this one constant.)
TEST_SEED = 100

# ======================================================================
# Douglas-Rachford solver (algorithm.DRConfig) -- used both for real
# deployment/evaluation AND to compute the plain-DR baseline trajectory
# during training (the "beat DR" signal the loss compares against)
# ======================================================================
DR_GAMMA = 4.0               # prox step size -- tuned so PLAIN DR itself
                              # converges reliably and reasonably fast; see
                              # the gamma-sweep note in runs/INDEX.md before
                              # changing this. A badly tuned baseline makes
                              # every downstream number meaningless.
DR_LAMBDA = 1.0               # DR relaxation parameter, must lie in (0,2)
DR_MAX_ITER = 4000             # hard iteration cap (safety net, not expected
                              # to bind for well-tuned gamma)
DR_TOLERANCE = 1e-5           # convergence tolerance on ||x^{k+1}-x^k||
DR_BISECT_STEPS = 30           # bisection steps inside each local prox solve

# ======================================================================
# Perturbation GNN (gnn.GNNConfig / gnn.PerturbationGNN)
# ======================================================================
GNN_HIDDEN_DIM = 32          # width of every MLP inside Phi_w (independent
                              # of N and |E| -- this is what makes one trained
                              # model deployable on any graph size/topology)

# fast channel: the one actually in use. SHORT/safe decay -- do not raise
# without also addressing the "generic pace" collapse documented in
# runs/fixed_market_longhorizon (raising it made things WORSE, not
# better, with or without cross-instance diversity).
GNN_RHO = 0.95
GNN_SCALE = 0.4

# slow channel: architecturally present (gnn.py) but INACTIVE by design --
# GNN_SCALE_SLOW = 0.0 makes its contribution exactly zero regardless of
# what it learns (v^k_slow = scale_slow * decay_slow(k) * direction, so
# scale_slow=0 zeroes it identically). See runs/a2b1_dual_channel_curriculum
# for why it is off: combined with a hard-instance curriculum it collapsed
# to an indiscriminate policy (8% win-rate).
GNN_RHO_SLOW = 0.99
GNN_SCALE_SLOW = 0.0

# ======================================================================
# Training (train.TrainConfig)
# ======================================================================
TRAIN_UNROLL = 200            # K: DR iterations differentiated through per
                              # training step. See GNN_RHO's note -- these
                              # two are linked, do not change one without
                              # thinking about the other.
TRAIN_GAMMA_MONO = 2.0        # monotonicity-penalty weight (both hinge and
                              # asymmetric loss use this the same way)

TRAIN_LOSS_MODE = "hinge"     # "hinge"      -- margin vs. plain DR, targets
                              #                 win-rate; magnitude capped at
                              #                 the margin by construction
                              # "asymmetric" -- uncapped, continuous reward/
                              #                 penalty; needs TRAIN_W_BEHIND/
                              #                 TRAIN_W_AHEAD
                              # "absolute", "relative", "log_ratio" -- kept
                              #                 for the record, not recommended;
                              #                 see train.py's module docstring
TRAIN_ETA = 0.15              # required fractional margin over plain DR
                              # ("hinge" mode only)
TRAIN_W_BEHIND = 2.0          # penalty weight for lagging plain DR
                              # ("asymmetric" mode only)
TRAIN_W_AHEAD = 1.0           # reward weight for beating plain DR
                              # ("asymmetric" mode only)

TRAIN_EPOCHS = 20
TRAIN_LR = 3e-3
TRAIN_GRAD_CLIP = 1.0
TRAIN_SEED = 0
