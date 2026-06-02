
"""
L2Play — Centralized configuration.
 
All hyperparameters, paths, and runtime settings live here.
Import this module from main.py (or from anywhere else that needs them).
"""
 
import os
import torch
 
 
# ------------------------------------------------------------------
# Device and reproducibility
# ------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_TRAIN = 42     # seed for training-time randomness
SEED_TEST  = 0      # seed for test-time aggregation
 
 
# ------------------------------------------------------------------
# Network setup
# ------------------------------------------------------------------
N_TRAIN  = 25        # network size used for meta-training
N_CHOICE = 23       # network size used at inference time
LOAD     = True     # if True and checkpoint exists, skip training
 
 
# ------------------------------------------------------------------
# Meta-training hyperparameters
# ------------------------------------------------------------------
NUM_SAMPLES = 50    # number of initial states in the training dataset
BATCH_SIZE  = 1     # samples per gradient update
T_UNROLL    = 30    # number of unrolled ADMM steps per sample
EPOCHS      = 40
LR          = 3e-3
ALPHA       = 0.95  # kept for API compatibility (loss is undiscounted)
 
 
# ------------------------------------------------------------------
# GNNUpdate hyperparameters
# ------------------------------------------------------------------
D_H            = 12  # GNN hidden dimension
RHO_GNN        = 0.5 # geometric decay m_k = phi_m * RHO_GNN^k  (summability)
NODE_FEAT_SIZE = 3   # [D, G, u_avg]
EDGE_FEAT_SIZE = 3   # [q, z, u]
 
 
# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
N_TEST = 5         # number of test instances over which to average
 
 
# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"
FIGURES_DIR    = "figures"
 
 
def checkpoint_path() -> str:
    """Filename for the trained GNN weights."""
    return os.path.join(
        CHECKPOINT_DIR,
        f"admm_N{N_TRAIN}_T{T_UNROLL}_e{EPOCHS}_s{NUM_SAMPLES}_learned_update"
        f"_rho{RHO_GNN}_lr{LR}.pt",
    )
 
 
def loss_history_path() -> str:
    """Filename for the training loss-history plot."""
    return os.path.join(
        CHECKPOINT_DIR,
        f"admm_N{N_TRAIN}_T{T_UNROLL}_e{EPOCHS}_s{NUM_SAMPLES}"
        f"_loss_history_rho{RHO_GNN}_lr{LR}.png",
    )
 
 
def aggregate_figure_path() -> str:
    """Filename for the aggregated inference residual plot."""
    return os.path.join(
        FIGURES_DIR,
        f"admm_N{N_CHOICE}_average_trainingN{N_TRAIN}.png",
    )
 