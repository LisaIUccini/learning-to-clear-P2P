"""Configuration of the official experiments (paper Sec. 5, App. C), one entry
per trading-graph density, and the helpers that rebuild their datasets.

The values reproduce the saved runs in results/official/ (checked by
tests/test_regression.py). They were recovered from the original launch
commands and run headers; the original commands are listed in
docs/official_runs.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from . import constants as C
from . import paths
from .fixed_point import DRConfig
from .gnn import GNNConfig, PerturbationGNN
from .market import MarketConfig, MarketSpec, sample_base
from .scenarios import sample_level_width_dataset

DENSITIES = (0.2, 0.4, 0.5, 0.6, 0.8, 1.0)

# Dataset sizes and in-distribution shift (paper Sec. 5.1)
N_TRAIN, N_VAL, N_TEST = 50, 10, 30
ID_PCT_LEVEL, ID_PCT_WIDTH = 0.10, 0.04
OOD_LEVELS = (0.20, 0.30, 0.40)    # OOD level shifts, all with OOD_PCT_WIDTH
OOD_PCT_WIDTH = 0.08


@dataclass(frozen=True)
class DensityConfig:
    """Everything that differs between densities."""

    base_seed: int        # seed of the base market
    gamma: float          # DR step size
    unroll: int           # K: training horizon
    rho: float            # fast-channel decay, 1 - 10/K
    scale: float          # fast-channel amplitude
    scale_slow: float     # slow-channel amplitude (0 = off)
    omega_mode: str       # iteration weights of the loss
    omega_power: float
    w_ahead: float        # loss slopes
    w_behind: float
    l2ws_unroll: int      # training horizon of the L2WS benchmarks
    a2dr_eta: float       # A2DR regularisation and memory
    a2dr_mmax: int


RHO_SLOW = 0.99
EPOCHS = 25
LR_FINAL_FRAC = 0.01

DENSITY: Dict[float, DensityConfig] = {
    0.2: DensityConfig(base_seed=3, gamma=4.0, unroll=426, rho=0.9765, scale=1.0, scale_slow=0.0,
                       omega_mode="fixed", omega_power=1.5, w_ahead=1.0, w_behind=2.0,
                       l2ws_unroll=46, a2dr_eta=1e-4, a2dr_mmax=3),
    0.4: DensityConfig(base_seed=3, gamma=4.0, unroll=318, rho=0.9686, scale=1.8, scale_slow=0.4,
                       omega_mode="fixed", omega_power=1.5, w_ahead=1.6, w_behind=6.0,
                       l2ws_unroll=318, a2dr_eta=1e-4, a2dr_mmax=5),
    0.5: DensityConfig(base_seed=1, gamma=6.0, unroll=570, rho=0.9825, scale=1.0, scale_slow=0.0,
                       omega_mode="fixed", omega_power=1.5, w_ahead=1.0, w_behind=2.0,
                       l2ws_unroll=570, a2dr_eta=1e-4, a2dr_mmax=5),
    0.6: DensityConfig(base_seed=3, gamma=4.0, unroll=321, rho=0.9688, scale=1.0, scale_slow=0.0,
                       omega_mode="fixed", omega_power=1.5, w_ahead=1.0, w_behind=2.0,
                       l2ws_unroll=321, a2dr_eta=3e-5, a2dr_mmax=5),
    0.8: DensityConfig(base_seed=0, gamma=4.0, unroll=1159, rho=0.9914, scale=1.8, scale_slow=0.4,
                       omega_mode="dr_slope", omega_power=0.0, w_ahead=1.6, w_behind=6.0,
                       l2ws_unroll=1159, a2dr_eta=1e-4, a2dr_mmax=3),
    1.0: DensityConfig(base_seed=2, gamma=6.0, unroll=1354, rho=0.9926, scale=1.8, scale_slow=0.4,
                       omega_mode="dr_slope", omega_power=0.0, w_ahead=1.6, w_behind=10.0,
                       l2ws_unroll=1354, a2dr_eta=3e-5, a2dr_mmax=5),
}

# L2WS benchmarks (shared across densities)
L2WS_EPOCHS = 50
L2WS_CENTRALIZED = dict(hidden_dim=64, n_hidden_layers=1, lr=3e-3,
                        gamma_mono=C.TRAIN_GAMMA_MONO, w_behind=C.TRAIN_W_BEHIND, w_ahead=C.TRAIN_W_AHEAD,
                        omega_mode="dr_slope", omega_power=0.0)
L2WS_DISTRIBUTED = dict(hidden_dim=32, n_rounds=3, lr=3e-3, shared_weights=True, update_type="residual",
                        gamma_mono=0.0, w_behind=2.0, w_ahead=2.0, omega_mode="dr_slope", omega_power=0.0)

# A2DR calibration grid (App. C.2): the chosen (eta, mmax) maximises the mean
# speed-up among the settings that converge on every validation instance;
# at density 1.0 no setting does and (3e-5, 5) was chosen by hand.
A2DR_ETA_GRID = (1e-8, 1e-6, 3e-5, 1e-4)
A2DR_MMAX_GRID = (3, 5, 10)


def market_config(density: float) -> MarketConfig:
    return MarketConfig(density=density)


def dr_config(density: float, **overrides) -> DRConfig:
    kw = dict(gamma=DENSITY[density].gamma, lam=C.DR_LAMBDA, n_iter=C.DR_MAX_ITER,
              tol=C.DR_TOLERANCE, bisect=C.DR_BISECT_STEPS)
    kw.update(overrides)
    return DRConfig(**kw)


def base_market(density: float) -> MarketSpec:
    return sample_base(C.BASE_MARKET_N, market_config(density), seed=DENSITY[density].base_seed)


def datasets(density: float, n_train: int = N_TRAIN, n_val: int = N_VAL,
             n_test: int = N_TEST) -> Tuple[List[MarketSpec], List[MarketSpec], List[MarketSpec]]:
    """(train, val, test): disjoint in-distribution sets seeded by TRAIN_SEED,
    +2 and +1; slot 0 of each is the base market."""
    base = base_market(density)

    def draw(n, seed):
        specs, _ = sample_level_width_dataset(base, n, seed=seed, pct_level=ID_PCT_LEVEL,
                                              pct_width=ID_PCT_WIDTH)
        return specs

    return draw(n_train, C.TRAIN_SEED), draw(n_val, C.TRAIN_SEED + 2), draw(n_test, C.TRAIN_SEED + 1)


def gnn_config(density: float) -> GNNConfig:
    d = DENSITY[density]
    return GNNConfig.from_market_config(market_config(density), rho=d.rho, scale=d.scale,
                                        rho_slow=RHO_SLOW, scale_slow=d.scale_slow,
                                        hidden_dim=C.GNN_HIDDEN_DIM)


def dr_gnn_dir(density: float):
    return paths.DR_GNN / paths.density_tag(density)


def load_model(path) -> PerturbationGNN:
    """A trained Phi_w, rebuilt from the configuration stored in its checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = PerturbationGNN(GNNConfig(**blob["cfg"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def load_official_model(density: float) -> PerturbationGNN:
    return load_model(dr_gnn_dir(density) / "phi_w.pt")
