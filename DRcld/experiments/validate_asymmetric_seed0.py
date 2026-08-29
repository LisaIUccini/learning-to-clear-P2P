"""
validate_asymmetric_seed0.py
==============================

QP validation for the asymmetric_loss breakthrough result: density=1.0,
seed=0, gamma=4.0, unroll=300, rho=0.9667, scale=1.0, loss=asymmetric
(runs/density_1.0/seed0_gamma4_unroll300_scale10_epochs60_asymmetric).
Confirms the reported 100% win-rate / 1.60x mean is genuine convergence to
the true v-GNE (cost gap <= tol), not a residual-based stopping-rule
artifact.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch

import config
import market
import algorithm
import gnn as gnn_mod
from demo import run_pair
from equilibrium import solve_equilibrium_qp

TOL = 1e-4
DENSITY = 1.0
SEED = 0
GAMMA = 4.0
device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
dr_cfg = algorithm.DRConfig(gamma=GAMMA, lam=config.DR_LAMBDA,
                             n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                             bisect=config.DR_BISECT_STEPS)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)
specs = [base_spec] * 30

model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=0.9667, scale=1.0, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
CKPT = "runs/density_1.0/seed0_gamma4_unroll300_scale10_epochs60_asymmetric/phi_w.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(blob["state_dict"])
model.eval()

n_ok = 0
worst = 0.0
worst_i = -1
for i, spec in enumerate(specs):
    res_dr, res_learned = run_pair(spec, model, dr_cfg, device,
                                    seed=config.TEST_SEED * 7919 + i, record=True)
    net = market.build_network(spec, device=device)
    j_star = solve_equilibrium_qp(net)
    gap = max(abs(res_dr.cost[-1] - j_star), abs(res_learned.cost[-1] - j_star))
    if gap > worst:
        worst = gap
        worst_i = i
    n_ok += gap <= TOL
print(f"[{CKPT}] {n_ok}/{len(specs)} genuine equilibria (tol={TOL:.0e}), worst gap {worst:.2e} (instance {worst_i+1})")
