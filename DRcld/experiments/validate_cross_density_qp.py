"""
validate_cross_density_qp.py
===============================

QP genuineness check (same pattern as validate_density10.py /
validate_asymmetric_seed0.py) for a cross_density checkpoint: confirms the
reported win-rate/speedup is genuine convergence to the true v-GNE, not a
residual-based stopping-rule artifact.

Usage: python3 validate_cross_density_qp.py <density> <seed> <gamma> <rho>
       <scale> <ckpt_path> [n_test]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import torch

import config
import market
import algorithm
import gnn as gnn_mod
from demo import run_pair
from equilibrium import solve_equilibrium_qp

TOL = 1e-4
DENSITY = float(sys.argv[1])
SEED = int(sys.argv[2])
GAMMA = float(sys.argv[3])
RHO = float(sys.argv[4])
SCALE = float(sys.argv[5])
CKPT = sys.argv[6]
N_TEST = int(sys.argv[7]) if len(sys.argv) > 7 else 30

device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
dr_cfg = algorithm.DRConfig(gamma=GAMMA, lam=config.DR_LAMBDA,
                             n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                             bisect=config.DR_BISECT_STEPS)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)
specs = [base_spec] * N_TEST

model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=RHO, scale=SCALE, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
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
        worst, worst_i = gap, i
    n_ok += gap <= TOL
print(f"[{CKPT}] {n_ok}/{N_TEST} genuine equilibria (tol={TOL:.0e}), worst gap {worst:.2e} (instance {worst_i+1})")
