"""
validate_epochs30_crossaxis.py
================================

QP validation for the epochs=30 zero-shot cross-axis checks
(fixed_market_epochs30_on_common_scale / _on_topology_scale, n=30 each).
"""

from __future__ import annotations

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
device = torch.device("cpu")
mc = market.MarketConfig(density=config.MARKET_DENSITY)
dr_cfg = algorithm.DRConfig(gamma=config.DR_GAMMA, lam=config.DR_LAMBDA,
                             n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                             bisect=config.DR_BISECT_STEPS)

base_nb, base_prim, base_spec = market.sample_base(config.BASE_MARKET_N, mc,
                                                    seed=config.BASE_MARKET_SEED)

model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=config.GNN_RHO, scale=config.GNN_SCALE,
    rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
blob = torch.load("runs/fixed_market_epochs30/phi_w.pt", map_location="cpu", weights_only=False)
model.load_state_dict(blob["state_dict"])
model.eval()


def check(name, specs):
    n_ok = 0
    worst = 0.0
    for i, spec in enumerate(specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device,
                                        seed=config.TEST_SEED * 7919 + i, record=True)
        net = market.build_network(spec, device=device)
        j_star = solve_equilibrium_qp(net)
        gap_dr = abs(res_dr.cost[-1] - j_star)
        gap_learned = abs(res_learned.cost[-1] - j_star)
        gap = max(gap_dr, gap_learned)
        worst = max(worst, gap)
        n_ok += gap <= TOL
    print(f"[{name}] {n_ok}/{len(specs)} genuine equilibria (tol={TOL:.0e}), worst gap {worst:.2e}")


specs_common = market.sample_common_scale_dataset(base_nb, base_prim, n_instances=30,
                                                    pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)
check("epochs30_on_common_scale", specs_common)

specs_topo = market.sample_common_scale_topology_dataset(base_nb, base_prim, mc, n_instances=30,
                                                           pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)
check("epochs30_on_topology_scale", specs_topo)
