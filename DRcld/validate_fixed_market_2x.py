"""
validate_fixed_market_2x.py
=============================

QP ground-truth validation for the fixed_market_2x experiment (2x epochs,
2x training instances) and its two zero-shot cross-axis checks, same
discipline as validate_five_tests.py.
"""

from __future__ import annotations

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


def load_model(weights_path):
    m = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=config.GNN_RHO, scale=config.GNN_SCALE,
        rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m


def check(name, specs, model):
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
        ok = gap <= TOL
        n_ok += ok
        if not ok:
            print(f"    instance {i+1}: gap_dr={gap_dr:.2e}  gap_learned={gap_learned:.2e}  "
                  f"J*={j_star:.6f}  NOT within tol")
    print(f"[{name}] {n_ok}/{len(specs)} genuine equilibria (tol={TOL:.0e}), worst gap {worst:.2e}")


model_2x = load_model("runs/fixed_market_2x/phi_w.pt")

specs_home = [base_spec] * 30
check("fixed_market_2x (home, n=30)", specs_home, model_2x)

specs_common = market.sample_common_scale_dataset(base_nb, base_prim, n_instances=30,
                                                    pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)
check("fixed_market_2x_on_common_scale (n=30)", specs_common, model_2x)

specs_topo = market.sample_common_scale_topology_dataset(base_nb, base_prim, mc, n_instances=30,
                                                           pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)
check("fixed_market_2x_on_topology_scale (n=30)", specs_topo, model_2x)
