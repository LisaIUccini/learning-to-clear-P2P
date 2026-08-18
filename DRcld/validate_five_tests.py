"""
validate_five_tests.py
========================

One-off QP ground-truth validation for the 5-test suite (fixed_market
zero-shot on common/topology/home-market test sets, plus full retraining
on common-scale and topology-scale datasets). For every instance in every
test, solves the exact convex QP equilibrium (`equilibrium.solve_equilibrium_qp`
-- valid because this game is a potential game, so the v-GNE is the unique
minimiser of a single QP) and checks that BOTH plain DR's and DR+learned's
final cost are within `tol` of that optimum, confirming every reported
"speed-up" is genuine convergence to the true equilibrium, not a
stopping-rule artifact.

Not part of any demo_*.py script (keeps the QP solve, which is slow, out
of the normal run path) -- run standalone after the fact, same discipline
used for every other MEASURED RESULT claim in this project.
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


# --- test 1: fixed_market weights, zero-shot on common-scale (n=30) -------
model_fm = load_model("runs/fixed_market/phi_w.pt")
specs1 = market.sample_common_scale_dataset(base_nb, base_prim, n_instances=30,
                                             pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)
check("1: fixed_market_on_common_scale_n30", specs1, model_fm)

# --- test 2: fixed_market weights, zero-shot on topology-scale (n=30) -----
specs2 = market.sample_common_scale_topology_dataset(base_nb, base_prim, mc, n_instances=30,
                                                      pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)
check("2: fixed_market_on_topology_scale_n30", specs2, model_fm)

# --- test 3: fixed_market weights, zero-shot on home market (n=30) --------
specs3 = [base_spec] * 30
check("3: fixed_market_bigtest", specs3, model_fm)

# --- test 4: retrained on common-scale (n=30) ------------------------------
model_cs = load_model("runs/common_scale_retrain_n30/phi_w.pt")
specs4 = market.sample_common_scale_dataset(base_nb, base_prim, n_instances=30,
                                             pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)
check("4: common_scale_retrain_n30", specs4, model_cs)

# --- test 5: retrained on topology-scale (n=30) ----------------------------
model_ts = load_model("runs/topology_scale/phi_w.pt")
specs5 = market.sample_common_scale_topology_dataset(base_nb, base_prim, mc, n_instances=30,
                                                      pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)
check("5: topology_scale", specs5, model_ts)
