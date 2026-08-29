"""
replot_percentage.py
======================

Regenerate results.png for an already-trained checkpoint with the new
percentage-based plot_results (no retraining -- reloads weights, rebuilds
the exact same deterministic test set, re-evaluates, replots).

Usage:
  python3 replot_percentage.py fixed <out_dir> <weights_path> <base_seed>
      <density> <dr_gamma> <gnn_rho> <gnn_scale> <n_test>
  python3 replot_percentage.py dstar_gres <out_dir> <weights_path>
      <base_seed> <density> <dr_gamma> <gnn_rho> <gnn_scale> <n_test> <pct>
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod
from demo import run_pair, evaluate, plot_results
from demo_fixed_market_on_dstar_gres_scale import build_dstar_gres_dataset

mode = sys.argv[1]
out_dir = sys.argv[2]
weights_path = sys.argv[3]
base_seed = int(sys.argv[4])
density = float(sys.argv[5])
dr_gamma = float(sys.argv[6])
gnn_rho = float(sys.argv[7])
gnn_scale = float(sys.argv[8])
n_test = int(sys.argv[9])

device = torch.device("cpu")
mc = market.MarketConfig(density=density)
dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                             n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                             bisect=config.DR_BISECT_STEPS)
model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=gnn_rho, scale=gnn_scale, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
blob = torch.load(weights_path, map_location="cpu", weights_only=False)
model.load_state_dict(blob["state_dict"])
model.eval()

_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)

if mode == "fixed":
    test_specs = [base_spec] * n_test
elif mode == "dstar_gres":
    pct = float(sys.argv[10])
    test_specs, _, _ = build_dstar_gres_dataset(base_spec, n_instances=n_test, pct=pct, seed=config.TRAIN_SEED + 3)
else:
    raise ValueError(mode)

test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="replot")

curves_dr, curves_learned = [], []
for i, spec in enumerate(test_specs):
    res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
    curves_dr.append(np.asarray(res_dr.residual))
    curves_learned.append(np.asarray(res_learned.residual))
test_res["curves_dr"] = curves_dr
test_res["curves_learned"] = curves_learned

out_path = os.path.join(out_dir, "results.png")
plot_results(test_res, dr_cfg.tol, out_path=out_path, subtitle="(replotted with % iteration reduction)")
print(f"replotted {out_path}")
