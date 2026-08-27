"""
demo_fixed_market_on_independent_params.py
=============================================

Zero-shot generalisation check: NO retraining. Loads the already-trained
`fixed_market` model (zero diversity of any kind during training) and
runs INFERENCE ONLY on a test set where the graph stays EXACTLY fixed but
each individual parameter of each prosumer/edge is jittered
INDEPENDENTLY within +/-pct (`market.sample_param_only_dataset`) -- NOT
one shared factor for the whole market (contrast
`demo_fixed_market_on_common_scale.py`). Some parameters can end up
barely changed, others near the +/-10% edge, uncorrelated across the
market.

Completes the 2x2 zero-shot picture on the `fixed_market` model:
    shared factor       vs.  independent per-parameter
    parameters only      -> demo_fixed_market_on_common_scale.py   vs.  this script
    topology only         -> demo_fixed_market_on_topology_scale.py
"""

from __future__ import annotations

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
from diagnostics import save_instance_plots
from demo import run_pair, evaluate, plot_results


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "fixed_market_on_independent_params"
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/")

    device = torch.device("cpu")
    mc = market.MarketConfig(density=config.MARKET_DENSITY)
    dr_cfg = algorithm.DRConfig(gamma=config.DR_GAMMA, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)

    print("=" * 70)
    print("loading the fixed_market model (trained on ZERO diversity) --")
    print("NO retraining here")
    print("=" * 70)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=config.GNN_RHO, scale=config.GNN_SCALE,
        rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load("runs/fixed_market/phi_w.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    print("\ndataset: graph EXACTLY fixed; EVERY individual parameter of")
    print("EVERY prosumer/edge jittered INDEPENDENTLY within +/-10%")
    print("(not a shared factor -- see market.sample_param_only_dataset)")
    base_nb, base_prim, base_spec = market.sample_base(config.BASE_MARKET_N, mc,
                                                        seed=config.BASE_MARKET_SEED)
    test_specs = market.sample_param_only_dataset(
        base_nb, base_prim, mc, n_instances=config.N_TEST_INSTANCES,
        pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)

    print("\nevaluating: plain DR vs DR + Phi_w (fixed_market weights), zero-shot")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         label="fixed-model-on-independent-params")

    print("\nrecording residual curves...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = f"trained on: 1 fixed market, zero diversity  |  tested on: params jittered independently +/-{config.DATASET_PCT*100:.0f}% each, n={config.N_TEST_INSTANCES} (zero-shot)"
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)


if __name__ == "__main__":
    main()
