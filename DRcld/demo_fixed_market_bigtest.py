"""
demo_fixed_market_bigtest.py
==============================

Zero-shot generalisation check, the "home turf" counterpart of
`demo_fixed_market_on_common_scale.py` / `demo_fixed_market_on_topology_scale.py`:
NO retraining. Loads the already-trained `fixed_market` model (zero
diversity of any kind during training -- only x0 varied) and runs
INFERENCE ONLY on a FRESH, LARGER test set built exactly the same way as
`demo_fixed_market.py`'s own test set -- the identical spec (same graph,
same parameters) repeated `n_test` times, with only the random initial
state x^0 varying per instance (`market.sample_base`, no perturbation
functions involved at all).

Whereas `demo_fixed_market.py`'s reported 92% win-rate / 1.40x-1.48x
came from only 12 test instances, this script re-tests the SAME trained
model on a much bigger (default 30-instance) fresh draw of x0's from the
same market, to get a more statistically robust read of "true" home-market
performance -- disjoint from the original 12 (different seed draws), so
it also doubles as an independent replication check.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod
from diagnostics import save_instance_plots
from demo import run_pair, evaluate, plot_results, write_run_notes


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "fixed_market_bigtest"
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else config.N_TEST_INSTANCES
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_test={n_test})")

    device = torch.device("cpu")
    mc = market.MarketConfig(density=config.MARKET_DENSITY)
    dr_cfg = algorithm.DRConfig(gamma=config.DR_GAMMA, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)

    print("=" * 70)
    print("loading the fixed_market model (trained on ZERO diversity,")
    print("only x0 varied during training) -- NO retraining here")
    print("=" * 70)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=config.GNN_RHO, scale=config.GNN_SCALE,
        rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load("runs/fixed_market/phi_w.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    print("\ndataset: the SAME spec as fixed_market's own base market (graph")
    print("and every parameter identical), repeated n_test times -- only the")
    print("random initial state x0 varies, exactly like demo_fixed_market.py's")
    print("own test set but bigger (fresh draws, not the original 12)")
    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=config.BASE_MARKET_SEED)
    test_specs = [base_spec] * n_test

    print("\nevaluating: plain DR vs DR + Phi_w (fixed_market weights), zero-shot")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="fixed-model-bigtest")

    print("\nrecording residual curves...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = f"trained on: 1 fixed market, zero diversity  |  tested on: same market, n={n_test} (zero-shot)"
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: zero-shot inference (NO retraining). Weights: `runs/fixed_market/phi_w.pt`
(trained on ZERO diversity of any kind -- only x0 varied).

**Dataset**: the identical spec used to train `fixed_market` (same graph,
same every parameter -- NO perturbation applied at all), repeated {n_test}
times. The only source of variation across instances is the random draw
of the initial state x^0 (`market.sample_base` + per-instance seeding via
`config.TEST_SEED`), exactly as in `demo_fixed_market.py`'s own test set,
just with more instances for a more statistically robust read.

**Question answered**: re-tests the fixed_market model on a bigger, freshly
drawn sample of starting points from its OWN training market, to check
whether the original 12-instance result (92% win-rate, 1.40x/1.48x mean/
median) holds up with more instances or was partly a small-sample artefact.
""")


if __name__ == "__main__":
    main()
