"""
demo_fixed_market_on_combined_scale.py
=========================================

Zero-shot generalisation check, the COMBINED counterpart of
`demo_fixed_market_on_common_scale.py` / `demo_fixed_market_on_topology_scale.py`:
NO retraining. Loads an already-trained fixed_market model (zero
diversity of any kind during training -- only x0 varied) and runs
INFERENCE ONLY on a test set where BOTH axes vary together -- market
parameters AND graph topology each shift via their OWN independently-
drawn shared factor in [1-pct, 1+pct] per instance
(`market.sample_common_scale_combined_dataset`), rather than either axis
alone.

Together with the two single-axis checks, this answers a question neither
can alone: does combining two individually-survivable coherent shifts
compound into a harder generalisation problem, or does the model handle
them roughly independently (e.g. topology-scale's degradation dominating,
since that was already the harder axis)?
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
from demo import run_pair, evaluate, plot_results, write_run_notes


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "fixed_market_on_combined_scale"
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else config.N_TEST_INSTANCES
    weights_path = sys.argv[3] if len(sys.argv) > 3 else "runs/fixed_market/phi_w.pt"
    density = float(sys.argv[4]) if len(sys.argv) > 4 else config.MARKET_DENSITY
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_test={n_test}  weights={weights_path}  density={density})")

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density)
    dr_cfg = algorithm.DRConfig(gamma=config.DR_GAMMA, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)

    print("=" * 70)
    print(f"loading {weights_path} (trained on ZERO diversity,")
    print("never saw parameter OR topology variation) -- NO retraining here")
    print("=" * 70)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=config.GNN_RHO, scale=config.GNN_SCALE,
        rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    print("\ndataset: BOTH axes vary together -- graph AND parameters each")
    print("shift via their OWN independent shared +/-10% factor per instance")
    base_nb, base_prim, base_spec = market.sample_base(config.BASE_MARKET_N, mc,
                                                        seed=config.BASE_MARKET_SEED)
    test_specs = market.sample_common_scale_combined_dataset(
        base_nb, base_prim, mc, n_instances=n_test,
        pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)

    print("\nevaluating: plain DR vs DR + Phi_w, zero-shot")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="fixed-model-on-combined-scale")

    print("\nrecording residual curves...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = f"trained on: 1 fixed market, zero diversity  |  tested on: graph AND params each shared +/-{config.DATASET_PCT*100:.0f}% factor, n={n_test} (zero-shot)"
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: zero-shot inference (NO retraining). Weights: `{weights_path}`
(trained on ZERO diversity of any kind -- only x0 varied).

**Dataset**: BOTH the graph and the market parameters vary per instance, each
via its OWN independently-drawn shared random factor in [1-{config.DATASET_PCT}, 1+{config.DATASET_PCT}]
(`market.sample_common_scale_combined_dataset`) -- two separate draws, not
one factor driving both (topology and parameter scale are conceptually
different kinds of shift). {n_test} test instances, disjoint draws from
the training data.

**Question answered**: does combining two individually-survivable coherent
shifts (parameters alone: ~97% zero-shot; topology alone: ~70-83% zero-shot,
depending on how much the base model was trained) compound into a harder
generalisation problem, or is the degradation dominated by whichever axis
is already the harder one? Directly comparable to
`fixed_market_on_common_scale` and `fixed_market_on_topology_scale` (the
single-axis zero-shot checks) using the SAME weights.
""")


if __name__ == "__main__":
    main()
