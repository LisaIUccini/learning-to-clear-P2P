"""
demo_fixed_market_on_dstar_gres_scale.py
==========================================

Zero-shot generalisation check, NO retraining: loads an already-trained
checkpoint and runs inference on a custom perturbed test set built from
the SAME base market (graph, costs a/a_tilde/b/c, capacities kappa, and
box bounds D_min/D_max/G_min/G_max ALL held exactly fixed) where, per
instance, only D_star and G_res are rescaled:

    D_star_new = clip(D_star * f_D, D_min, D_max),   f_D ~ U(1-pct, 1+pct)
    G_res_new  = G_res * f_G,                        f_G ~ U(1-pct, 1+pct)

f_D and f_G are two INDEPENDENT draws, each SHARED across all N nodes
(one f_D and one f_G per instance, not per node). Unlike
`market.sample_common_scale_dataset` (which perturbs primitives and
re-derives D_min/D_max/G_min/G_max through build_spec), this perturbs the
already-built spec's D_star/G_res directly and leaves every other field,
including the boxes, untouched -- exactly what was asked for.

Usage: python3 demo_fixed_market_on_dstar_gres_scale.py <run_name>
       <weights_path> <base_seed> <density> <dr_gamma> <gnn_rho>
       <gnn_scale> [n_test] [pct]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import dataclasses
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


def build_dstar_gres_dataset(base_spec, n_instances: int, pct: float, seed: int):
    specs, log_rows = [], []
    n_infeasible = 0
    for i in range(n_instances):
        rng = np.random.default_rng(seed * 100_003 + i)
        f_D = float(rng.uniform(1 - pct, 1 + pct))
        f_G = float(rng.uniform(1 - pct, 1 + pct))
        D_star_new = np.clip(base_spec.D_star * f_D, base_spec.D_min, base_spec.D_max)
        G_res_new = base_spec.G_res * f_G
        spec = dataclasses.replace(base_spec, D_star=D_star_new, G_res=G_res_new)
        ok = market.spec_is_feasible(spec)
        n_infeasible += (not ok)
        specs.append(spec)
        log_rows.append((i, f_D, f_G, ok))
    return specs, log_rows, n_infeasible


def main():
    run_name = sys.argv[1]
    weights_path = sys.argv[2]
    base_seed = int(sys.argv[3])
    density = float(sys.argv[4])
    dr_gamma = float(sys.argv[5])
    gnn_rho = float(sys.argv[6])
    gnn_scale = float(sys.argv[7])
    n_test = int(sys.argv[8]) if len(sys.argv) > 8 else 30
    pct = float(sys.argv[9]) if len(sys.argv) > 9 else 0.10

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (weights={weights_path}  base_seed={base_seed}  "
          f"density={density}  dr_gamma={dr_gamma}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"n_test={n_test}  pct={pct})")

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)

    print("=" * 70)
    print(f"loading {weights_path} -- NO retraining, inference only")
    print("=" * 70)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale,
        rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}")

    print(f"\ndataset: graph/costs/capacities/boxes ALL fixed; D_star *= f_D, G_res *= f_G, "
          f"f_D,f_G ~ U({1-pct:.2f},{1+pct:.2f}) independently, shared across nodes, per instance")
    test_specs, log_rows, n_infeasible = build_dstar_gres_dataset(
        base_spec, n_instances=n_test, pct=pct, seed=config.TRAIN_SEED + 3)
    print(f"  {n_infeasible}/{n_test} instances flagged infeasible (Assumption 1 check)")
    for i, f_D, f_G, ok in log_rows:
        print(f"  instance {i + 1:2d}/{n_test}  f_D={f_D:.4f}  f_G={f_G:.4f}  feasible={ok}")

    print("\nevaluating: plain DR vs DR + Phi_w, zero-shot (D*/G_res shifted, no retraining)")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="dstar-gres-scale")

    print("\nrecording residual curves...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = (f"weights: {os.path.basename(os.path.dirname(weights_path))}  |  tested on: D*,G_res each "
                f"scaled by their OWN shared +/-{pct*100:.0f}% factor, boxes/graph/costs fixed, n={n_test} (zero-shot)")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: zero-shot inference (NO retraining). Weights: `{weights_path}`.

**Dataset**: base market density={density}, base_seed={base_seed} (N={base_spec.N}).
Graph, a/a_tilde/b/c, kappa, and box bounds (D_min/D_max/G_min/G_max) ALL
held exactly fixed. Per instance, D_star *= f_D and G_res *= f_G, with
f_D, f_G independently drawn ~ U({1-pct:.2f}, {1+pct:.2f}), each SHARED
across all N nodes (one f_D, one f_G per instance). D_star is clipped back
into [D_min, D_max] after scaling. {n_infeasible}/{n_test} instances
flagged infeasible by the Assumption-1 check.

**Question answered**: how does a model trained only on x0-diversity (this
exact market, unperturbed) perform when D* and G_res individually drift by
up to {pct*100:.0f}%, with everything else -- including the boxes it was
implicitly calibrated against -- held fixed?
""")


if __name__ == "__main__":
    main()
