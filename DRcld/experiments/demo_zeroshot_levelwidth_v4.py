"""
demo_zeroshot_levelwidth_v4.py
==========================================

ZERO-SHOT, INFERENCE ONLY (no training): like `demo_zeroshot_dstar_gres_v4.py`,
but perturbs ALL SIX quantities (D_star, G_res, D_min, D_max, G_min, G_max)
via the project's "level+width" scheme (`sample_level_width_perturbed_dataset`)
instead of D*/G_res alone -- box bounds move too. x0 FRESH RANDOM per test
instance (matching the `realistic_market_x0_diversity_v4_wbehind6` checkpoint's
own x0-diversity training convention).

Usage: python3 demo_zeroshot_levelwidth_v4.py
       [weights_path] [n_test] [pct_level] [pct_width] [dr_gamma]
       [gnn_rho] [gnn_scale] [gnn_rho_slow] [gnn_scale_slow]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

import torch

import config
import algorithm
import gnn as gnn_mod
from demo import plot_results
from plot_style import stats_from
from demo_train_realistic_market_x0_diversity import evaluate_with_curves
from build_realistic_training_dataset import build_realistic_base_market, sample_level_width_perturbed_dataset, REALISTIC_MARKET_CONFIG


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "runs/cross_density/realistic_market_x0_diversity_v4_wbehind6/phi_w.pt"
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    pct_level = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20
    pct_width = float(sys.argv[4]) if len(sys.argv) > 4 else 0.08
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    gnn_rho = float(sys.argv[6]) if len(sys.argv) > 6 else 0.973
    gnn_scale = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    gnn_rho_slow = float(sys.argv[8]) if len(sys.argv) > 8 else 0.99
    gnn_scale_slow = float(sys.argv[9]) if len(sys.argv) > 9 else 0.4
    out_path = sys.argv[10] if len(sys.argv) > 10 else "results.png"
    density = float(sys.argv[11]) if len(sys.argv) > 11 else 1.0
    base_seed = int(sys.argv[12]) if len(sys.argv) > 12 else 0

    print(f"zero-shot level+width test  (weights={weights_path}  n_test={n_test}  "
          f"pct_level={pct_level:.0%}  pct_width={pct_width:.0%}  dr_gamma={dr_gamma}  "
          f"gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  gnn_rho_slow={gnn_rho_slow}  "
          f"gnn_scale_slow={gnn_scale_slow})", flush=True)

    device = torch.device("cpu")
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint", flush=True)

    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=base_seed, density=density)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    test_specs, log_rows, n_infeasible = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TEST_SEED + 4, pct_level=pct_level, pct_width=pct_width)
    test_specs = [base_spec] + test_specs
    print(f"  test set: {n_test} instances (slot 0 = original), ALL SIX quantities perturbed via "
          f"level+width (+/-{pct_level:.0%}/+/-{pct_width:.0%}), x0 FRESH RANDOM per instance "
          f"({n_infeasible} infeasible/negative draws discarded+resampled)", flush=True)
    print("-" * 70)

    test_res = evaluate_with_curves(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="zeroshot-levelwidth")
    st = stats_from(test_res)
    print(f"  [zeroshot-levelwidth] CORRECT mean % reduction (per-instance average, not 1-1/mean_speedup): "
          f"mean={st['pct_mean']:.1f}%  median={st['pct_median']:.1f}%", flush=True)
    subtitle = (f"checkpoint: {weights_path}  |  "
                f"zero-shot tested on: level+width +/-{pct_level:.0%}/+/-{pct_width:.0%}, density={density}, n={n_test}, fresh random x0")
    plot_results(test_res, dr_cfg.tol, out_path=out_path, subtitle=subtitle)


if __name__ == "__main__":
    main()
