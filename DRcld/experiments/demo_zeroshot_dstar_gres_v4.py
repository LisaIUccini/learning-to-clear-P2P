"""
demo_zeroshot_dstar_gres_v4.py
==========================================

ZERO-SHOT, INFERENCE ONLY (no training): loads the checkpoint from
`realistic_market_x0_diversity_v4_wbehind6` (x0-diversity, ONE fixed
realistic market, w_behind=6.0/w_ahead=1.4 -- home-market result
1.77x/100%, mean 43.5% iteration reduction) and tests it on a dataset
where D_star and G_res are perturbed by +/-pct (default 20%, one shared
factor each per instance, box bounds D_min/D_max/G_min/G_max held EXACTLY
fixed -- `sample_dstar_gres_only_dataset`, matching the project's original
graceful-degradation sweep methodology), x0 FRESH RANDOM per test instance
(matching this checkpoint's own training convention, via `demo.evaluate`/
`demo.run_pair` -- NOT the fixed-x0 convention used elsewhere this
session).

Answers: does the x0-diversity-trained checkpoint (which only ever saw ONE
market during training) generalize zero-shot to a market with genuinely
different D*/G_res, without any retraining?

Usage: python3 demo_zeroshot_dstar_gres_v4.py
       [weights_path] [n_test] [pct] [dr_gamma] [gnn_rho] [gnn_scale]
       [gnn_rho_slow] [gnn_scale_slow]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

import torch

import config
import algorithm
import gnn as gnn_mod
from demo import evaluate
from build_realistic_training_dataset import build_realistic_base_market, sample_dstar_gres_only_dataset, REALISTIC_MARKET_CONFIG


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "runs/cross_density/realistic_market_x0_diversity_v4_wbehind6/phi_w.pt"
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20
    dr_gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    # architecture/decay params MUST match what this checkpoint was actually
    # trained with (demo_train_realistic_market_x0_diversity.py's call site
    # only overrides rho/scale -- rho_slow/scale_slow fall back to
    # GNNConfig's own dataclass defaults, 0.99/0.4, so the slow channel WAS
    # active throughout v4's training, not skipped)
    gnn_rho = float(sys.argv[5]) if len(sys.argv) > 5 else 0.973
    gnn_scale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    gnn_rho_slow = float(sys.argv[7]) if len(sys.argv) > 7 else 0.99
    gnn_scale_slow = float(sys.argv[8]) if len(sys.argv) > 8 else 0.4

    print(f"zero-shot D*/G_res test  (weights={weights_path}  n_test={n_test}  pct={pct:.0%}  "
          f"dr_gamma={dr_gamma}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"gnn_rho_slow={gnn_rho_slow}  gnn_scale_slow={gnn_scale_slow})", flush=True)

    device = torch.device("cpu")
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint (trained {blob.get('epochs', '?')} epochs)", flush=True)

    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=0)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    test_specs, log_rows, n_infeasible = sample_dstar_gres_only_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TEST_SEED + 4, pct=pct)
    test_specs = [base_spec] + test_specs
    print(f"  test set: {n_test} instances (slot 0 = original, unperturbed), D*/G_res perturbed "
          f"+/-{pct:.0%}, box bounds EXACTLY fixed, x0 FRESH RANDOM per instance "
          f"({n_infeasible} infeasible/negative draws discarded+resampled)", flush=True)
    print("-" * 70)

    evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="zeroshot-dstar-gres")


if __name__ == "__main__":
    main()
