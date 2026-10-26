"""
demo_zeroshot_levelwidth_oldgen.py
==========================================

Zero-shot (inference-only) OOD test for the "official" Phi_w checkpoints
(train_official_pilot.py, generator="old" + c<b fix), adapted from
`demo_zeroshot_levelwidth_v5_resample.py` (which targets the calibrated/
realistic generator instead -- every official_N10_d*_replicate* run used
market.py's OLD generator with c_old_range=(0.05,0.18), so this script
reproduces that SAME market exactly before testing the checkpoint on a
level+width perturbation WIDER than the +/-10%/+/-4% training distribution
(e.g. +/-20%/+/-8%, matching this project's own "levelNN" OOD severity
convention -- see runs/cross_density/zeroshot_density08_level20 etc.).

Same discard+resample contract as v5: any draw where plain DR itself fails
to converge within DR_MAX_ITER is discarded and redrawn, so a capped DR
iteration count never masquerades as a real data point.

Usage: python3 demo_zeroshot_levelwidth_oldgen.py [weights_path] [density]
       [base_seed] [dr_gamma] [gnn_rho] [gnn_scale] [gnn_rho_slow]
       [gnn_scale_slow] [n_test] [pct_level] [pct_width] [c_old_range]
       [out_path] [max_retries]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod
from demo import plot_results, run_pair
from plot_style import stats_from
from build_realistic_training_dataset import perturb_six_level_width


def sample_converged_test_set(base_spec, model, dr_cfg, device, n_test, seed,
                               pct_level, pct_width, max_retries=30):
    """Slot 0 = the exact base market (unperturbed). Slots 1..n_test-1 are
    level+width perturbed draws, discard+resample on EITHER infeasibility
    OR plain-DR non-convergence."""
    results = [run_pair(base_spec, model, dr_cfg, device, seed=seed * 7919, record=True)]
    n_infeasible = 0
    n_nonconverged = 0

    for i in range(1, n_test):
        kept = False
        for attempt in range(max_retries):
            rng = np.random.default_rng(seed * 100_003 + i * 1009 + attempt)
            f_level_D = float(rng.uniform(1 - pct_level, 1 + pct_level))
            f_width_D = float(rng.uniform(1 - pct_width, 1 + pct_width))
            f_level_G = float(rng.uniform(1 - pct_level, 1 + pct_level))
            f_width_G = float(rng.uniform(1 - pct_width, 1 + pct_width))
            f_gres = float(rng.uniform(1 - pct_level, 1 + pct_level))
            spec = perturb_six_level_width(base_spec, f_level_D, f_width_D, f_level_G, f_width_G, f_gres)
            all_ge0 = bool((spec.D_star >= 0).all() and (spec.G_res >= 0).all()
                           and (spec.D_min >= 0).all() and (spec.D_max >= 0).all()
                           and (spec.G_min >= 0).all() and (spec.G_max >= 0).all())
            if not (all_ge0 and market.spec_is_feasible(spec)):
                n_infeasible += 1
                continue

            res_dr, res_learned = run_pair(spec, model, dr_cfg, device,
                                            seed=seed * 7919 + i * 131 + attempt, record=True)
            if not res_dr.converged:
                n_nonconverged += 1
                print(f"  ... instance {i}: draw discarded, DR did not converge "
                      f"(hit {res_dr.n_iter} iters), retrying (attempt {attempt + 1})", flush=True)
                continue

            results.append((res_dr, res_learned))
            kept = True
            print(f"  [ood] {i:2d}/{n_test - 1} kept (attempt {attempt + 1})  "
                  f"DR {res_dr.n_iter:5d} it  |  learned {res_learned.n_iter:5d} it", flush=True)
            break
        if not kept:
            raise RuntimeError(f"instance {i}: no feasible+converged draw found in {max_retries} retries "
                                f"at pct_level={pct_level}, pct_width={pct_width}")

    return results, n_infeasible, n_nonconverged


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "runs/official/official_N10_d0.2_seed3_replicate/phi_w.pt"
    density = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    base_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    dr_gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    gnn_rho = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9833
    gnn_scale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.8
    gnn_rho_slow = float(sys.argv[7]) if len(sys.argv) > 7 else 0.99
    gnn_scale_slow = float(sys.argv[8]) if len(sys.argv) > 8 else 0.4
    n_test = int(sys.argv[9]) if len(sys.argv) > 9 else 30
    pct_level = float(sys.argv[10]) if len(sys.argv) > 10 else 0.20
    pct_width = float(sys.argv[11]) if len(sys.argv) > 11 else 0.08
    c_old_range = (tuple(float(x) for x in sys.argv[12].split(",")) if len(sys.argv) > 12
                   else (0.05, 0.18))
    out_path = sys.argv[13] if len(sys.argv) > 13 else "ood_results.png"
    max_retries = int(sys.argv[14]) if len(sys.argv) > 14 else 30

    print(f"zero-shot OOD test (generator=old, c<b fix), discard+resample on DR non-convergence  "
          f"(weights={weights_path}  density={density}  base_seed={base_seed}  n_test={n_test}  "
          f"pct_level={pct_level:.0%}  pct_width={pct_width:.0%}  dr_gamma={dr_gamma}  gnn_rho={gnn_rho}  "
          f"gnn_scale={gnn_scale}  gnn_rho_slow={gnn_rho_slow}  gnn_scale_slow={gnn_scale_slow}  "
          f"DR_MAX_ITER={config.DR_MAX_ITER})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow,
        hidden_dim=config.GNN_HIDDEN_DIM))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint", flush=True)

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    results, n_infeasible, n_nonconverged = sample_converged_test_set(
        base_spec, model, dr_cfg, device, n_test, seed=config.TEST_SEED + 4,
        pct_level=pct_level, pct_width=pct_width)
    print(f"  test set: {n_test} instances (slot 0 = original), level+width "
          f"(+/-{pct_level:.0%}/+/-{pct_width:.0%}), x0 FRESH RANDOM per instance "
          f"({n_infeasible} infeasible discarded, {n_nonconverged} DR-non-converged discarded+resampled)",
          flush=True)
    print("-" * 70)

    dr_iters, learned_iters, speedups = [], [], []
    for i, (res_dr, res_learned) in enumerate(results):
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter); speedups.append(sp)
        print(f"  [ood] {i + 1:2d}/{len(results)}  DR {res_dr.n_iter:5d} it  |  "
              f"learned {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)
    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    test_res = {"dr": dr_iters, "learned": learned_iters, "speedup": speedups,
                "curves_dr": [np.asarray(r.residual) for r, _ in results],
                "curves_learned": [np.asarray(l.residual) for _, l in results]}
    print(f"  [ood] mean speed-up {np.mean(speedups):.2f}x   median {np.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)

    st = stats_from(test_res)
    print(f"  [ood] mean % reduction (per-instance average): mean={st['pct_mean']:.1f}%  "
          f"median={st['pct_median']:.1f}%", flush=True)
    subtitle = (f"checkpoint: {weights_path}  |  OOD test: level+width "
                f"+/-{pct_level:.0%}/+/-{pct_width:.0%}, density={density}, n={n_test}, "
                f"fresh random x0, DR-non-convergent draws discarded+resampled")
    plot_results(test_res, dr_cfg.tol, out_path=out_path, subtitle=subtitle)


if __name__ == "__main__":
    main()
