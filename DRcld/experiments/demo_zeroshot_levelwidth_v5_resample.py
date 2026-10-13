"""
demo_zeroshot_levelwidth_v5_resample.py
==========================================

Same as `demo_zeroshot_levelwidth_v4.py` (zero-shot, inference-only,
level+width OOD test), EXCEPT: any perturbation draw where PLAIN DR itself
fails to converge within DR_MAX_ITER is discarded and a fresh draw is taken
instead (same discard+resample contract the dataset sampler already uses for
infeasible/negative draws, just extended to cover non-convergence too).

Why: for a base market whose natural DR difficulty is already high/variable
(e.g. density=1.0 base_seed=2, mean~1624 iters), pushing the OOD level up
causes a growing share of draws to hit the DR_MAX_ITER safety cap. A capped
DR iteration count is not the true difficulty, just a truncation -- keeping
those instances in the test set silently corrupts both the DR baseline and,
when the learned model ALSO caps out, produces a fake "tie" (speed-up=1.00,
pct=0.0%) that masks a double failure. Discarding and redrawing keeps the
test set restricted to instances where the comparison is actually well
defined (plain DR has a true, finite answer to compare against).

Usage: python3 demo_zeroshot_levelwidth_v5_resample.py
       [weights_path] [n_test] [pct_level] [pct_width] [dr_gamma]
       [gnn_rho] [gnn_scale] [gnn_rho_slow] [gnn_scale_slow] [out_path]
       [density] [base_seed] [max_retries]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

import numpy as np
import torch

import config
import algorithm
import gnn as gnn_mod
from demo import plot_results, run_pair
from plot_style import stats_from
from build_realistic_training_dataset import (
    build_realistic_base_market, perturb_six_level_width, REALISTIC_MARKET_CONFIG,
)
import market


def sample_converged_test_set(base_spec, model, dr_cfg, device, n_test, seed,
                               pct_level, pct_width, max_retries=30):
    """Slot 0 = the exact base market (unperturbed), matching every other
    condition's test set so they share one common anchor instance. Slots
    1..n_test-1 are level+width perturbed draws, discard+resample on EITHER
    infeasibility (structural: a negative D*/G_res/box) OR plain-DR
    non-convergence (res_dr.converged == False, i.e. it hit DR_MAX_ITER)."""
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
            print(f"  [resample] {i:2d}/{n_test - 1} kept (attempt {attempt + 1})  "
                  f"DR {res_dr.n_iter:5d} it  |  learned {res_learned.n_iter:5d} it", flush=True)
            break
        if not kept:
            raise RuntimeError(f"instance {i}: no feasible+converged draw found in {max_retries} retries "
                                f"at pct_level={pct_level}, pct_width={pct_width}")

    return results, n_infeasible, n_nonconverged


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
    max_retries = int(sys.argv[13]) if len(sys.argv) > 13 else 30

    print(f"zero-shot level+width test, DISCARD+RESAMPLE on DR non-convergence  "
          f"(weights={weights_path}  n_test={n_test}  pct_level={pct_level:.0%}  "
          f"pct_width={pct_width:.0%}  dr_gamma={dr_gamma}  gnn_rho={gnn_rho}  "
          f"gnn_scale={gnn_scale}  gnn_rho_slow={gnn_rho_slow}  gnn_scale_slow={gnn_scale_slow}  "
          f"DR_MAX_ITER={config.DR_MAX_ITER})", flush=True)

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

    results, n_infeasible, n_nonconverged = sample_converged_test_set(
        base_spec, model, dr_cfg, device, n_test, seed=config.TEST_SEED + 4,
        pct_level=pct_level, pct_width=pct_width)
    print(f"  test set: {n_test} instances (slot 0 = original), ALL SIX quantities perturbed via "
          f"level+width (+/-{pct_level:.0%}/+/-{pct_width:.0%}), x0 FRESH RANDOM per instance "
          f"({n_infeasible} infeasible discarded, {n_nonconverged} DR-non-converged discarded+resampled)",
          flush=True)
    print("-" * 70)

    dr_iters, learned_iters, speedups = [], [], []
    for i, (res_dr, res_learned) in enumerate(results):
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter); speedups.append(sp)
        print(f"  [resample] {i + 1:2d}/{len(results)}  DR {res_dr.n_iter:5d} it  |  "
              f"learned {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)
    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    test_res = {"dr": dr_iters, "learned": learned_iters, "speedup": speedups,
                "curves_dr": [np.asarray(r.residual) for r, _ in results],
                "curves_learned": [np.asarray(l.residual) for _, l in results]}
    print(f"  [resample] mean speed-up {np.mean(speedups):.2f}x   median {np.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)

    st = stats_from(test_res)
    print(f"  [resample] CORRECT mean % reduction (per-instance average, not 1-1/mean_speedup): "
          f"mean={st['pct_mean']:.1f}%  median={st['pct_median']:.1f}%", flush=True)
    subtitle = (f"checkpoint: {weights_path}  |  zero-shot tested on: level+width "
                f"+/-{pct_level:.0%}/+/-{pct_width:.0%}, density={density}, n={n_test}, "
                f"fresh random x0, DR-non-convergent draws discarded+resampled")
    plot_results(test_res, dr_cfg.tol, out_path=out_path, subtitle=subtitle)


if __name__ == "__main__":
    main()
