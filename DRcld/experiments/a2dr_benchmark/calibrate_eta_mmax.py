"""
calibrate_eta_mmax.py
==========================================

Proper eta/mmax selection for A2DR, on a VALIDATION set distinct from the
test set compare_a2dr_vs_official.py reports its final numbers on --
fixes the methodological gap flagged 2026-10-19: the historical eta=3e-5/
mmax=5 default came from a sweep on an OLD market (density=1.0, seed=0,
pre-c<b-fix) whose own x0 draws may have overlapped the SAME test set that
sweep's own headline number was later reported on (the sweep script wasn't
preserved, so this can no longer be verified either way), and today's
6-density benchmark reused that single old number for every density
without re-sweeping at all -- directly against this project's own standing
caution (see memory: "do NOT assume eta~3e-5 transfers automatically").

Validation instances: SAME level+width perturbed draw mechanism as the
official/A2DR test set (`sample_level_width_perturbed_dataset`), but seed
= TRAIN_SEED+777 (not TRAIN_SEED+1, which is the test set's own seed, nor
TRAIN_SEED/+2, the GNN pipeline's train/val split) -- a seed never used
for anything this project reports as a final number. x0 draws use
TEST_SEED+5000 as their base instead of TEST_SEED*7919+i directly, for the
same reason (no overlap with the actual test evaluation's own x0 draws).

Usage: python3 calibrate_eta_mmax.py <density> <base_seed> <dr_gamma>
       <eta> <mmax> [n_val=10] [c_old_range=0.05,0.18] [out_path]
"""
from __future__ import annotations

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_EXPERIMENTS = _os.path.dirname(_HERE)
_ROOT = _os.path.dirname(_EXPERIMENTS)
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _EXPERIMENTS)

import json
import os
import statistics as stats
import sys

import torch
torch.set_num_threads(1)

import config
import market
import algorithm
import a2dr
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

A2DR_VAL_SEED = config.TRAIN_SEED + 777     # distinct from TRAIN_SEED/+1/+2
X0_VAL_SEED_BASE = config.TEST_SEED + 5000  # distinct from TEST_SEED*7919+i


def main():
    density = float(sys.argv[1])
    base_seed = int(sys.argv[2])
    dr_gamma = float(sys.argv[3])
    eta = float(sys.argv[4])
    mmax = int(sys.argv[5])
    n_val = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    c_old_range = (tuple(float(x) for x in sys.argv[7].split(",")) if len(sys.argv) > 7
                   else (0.05, 0.18))
    out_path = sys.argv[8] if len(sys.argv) > 8 else "calib_result.json"

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    a2_cfg = a2dr.A2DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                              tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS, mmax=mmax, eta=eta)

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    val_specs, _, n_discarded = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_val - 1, seed=A2DR_VAL_SEED, pct_level=0.10, pct_width=0.04)
    val_specs = [base_spec] + val_specs

    print(f"calibrating eta={eta:g} mmax={mmax} on density={density}  "
          f"(n_val={n_val}, {n_discarded} infeasible discarded)", flush=True)

    dr_iters, a2_iters, converged_flags = [], [], []
    for i, spec in enumerate(val_specs):
        seed = X0_VAL_SEED_BASE * 7919 + i
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        net = market.build_network(spec, device=device, rng=gen)
        x0 = net.flatten().clone()

        res_dr = algorithm.run_dr(net, dr_cfg, record=False)

        net_a2 = market.build_network(spec, device=device)
        net_a2.unflatten(x0)
        res_a2 = a2dr.run_a2dr(net_a2, a2_cfg, record=False)

        dr_iters.append(res_dr.n_iter)
        a2_iters.append(res_a2.n_iter)
        converged_flags.append(bool(res_dr.converged and res_a2.converged))
        print(f"  {i + 1:2d}/{n_val}  DR {res_dr.n_iter:5d} it  |  A2DR {res_a2.n_iter:5d} it  "
              f"x{res_dr.n_iter / max(res_a2.n_iter, 1):.2f}"
              f"{'' if res_a2.converged else '  [A2DR DID NOT CONVERGE]'}", flush=True)

    sp = [d / max(a, 1) for d, a in zip(dr_iters, a2_iters)]
    win_rate = sum(s >= 1.0 for s in sp) / len(sp)
    all_converged = all(converged_flags)
    result = {
        "density": density, "eta": eta, "mmax": mmax, "n_val": n_val,
        "mean_speedup": stats.mean(sp), "median_speedup": stats.median(sp),
        "min_speedup": min(sp), "max_speedup": max(sp), "win_rate": win_rate,
        "all_converged": all_converged, "n_nonconverged": sum(not c for c in converged_flags),
        "dr_iters": dr_iters, "a2dr_iters": a2_iters,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> mean={stats.mean(sp):.2f}x  median={stats.median(sp):.2f}x  win_rate={win_rate:.0%}  "
          f"all_converged={all_converged}", flush=True)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
