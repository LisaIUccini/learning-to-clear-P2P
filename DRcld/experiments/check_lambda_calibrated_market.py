"""
check_lambda_calibrated_market.py
==========================================

Quick (optional, user-specified 2026-10-14) lambda check, run right after
the gamma calibration: for each density, FIX gamma at its own resolved
gamma*(density, seed=0) (`calibrate_gamma_calibrated_market.py`'s final
table) and sweep lambda over a small grid around the project default
(config.DR_LAMBDA = 1.0), reporting median k_DR on the same training set
(seed=0 base market, n_train instances, seed=config.TRAIN_SEED
perturbations -- identical convention to the gamma search). Purpose:
confirm lambda=1.0 is not leaving easy iterations on the table now that
gamma itself has actually been tuned (lambda was never swept before --
every DR run in this project, before and after the recalibration, used
lambda=1.0 unquestioned).

Same x0-seeding fix as the gamma search (tied to (density, idx), NOT to
lambda, so every lambda tested on a given instance starts from the exact
same x0).

Usage: python3 check_lambda_calibrated_market.py [n_train=25]
       [densities=0.2,0.4,0.5,0.6,0.8,1.0] [base_seed=0] [n_workers=32]
       [lambda_grid=0.5,0.8,1.0,1.2,1.5,1.8]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import multiprocessing as mp
import statistics as stats
import sys

import torch

import config
import market
import algorithm
from build_calibrated_market import sample_calibrated_base_market, CalibratedConfig
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

LAMBDA_GRID = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8]

# resolved gamma*(density), seed=0, from calibrate_gamma_calibrated_market.py's
# final (merged, edge-extended) table -- 2026-10-14
GAMMA_STAR = {0.2: 84.0, 0.4: 92.0, 0.5: 36.0, 0.6: 300.0, 0.8: 22.0, 1.0: 16.0}


def _worker_init():
    torch.set_num_threads(1)


def _run_one(job):
    density, lam, idx, spec, x0_seed, gamma = job
    gen = torch.Generator(device="cpu")
    gen.manual_seed(x0_seed)
    net = market.build_network(spec, device=torch.device("cpu"), rng=gen)
    cfg = algorithm.DRConfig(gamma=gamma, lam=lam, n_iter=config.DR_MAX_ITER,
                              tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    res = algorithm.run_dr(net, cfg, record=False)
    return density, lam, idx, res.n_iter, res.converged


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    densities = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    base_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    lambda_grid = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else LAMBDA_GRID

    print(f"lambda check  grid={lambda_grid}  gamma*(density)={GAMMA_STAR}  n_train={n_train}  "
          f"densities={densities}  base_seed={base_seed}  n_workers={n_workers}", flush=True)

    jobs = []
    for density in densities:
        gamma = GAMMA_STAR[density]
        cfg = CalibratedConfig(density=density)
        neighbors, base_spec, tier_names = sample_calibrated_base_market(base_seed, cfg)
        train_specs, train_log, n_discarded = sample_level_width_perturbed_dataset(
            base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct_level=0.10, pct_width=0.04)
        train_specs = [base_spec] + train_specs
        for idx, spec in enumerate(train_specs):
            x0_seed = int(round(density * 1000)) * 1_000_003 + base_seed * 100_003 + idx
            for lam in lambda_grid:
                jobs.append((density, lam, idx, spec, x0_seed, gamma))

    print(f"dispatching {len(jobs)} DR solves across {n_workers} worker processes...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run_one, jobs, chunksize=1)

    by_dl = {}
    for density, lam, idx, n_iter, converged in results:
        by_dl.setdefault((density, lam), []).append((n_iter, converged))

    for density in densities:
        gamma = GAMMA_STAR[density]
        print(f"\n=== density={density}  (gamma*={gamma}) ===")
        medians = {}
        for lam in lambda_grid:
            entries = by_dl[(density, lam)]
            ks = [k for k, _ in entries]
            n_nonconverged = sum(1 for _, c in entries if not c)
            med = stats.median(ks)
            medians[lam] = med
            flag = (f"  ({n_nonconverged} hit the {config.DR_MAX_ITER}-iter cap without converging!)"
                     if n_nonconverged else "")
            marker = "  <-- project default" if lam == 1.0 else ""
            print(f"  lambda={lam:.2f}  median k_DR={med:7.1f}  mean={stats.mean(ks):7.1f}  "
                  f"min={min(ks)}  max={max(ks)}{flag}{marker}")
        best_lam = min(medians, key=medians.get)
        print(f"  -> best lambda = {best_lam} (median={medians[best_lam]:.1f})  vs  "
              f"lambda=1.0 (median={medians[1.0]:.1f})")


if __name__ == "__main__":
    main()
