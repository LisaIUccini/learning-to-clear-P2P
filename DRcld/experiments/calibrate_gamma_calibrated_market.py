"""
calibrate_gamma_calibrated_market.py
==========================================

Official gamma grid search for the recalibrated market generator
(`build_calibrated_market.py`), the first step of the "official" paper-plot
pipeline (2026-10-14): for each density, build the training set the real
per-density training run will use (level+width +/-10%/+/-4%, n_train
instances, seed=config.TRAIN_SEED -- identical convention to every
existing demo_train_*.py script), and run PLAIN DR (no GNN, lam=1.0 fixed
per the user's spec) once per training instance for every gamma in
`gamma_grid`.

For each (density, base_seed, gamma): the reported statistic is the
MEDIAN number of DR iterations to convergence (tol=config.DR_TOLERANCE)
across the training set. Selection uses a ROBUST rule (user-specified,
2026-10-14), not plain argmin: take every gamma whose median sits within
5% of the grid's overall minimum median (the "plateau"), and choose the
CENTER of that plateau's [min, max] gamma range -- plain argmin can land
on a fragile point right at a sharp, noisy minimum that is an artifact of
this particular x0/instance draw rather than a stable property of the
gamma-convergence curve; the plateau center is far less sensitive to
exactly where that noisy minimum happened to fall.

MULTIPLE BASE MARKETS per density (`base_seeds`, user-specified
2026-10-14): a single base market (one random draw from the generator) is
one sample of what a real community's market could look like -- repeating
the calibration across 3 independent base-market draws per density is
"the calibration an operator would actually do on their own community's
historical data" (the user's own framing), and is cheap relative to the
DR solves themselves (only the market-generation step repeats, not the
heavy part). Reports gamma per (density, base_seed) AND an aggregate
(median of the 3 robust choices) per density.

x0 is seeded deterministically, tied to (density, base_seed, idx) only --
NOT to gamma -- so every gamma tested on a given instance starts from the
EXACT SAME x0, isolating gamma's own effect on k_DR.
`market.build_network`'s default (rng=None) draws x0 from torch's GLOBAL
unseeded RNG, which differs per spawned worker process; caught
empirically 2026-10-14 (same spec+gamma, 5 repeated calls: k_DR ranged
677-1156, a ~70% spread from x0 noise alone) BEFORE this fix, which
invalidated every grid-search comparison done before it (new worker
processes = new unseeded draws each time, so even two gammas tested
"in the same run" were not actually using the same starting point).

Does not train or touch the GNN at all -- DR-only, parallelised with
multiprocessing across every (density, base_seed, gamma, instance) job at
once, torch pinned to 1 thread/worker since these tensors are far too
small for intra-op parallelism to help, each worker rebuilding the
Network from a picklable MarketSpec rather than sharing torch state
across processes (spawn, not fork -- forking a process whose torch
thread pool is already initialised is a known source of deadlocked
children).

Usage: python3 calibrate_gamma_calibrated_market.py [n_train=25]
       [densities=0.2,0.4,0.5,0.6,0.8,1.0] [base_seeds=0,1,2]
       [n_workers=32] [gamma_grid=1,2,4,6,8,10,12,16,20,24,32,40,56,72]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import multiprocessing as mp
import statistics as stats
import sys
from typing import Dict, List, Tuple

import torch

import config
import market
import algorithm
from build_calibrated_market import sample_calibrated_base_market, CalibratedConfig
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

GAMMA_GRID = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 56.0, 72.0]
PLATEAU_TOL = 0.05   # robust-selection rule: within 5% of the minimum median


def _worker_init():
    torch.set_num_threads(1)


def _run_one(job):
    density, base_seed, gamma, idx, spec, x0_seed = job
    gen = torch.Generator(device="cpu")
    gen.manual_seed(x0_seed)
    net = market.build_network(spec, device=torch.device("cpu"), rng=gen)
    cfg = algorithm.DRConfig(gamma=gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                              tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    res = algorithm.run_dr(net, cfg, record=False)
    return density, base_seed, gamma, idx, res.n_iter, res.converged


def robust_gamma(medians: Dict[float, float]) -> Tuple[float, float, float]:
    """Center of the plateau of gammas within PLATEAU_TOL of the minimum
    median k_DR -- see module docstring for why this replaces plain
    argmin."""
    m_min = min(medians.values())
    threshold = m_min * (1.0 + PLATEAU_TOL)
    plateau = sorted(g for g, m in medians.items() if m <= threshold)
    lo, hi = plateau[0], plateau[-1]
    return (lo + hi) / 2.0, lo, hi


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    densities = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    base_seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0, 1, 2]
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    gamma_grid = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else GAMMA_GRID
    n_nodes = int(sys.argv[6]) if len(sys.argv) > 6 else None   # override config.BASE_MARKET_N, 2026-10-14
    c_scheme = sys.argv[7] if len(sys.argv) > 7 else "seller_discount"   # 2026-10-14 ablation
    curvature_regime = sys.argv[8] if len(sys.argv) > 8 else "literature"   # 2026-10-14 ablation
    generator = sys.argv[9] if len(sys.argv) > 9 else "calibrated"   # 2026-10-14: "calibrated" | "old"
    c_old_range = (tuple(float(x) for x in sys.argv[10].split(",")) if len(sys.argv) > 10
                    else (0.05, 0.18))

    print(f"gamma grid search  grid={gamma_grid}  lambda={config.DR_LAMBDA}  n_train={n_train}  "
          f"densities={densities}  base_seeds={base_seeds}  tol={config.DR_TOLERANCE}  "
          f"cap={config.DR_MAX_ITER}  n_workers={n_workers}  plateau_tol={PLATEAU_TOL:.0%}  "
          f"n_nodes={n_nodes or 'config.BASE_MARKET_N'}  c_scheme={c_scheme}  generator={generator}  "
          f"c_old_range={c_old_range}", flush=True)

    jobs = []
    tier_counts = {}
    for density in densities:
        for base_seed in base_seeds:
            if generator == "calibrated":
                cfg = CalibratedConfig(density=density, c_scheme=c_scheme, curvature_regime=curvature_regime)
                neighbors, base_spec, tier_names = sample_calibrated_base_market(base_seed, cfg, N=n_nodes)
            elif generator == "old":
                mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                                          G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0),
                                          c_range=c_old_range)
                _, _, base_spec = market.sample_base(n_nodes or config.BASE_MARKET_N, mc, seed=base_seed)
                tier_names = []
            else:
                raise ValueError(f"unknown generator {generator!r}")
            train_specs, train_log, n_discarded = sample_level_width_perturbed_dataset(
                base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct_level=0.10, pct_width=0.04)
            train_specs = [base_spec] + train_specs
            tier_counts[(density, base_seed)] = (tier_names.count("low"), tier_names.count("medium"),
                                                   tier_names.count("high"), n_discarded)
            for idx, spec in enumerate(train_specs):
                x0_seed = int(round(density * 1000)) * 1_000_003 + base_seed * 100_003 + idx
                for gamma in gamma_grid:
                    jobs.append((density, base_seed, gamma, idx, spec, x0_seed))

    print(f"dispatching {len(jobs)} DR solves across {n_workers} worker processes...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run_one, jobs, chunksize=1)

    by_dbg = {}
    for density, base_seed, gamma, idx, n_iter, converged in results:
        by_dbg.setdefault((density, base_seed, gamma), []).append((n_iter, converged))

    per_density_robust: Dict[float, List[float]] = {d: [] for d in densities}
    for density in densities:
        for base_seed in base_seeds:
            low, med, high, n_discarded = tier_counts[(density, base_seed)]
            print(f"\n=== density={density}  base_seed={base_seed} ===  (n_train={n_train}, "
                  f"{n_discarded} infeasible discarded, base tiers: low={low} medium={med} high={high})")
            medians = {}
            for gamma in gamma_grid:
                entries = by_dbg[(density, base_seed, gamma)]
                ks = [k for k, _ in entries]
                n_nonconverged = sum(1 for _, c in entries if not c)
                med = stats.median(ks)
                medians[gamma] = med
                flag = (f"  ({n_nonconverged} hit the {config.DR_MAX_ITER}-iter cap without converging!)"
                         if n_nonconverged else "")
                print(f"  gamma={gamma:>5.1f}  median k_DR={med:7.1f}  "
                      f"mean={stats.mean(ks):7.1f}  min={min(ks)}  max={max(ks)}{flag}")
            center, lo, hi = robust_gamma(medians)
            argmin_gamma = min(medians, key=medians.get)
            print(f"  -> argmin gamma = {argmin_gamma} (median={medians[argmin_gamma]:.1f})  |  "
                  f"ROBUST gamma*(density={density}, seed={base_seed}) = {center:.1f}  "
                  f"(plateau within {PLATEAU_TOL:.0%} of min spans [{lo:.1f}, {hi:.1f}])")
            per_density_robust[density].append(center)

    print("\n" + "=" * 70)
    print("SUMMARY -- robust gamma*(density), aggregated (median) across base_seeds:")
    for d in densities:
        vals = per_density_robust[d]
        agg = stats.median(vals)
        print(f"  density={d}: per-seed robust gamma*={[round(v,1) for v in vals]}  ->  aggregate={agg:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
