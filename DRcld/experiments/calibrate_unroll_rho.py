"""
calibrate_unroll_rho.py
==========================================

Recalibrate the GNN's training-unroll length K and fast-channel decay rho
for each density, now that gamma itself has been properly tuned
(`calibrate_gamma_calibrated_market.py`'s resolved gamma*(density, seed=0)
table) -- the project's standing rule (unroll=mean_DR/2, rho=1-10/unroll,
used throughout, e.g. `demo_calibrated_rho.py`'s `estimate_rho`) was never
re-measured against these new gamma values, and k_DR itself changed a lot
under them (e.g. density=0.6 needs gamma=300, an order of magnitude
above every other density).

Measurement convention (matching every earlier "n=40 draws" calibration
note in this project, e.g. demo_train_combined_levelwidth_x0diversity.py's
module docstring): n_measure=40 draws per density, EACH a FRESH
level+width-perturbed instance (+/-10%/+/-4%, own seed per draw) PLUS a
FRESH random x0 (also seeded per draw, for reproducibility -- earlier
measurements used truly unseeded x0, this pins it down without changing
what's being measured). Plain DR (no GNN), gamma=gamma*(density),
lambda=config.DR_LAMBDA (kept at 1.0 project-wide per the 2026-10-14
decision not to adopt the lambda sweep's gains yet).

Reports mean/cv/min/max of k_DR, then:
    unroll = mean / 2
    rho_10 = 1 - 10/unroll           (the project's standing rule)
    rho_5  = 1 - 5/unroll            (user-specified alternative to try
                                       in the pilot, 2026-10-14 -- slower
                                       decay, the correction stays active
                                       longer relative to the horizon)

Usage: python3 calibrate_unroll_rho.py [n_measure=40]
       [densities=0.2,0.4,0.5,0.6,0.8,1.0] [base_seed=0] [n_workers=32]
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

GAMMA_STAR = {0.2: 84.0, 0.4: 92.0, 0.5: 36.0, 0.6: 300.0, 0.8: 22.0, 1.0: 16.0}
CALIB_SEED_OFFSET = 50   # distinct from TRAIN_SEED(+0 train/+1 test/+2 val)


def _worker_init():
    torch.set_num_threads(1)


def _run_one(job):
    density, gamma, idx, spec, x0_seed = job
    gen = torch.Generator(device="cpu")
    gen.manual_seed(x0_seed)
    net = market.build_network(spec, device=torch.device("cpu"), rng=gen)
    cfg = algorithm.DRConfig(gamma=gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                              tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    res = algorithm.run_dr(net, cfg, record=False)
    return density, idx, res.n_iter, res.converged


def main():
    n_measure = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    densities = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    base_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    # gamma_override: comma list aligned to `densities`, overriding GAMMA_STAR --
    # added 2026-10-14 for one-off market sizes (N != config.BASE_MARKET_N)
    # whose gamma* isn't in GAMMA_STAR (that dict is N=10-specific).
    gamma_override = ([float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 and sys.argv[5] != "none"
                       else None)
    n_nodes = int(sys.argv[6]) if len(sys.argv) > 6 else None   # override config.BASE_MARKET_N
    c_scheme = sys.argv[7] if len(sys.argv) > 7 else "seller_discount"   # 2026-10-14 ablation
    curvature_regime = sys.argv[8] if len(sys.argv) > 8 else "literature"   # 2026-10-14 ablation
    generator = sys.argv[9] if len(sys.argv) > 9 else "calibrated"   # 2026-10-14: "calibrated" | "old"
    c_old_range = (tuple(float(x) for x in sys.argv[10].split(",")) if len(sys.argv) > 10
                    else (0.05, 0.18))

    gamma_map = GAMMA_STAR if gamma_override is None else dict(zip(densities, gamma_override))
    print(f"unroll/rho calibration  n_measure={n_measure}  densities={densities}  base_seed={base_seed}  "
          f"gamma*={gamma_map}  lambda={config.DR_LAMBDA}  n_workers={n_workers}  "
          f"n_nodes={n_nodes or 'config.BASE_MARKET_N'}  c_scheme={c_scheme}  generator={generator}  "
          f"c_old_range={c_old_range}", flush=True)

    jobs = []
    for density in densities:
        gamma = gamma_map[density]
        if generator == "calibrated":
            cfg = CalibratedConfig(density=density, c_scheme=c_scheme, curvature_regime=curvature_regime)
            neighbors, base_spec, tier_names = sample_calibrated_base_market(base_seed, cfg, N=n_nodes)
        elif generator == "old":
            mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                                      G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0),
                                      c_range=c_old_range)
            _, _, base_spec = market.sample_base(n_nodes or config.BASE_MARKET_N, mc, seed=base_seed)
        else:
            raise ValueError(f"unknown generator {generator!r}")
        # n_measure FRESH level+width-perturbed single-use instances, own
        # seed namespace (CALIB_SEED_OFFSET), distinct from train/test/val
        measure_specs, _, n_discarded = sample_level_width_perturbed_dataset(
            base_spec, n_instances=n_measure, seed=config.TRAIN_SEED + CALIB_SEED_OFFSET,
            pct_level=0.10, pct_width=0.04)
        for idx, spec in enumerate(measure_specs):
            x0_seed = int(round(density * 1000)) * 1_000_003 + 999_000 + idx   # own namespace, fresh x0 per draw
            jobs.append((density, gamma, idx, spec, x0_seed))

    print(f"dispatching {len(jobs)} DR solves across {n_workers} worker processes...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run_one, jobs, chunksize=1)

    by_density = {}
    for density, idx, n_iter, converged in results:
        by_density.setdefault(density, []).append((n_iter, converged))

    print("\n" + "=" * 70)
    for density in densities:
        entries = by_density[density]
        ks = [k for k, _ in entries]
        n_nonconverged = sum(1 for _, c in entries if not c)
        mean_k = stats.mean(ks)
        cv = stats.pstdev(ks) / mean_k if mean_k > 0 else float("nan")
        unroll = mean_k / 2.0
        rho_10 = 1.0 - 10.0 / unroll
        rho_5 = 1.0 - 5.0 / unroll
        flag = f"  ({n_nonconverged} hit the {config.DR_MAX_ITER}-iter cap!)" if n_nonconverged else ""
        print(f"density={density}  gamma*={gamma_map[density]}  n={len(ks)}  "
              f"mean={mean_k:.1f}  cv={cv:.4f}  min={min(ks)}  max={max(ks)}{flag}")
        print(f"  -> unroll={unroll:.0f}  rho_10(standing rule)={rho_10:.4f}  "
              f"rho_5(pilot alt.)={rho_5:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
