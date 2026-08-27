"""
gamma_sweep_seed.py
====================

Generalized version of validate_density10_gamma_sweep.py / _seed1_gamma_sweep.py:
plain-DR gamma sweep (no training) for an arbitrary (density, seed) pair, used to
screen many market instances cheaply before committing to a full train+evaluate run.

Usage: python3 gamma_sweep_seed.py <density> <seed>
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import statistics
import torch

import config
import market
import algorithm

DENSITY = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)
n_edges = sum(len(v) for v in base_spec.neighbors.values()) // 2
print(f"[density={DENSITY} seed={SEED}] N={base_spec.N} edges={n_edges}")

N_SAMPLES = 30
CANDIDATE_GAMMAS = [8.0, 10.0, 12.0, 14.0, 16.0]

best = None
for gamma in CANDIDATE_GAMMAS:
    dr_cfg = algorithm.DRConfig(gamma=gamma, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)
    iters = []
    n_unconverged = 0
    for i in range(N_SAMPLES):
        gen = torch.Generator(device=device); gen.manual_seed(config.TRAIN_SEED * 7919 + i + 1000)
        net = market.build_network(base_spec, device=device, rng=gen)
        res = algorithm.run_dr(net, dr_cfg)
        if not res.converged:
            n_unconverged += 1
        iters.append(res.n_iter)
    iters_sorted = sorted(iters)
    mean_i = statistics.mean(iters)
    median_i = statistics.median(iters)
    stdev_i = statistics.stdev(iters)
    cv = stdev_i / mean_i
    print(f"gamma={gamma:5.1f}  mean={mean_i:7.1f}  median={median_i:7.1f}  "
          f"stdev={stdev_i:7.1f}  cv={cv:.3f}  "
          f"min={iters_sorted[0]:5d}  max={iters_sorted[-1]:5d}  "
          f"unconverged={n_unconverged}")
    if n_unconverged == 0 and (best is None or cv < best[1]):
        best = (gamma, cv, mean_i)

if best is not None:
    gamma, cv, mean_i = best
    unroll = round(mean_i / 2)
    rho = 1 - 10 / unroll
    print(f"\n[best] gamma={gamma}  cv={cv:.3f}  mean={mean_i:.1f}  "
          f"-> suggested unroll={unroll}  rho={rho:.4f}")
