"""
validate_density10_seed1_gamma_sweep.py
==========================================

Gamma sweep for density=1.0 (complete graph) with a DIFFERENT seed (1),
testing whether density 1.0's own ceiling/optimal-gamma pattern
generalises beyond seed=0.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import statistics
import torch

import config
import market
import algorithm

DENSITY = 1.0
SEED = 1
device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)

N_SAMPLES = 30
CANDIDATE_GAMMAS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

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
    print(f"gamma={gamma:5.1f}  mean={mean_i:7.1f}  median={median_i:7.1f}  "
          f"stdev={stdev_i:7.1f}  cv={stdev_i/mean_i:.3f}  "
          f"min={iters_sorted[0]:5d}  max={iters_sorted[-1]:5d}  "
          f"unconverged={n_unconverged}")
