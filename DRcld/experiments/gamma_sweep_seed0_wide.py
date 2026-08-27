"""
gamma_sweep_seed0_wide.py
===========================

Cheap diagnostic (no training): for density=1.0, seed=0 -- the exact
market behind the known-good 100%-win/1.03x-mean baseline -- check
whether any gamma trades a bit of uniformity for notably MORE headroom
(higher mean iteration count), mirroring the density=0.5 finding that the
cv-minimizing gamma is not always the one that leaves the most room for a
learned correction to exploit. If every reasonable gamma clusters tightly
regardless, this market is just intrinsically uniform and this lever
won't help much.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import statistics
import torch

import config
import market
import algorithm

DENSITY = 1.0
SEED = 0
device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)

N_SAMPLES = 30
CANDIDATE_GAMMAS = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]

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
    stdev_i = statistics.stdev(iters)
    cv = stdev_i / mean_i
    print(f"gamma={gamma:5.1f}  mean={mean_i:7.1f}  stdev={stdev_i:7.1f}  cv={cv:.3f}  "
          f"min={iters_sorted[0]:5d}  max={iters_sorted[-1]:5d}  unconverged={n_unconverged}", flush=True)
