"""
shape_sweep.py
================

For a given density, print the FULL sorted list of plain-DR iteration
counts (not just mean/cv) across several candidate gammas, so bimodality
or other multi-modal structure is visible directly, not just inferred
from a single cv number. Also prints a coarse histogram.

Usage: python3 shape_sweep.py <density> <gamma1> <gamma2> ... [--seed N]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import statistics
import torch

import config
import market
import algorithm

args = sys.argv[1:]
SEED = 0
if "--seed" in args:
    i = args.index("--seed")
    SEED = int(args[i + 1])
    args = args[:i] + args[i + 2:]
DENSITY = float(args[0])
GAMMAS = [float(g) for g in args[1:]]
device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)
print(f"[density={DENSITY} seed={SEED}]")

N_SAMPLES = 40
for gamma in GAMMAS:
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
    mean_i = statistics.mean(iters); stdev_i = statistics.stdev(iters)
    lo, hi = iters_sorted[0], iters_sorted[-1]
    nbins = 10
    width = max((hi - lo) / nbins, 1)
    bins = [0] * nbins
    for v in iters:
        b = min(int((v - lo) / width), nbins - 1)
        bins[b] += 1
    hist = " ".join(f"{c:2d}" for c in bins)
    print(f"gamma={gamma:5.1f}  mean={mean_i:7.1f}  cv={stdev_i/mean_i:.3f}  "
          f"min={lo} max={hi}  unconverged={n_unconverged}", flush=True)
    print(f"  sorted: {iters_sorted}", flush=True)
    print(f"  hist({lo}-{hi}, {nbins} bins): {hist}", flush=True)
