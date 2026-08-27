"""
analyze_bifurcation.py
========================

Diagnostic: at gamma=4.0 (where fast/slow x0 draws genuinely differ in
iteration count), find the fastest and slowest x0 draw for a given market
and record their FULL residual trajectories, to see WHEN the two start
to diverge from each other -- early divergence (within the correction's
active window) would explain why a bounded, short-horizon perturbation
can meaningfully redirect the trajectory; late divergence would explain
why it structurally cannot, regardless of hyperparameter tuning.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch

import config
import market
import algorithm

DENSITY = 1.0
GAMMA = 4.0
N_SAMPLES = 30
device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
dr_cfg = algorithm.DRConfig(gamma=GAMMA, lam=config.DR_LAMBDA,
                             n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                             bisect=config.DR_BISECT_STEPS)


def find_fast_slow(seed):
    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=seed)
    results = []
    for i in range(N_SAMPLES):
        gen = torch.Generator(device=device); gen.manual_seed(config.TRAIN_SEED * 7919 + i + 1000)
        net = market.build_network(base_spec, device=device, rng=gen)
        res = algorithm.run_dr(net, dr_cfg)
        results.append((i, res.n_iter))
    results.sort(key=lambda x: x[1])
    fast_i, fast_n = results[0]
    slow_i, slow_n = results[-1]
    return base_spec, fast_i, fast_n, slow_i, slow_n


def get_trajectory(base_spec, i):
    gen = torch.Generator(device=device); gen.manual_seed(config.TRAIN_SEED * 7919 + i + 1000)
    net = market.build_network(base_spec, device=device, rng=gen)
    res = algorithm.run_dr(net, dr_cfg, record=True)
    return res.residual


for seed in [0, 1]:
    base_spec, fast_i, fast_n, slow_i, slow_n = find_fast_slow(seed)
    print(f"=== seed={seed}: fastest x0 (i={fast_i}, {fast_n} iters), slowest x0 (i={slow_i}, {slow_n} iters) ===")
    fast_traj = get_trajectory(base_spec, fast_i)
    slow_traj = get_trajectory(base_spec, slow_i)
    # compare at matching checkpoints (up to the shorter trajectory's length)
    checkpoints = [10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400]
    print(f"{'k':>5}  {'fast residual':>15}  {'slow residual':>15}  {'ratio slow/fast':>16}")
    for k in checkpoints:
        if k <= len(fast_traj) and k <= len(slow_traj):
            f = fast_traj[k - 1]
            s = slow_traj[k - 1]
            print(f"{k:5d}  {f:15.6e}  {s:15.6e}  {s / f:16.3f}")
    print()
