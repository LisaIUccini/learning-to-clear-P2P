"""
demo_zeroshot_crossmarket.py
==========================================

ZERO-SHOT, INFERENCE ONLY: tests a checkpoint trained on ONE base market
(one base_seed) against N COMPLETELY DIFFERENT random base markets at the
SAME density -- different topology draw (same density, different edges),
different cost coefficients (a_tilde/a/b/c/kappa), different D*/G_res/box
ranges, all freshly redrawn (NOT a perturbation of the training market).
x0 fresh random per test market. This is a stronger generalization test
than the project's usual "perturb the SAME base market" zero-shot checks.

Usage: python3 demo_zeroshot_crossmarket.py <weights_path> <density>
       <gnn_rho> <gnn_scale> [dr_gamma] [gnn_rho_slow] [gnn_scale_slow]
       [n_markets] [seed_start]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import statistics as stats
import sys

import torch

import config
import market
import algorithm
import gnn as gnn_mod
from build_realistic_training_dataset import build_realistic_base_market


def main():
    weights_path = sys.argv[1]
    density = float(sys.argv[2])
    gnn_rho = float(sys.argv[3])
    gnn_scale = float(sys.argv[4])
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    gnn_rho_slow = float(sys.argv[6]) if len(sys.argv) > 6 else 0.99
    gnn_scale_slow = float(sys.argv[7]) if len(sys.argv) > 7 else 0.4
    n_markets = int(sys.argv[8]) if len(sys.argv) > 8 else 10
    seed_start = int(sys.argv[9]) if len(sys.argv) > 9 else 100

    print(f"cross-market zero-shot test  (weights={weights_path}  density={density}  "
          f"gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  dr_gamma={dr_gamma}  "
          f"n_markets={n_markets}  seed_start={seed_start})", flush=True)

    device = torch.device("cpu")
    from build_realistic_training_dataset import REALISTIC_MARKET_CONFIG
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint", flush=True)

    dr_iters, learned_iters, speedups, pcts = [], [], [], []
    for i in range(n_markets):
        seed = seed_start + i
        base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=seed, density=density)

        gen = torch.Generator(device=device); gen.manual_seed(config.TEST_SEED * 7919 + i)
        net_dr = market.build_network(base_spec, device=device, rng=gen)
        x0 = net_dr.flatten().clone()
        gnn_mod.undeploy(net_dr)
        res_dr = algorithm.run_dr(net_dr, dr_cfg, record=False)

        net_l = market.build_network(base_spec, device=device)
        net_l.unflatten(x0)
        gnn_mod.deploy(model, net_l)
        res_l = algorithm.run_dr(net_l, dr_cfg, record=False)

        sp = res_dr.n_iter / max(res_l.n_iter, 1)
        pct = (res_dr.n_iter - res_l.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_l.n_iter)
        speedups.append(sp); pcts.append(pct)
        print(f"  market_seed={seed:3d}  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values())//2}  "
              f"DR {res_dr.n_iter:5d} it  |  learned {res_l.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)

    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"\n[cross-market] mean speed-up {stats.mean(speedups):.2f}x   median {stats.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)
    print(f"[cross-market] CORRECT mean % reduction (per-instance average): "
          f"mean={stats.mean(pcts):.1f}%  median={stats.median(pcts):.1f}%", flush=True)


if __name__ == "__main__":
    main()
