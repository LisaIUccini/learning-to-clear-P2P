"""
demo_zeroshot_node_rewire.py
==========================================

ZERO-SHOT, INFERENCE ONLY: the MILDEST topology-generalization test in this
series. N stays exactly at the training value, and EVERY node keeps its
EXACT training-time degree (number of neighbors) -- only WHICH specific
nodes it's paired with changes, via a degree-preserving double-edge-swap
(the standard random-graph technique: pick two edges (a,b) and (c,d) with
4 distinct endpoints and rewire to (a,d)+(c,b), so a and c each keep their
degree, b and d each keep theirs). All per-node primitives (D/G boxes,
a_tilde/a/b) are UNCHANGED -- only the trading cost/capacity (c, kappa) of
the specific edges that moved get a fresh draw, since there's no old value
for a brand-new pair.

This isolates "does the model generalize to a different WIRING of the same
degree sequence" from the harsher node-removal/addition tests (which also
change N and/or the degree sequence itself). Not applicable to density=1.0
(the complete graph is the only graph with that degree sequence on N
nodes -- there is nothing to rewire without removing edges).

Usage: python3 demo_zeroshot_node_rewire.py <weights_path> <density>
       <gnn_rho> <gnn_scale> [n_test=15] [n_swaps=3] [dr_gamma=4.0]
       [gnn_rho_slow=0.99] [gnn_scale_slow=0.4] [base_seed=0] [out_path]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod
from demo import run_pair, plot_results
from plot_style import stats_from
from build_realistic_training_dataset import build_realistic_base_market, REALISTIC_MARKET_CONFIG


def degree_preserving_rewire(spec: market.MarketSpec, rng: np.random.Generator, n_swaps: int):
    """Returns a new MarketSpec with the SAME degree sequence (every node
    keeps its exact number of neighbors) but `n_swaps` double-edge-swaps
    applied, or None if a connected+feasible result couldn't be reached."""
    cfg = REALISTIC_MARKET_CONFIG
    neighbors = {n: set(v) for n, v in spec.neighbors.items()}
    edges = [tuple(sorted((m, n))) for n in spec.neighbors for m in spec.neighbors[n] if m < n]
    c_new = dict(spec.c)
    kappa_new = dict(spec.kappa)

    done, attempts = 0, 0
    while done < n_swaps and attempts < n_swaps * 60:
        attempts += 1
        i1, i2 = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[i1]
        c, d = edges[i2]
        if len({a, b, c, d}) != 4:
            continue
        if rng.random() < 0.5:
            new1, new2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        else:
            new1, new2 = tuple(sorted((a, c))), tuple(sorted((b, d)))
        na, nb = new1
        nc, nd = new2
        if nb in neighbors[na] or nd in neighbors[nc]:
            continue  # already an edge (or would create a duplicate/self-loop)

        neighbors[a].discard(b); neighbors[b].discard(a)
        neighbors[c].discard(d); neighbors[d].discard(c)
        neighbors[na].add(nb); neighbors[nb].add(na)
        neighbors[nc].add(nd); neighbors[nd].add(nc)

        for p, q in [(a, b), (b, a), (c, d), (d, c)]:
            c_new.pop((p, q), None); kappa_new.pop((p, q), None)
        for (p, q) in [new1, new2]:
            c_new[(p, q)] = float(rng.uniform(*cfg.c_range))
            c_new[(q, p)] = float(rng.uniform(*cfg.c_range))
            cap = float(rng.uniform(*cfg.kappa_range))
            kappa_new[(p, q)] = cap; kappa_new[(q, p)] = cap

        edges[i1], edges[i2] = new1, new2
        done += 1

    if done < n_swaps:
        return None
    neighbors_final = {n: sorted(v) for n, v in neighbors.items()}
    if not market._is_connected(neighbors_final, spec.N):
        return None
    new_spec = market.MarketSpec(
        N=spec.N, neighbors=neighbors_final,
        D_min=spec.D_min, D_max=spec.D_max, D_star=spec.D_star,
        G_min=spec.G_min, G_max=spec.G_max, G_res=spec.G_res,
        a_tilde=spec.a_tilde, a=spec.a, b=spec.b,
        c=c_new, kappa=kappa_new,
    )
    if not market.spec_is_feasible(new_spec):
        return None
    return new_spec


def main():
    weights_path = sys.argv[1]
    density = float(sys.argv[2])
    gnn_rho = float(sys.argv[3])
    gnn_scale = float(sys.argv[4])
    n_test = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    n_swaps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    dr_gamma = float(sys.argv[7]) if len(sys.argv) > 7 else 4.0
    gnn_rho_slow = float(sys.argv[8]) if len(sys.argv) > 8 else 0.99
    gnn_scale_slow = float(sys.argv[9]) if len(sys.argv) > 9 else 0.4
    base_seed = int(sys.argv[10]) if len(sys.argv) > 10 else 0
    out_path = sys.argv[11] if len(sys.argv) > 11 else "results.png"

    print(f"degree-preserving rewire zero-shot test  (weights={weights_path}  density={density}  "
          f"n_test={n_test}  n_swaps={n_swaps}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"base_seed={base_seed})", flush=True)

    device = torch.device("cpu")
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint", flush=True)

    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=base_seed, density=density)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}",
          flush=True)

    rng = np.random.default_rng(config.TEST_SEED + 13)
    specs = []
    attempts = 0
    while len(specs) < n_test and attempts < n_test * 30:
        attempts += 1
        spec = degree_preserving_rewire(base_spec, rng, n_swaps)
        if spec is not None:
            specs.append(spec)

    print(f"  {len(specs)}/{n_test} valid rewired instances found", flush=True)
    print("-" * 70)

    dr_iters, learned_iters, speedups, pcts = [], [], [], []
    for i, spec in enumerate(specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=False)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter)
        speedups.append(sp); pcts.append(pct)
        print(f"  {i + 1:2d}/{len(specs)}  DR {res_dr.n_iter:5d} it  |  learned {res_learned.n_iter:5d} it  |  "
              f"x{sp:5.2f} ({pct:+.1f}%)", flush=True)

    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"\n[rewire] mean speed-up {np.mean(speedups):.2f}x   median {np.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)
    print(f"[rewire] CORRECT mean % reduction (per-instance average): "
          f"mean={np.mean(pcts):.1f}%  median={np.median(pcts):.1f}%", flush=True)


if __name__ == "__main__":
    main()
