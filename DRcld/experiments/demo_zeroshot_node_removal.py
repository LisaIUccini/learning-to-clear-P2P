"""
demo_zeroshot_node_removal.py
==========================================

ZERO-SHOT, INFERENCE ONLY: tests a checkpoint on a TOPOLOGY it was never
trained on -- the SAME base market (same costs, capacities, cost-curvature
parameters), but with 1 or 2 nodes REMOVED (and every edge touching them).
No retraining, no parameter perturbation -- pure node removal, isolating
generalization to a different N from the usual level+width generalization
(which always keeps N fixed).

The GNN architecture has no N-dependent weights (hidden_dim is "fixed,
independent of N and |E|", gnn.py's GNNConfig) -- it is a per-node/per-edge
message-passing network, so running it on a smaller N is architecturally
valid without any code changes to the model itself.

remove_k=1: EXHAUSTIVE -- every one of the N nodes removed exactly once
            (n_test = N instances).
remove_k=2: a random sample of n_test distinct node PAIRS (without
            replacement) out of the C(N,2) possible pairs.

Each removal is checked for both connectivity (the remaining graph must
stay connected) and feasibility (market.spec_is_feasible); draws that fail
either check are discarded and a fresh pair is drawn instead (remove_k=2
only -- remove_k=1 is exhaustive so there's nothing to resample, a failing
node is just skipped and reported).

Usage: python3 demo_zeroshot_node_removal.py <weights_path> <density>
       <gnn_rho> <gnn_scale> [remove_k=1] [n_test=10] [dr_gamma=4.0]
       [gnn_rho_slow=0.99] [gnn_scale_slow=0.4] [base_seed=0] [out_path]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import itertools
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


def remove_nodes(spec: market.MarketSpec, removed: set[int]):
    """Returns a new MarketSpec with `removed` nodes (and every edge
    touching them) deleted, remaining nodes renumbered 0..N_new-1, or None
    if the result is disconnected."""
    keep = sorted(set(range(spec.N)) - removed)
    remap = {old: new for new, old in enumerate(keep)}
    N_new = len(keep)

    neighbors_new = {remap[n]: sorted(remap[m] for m in spec.neighbors[n] if m not in removed)
                      for n in keep}
    if not market._is_connected(neighbors_new, N_new):
        return None

    idx = np.array(keep)
    c_new = {(remap[m], remap[n]): v for (m, n), v in spec.c.items() if m not in removed and n not in removed}
    kappa_new = {(remap[m], remap[n]): v for (m, n), v in spec.kappa.items() if m not in removed and n not in removed}

    new_spec = market.MarketSpec(
        N=N_new, neighbors=neighbors_new,
        D_min=spec.D_min[idx], D_max=spec.D_max[idx], D_star=spec.D_star[idx],
        G_min=spec.G_min[idx], G_max=spec.G_max[idx], G_res=spec.G_res[idx],
        a_tilde=spec.a_tilde[idx], a=spec.a[idx], b=spec.b[idx],
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
    remove_k = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    n_test = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    dr_gamma = float(sys.argv[7]) if len(sys.argv) > 7 else 4.0
    gnn_rho_slow = float(sys.argv[8]) if len(sys.argv) > 8 else 0.99
    gnn_scale_slow = float(sys.argv[9]) if len(sys.argv) > 9 else 0.4
    base_seed = int(sys.argv[10]) if len(sys.argv) > 10 else 0
    out_path = sys.argv[11] if len(sys.argv) > 11 else "results.png"

    print(f"node-removal zero-shot test  (weights={weights_path}  density={density}  "
          f"remove_k={remove_k}  n_test={n_test}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
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
    N = base_spec.N
    print(f"  base market: N={N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    # candidate removal sets
    if remove_k == 1:
        candidates = [frozenset([i]) for i in range(N)]
    else:
        rng = np.random.default_rng(config.TEST_SEED + 99)
        all_pairs = list(itertools.combinations(range(N), remove_k))
        order = rng.permutation(len(all_pairs))
        candidates = [frozenset(all_pairs[i]) for i in order]

    specs = []
    n_disconnected_or_infeasible = 0
    for removed in candidates:
        if len(specs) >= n_test:
            break
        spec = remove_nodes(base_spec, set(removed))
        if spec is None:
            n_disconnected_or_infeasible += 1
            continue
        specs.append((removed, spec))

    print(f"  {len(specs)}/{n_test} valid removals found ({n_disconnected_or_infeasible} discarded: "
          f"disconnected or infeasible after removal)", flush=True)
    print("-" * 70)

    dr_iters, learned_iters, speedups, pcts = [], [], [], []
    for i, (removed, spec) in enumerate(specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=False)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter)
        speedups.append(sp); pcts.append(pct)
        print(f"  removed={sorted(removed)}  N={spec.N}  DR {res_dr.n_iter:5d} it  |  "
              f"learned {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)

    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"\n[node-removal] mean speed-up {np.mean(speedups):.2f}x   median {np.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)
    print(f"[node-removal] CORRECT mean % reduction (per-instance average): "
          f"mean={np.mean(pcts):.1f}%  median={np.median(pcts):.1f}%", flush=True)


if __name__ == "__main__":
    main()
