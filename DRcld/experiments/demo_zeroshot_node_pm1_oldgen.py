"""
demo_zeroshot_node_pm1_oldgen.py
==========================================

Same as `demo_zeroshot_node_pm1.py` (zero-shot, inference-only, +/-1 node
topology-generalization test), adapted to market.py's OLD generator +
c<b fix (c_old_range=(0.05,0.18)) -- every official_N10_d*_replicate*
checkpoint was trained on THIS generator, not the calibrated/realistic one
demo_zeroshot_node_pm1.py targets.

Usage: python3 demo_zeroshot_node_pm1_oldgen.py <weights_path> <density>
       <gnn_rho> <gnn_scale> [n_test=30] [dr_gamma=4.0] [gnn_rho_slow=0.99]
       [gnn_scale_slow=0.4] [base_seed=0] [out_path] [c_old_range]
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
from demo import run_pair


def remove_one_node(spec: market.MarketSpec, removed: int):
    keep = sorted(set(range(spec.N)) - {removed})
    remap = {old: new for new, old in enumerate(keep)}
    N_new = len(keep)
    neighbors_new = {remap[n]: sorted(remap[m] for m in spec.neighbors[n] if m != removed) for n in keep}
    if not market._is_connected(neighbors_new, N_new):
        return None
    idx = np.array(keep)
    c_new = {(remap[m], remap[n]): v for (m, n), v in spec.c.items() if m != removed and n != removed}
    kappa_new = {(remap[m], remap[n]): v for (m, n), v in spec.kappa.items() if m != removed and n != removed}
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


def add_one_node(spec: market.MarketSpec, rng: np.random.Generator, new_degree: int, cfg: market.MarketConfig):
    """Appends one freshly-drawn node (same primitive ranges as every other
    node, from the OLD generator's own MarketConfig) as index N, attached to
    `new_degree` distinct existing nodes chosen at random, with freshly
    drawn c/kappa for each new edge."""
    N_old = spec.N
    new_degree = max(1, min(new_degree, N_old))
    attach_to = rng.choice(N_old, size=new_degree, replace=False).tolist()

    a_tilde = float(rng.uniform(*cfg.a_tilde_range))
    a = float(rng.uniform(*cfg.a_range))
    b = float(rng.uniform(*cfg.b_range))
    G_max = float(rng.uniform(*cfg.Gmax_range))
    G_min_raw = float(rng.uniform(*cfg.Gmin_range))
    G_min = min(G_min_raw, G_max - 1e-3)
    G_res = float(rng.uniform(*cfg.G_res_range))
    G0_frac = float(rng.uniform(0.2, 0.8))
    D_spread = float(rng.uniform(*cfg.D_spread_range))
    D_star_jitter = float(rng.uniform(-0.3, 0.3))
    G0 = G_min + G0_frac * (G_max - G_min)
    D0 = G0 + G_res
    D_min = max(0.0, D0 - D_spread)
    D_max = D0 + D_spread
    D_star = float(np.clip(D0 + D_star_jitter, D_min, D_max))

    neighbors_new = {n: list(v) for n, v in spec.neighbors.items()}
    neighbors_new[N_old] = sorted(attach_to)
    for m in attach_to:
        neighbors_new[m] = sorted(neighbors_new[m] + [N_old])

    c_new = dict(spec.c)
    kappa_new = dict(spec.kappa)
    for m in attach_to:
        c_new[(m, N_old)] = float(rng.uniform(*cfg.c_range))
        c_new[(N_old, m)] = float(rng.uniform(*cfg.c_range))
        cap = float(rng.uniform(*cfg.kappa_range))
        kappa_new[(m, N_old)] = cap
        kappa_new[(N_old, m)] = cap

    new_spec = market.MarketSpec(
        N=N_old + 1, neighbors=neighbors_new,
        D_min=np.append(spec.D_min, D_min), D_max=np.append(spec.D_max, D_max),
        D_star=np.append(spec.D_star, D_star),
        G_min=np.append(spec.G_min, G_min), G_max=np.append(spec.G_max, G_max),
        G_res=np.append(spec.G_res, G_res),
        a_tilde=np.append(spec.a_tilde, a_tilde), a=np.append(spec.a, a), b=np.append(spec.b, b),
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
    n_test = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    dr_gamma = float(sys.argv[6]) if len(sys.argv) > 6 else 4.0
    gnn_rho_slow = float(sys.argv[7]) if len(sys.argv) > 7 else 0.99
    gnn_scale_slow = float(sys.argv[8]) if len(sys.argv) > 8 else 0.4
    base_seed = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    out_path = sys.argv[10] if len(sys.argv) > 10 else "results.png"
    c_old_range = (tuple(float(x) for x in sys.argv[11].split(",")) if len(sys.argv) > 11
                   else (0.05, 0.18))

    print(f"+/-1 node zero-shot test (generator=old, c<b fix)  (weights={weights_path}  density={density}  "
          f"n_test={n_test}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  base_seed={base_seed})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow,
        hidden_dim=config.GNN_HIDDEN_DIM))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint", flush=True)

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    N = base_spec.N
    degrees = [len(v) for v in base_spec.neighbors.values()]
    mean_degree = round(sum(degrees) / len(degrees))
    print(f"  base market: N={N}  edges={sum(degrees) // 2}  mean_degree={mean_degree}", flush=True)

    rng = np.random.default_rng(config.TEST_SEED + 7)
    specs, ops = [], []
    attempts = 0
    while len(specs) < n_test and attempts < n_test * 20:
        attempts += 1
        do_add = bool(rng.integers(0, 2))
        if do_add:
            spec = add_one_node(base_spec, rng, mean_degree, mc)
            op = "add"
        else:
            removed = int(rng.integers(0, N))
            spec = remove_one_node(base_spec, removed)
            op = "remove"
        if spec is None:
            continue
        specs.append(spec)
        ops.append(op)

    print(f"  {len(specs)}/{n_test} valid +/-1 instances found "
          f"({sum(o == 'add' for o in ops)} add, {sum(o == 'remove' for o in ops)} remove)", flush=True)
    print("-" * 70)

    dr_iters, learned_iters, speedups, pcts = [], [], [], []
    for i, (op, spec) in enumerate(zip(ops, specs)):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=False)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter)
        speedups.append(sp); pcts.append(pct)
        print(f"  [{op:6s}]  N={spec.N}  DR {res_dr.n_iter:5d} it  |  "
              f"learned {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)

    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"\n[pm1] mean speed-up {np.mean(speedups):.2f}x   median {np.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)
    print(f"[pm1] CORRECT mean % reduction (per-instance average): "
          f"mean={np.mean(pcts):.1f}%  median={np.median(pcts):.1f}%", flush=True)


if __name__ == "__main__":
    main()
