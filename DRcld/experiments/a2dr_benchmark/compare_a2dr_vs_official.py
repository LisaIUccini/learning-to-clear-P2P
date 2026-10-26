"""
compare_a2dr_vs_official.py
==========================================

Three-way benchmark: plain DR (distributed baseline), A2DR (Anderson
accelerated DR, Fu/Zhang/Boyd 2020, Algorithm 3.1 -- MODEL-FREE, no
training at all), and DR + the trained GNN correction (this project's own
Phi_w, "official" checkpoints from train_official_pilot.py). Centralised
DR was dropped (2026-10-16, user's own instruction: "non mi interessa il
dr centralizzato") -- it was only ever a sanity-check reference, not a
real contender.

SAME test set as the official Phi_w/L2WS runs: level+width perturbed
(+/-10%/+/-4%) from the SAME base market, seed=config.TRAIN_SEED+1 (the
project's own "test" seed slot, distinct from TRAIN_SEED/+2 used for
train/val), slot 0 = the unperturbed base instance -- this makes A2DR's
numbers directly comparable, INSTANCE BY INSTANCE, to the already-measured
official/L2WS results on this exact same test set, rather than 30 fresh x0
draws on a single market as the first version of this script did.

THREE comparison axes, all requested by the user (2026-10-16):
  1. Iterations to converge (what a shared-memory/simulated run sees).
  2. Communication ROUNDS (the real distributed-deployment cost): plain DR
     and DR+GNN need exactly 1 round/iteration (one-hop neighbour
     exchange, see algorithm.py/gnn.py's own module docstrings); A2DR
     needs `1 + 2*diameter` rounds/iteration (1 plain-DR round + a
     network-wide gather + broadcast for its (3.4) least-squares step's
     global inner products, each <= the trading graph's diameter hops).
  3. WALL-CLOCK TIME, measured directly (time.perf_counter() around each
     method's actual run, this process, single-threaded torch) -- distinct
     from the communication-rounds MODEL: this captures each method's real
     local COMPUTE cost per step too (A2DR's least-squares solve has a
     nontrivial per-iteration cost of its own, independent of network
     latency; the GNN forward pass does too), not just the idealised
     round-counting abstraction. Network latency itself is NOT modelled
     here (this machine runs everything in one process) -- wall-clock
     is a proxy for compute cost only, communication rounds for network
     cost only; a real deployment's online time is dominated by whichever
     of the two is larger depending on the network's actual latency.

A2DR hyperparameters (eta, mmax) are exposed as CLI args; the validated
defaults (eta=3e-5, mmax=5) come from a sweep on this project's own DR map
(see a2dr.py's module docstring) and are NOT density-specific in
principle (they depend on the local prox's bisection geometry, not which
market is drawn) -- re-sweep per density only if results look anomalous.

Usage: python3 compare_a2dr_vs_official.py [density] [base_seed] [gamma]
       [gnn_rho] [gnn_scale] [rho_slow] [scale_slow] [checkpoint_path]
       [n_test=30] [eta=3e-5] [mmax=5] [c_old_range=0.05,0.18] [out_dir]
"""
from __future__ import annotations

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))          # experiments/a2dr_benchmark/
_EXPERIMENTS = _os.path.dirname(_HERE)                          # experiments/
_ROOT = _os.path.dirname(_EXPERIMENTS)                           # DRcld/ (market.py, a2dr.py, ...)
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _EXPERIMENTS)   # build_realistic_training_dataset.py lives in experiments/

import collections
import json
import os
import statistics as stats
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
torch.set_num_threads(1)

import config
import market
import algorithm
import a2dr
import gnn as gnn_mod
from plot_style import C_DR, C_LEARNED, C_WIN, C_LOSS, INK, INK_MUTED, clean_axes
from build_realistic_training_dataset import sample_level_width_perturbed_dataset


def graph_diameter(neighbors: dict) -> int:
    diam = 0
    for src in neighbors:
        dist = {src: 0}
        q = collections.deque([src])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        diam = max(diam, max(dist.values()))
    return diam


def a2dr_rounds_per_iter(diameter: int) -> int:
    return 1 + 2 * diameter


def three_way(spec, model, dr_cfg, a2_cfg, device, seed):
    """Plain DR, A2DR, DR+GNN, all three from the SAME fresh random x0 for
    this test instance (matching the official runs' own "fresh random x0
    per test instance at eval" convention). Returns (res_dr, res_a2,
    res_l, t_dr, t_a2, t_l) -- the last three are wall-clock seconds."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = market.build_network(spec, device=device, rng=gen)
    x0 = net.flatten().clone()

    gnn_mod.undeploy(net)
    t0 = time.perf_counter()
    res_dr = algorithm.run_dr(net, dr_cfg, record=False)
    t_dr = time.perf_counter() - t0

    net_a2 = market.build_network(spec, device=device)
    net_a2.unflatten(x0)
    t0 = time.perf_counter()
    res_a2 = a2dr.run_a2dr(net_a2, a2_cfg, record=False)
    t_a2 = time.perf_counter() - t0

    net_l = market.build_network(spec, device=device)
    net_l.unflatten(x0)
    gnn_mod.deploy(model, net_l)
    t0 = time.perf_counter()
    res_l = algorithm.run_dr(net_l, dr_cfg, record=False)
    t_l = time.perf_counter() - t0

    return res_dr, res_a2, res_l, t_dr, t_a2, t_l


def plot_three_metrics(dr_iters, a2_iters, l_iters, dr_rounds, a2_rounds, l_rounds,
                        dr_time, a2_time, l_time, diameter, rounds_per_iter, density, out_path):
    """Three side-by-side panels, one per comparison axis -- iterations,
    communication rounds, wall-clock time -- same 3 methods/colors in each.
    The rounds and time panels also annotate each bar with the mean
    iteration count (muted, in parentheses below the panel's own value),
    so the iteration figure is always visible alongside the other two axes
    for direct cross-reference, without needing to look back at panel 1."""
    labels = ["Plain DR", "A2DR\n(no training)", "DR + GNN\n(official)"]
    colors = [C_DR, C_LOSS, C_LEARNED]
    iter_means = [np.mean(dr_iters), np.mean(a2_iters), np.mean(l_iters)]
    panels = [
        ("mean iterations to converge", iter_means, False, False),
        ("mean communication rounds (log)", [np.mean(dr_rounds), np.mean(a2_rounds), np.mean(l_rounds)], True, True),
        ("mean wall-clock time, seconds (log)", [np.mean(dr_time), np.mean(a2_time), np.mean(l_time)], True, True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (title, means, logscale, show_iters) in zip(axes, panels):
        x = np.arange(3)
        bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        if logscale:
            ax.set_yscale("log")
            # narrow dynamic-range bar triplets (e.g. wall-clock seconds, all
            # within 1 decade) otherwise get a cluttered, overlapping tick
            # label set from the default LogLocator -- cap at a few major
            # ticks, plain comma-formatted (not scientific notation).
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_title(title, fontsize=10.5)
        # headroom in AXES (not data) space so labels can't land outside the
        # autoscaled range -- a data-coordinate offset (e.g. m*1.12) can
        # exceed the axis limits on a narrow-range log panel, which forces
        # tight_layout to reserve a huge blank margin to avoid clipping it.
        ax.margins(y=0.35)
        for b, m, it in zip(bars, means, iter_means):
            fmt = f"{m:,.0f}" if m >= 10 else f"{m:.3g}"
            ax.annotate(fmt, xy=(b.get_x() + b.get_width() / 2, m), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color=INK)
            if show_iters:
                ax.annotate(f"({it:,.0f} it)", xy=(b.get_x() + b.get_width() / 2, m), xytext=(0, 18),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=7.5, color=INK_MUTED, style="italic")
        clean_axes(ax)
    fig.suptitle(f"density={density}  --  trading-graph diameter={diameter}  ->  "
                 f"A2DR: {rounds_per_iter} comm. rounds/iteration vs 1 for plain DR / DR+GNN", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(label, sps):
    win_rate = sum(s >= 1.0 for s in sps) / len(sps)
    print(f"  [{label}] mean speed-up {stats.mean(sps):.2f}x   median {stats.median(sps):.2f}x   "
          f"min {min(sps):.2f}x   max {max(sps):.2f}x   "
          f"win-rate {win_rate:.0%} ({sum(s >= 1.0 for s in sps)}/{len(sps)})", flush=True)
    return {"mean": stats.mean(sps), "median": stats.median(sps), "min": min(sps),
            "max": max(sps), "win_rate": win_rate}


def main():
    density = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    base_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    dr_gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    gnn_rho = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9765
    gnn_scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    rho_slow = float(sys.argv[6]) if len(sys.argv) > 6 else 0.99
    scale_slow = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    checkpoint_path = sys.argv[8] if len(sys.argv) > 8 else \
        _os.path.join(_EXPERIMENTS, "runs", "official", "official_N10_d0.2_seed3_replicate", "phi_w.pt")
    n_test = int(sys.argv[9]) if len(sys.argv) > 9 else 30
    eta = float(sys.argv[10]) if len(sys.argv) > 10 else 3e-5
    mmax = int(sys.argv[11]) if len(sys.argv) > 11 else 5
    c_old_range = (tuple(float(x) for x in sys.argv[12].split(",")) if len(sys.argv) > 12
                   else (0.05, 0.18))
    out_dir = sys.argv[13] if len(sys.argv) > 13 else \
        _os.path.join(_HERE, "runs", f"a2dr_vs_official_d{density}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (density={density}  base_seed={base_seed}  "
          f"gamma={dr_gamma}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  rho_slow={rho_slow}  "
          f"scale_slow={scale_slow}  n_test={n_test}  eta={eta}  mmax={mmax})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)
    a2_cfg = a2dr.A2DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                              n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                              bisect=config.DR_BISECT_STEPS, mmax=mmax, eta=eta)

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    diameter = graph_diameter(base_spec.neighbors)
    rounds_per_iter = a2dr_rounds_per_iter(diameter)
    n_edges = sum(len(v) for v in base_spec.neighbors.values()) // 2
    print(f"  base market: N={base_spec.N}  edges={n_edges}  trading-graph diameter={diameter}", flush=True)
    print(f"  -> A2DR communication cost model: {rounds_per_iter} rounds/iteration "
          f"(1 plain-DR round + gather + broadcast, each <= diameter hops) vs 1 for plain DR / DR+GNN",
          flush=True)

    # SAME test set as the official Phi_w/L2WS runs: level+width perturbed,
    # seed=TRAIN_SEED+1, slot 0 = unperturbed base instance.
    test_specs, _, n_discarded = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
    test_specs = [base_spec] + test_specs
    print(f"  test set: {n_test} instances (slot 0 = original, SAME level+width perturbed set the "
          f"official/L2WS runs used), {n_discarded} infeasible discarded", flush=True)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=rho_slow, scale_slow=scale_slow,
        hidden_dim=config.GNN_HIDDEN_DIM))
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint {checkpoint_path}", flush=True)

    dr_iters, a2_iters, l_iters = [], [], []
    dr_time, a2_time, l_time = [], [], []
    for i, spec in enumerate(test_specs):
        seed = config.TEST_SEED * 7919 + i
        res_dr, res_a2, res_l, t_dr, t_a2, t_l = three_way(spec, model, dr_cfg, a2_cfg, device, seed)
        dr_iters.append(res_dr.n_iter); a2_iters.append(res_a2.n_iter); l_iters.append(res_l.n_iter)
        dr_time.append(t_dr); a2_time.append(t_a2); l_time.append(t_l)
        s_a2 = res_dr.n_iter / max(res_a2.n_iter, 1)
        s_l = res_dr.n_iter / max(res_l.n_iter, 1)
        r_a2 = res_dr.n_iter / max(res_a2.n_iter * rounds_per_iter, 1)
        time_a2 = t_dr / max(t_a2, 1e-9)
        time_l = t_dr / max(t_l, 1e-9)
        flag = "" if (res_dr.converged and res_a2.converged and res_l.converged) else "  [DID NOT CONVERGE]"
        print(f"  {i + 1:2d}/{n_test}  DR {res_dr.n_iter:5d} it ({t_dr:6.2f}s)  |  "
              f"A2DR {res_a2.n_iter:5d} it ({t_a2:6.2f}s)  x{s_a2:5.2f} iters, x{r_a2:5.2f} rounds, "
              f"x{time_a2:5.2f} time  |  "
              f"GNN {res_l.n_iter:5d} it ({t_l:6.2f}s)  x{s_l:5.2f} iters, x{time_l:5.2f} time{flag}", flush=True)

    sp_a2 = [d / max(a, 1) for d, a in zip(dr_iters, a2_iters)]
    sp_l = [d / max(l, 1) for d, l in zip(dr_iters, l_iters)]
    rsp_a2 = [d / max(a * rounds_per_iter, 1) for d, a in zip(dr_iters, a2_iters)]
    tsp_a2 = [d / max(a, 1e-9) for d, a in zip(dr_time, a2_time)]
    tsp_l = [d / max(l, 1e-9) for d, l in zip(dr_time, l_time)]

    dr_rounds = [d * 1 for d in dr_iters]
    a2_rounds = [a * rounds_per_iter for a in a2_iters]
    l_rounds = [l * 1 for l in l_iters]

    print(flush=True)
    a2_summary = summarize("A2DR -- iterations", sp_a2)
    a2_round_summary = summarize("A2DR -- communication rounds", rsp_a2)
    a2_time_summary = summarize("A2DR -- wall-clock time", tsp_a2)
    l_summary = summarize("DR + GNN -- iterations", sp_l)
    l_time_summary = summarize("DR + GNN -- wall-clock time", tsp_l)

    results = {
        "n_test": n_test, "density": density, "base_seed": base_seed, "dr_gamma": dr_gamma,
        "a2dr_eta": eta, "a2dr_mmax": mmax,
        "trading_graph_diameter": diameter, "a2dr_rounds_per_iter": rounds_per_iter,
        "dr_iters": dr_iters, "a2dr_iters": a2_iters, "learned_iters": l_iters,
        "dr_time_s": dr_time, "a2dr_time_s": a2_time, "learned_time_s": l_time,
        "dr_total_rounds": dr_rounds, "a2dr_total_rounds": a2_rounds, "learned_total_rounds": l_rounds,
        "a2dr_speedup_iters": sp_a2, "a2dr_speedup_rounds": rsp_a2, "a2dr_speedup_time": tsp_a2,
        "learned_speedup_iters": sp_l, "learned_speedup_time": tsp_l,
        "a2dr_summary_iters": a2_summary, "a2dr_summary_rounds": a2_round_summary, "a2dr_summary_time": a2_time_summary,
        "learned_summary_iters": l_summary, "learned_summary_time": l_time_summary,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {os.path.join(out_dir, 'results.json')}", flush=True)

    plot_three_metrics(dr_iters, a2_iters, l_iters, dr_rounds, a2_rounds, l_rounds,
                        dr_time, a2_time, l_time, diameter, rounds_per_iter, density,
                        os.path.join(out_dir, "three_metrics.png"))
    print(f"saved {os.path.join(out_dir, 'three_metrics.png')}", flush=True)
    print(f"\nDONE -- {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
