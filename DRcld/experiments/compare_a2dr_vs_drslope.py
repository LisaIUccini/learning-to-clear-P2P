"""
compare_a2dr_vs_drslope.py
==========================

How much of `seed0_density1.0_seed0_drslope`'s speed-up over plain DR
(2.09x mean, 100% win-rate, see runs/INDEX.md and
runs/cross_density/seed0_density1.0_seed0_drslope/NOTES.md) can a
classical, MODEL-FREE technique -- Anderson accelerated DR (A2DR, `a2dr.py`,
Fu/Zhang/Boyd 2020) -- already buy for free, with no training at all?

Four-way comparison, all four from the SAME random x0 per test instance:
plain (distributed) DR, A2DR, DR + the trained GNN correction
(`seed0_density1.0_seed0_drslope`'s own checkpoint), and centralised DR.
SAME market (density=1.0, N0=10, seed=0) and SAME DR gamma=4.0 that
checkpoint was trained/evaluated with.

Also reports a REALISTIC-DEPLOYMENT metric alongside the usual iteration
count: communication ROUNDS. Plain DR and DR+learned need exactly one
round of one-hop neighbour exchange per iteration (the GNN correction adds
no communication of its own, by design -- see gnn.py's module docstring).
A2DR's Anderson-extrapolation step, by contrast, needs a network-wide
reduction (to form the small Gram matrix in (3.4), which involves inner
products over the ENTIRE state vector, not just a node's neighbourhood)
and a network-wide broadcast of the resulting coefficients EVERY
iteration -- neither is a one-hop operation. Modelled here as `1 + 2*diam`
rounds per A2DR iteration (1 plain-DR round + a gather + a broadcast, each
<= the trading graph's diameter hops) -- see `a2dr_rounds_per_iter` below.

A FOURTH method, Centralised DR (`dr_centralized.py`), is included for the
opposite extreme: the SAME Douglas-Rachford map, but run by one coordinator
holding the entire state -- so it needs the SAME iteration count as plain
DR (validated: matches within +/-2 iterations on this market, see
`dr_centralized.py`'s own docstring) but ZERO communication PER ITERATION,
since nothing needs to travel between iterations. Its only communication
is a ONE-TIME gather (every prosumer's private cost/box data -> the
coordinator) and a ONE-TIME broadcast of the final result -- a FIXED cost
(`2*diameter` rounds total) independent of how many iterations it takes,
unlike the other three methods, whose cost scales with iteration count.
The catch: every prosumer must reveal its private data to one coordinator,
which is precisely the property a distributed algorithm exists to avoid --
Centralised DR is not "free," it trades away decentralisation/privacy for
communication efficiency, the polar opposite trade-off from A2DR (which
trades away decentralisation for a WORSE communication outcome than even
staying fully distributed).

A2DR hyperparameters: NOT the paper's own defaults (eta=1e-8, mmax=10,
Sec. 7) -- those oscillate for 1000+ iterations before eventually
converging on this market (a highly kinked, nonsmooth map: every local
prox is solved by 20-40 bisection steps under box/capacity constraints,
not a closed form or single Newton step like the paper's own test
problems, which were also equilibrated/preconditioned first, Sec. 5 --
this project's DR has neither). A regularization sweep (`eta` in the
adaptive-regularization term, (3.4) in the paper) on the density=0.3
market found the oscillation is an under-regularization artifact, not a
property of A2DR itself: eta=1e-8 gives 0.30-0.86x (SLOWER than plain DR);
eta~1e-4 gives a clean, stable ~1.2-1.3x; larger eta (>=0.1) over-
regularizes back toward plain DR's own 1.0x. A follow-up sweep on THIS
script's actual target market (density=1.0, seed=0, 6 x0 draws) confirmed
and sharpened this: eta=3e-5 with mmax=5 gives 1.56x mean / 1.50x min,
100% converged (mmax=3: 1.47x/1.41x; mmax=8: 1.55x/1.49x, i.e. mmax=5 is
already past the point of diminishing returns) -- these are the defaults
below. `--eta`/`--mmax` stay exposed as CLI args to re-sweep if needed.

Usage: python3 compare_a2dr_vs_drslope.py [n_test] [eta] [mmax]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import collections
import json
import os
import statistics as stats
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_num_threads(1)  # every tensor here is tiny (dim ~ 2N+E); BLAS
                          # thread-spawning overhead dominates actual compute
                          # otherwise (measured 5-6x wall-clock slowdown)

import config
import market
import algorithm
import a2dr
import dr_centralized
import gnn as gnn_mod
from plot_style import C_DR, C_LEARNED, C_WARMSTART, C_WIN, C_LOSS, C_TRAIN, C_VAL, INK, INK_MUTED, clean_axes

DENSITY = 1.0
BASE_SEED = 0
DR_GAMMA = 4.0
DRSLOPE_RUN = "seed0_density1.0_seed0_drslope"
GNN_RHO = 0.9667
GNN_SCALE = 1.0


def graph_diameter(neighbors: dict) -> int:
    """Exact diameter (max shortest-path length, in hops) of the trading
    graph, via BFS from every node -- cheap at this project's scale
    (N0=10). Used to model A2DR's per-iteration communication cost: a
    network-wide gather or broadcast needs at most `diameter` hops along
    a shortest-path (spanning-tree) route, whichever node ends up
    coordinating."""
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
    """Communication rounds ONE A2DR iteration costs, vs plain DR/DR+
    learned's fixed 1 (one-hop neighbour exchange only, see module
    docstring): 1 (the plain-DR round itself, identical to the baseline)
    + 1 gather (every node's local contribution to the (3.4) Gram matrix
    reaches a coordinator, <= diameter hops) + 1 broadcast (the resulting
    alpha^k coefficients reach every node again, <= diameter hops)."""
    return 1 + 2 * diameter


def four_way(spec, model, dr_cfg, a2_cfg, device, seed):
    """Plain (distributed) DR, A2DR, DR+learned, and centralised DR, all
    four from the SAME x0."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = market.build_network(spec, device=device, rng=gen)
    x0 = net.flatten().clone()

    gnn_mod.undeploy(net)
    res_dr = algorithm.run_dr(net, dr_cfg, record=False)

    net_a2 = market.build_network(spec, device=device)
    net_a2.unflatten(x0)
    res_a2 = a2dr.run_a2dr(net_a2, a2_cfg, record=False)

    net_l = market.build_network(spec, device=device)
    net_l.unflatten(x0)
    gnn_mod.deploy(model, net_l)
    res_l = algorithm.run_dr(net_l, dr_cfg, record=False)

    net_c = market.build_network(spec, device=device)
    net_c.unflatten(x0)
    res_c = dr_centralized.run_dr_centralized(net_c, dr_cfg, record=False)

    return res_dr, res_a2, res_l, res_c


def plot_bars(dr_iters, a2_iters, l_iters, cent_iters, out_path):
    """Per-instance grouped bars, sorted by DR's own difficulty (matches
    this project's established convention of ranking instances by DR
    iteration count, e.g. plot_style.stats_from's per-instance panels).
    Centralised DR's bars land almost exactly on plain DR's (by
    construction, same fixed-point map) -- the point of this panel is
    iteration count only; see total_communication_rounds.png for where
    the four methods actually differ."""
    n = len(dr_iters)
    order = np.argsort(dr_iters)
    dr_s = np.asarray(dr_iters)[order]
    a2_s = np.asarray(a2_iters)[order]
    l_s = np.asarray(l_iters)[order]
    c_s = np.asarray(cent_iters)[order]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n)
    w = 0.2
    ax.bar(x - 1.5 * w, dr_s, width=w, color=C_DR, label="plain DR (distributed)", zorder=3)
    ax.bar(x - 0.5 * w, c_s, width=w, color=C_WIN, label="centralised DR", zorder=3)
    ax.bar(x + 0.5 * w, a2_s, width=w, color=C_WARMSTART, label="A2DR (no training)", zorder=3)
    ax.bar(x + 1.5 * w, l_s, width=w, color=C_LEARNED, label="DR + learned (dr_slope)", zorder=3)
    ax.set_xlabel("test instance (sorted by plain-DR iteration count)")
    ax.set_ylabel("iterations to converge")
    ax.set_title(f"density={DENSITY}  seed={BASE_SEED}  gamma={DR_GAMMA}  ({n} test instances, shared x0 per instance)")
    ax.legend(frameon=False, fontsize=9.5)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_speedup_scatter(sp_a2, sp_l, out_path):
    """Per-instance A2DR speed-up vs. learned speed-up -- above the y=x
    line means the learned correction beats A2DR on that instance, below
    means A2DR (free, untrained) already matches or beats it."""
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    lo = 0.8 * min(min(sp_a2), min(sp_l), 1.0)
    hi = 1.15 * max(max(sp_a2), max(sp_l))
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linestyle="--", linewidth=1.0, zorder=2)
    ax.axhline(1.0, color=INK_MUTED, linewidth=0.8, zorder=1)
    ax.axvline(1.0, color=INK_MUTED, linewidth=0.8, zorder=1)
    colors = [C_WIN if a >= 1.0 else C_LOSS for a in sp_a2]
    ax.scatter(sp_a2, sp_l, c=colors, edgecolor=INK, linewidth=0.5, s=55, zorder=3)
    ax.set_xlabel("A2DR speed-up over plain DR (no training)")
    ax.set_ylabel("DR + learned speed-up over plain DR (dr_slope)")
    ax.set_title("per-instance: classical acceleration vs. learned correction")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_total_communication_rounds(dr_rounds, a2_rounds, l_rounds, cent_rounds,
                                     diameter, rounds_per_iter, out_path):
    """The whole point in one picture, across all FOUR methods: mean TOTAL
    communication rounds needed to reach convergence (log scale -- the
    range spans ~2 to ~2000). Centralised DR needs a handful of rounds
    (fixed, independent of iteration count) at the cost of every prosumer
    revealing private data to one coordinator; among the fully-distributed
    options (everything else), DR+learned is the cheapest, plain DR is the
    honest baseline, and A2DR -- despite winning on iteration count -- is
    the most expensive of all four."""
    labels = ["Centralised DR\n(privacy given up)", "DR + learned\n(dr_slope)",
              "Plain DR\n(distributed)", "A2DR\n(no training)"]
    means = [np.mean(cent_rounds), np.mean(l_rounds), np.mean(dr_rounds), np.mean(a2_rounds)]
    colors = [C_WIN, C_LEARNED, C_DR, C_LOSS]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    x = np.arange(4)
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(np.mean(dr_rounds), color=INK_MUTED, linestyle="--", linewidth=1.0, zorder=2,
               label="plain-DR (distributed) baseline")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean total communication rounds (log scale)")
    ax.set_title(f"real communication cost to converge, all 4 methods\n"
                 f"trading-graph diameter={diameter}  ->  A2DR: {rounds_per_iter} rounds/iteration; "
                 f"centralised DR: 2*diameter={2 * diameter} rounds total, ONCE")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m * 1.15, f"{m:,.0f}", ha="center", fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=9)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(label, sps):
    win_rate = sum(s >= 1.0 for s in sps) / len(sps)
    print(f"  [{label}] mean speed-up {stats.mean(sps):.2f}x   median {stats.median(sps):.2f}x   "
          f"min {min(sps):.2f}x   max {max(sps):.2f}x   "
          f"win-rate {win_rate:.0%} ({sum(s >= 1.0 for s in sps)}/{len(sps)})", flush=True)
    return {"mean": stats.mean(sps), "median": stats.median(sps), "min": min(sps),
            "max": max(sps), "win_rate": win_rate}


def write_outputs(out_dir, n_test, eta, mmax, diameter, rounds_per_iter,
                   dr_iters, a2_iters, l_iters, cent_iters):
    """Derive speed-ups (iteration- and communication-round-based) from
    raw per-instance iteration counts and write results.json/NOTES.md/
    plots. Pulled out of `main()` so a re-analysis (e.g. adding a new
    derived metric) can reuse already-computed `dr_iters`/`a2_iters`/
    `l_iters`/`cent_iters` instead of re-running the whole (deterministic,
    so bit-identical) simulation."""
    sp_a2 = [d / max(a, 1) for d, a in zip(dr_iters, a2_iters)]
    sp_l = [d / max(l, 1) for d, l in zip(dr_iters, l_iters)]
    sp_c = [d / max(c, 1) for d, c in zip(dr_iters, cent_iters)]
    rsp_a2 = [d / max(a * rounds_per_iter, 1) for d, a in zip(dr_iters, a2_iters)]

    dr_rounds = [d * 1 for d in dr_iters]
    a2_rounds = [a * rounds_per_iter for a in a2_iters]
    l_rounds = [l * 1 for l in l_iters]
    cent_rounds = [2 * diameter for _ in cent_iters]  # fixed, independent of iteration count

    print(flush=True)
    a2_summary = summarize("A2DR (no training, iterations)", sp_a2)
    a2_round_summary = summarize("A2DR (no training, communication rounds)", rsp_a2)
    l_summary = summarize("DR + learned (dr_slope)", sp_l)
    c_summary = summarize("Centralised DR (iterations, sanity check vs plain DR)", sp_c)
    print(f"  [Centralised DR] mean total communication rounds: {np.mean(cent_rounds):.0f} "
          f"(fixed = 2*diameter, vs plain DR's mean {np.mean(dr_rounds):,.0f})", flush=True)

    results = {
        "n_test": n_test, "density": DENSITY, "base_seed": BASE_SEED, "dr_gamma": DR_GAMMA,
        "a2dr_eta": eta, "a2dr_mmax": mmax,
        "trading_graph_diameter": diameter, "a2dr_rounds_per_iter": rounds_per_iter,
        "dr_iters": dr_iters, "a2dr_iters": a2_iters, "learned_iters": l_iters, "centralized_dr_iters": cent_iters,
        "dr_total_rounds": dr_rounds, "a2dr_total_rounds": a2_rounds,
        "learned_total_rounds": l_rounds, "centralized_dr_total_rounds": cent_rounds,
        "a2dr_speedup": sp_a2, "learned_speedup": sp_l, "a2dr_round_speedup": rsp_a2,
        "centralized_dr_speedup_iterations": sp_c,
        "a2dr_summary": a2_summary, "learned_summary": l_summary, "a2dr_round_summary": a2_round_summary,
        "centralized_dr_summary": c_summary,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {os.path.join(out_dir, 'results.json')}", flush=True)

    plot_bars(dr_iters, a2_iters, l_iters, cent_iters, os.path.join(out_dir, "comparison_bars.png"))
    plot_speedup_scatter(sp_a2, sp_l, os.path.join(out_dir, "speedup_scatter.png"))
    plot_total_communication_rounds(dr_rounds, a2_rounds, l_rounds, cent_rounds, diameter, rounds_per_iter,
                                     os.path.join(out_dir, "total_communication_rounds.png"))
    print(f"saved {os.path.join(out_dir, 'comparison_bars.png')}", flush=True)
    print(f"saved {os.path.join(out_dir, 'speedup_scatter.png')}", flush=True)
    print(f"saved {os.path.join(out_dir, 'total_communication_rounds.png')}", flush=True)

    notes = f"""# cross_density/seed0_density1.0_seed0_a2dr_vs_drslope

**Test**: four-way comparison on the SAME market as `{DRSLOPE_RUN}`
(density={DENSITY}, N0={config.BASE_MARKET_N}, seed={BASE_SEED}, DR
gamma={DR_GAMMA}), {n_test} test instances, each algorithm run from the
SAME random x0 per instance (seed = TEST_SEED*7919+i, this project's usual
convention): plain DR (distributed, `algorithm.run_dr`), A2DR (Anderson
accelerated DR, `a2dr.py` -- Algorithm 3.1 of Fu/Zhang/Boyd 2020,
MODEL-FREE, no training at all, eta={eta}, mmax={mmax}), DR + the trained
GNN correction from `{DRSLOPE_RUN}/phi_w.pt` (rho={GNN_RHO}, scale={GNN_SCALE}),
and centralised DR (`dr_centralized.py` -- the SAME DR map, run by one
coordinator holding the whole state; validated to match plain DR's
iteration count within +/-2 iterations on this market).

**Why eta={eta}, not the paper's own default (1e-8)**: a regularization
sweep found the paper's default oscillates for 1000+ iterations before
eventually converging (0.3-0.9x, i.e. SLOWER than plain DR) -- this DR's
local prox is solved by 20-40 bisection steps under box/capacity
constraints (a highly kinked, nonsmooth map), unlike the paper's own
closed-form/single-Newton test problems, which were also equilibrated
first (Sec. 5); this project's DR has neither. Raising eta (the (3.4)
regularization strength) to eta~3e-5-1e-4 removes the oscillation and
gives a clean, stable speed-up; eta>=0.1 over-regularizes back toward
plain DR's own 1.0x. mmax={mmax} was similarly chosen from a sweep on
this exact market (mmax=3: 1.47x mean; mmax=5: 1.56x; mmax=8: 1.55x --
diminishing returns past 5).

**Result, in ITERATIONS** (what a shared-memory/simulated run of each
would see -- centralised DR ties plain DR by construction, same map):

| method | mean | median | min | max | win-rate |
|---|---|---|---|---|---|
| Centralised DR | {c_summary['mean']:.2f}x | {c_summary['median']:.2f}x | {c_summary['min']:.2f}x | {c_summary['max']:.2f}x | {c_summary['win_rate']:.0%} |
| A2DR (no training) | {a2_summary['mean']:.2f}x | {a2_summary['median']:.2f}x | {a2_summary['min']:.2f}x | {a2_summary['max']:.2f}x | {a2_summary['win_rate']:.0%} |
| DR + learned (dr_slope) | {l_summary['mean']:.2f}x | {l_summary['median']:.2f}x | {l_summary['min']:.2f}x | {l_summary['max']:.2f}x | {l_summary['win_rate']:.0%} |

**Result, in TOTAL COMMUNICATION ROUNDS** (what a REAL P2P deployment
would see -- trading-graph diameter={diameter}): plain DR and DR+learned
cost 1 round/iteration; A2DR costs {rounds_per_iter} rounds/iteration [1
plain-DR round + a gather + a broadcast for the (3.4) least-squares
step's global inner products, each <= diameter hops]; centralised DR
costs a FIXED `2*diameter` = {2 * diameter} rounds TOTAL, independent of
iteration count (one gather of every prosumer's private data + one final
broadcast), because nothing needs to travel between iterations once the
coordinator holds the whole state:

| method | mean total rounds | requires |
|---|---|---|
| Centralised DR | **{np.mean(cent_rounds):.0f}** | every prosumer reveals private data to 1 coordinator |
| DR + learned (dr_slope) | {np.mean(l_rounds):,.0f} | nothing extra -- fully distributed, one-hop only |
| Plain DR (distributed) | {np.mean(dr_rounds):,.0f} | fully distributed, one-hop only (the baseline) |
| A2DR (no training) | {np.mean(a2_rounds):,.0f} | a global gather+broadcast EVERY iteration |

**Honest headline**: there is a real, three-way trade-off, not a single
ranking. Centralised DR is dramatically cheaper in communication (a fixed
handful of rounds vs. hundreds) but only because every prosumer gives up
exactly the privacy/decentralisation a distributed algorithm exists to
protect -- it is not a fair peer in the SAME category as the other three.
Among the methods that DO stay fully distributed (plain DR, A2DR,
DR+learned): DR+learned is the cheapest AND fastest; A2DR, despite winning
on raw iteration count ({a2_summary['mean']:.2f}x), is actually the MOST
expensive of all four once its non-local reduction/broadcast is honestly
counted -- worse than even staying with plain DR. A sparser/less-connected
market (diameter>1, more realistic for a real P2P grid) would only widen
every one of these gaps.

`comparison_bars.png`: per-instance iteration counts, all four methods,
sorted by plain-DR difficulty (centralised DR's bars sit almost exactly on
plain DR's, by construction). `speedup_scatter.png`: per-instance A2DR
speed-up vs. learned speed-up in iterations (y=x line splits which method
wins where). `total_communication_rounds.png`: mean total communication
rounds, all four methods, log scale -- the real headline picture.
"""
    with open(os.path.join(out_dir, "NOTES.md"), "w") as f:
        f.write(notes.strip() + "\n")
    print(f"saved {os.path.join(out_dir, 'NOTES.md')}", flush=True)


def main():
    n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    eta = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-5
    mmax = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    out_dir = os.path.join("runs", "cross_density", "seed0_density1.0_seed0_a2dr_vs_drslope")
    os.makedirs(out_dir, exist_ok=True)

    print(f"writing all artifacts to {out_dir}/  (n_test={n_test}  eta={eta}  mmax={mmax})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=DENSITY)
    dr_cfg = algorithm.DRConfig(gamma=DR_GAMMA, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)
    a2_cfg = a2dr.A2DRConfig(gamma=DR_GAMMA, lam=config.DR_LAMBDA,
                              n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                              bisect=config.DR_BISECT_STEPS, mmax=mmax, eta=eta)

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=BASE_SEED)
    diameter = graph_diameter(base_spec.neighbors)
    rounds_per_iter = a2dr_rounds_per_iter(diameter)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}  "
          f"trading-graph diameter={diameter}", flush=True)
    print(f"  -> A2DR communication cost model: {rounds_per_iter} rounds/iteration "
          f"(1 plain-DR round + gather + broadcast, each <= diameter hops) vs 1 for plain DR / DR+learned",
          flush=True)

    weights_path = os.path.join("runs", "cross_density", DRSLOPE_RUN, "phi_w.pt")
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=GNN_RHO, scale=GNN_SCALE, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint {weights_path}", flush=True)

    dr_iters, a2_iters, l_iters, cent_iters = [], [], [], []
    for i in range(n_test):
        seed = config.TEST_SEED * 7919 + i
        res_dr, res_a2, res_l, res_c = four_way(base_spec, model, dr_cfg, a2_cfg, device, seed)
        dr_iters.append(res_dr.n_iter); a2_iters.append(res_a2.n_iter)
        l_iters.append(res_l.n_iter); cent_iters.append(res_c.n_iter)
        s_a2 = res_dr.n_iter / max(res_a2.n_iter, 1)
        s_l = res_dr.n_iter / max(res_l.n_iter, 1)
        r_a2 = res_dr.n_iter / max(res_a2.n_iter * rounds_per_iter, 1)  # speed-up in COMMUNICATION ROUNDS
        flag = "" if (res_dr.converged and res_a2.converged and res_l.converged and res_c.converged) \
            else "  [DID NOT CONVERGE]"
        print(f"  {i + 1:2d}/{n_test}  DR {res_dr.n_iter:5d} it  |  centralized {res_c.n_iter:5d} it  |  "
              f"A2DR {res_a2.n_iter:5d} it (x{s_a2:5.2f} iters, x{r_a2:5.2f} comm rounds)  |  "
              f"learned {res_l.n_iter:5d} it (x{s_l:5.2f}){flag}", flush=True)

    write_outputs(out_dir, n_test, eta, mmax, diameter, rounds_per_iter, dr_iters, a2_iters, l_iters, cent_iters)


if __name__ == "__main__":
    main()
