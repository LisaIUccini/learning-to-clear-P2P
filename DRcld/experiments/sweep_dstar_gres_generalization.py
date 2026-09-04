"""
sweep_dstar_gres_generalization.py
=====================================

How well does a trained checkpoint generalise as the D*/G_res
perturbation grows? Zero-shot inference (NO retraining) on a SWEEP of
test sets built exactly like `demo_fixed_market_on_dstar_gres_scale.py`
(`build_dstar_gres_dataset`: D_star *= f_D, G_res *= f_G, independent
shared factors per instance, graph/costs/capacities/boxes held fixed),
at increasing perturbation magnitudes pct = 0%, 5%, 10%, ..., up to
`--max-pct`. pct=0% is the exact home market (only x0 varies).

For each pct level: n_test instances, DR vs DR+learned, per-instance
iteration-count reduction pct_reduction = (dr_iters-learned_iters)/dr_iters*100.
Plots mean pct_reduction (y) against the perturbation magnitude (x).

Usage: python3 sweep_dstar_gres_generalization.py <run_name>
       <weights_path> <base_seed> <density> <dr_gamma> <gnn_rho>
       <gnn_scale> [n_test] [max_pct] [step_pct]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import statistics as stats
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
import market
import algorithm
import gnn as gnn_mod
from demo import run_pair
from demo_fixed_market_on_dstar_gres_scale import build_dstar_gres_dataset
from plot_style import C_LEARNED, INK_MUTED, clean_axes


def plot_bars(results, out_dir, density, base_seed, weights_path):
    """Bar chart: one bar per pct level, height = mean pct-reduction across
    the n_test inferences at that level, whiskers spanning the observed
    MIN-MAX range (not a symmetric error bar -- the raw extremes actually
    seen across the n_test draws, since the per-instance distribution at a
    given perturbation level is not assumed symmetric)."""
    xs = [r["pct"] * 100 for r in results]
    means = [r["mean_pct_reduction"] for r in results]
    mins = [r["min_pct_reduction"] for r in results]
    maxs = [r["max_pct_reduction"] for r in results]
    yerr_lo = [m - lo for m, lo in zip(means, mins)]
    yerr_hi = [hi - m for m, hi in zip(means, maxs)]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    width = min([xs[i + 1] - xs[i] for i in range(len(xs) - 1)], default=5) * 0.6 if len(xs) > 1 else 3.0
    ax.bar(xs, means, width=width, color=C_LEARNED, edgecolor="white", linewidth=0.8, zorder=3)
    ax.errorbar(xs, means, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor=INK_MUTED,
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)
    ax.axhline(0, color=INK_MUTED, linestyle="-", linewidth=0.9, zorder=2)
    ax.set_xlabel("D*/G_res perturbation magnitude (%)")
    ax.set_ylabel("iteration-count reduction (%)")
    ax.set_title(f"density={density}  base_seed={base_seed}  ({os.path.basename(os.path.dirname(weights_path))})\n"
                 f"bar = mean across instances, whiskers = observed min-max range")
    clean_axes(ax)
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "generalization_sweep_bars.png"), dpi=150)
    plt.close(fig)


def main():
    run_name = sys.argv[1]
    weights_path = sys.argv[2]
    base_seed = int(sys.argv[3])
    density = float(sys.argv[4])
    dr_gamma = float(sys.argv[5])
    gnn_rho = float(sys.argv[6])
    gnn_scale = float(sys.argv[7])
    n_test = int(sys.argv[8]) if len(sys.argv) > 8 else 30
    max_pct = float(sys.argv[9]) if len(sys.argv) > 9 else 0.50
    step_pct = float(sys.argv[10]) if len(sys.argv) > 10 else 0.05

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    pct_levels = []
    p = 0.0
    while p <= max_pct + 1e-9:
        pct_levels.append(round(p, 4))
        p += step_pct

    print(f"writing all artifacts to {out_dir}/  (weights={weights_path}  base_seed={base_seed}  "
          f"density={density}  dr_gamma={dr_gamma}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"n_test={n_test}  pct_levels={pct_levels})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}",
          flush=True)

    results = []
    for level_i, pct in enumerate(pct_levels):
        print(f"\n--- level {level_i + 1}/{len(pct_levels)}: pct={pct*100:.1f}% ---", flush=True)
        if pct == 0.0:
            test_specs = [base_spec] * n_test
            n_infeasible = 0
        else:
            test_specs, log_rows, n_infeasible = build_dstar_gres_dataset(
                base_spec, n_instances=n_test, pct=pct, seed=config.TRAIN_SEED + 5)

        pct_reductions, speedups, dr_iters_l, learned_iters_l = [], [], [], []
        for i, spec in enumerate(test_specs):
            res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i)
            dr_it, learned_it = res_dr.n_iter, res_learned.n_iter
            pr = (dr_it - learned_it) / max(dr_it, 1) * 100.0
            sp = dr_it / max(learned_it, 1)
            pct_reductions.append(pr); speedups.append(sp)
            dr_iters_l.append(dr_it); learned_iters_l.append(learned_it)
            print(f"  {i + 1:2d}/{n_test}  DR {dr_it:5d} it  |  learned {learned_it:5d} it  "
                  f"|  reduction {pr:5.1f}%", flush=True)

        mean_pr = stats.mean(pct_reductions)
        median_pr = stats.median(pct_reductions)
        mean_sp = stats.mean(speedups)
        win_rate = sum(s >= 1.0 for s in speedups) / len(speedups)
        results.append({
            "pct": pct, "n_infeasible": n_infeasible,
            "mean_pct_reduction": mean_pr, "median_pct_reduction": median_pr,
            "min_pct_reduction": min(pct_reductions), "max_pct_reduction": max(pct_reductions),
            "mean_speedup": mean_sp, "win_rate": win_rate,
            "mean_dr_iters": stats.mean(dr_iters_l), "mean_learned_iters": stats.mean(learned_iters_l),
        })
        print(f"  [level {level_i + 1}/{len(pct_levels)}] pct={pct*100:5.1f}%  n_infeasible={n_infeasible:2d}/{n_test}  "
              f"mean_pct_reduction={mean_pr:6.2f}%  median={median_pr:6.2f}%  "
              f"mean_speedup={mean_sp:5.2f}x  win-rate={win_rate:.0%}", flush=True)

        # write after EVERY level, not just at the end -- so a partial sweep
        # (still running, or killed early) always leaves a readable/plottable
        # result for whatever levels finished so far.
        with open(os.path.join(out_dir, "sweep_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        plot_bars(results, out_dir, density, base_seed, weights_path)

    xs = [r["pct"] * 100 for r in results]
    ys = [r["mean_pct_reduction"] for r in results]
    ys_lo = [r["min_pct_reduction"] for r in results]
    ys_hi = [r["max_pct_reduction"] for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.fill_between(xs, ys_lo, ys_hi, color=C_LEARNED, alpha=0.15, label="min-max range")
    ax.plot(xs, ys, color=C_LEARNED, linewidth=2.2, marker="o", markersize=5, label="mean")
    ax.axhline(0, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.set_xlabel("D*/G_res perturbation magnitude (%)")
    ax.set_ylabel("mean iteration-count reduction (%)")
    ax.set_title(f"density={density}  base_seed={base_seed}  ({os.path.basename(os.path.dirname(weights_path))})")
    ax.legend(frameon=False, fontsize=9.5)
    clean_axes(ax)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "generalization_sweep.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out_path}", flush=True)
    print(f"saved {os.path.join(out_dir, 'sweep_results.json')}", flush=True)


if __name__ == "__main__":
    main()
