"""
build_methods_comparison_rounds_plot.py
==========================================

Same structure/style as build_methods_comparison_time_plot.py, but the
y-axis is total COMMUNICATION ROUNDS (median+IQR, log scale) instead of
wall-clock seconds -- the metric that actually matters for a genuinely
distributed deployment (network latency, not local compute, usually
dominates online time for small per-node payloads). Unlike the
parallel-equivalent TIME figure, A2DR belongs here: its round cost model
(1 + 2*diameter per iteration -- 1 plain-DR round + a network-wide
gather+broadcast for its least-squares step, every single iteration) was
the whole reason compare_a2dr_vs_official.py measured trading-graph
diameter in the first place.

Six categories per panel, three different round-cost models:
  - Plain DR / DR+GNN: 1 round/iteration (one-hop neighbour exchange only,
    `algorithm.run_dr`'s own design) -> rounds = iterations exactly.
  - A2DR: 1+2*diameter rounds/iteration (plain DR's own round, PLUS a
    gather+broadcast for the global least-squares extrapolation, paid
    EVERY iteration) -> a2dr_benchmark/*/results.json's own
    `a2dr_total_rounds` (already computed there).
  - L2WS (3 loss variants): a ONE-TIME 2*diameter gather+broadcast (collect
    every node's private theta to a central point for the MLP, then
    broadcast the predicted warm-start z^0 back out -- model.py's
    `build_theta` reads all nodes/edges at once, so there is no way to
    avoid this centralisation), followed by ordinary 1-round/iteration
    plain DR for the rest of the run -> rounds = 2*diameter +
    l2ws_iters, computed here from time_l2ws_vs_dr.py's own
    timing_results.json (l2ws_iters) + the SAME trading_graph_diameter
    a2dr_benchmark already measured for that density's market (same
    base_seed/density -> same graph, so the SAME diameter applies).

Usage: python3 build_methods_comparison_rounds_plot.py [out_name]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import plot_style as ps

OUT_DIR = "runs/paper_plots"
DENSITIES = ["0.2", "0.4", "0.5", "0.6", "0.8", "1.0"]
METHODS = ["plain_dr", "phiw", "a2dr", "l2ws_reg", "l2ws_asym", "l2ws_fpres"]
METHOD_LABEL = {"plain_dr": "Plain DR", "phiw": "DR+GNN", "a2dr": "A2DR (no training)",
                "l2ws_reg": "L2WS (regression)", "l2ws_asym": "L2WS (asymmetric)",
                "l2ws_fpres": "L2WS (fp-residual)"}
METHOD_SHORT = {"plain_dr": "Plain DR", "phiw": "DR+GNN", "a2dr": "A2DR", "l2ws_reg": "L2WS-reg",
                "l2ws_asym": "L2WS-asym", "l2ws_fpres": "L2WS-fpres"}
METHOD_COLOR = {"plain_dr": "#000000", "phiw": "#0072B2", "a2dr": "#CC79A7", "l2ws_reg": "#E69F00",
                "l2ws_asym": "#009E73", "l2ws_fpres": "#D55E00"}

A2DR_JSON = {d: f"a2dr_benchmark/runs/a2dr_vs_official_d{d}/results.json" for d in DENSITIES}

L2WS_JSON = {
    "0.2": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_regression/timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_asymmetric/timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_fpresidual_FAILED/timing_results.json"},
    "0.4": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_regression/timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_asymmetric/timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_fpresidual/timing_results.json"},
    "0.5": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_regression/timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_asymmetric/timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_fpresidual/timing_results.json"},
    "0.6": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_regression/timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_asymmetric/timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_fpresidual/timing_results.json"},
    "0.8": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_regression/timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_asymmetric/timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_fpresidual/timing_results.json"},
    "1.0": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_regression/timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_asymmetric/timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_fpresidual/timing_results.json"},
}


def load_json(path):
    if not os.path.exists(path):
        return None, f"missing: {path}"
    with open(path) as f:
        return json.load(f), None


def load_data():
    data = {}
    warnings = []
    for dkey in DENSITIES:
        data[dkey] = {}
        a2dr_blob, warn = load_json(A2DR_JSON[dkey])
        if warn:
            warnings.append(warn)
            continue
        if len(a2dr_blob["dr_total_rounds"]) != 30:
            warnings.append(f"WARNING: expected 30, found {len(a2dr_blob['dr_total_rounds'])} in {A2DR_JSON[dkey]}")
            continue
        diameter = a2dr_blob["trading_graph_diameter"]
        data[dkey]["plain_dr"] = a2dr_blob["dr_total_rounds"]
        data[dkey]["phiw"] = a2dr_blob["learned_total_rounds"]
        data[dkey]["a2dr"] = a2dr_blob["a2dr_total_rounds"]

        for m in ("l2ws_reg", "l2ws_asym", "l2ws_fpres"):
            path = L2WS_JSON[dkey][m]
            blob, warn = load_json(path)
            if warn:
                warnings.append(warn)
                continue
            if len(blob["l2ws_iters"]) == 30:
                data[dkey][m] = [2 * diameter + it for it in blob["l2ws_iters"]]
            else:
                warnings.append(f"WARNING: expected 30, found {len(blob['l2ws_iters'])} in {path}")
    return data, warnings


def plot_rounds_comparison_paperstyle(data, out_name):
    n = len(DENSITIES)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    _rc_old = {k: plt.rcParams[k] for k in ("font.family", "mathtext.fontset")}
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False)

    x = np.arange(len(METHODS))
    for i, dkey in enumerate(DENSITIES):
        ax = axes[i // ncols][i % ncols]
        ax.set_yscale("log")

        dr_rounds = data.get(dkey, {}).get("plain_dr")
        if dr_rounds:
            ax.axhline(np.median(dr_rounds), color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)

        for j, m in enumerate(METHODS):
            rounds = data.get(dkey, {}).get(m)
            if not rounds:
                continue
            rounds = np.asarray(rounds, dtype=float)
            median = np.median(rounds)
            p25, p75 = np.percentile(rounds, 25), np.percentile(rounds, 75)
            err = np.array([[median - p25], [p75 - median]])
            ax.errorbar(j, median, yerr=err, fmt="s", ms=11, color=METHOD_COLOR[m],
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in METHODS], fontsize=9, rotation=20)
        ax.tick_params(axis="y", labelsize=10)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        panel_letter = chr(ord("a") + i)
        ax.set_xlabel(f"({panel_letter}) $\\delta = {dkey}$", fontsize=11, labelpad=10)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("total communication rounds (log)", fontsize=11)

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=METHOD_COLOR[m],
                          markeredgecolor="white", markersize=10, label=METHOD_LABEL[m]) for m in METHODS]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(_rc_old)
    print(f"wrote {out_path}")


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "fig_methods_comparison_rounds_paperstyle.png"
    data, warnings = load_data()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    print("\nSummary (median communication rounds, IQR):")
    for dkey in DENSITIES:
        for m in METHODS:
            rounds = data.get(dkey, {}).get(m)
            if not rounds:
                print(f"  density={dkey:4s} {m:9s}  MISSING")
                continue
            rounds = np.asarray(rounds, dtype=float)
            print(f"  density={dkey:4s} {m:9s} n={len(rounds):2d}  "
                  f"median={np.median(rounds):9,.0f}  IQR=[{np.percentile(rounds,25):8,.0f}, "
                  f"{np.percentile(rounds,75):8,.0f}]")
    plot_rounds_comparison_paperstyle(data, out_name)


if __name__ == "__main__":
    main()
