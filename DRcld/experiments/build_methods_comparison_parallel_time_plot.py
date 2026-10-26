"""
build_methods_comparison_parallel_time_plot.py
==========================================

Same structure/style as build_methods_comparison_time_plot.py, but reads
PARALLEL-EQUIVALENT (critical-path, max-over-nodes-per-iteration) wall-
clock time instead of the earlier purely SEQUENTIAL (sum-over-nodes)
measurement -- i.e. an estimate of what each method's time would look like
on dedicated-hardware-per-node distributed deployment, where every
prosumer's local compute at a given DR round happens simultaneously, not
one after another in a single Python process.

    Plain DR / DR+GNN: a2dr_benchmark/time_parallel_equivalent.py's own
        results (dr_par_time_s / learned_par_time_s) -- every prosumer's
        dr_step() timed individually each round, MAX taken across nodes.
    A2DR: NOT included here. Its Anderson-extrapolation step is an
        inherently CENTRALISED linear-algebra operation on the full global
        state (the regularised least-squares solve, a2dr.py's own
        module docstring) -- there is no per-node decomposition to take a
        max over for that part, so a naive "max-over-nodes" timing would
        silently understate its real cost. The already-published
        SEQUENTIAL A2DR number (fig_methods_comparison_time_paperstyle.png)
        remains the honest one until a proper centralised+distributed
        hybrid timing model is built for it specifically.
    L2WS (3 loss variants): l2ws_benchmark/time_l2ws_parallel_equivalent.py
        -- the warm-start MLP call itself is centralised (one global theta
        vector, one MLP, model.py's own build_theta) and is charged IN
        FULL (no parallel speedup), while the subsequent post-warm-start
        plain-DR iterations use the same max-over-nodes model as Plain DR.

Usage: python3 build_methods_comparison_parallel_time_plot.py [out_name]
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
METHODS = ["plain_dr", "phiw", "l2ws_reg", "l2ws_asym", "l2ws_fpres"]
METHOD_LABEL = {"plain_dr": "Plain DR", "phiw": "DR+GNN", "l2ws_reg": "L2WS (regression)",
                "l2ws_asym": "L2WS (asymmetric)", "l2ws_fpres": "L2WS (fp-residual)"}
METHOD_SHORT = {"plain_dr": "Plain DR", "phiw": "DR+GNN", "l2ws_reg": "L2WS-reg",
                "l2ws_asym": "L2WS-asym", "l2ws_fpres": "L2WS-fpres"}
METHOD_COLOR = {"plain_dr": "#000000", "phiw": "#0072B2", "l2ws_reg": "#E69F00",
                "l2ws_asym": "#009E73", "l2ws_fpres": "#D55E00"}

DRGNN_JSON = {d: f"a2dr_benchmark/runs/a2dr_vs_official_d{d}/parallel_timing_results.json" for d in DENSITIES}

L2WS_JSON = {
    "0.2": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_regression/parallel_timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_asymmetric/parallel_timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_fpresidual_FAILED/parallel_timing_results.json"},
    "0.4": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_regression/parallel_timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_asymmetric/parallel_timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_fpresidual/parallel_timing_results.json"},
    "0.5": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_regression/parallel_timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_asymmetric/parallel_timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_fpresidual/parallel_timing_results.json"},
    "0.6": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_regression/parallel_timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_asymmetric/parallel_timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_fpresidual/parallel_timing_results.json"},
    "0.8": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_regression/parallel_timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_asymmetric/parallel_timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_fpresidual/parallel_timing_results.json"},
    "1.0": {"l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_regression/parallel_timing_results.json",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_asymmetric/parallel_timing_results.json",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_fpresidual/parallel_timing_results.json"},
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
        blob, warn = load_json(DRGNN_JSON[dkey])
        if warn:
            warnings.append(warn)
        elif len(blob["dr_par_time_s"]) == 30:
            data[dkey]["plain_dr"] = blob["dr_par_time_s"]
            data[dkey]["phiw"] = blob["learned_par_time_s"]
        else:
            warnings.append(f"WARNING: expected 30, found {len(blob['dr_par_time_s'])} in {DRGNN_JSON[dkey]}")

        for m in ("l2ws_reg", "l2ws_asym", "l2ws_fpres"):
            path = L2WS_JSON[dkey][m]
            blob, warn = load_json(path)
            if warn:
                warnings.append(warn)
                continue
            if len(blob["l2ws_par_time_s"]) == 30:
                data[dkey][m] = blob["l2ws_par_time_s"]
            else:
                warnings.append(f"WARNING: expected 30, found {len(blob['l2ws_par_time_s'])} in {path}")
    return data, warnings


def plot_parallel_time_comparison_paperstyle(data, out_name):
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

        dr_times = data.get(dkey, {}).get("plain_dr")
        if dr_times:
            ax.axhline(np.median(dr_times), color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)

        for j, m in enumerate(METHODS):
            times = data.get(dkey, {}).get(m)
            if not times:
                continue
            times = np.asarray(times, dtype=float)
            median = np.median(times)
            p25, p75 = np.percentile(times, 25), np.percentile(times, 75)
            err = np.array([[median - p25], [p75 - median]])
            ax.errorbar(j, median, yerr=err, fmt="s", ms=11, color=METHOD_COLOR[m],
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in METHODS], fontsize=9, rotation=20)
        ax.tick_params(axis="y", labelsize=10)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:,.2f}" if v < 1 else (f"{v:,.0f}" if v >= 10 else f"{v:,.1f}")))
        panel_letter = chr(ord("a") + i)
        ax.set_xlabel(f"({panel_letter}) $\\delta = {dkey}$", fontsize=11, labelpad=10)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("parallel-equivalent time, seconds (log)", fontsize=11)

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
    out_name = sys.argv[1] if len(sys.argv) > 1 else "fig_methods_comparison_parallel_time_paperstyle.png"
    data, warnings = load_data()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    print("\nSummary (median parallel-equivalent seconds, IQR):")
    for dkey in DENSITIES:
        for m in METHODS:
            times = data.get(dkey, {}).get(m)
            if not times:
                print(f"  density={dkey:4s} {m:9s}  MISSING")
                continue
            times = np.asarray(times, dtype=float)
            print(f"  density={dkey:4s} {m:9s} n={len(times):2d}  "
                  f"median={np.median(times):7.3f}s  IQR=[{np.percentile(times,25):6.3f}, "
                  f"{np.percentile(times,75):6.3f}]")
    plot_parallel_time_comparison_paperstyle(data, out_name)


if __name__ == "__main__":
    main()
