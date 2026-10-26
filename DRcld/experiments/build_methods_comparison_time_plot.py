"""
build_methods_comparison_time_plot.py
==========================================

Same visual style/faceting as build_methods_comparison_plot.py (median+IQR
block markers, Computer Modern serif, one panel per density, panel labels
"(a) delta = 0.2"), but the y-axis is now REAL WALL-CLOCK TIME in seconds
(log scale) instead of % iteration reduction vs. plain DR -- and since an
absolute-time axis has no natural "0 = no-op" reference the way % reduction
implicitly has plain DR as its own zero-line, Plain DR is added as its own
plotted category here (it wasn't a category at all in the % version).

Six categories per panel: Plain DR, DR+GNN (Phi_w), A2DR (no training),
L2WS-regression, L2WS-asymmetric, L2WS-fp-residual.

Two data sources, all REAL measured time.perf_counter() wall-clock (never
estimated from iteration counts x a constant):
  - Plain DR / DR+GNN / A2DR: compare_a2dr_vs_official.py's own
    results.json (a2dr_benchmark/runs/a2dr_vs_official_d<density>/), keys
    dr_time_s / learned_time_s / a2dr_time_s -- all three timed together,
    same process, same instances, so internally consistent.
  - L2WS (3 loss variants): time_l2ws_vs_dr.py's timing_results.json
    (l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d<density>_seed<seed>_<loss>/),
    retrofitted onto the already-trained checkpoints (train_l2ws.py's own
    train.log only ever recorded iteration counts, never timing) -- key
    l2ws_time_s. Plain DR is NOT re-plotted from this second source (it's
    the same quantity, just measured in a different process/moment); the
    Plain DR category uses ONLY the a2dr_benchmark measurement so the
    baseline bar is one single consistent source per density, not an
    average across two separately-timed runs.

Every source already uses the SAME level+width perturbed test set per
density (verified instance-by-instance: compare_a2dr_vs_official.py and
time_l2ws_vs_dr.py produce identical dr_iters sequences), so all six
categories in a given panel are directly comparable.

Usage: python3 build_methods_comparison_time_plot.py [out_name]
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
# Okabe-Ito colorblind-safe qualitative palette, same assignment as
# build_methods_comparison_plot.py for the shared methods (phiw/a2dr/l2ws_*),
# plus black for Plain DR -- it's the reference/baseline category, not one
# of the "competing methods" the other five colors are reserved for.
METHOD_COLOR = {"plain_dr": "#000000", "phiw": "#0072B2", "a2dr": "#CC79A7", "l2ws_reg": "#E69F00",
                "l2ws_asym": "#009E73", "l2ws_fpres": "#D55E00"}

A2DR_JSON = {
    "0.2": "a2dr_benchmark/runs/a2dr_vs_official_d0.2/results.json",
    "0.4": "a2dr_benchmark/runs/a2dr_vs_official_d0.4/results.json",
    "0.5": "a2dr_benchmark/runs/a2dr_vs_official_d0.5/results.json",
    "0.6": "a2dr_benchmark/runs/a2dr_vs_official_d0.6/results.json",
    "0.8": "a2dr_benchmark/runs/a2dr_vs_official_d0.8/results.json",
    "1.0": "a2dr_benchmark/runs/a2dr_vs_official_d1.0/results.json",
}

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


def load_json(path, n=30):
    if not os.path.exists(path):
        return None, f"missing: {path}"
    with open(path) as f:
        d = json.load(f)
    return d, None


def load_data():
    data = {}
    warnings = []
    for dkey in DENSITIES:
        data[dkey] = {}
        a2dr_blob, warn = load_json(A2DR_JSON[dkey])
        if warn:
            warnings.append(warn)
        elif len(a2dr_blob["dr_time_s"]) == 30:
            data[dkey]["plain_dr"] = a2dr_blob["dr_time_s"]
            data[dkey]["phiw"] = a2dr_blob["learned_time_s"]
            data[dkey]["a2dr"] = a2dr_blob["a2dr_time_s"]
        else:
            warnings.append(f"WARNING: expected 30 instances, found "
                             f"{len(a2dr_blob['dr_time_s'])} in {A2DR_JSON[dkey]}")

        for m in ("l2ws_reg", "l2ws_asym", "l2ws_fpres"):
            path = L2WS_JSON[dkey][m]
            blob, warn = load_json(path)
            if warn:
                warnings.append(warn)
                continue
            if len(blob["l2ws_time_s"]) == 30:
                data[dkey][m] = blob["l2ws_time_s"]
            else:
                warnings.append(f"WARNING: expected 30 instances, found "
                                 f"{len(blob['l2ws_time_s'])} in {path}")
    return data, warnings


def plot_time_comparison_paperstyle(data, out_name):
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
            dr_median = np.median(dr_times)
            # reference line at the Plain DR baseline's own median -- plays
            # the same role here that axhline(0) plays in the % version
            # (instant visual "faster/slower than DR" read), since 0 has no
            # meaning on a log-time axis.
            ax.axhline(dr_median, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)

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
            lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
        panel_letter = chr(ord("a") + i)
        ax.set_xlabel(f"({panel_letter}) $\\delta = {dkey}$", fontsize=11, labelpad=10)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("wall-clock time, seconds (log scale)", fontsize=11)

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
    out_name = sys.argv[1] if len(sys.argv) > 1 else "fig_methods_comparison_time_paperstyle.png"
    data, warnings = load_data()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    print("\nSummary (median wall-clock seconds, IQR):")
    for dkey in DENSITIES:
        for m in METHODS:
            times = data.get(dkey, {}).get(m)
            if not times:
                print(f"  density={dkey:4s} {m:9s}  MISSING")
                continue
            times = np.asarray(times, dtype=float)
            print(f"  density={dkey:4s} {m:9s} n={len(times):2d}  "
                  f"median={np.median(times):7.2f}s  IQR=[{np.percentile(times,25):6.2f}, "
                  f"{np.percentile(times,75):6.2f}]")
    plot_time_comparison_paperstyle(data, out_name)


if __name__ == "__main__":
    main()
