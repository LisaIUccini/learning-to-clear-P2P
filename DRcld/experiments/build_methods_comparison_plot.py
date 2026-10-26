"""
build_methods_comparison_plot.py
==========================================

Same visual style as build_paper_plots.py's
`plot_blocks_faceted_median_iqr_paperstyle` (faceted by density, one panel
per density, median+IQR block markers, Computer Modern serif, panel labels
"(a) delta = 0.2"), but faceting on METHOD instead of OOD condition, and
IN-DISTRIBUTION ONLY (each method's own "home"/same in-distribution test
set -- no OOD):

    Phi_w (DR+GNN, this project's own per-step correction)
    A2DR (Anderson accelerated DR, Fu/Zhang/Boyd 2020 -- classical,
          model-free, no training at all)
    L2WS -- regression loss (knows the true DR fixed point during training)
    L2WS -- asymmetric loss (self-supervised, Phi_w's own loss mechanism
            adapted to a warm-start-only architecture)
    L2WS -- fp-residual loss

Two different data sources, same per-instance (dr_iters, method_iters) pair
feeding the same median+IQR % reduction statistic either way:
  - phiw/l2ws_*: parses the per-instance "DR XXX it | learned YYY it |
    xS.SS (+PP.P%)" lines straight out of each run's own train.log (same
    line format/regex as build_paper_data.py uses for the OOD pipeline),
    last 30 matches = that run's own final in-distribution test-set
    evaluation.
  - a2dr: reads the already-tabulated `dr_iters`/`a2dr_iters` arrays out of
    `a2dr_benchmark/runs/a2dr_vs_official_d<density>/results.json`
    (compare_a2dr_vs_official.py's own output), which already used the
    SAME level+width perturbed test set as the phiw/l2ws runs.

Usage: python3 build_methods_comparison_plot.py [out_name]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import plot_style as ps

OUT_DIR = "runs/paper_plots"
DENSITIES = ["0.2", "0.4", "0.5", "0.6", "0.8", "1.0"]
METHODS = ["phiw", "a2dr", "l2ws_reg", "l2ws_asym", "l2ws_fpres"]
METHOD_LABEL = {"phiw": "DR+GNN", "a2dr": "A2DR (no training)", "l2ws_reg": "L2WS (regression)",
                "l2ws_asym": "L2WS (asymmetric)", "l2ws_fpres": "L2WS (fp-residual)"}
METHOD_SHORT = {"phiw": "DR+GNN", "a2dr": "A2DR", "l2ws_reg": "L2WS-reg", "l2ws_asym": "L2WS-asym",
                "l2ws_fpres": "L2WS-fpres"}
# Okabe-Ito colorblind-safe qualitative palette (same source build_paper_plots.py
# uses for its own density-axis scatter), 5 distinct slots reserved for method
# identity -- deliberately NOT the plasma severity scale COND_COLOR uses
# elsewhere, since this figure's categorical axis is method, not OOD severity.
METHOD_COLOR = {"phiw": "#0072B2", "a2dr": "#CC79A7", "l2ws_reg": "#E69F00", "l2ws_asym": "#009E73",
                "l2ws_fpres": "#D55E00"}

LOG_PATHS = {
    "0.2": {"phiw": "runs/official/official_N10_d0.2_seed3_replicate/train.log",
            "a2dr": "a2dr_benchmark/runs/a2dr_vs_official_d0.2/results.json",
            "l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_regression/train.log",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_asymmetric/train.log",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.2_seed3_fpresidual_FAILED/train.log"},
    "0.4": {"phiw": "runs/official/official_N10_d0.4_seed3_replicate/train.log",
            "a2dr": "a2dr_benchmark/runs/a2dr_vs_official_d0.4/results.json",
            "l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_regression/train.log",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_asymmetric/train.log",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.4_seed3_fpresidual/train.log"},
    "0.5": {"phiw": "runs/official/official_N10_d0.5_seed1_replicate/train.log",
            "a2dr": "a2dr_benchmark/runs/a2dr_vs_official_d0.5/results.json",
            "l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_regression/train.log",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_asymmetric/train.log",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.5_seed1_fpresidual/train.log"},
    "0.6": {"phiw": "runs/official/official_N10_d0.6_seed3_replicate/train.log",
            "a2dr": "a2dr_benchmark/runs/a2dr_vs_official_d0.6/results.json",
            "l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_regression/train.log",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_asymmetric/train.log",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.6_seed3_fpresidual/train.log"},
    "0.8": {"phiw": "runs/official/official_N10_d0.8_seed0_replicate_fixed/train.log",
            "a2dr": "a2dr_benchmark/runs/a2dr_vs_official_d0.8/results.json",
            "l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_regression/train.log",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_asymmetric/train.log",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d0.8_seed0_fpresidual/train.log"},
    "1.0": {"phiw": "runs/official/official_N10_d1.0_seed2_replicate/train.log",
            "a2dr": "a2dr_benchmark/runs/a2dr_vs_official_d1.0/results.json",
            "l2ws_reg": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_regression/train.log",
            "l2ws_asym": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_asymmetric/train.log",
            "l2ws_fpres": "l2ws_benchmark/runs/l2ws/centralized/l2ws_N10_d1.0_seed2_fpresidual/train.log"},
}

LINE_RE = re.compile(
    r"DR\s+(\d+)\s*it\s*\|\s*(?:learned|L2WS)\s+(\d+)\s*it\s*\|\s*x\s*([\d.]+)\s*\(([+-][\d.]+)%\)"
)


def parse_last_n(log_path: str, n: int = 30):
    if not os.path.exists(log_path):
        return None, f"missing: {log_path}"
    with open(log_path) as f:
        lines = f.readlines()
    matches = []
    for ln in lines:
        m = LINE_RE.search(ln)
        if m:
            dr_it, learned_it, sp, pct = m.groups()
            matches.append({"dr": int(dr_it), "learned": int(learned_it),
                             "speedup": float(sp), "pct": float(pct)})
    if len(matches) < n:
        return matches, f"WARNING: expected >= {n} instances, parsed {len(matches)} from {log_path}"
    return matches[-n:], None


def load_a2dr_json(json_path: str, n: int = 30):
    """A2DR's own data isn't a train.log (no training happens) -- it's
    compare_a2dr_vs_official.py's results.json, which already holds the
    raw per-instance dr_iters/a2dr_iters arrays on the SAME test set the
    phiw/l2ws runs use. Reduction here is vs. plain DR, exactly like every
    other method's own "DR XXX it | method YYY it" line."""
    if not os.path.exists(json_path):
        return None, f"missing: {json_path}"
    with open(json_path) as f:
        d = json.load(f)
    dr_iters, a2_iters = d["dr_iters"], d["a2dr_iters"]
    if len(dr_iters) != n:
        return dr_iters, f"WARNING: expected {n} instances, found {len(dr_iters)} in {json_path}"
    matches = []
    for dr_it, a2_it in zip(dr_iters, a2_iters):
        matches.append({"dr": dr_it, "learned": a2_it,
                         "speedup": dr_it / max(a2_it, 1),
                         "pct": (dr_it - a2_it) / max(dr_it, 1) * 100.0})
    return matches, None


def load_data():
    data = {}
    warnings = []
    for dkey in DENSITIES:
        data[dkey] = {}
        for m in METHODS:
            path = LOG_PATHS[dkey][m]
            loader = load_a2dr_json if m == "a2dr" else parse_last_n
            instances, warn = loader(path, n=30)
            if warn:
                warnings.append(warn)
            # only plot a COMPLETE 30-instance evaluation -- a run still mid-training
            # (e.g. l2ws_fpres for d0.8/d1.0, still in progress) would otherwise show
            # a noisy/incomplete block from whatever partial matches exist so far
            if instances and len(instances) == 30:
                data[dkey][m] = instances
    return data, warnings


def plot_methods_comparison_paperstyle(data, out_name):
    n = len(DENSITIES)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    _rc_old = {k: plt.rcParams[k] for k in ("font.family", "mathtext.fontset")}
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows), sharey=True, squeeze=False)

    x = np.arange(len(METHODS))
    for i, dkey in enumerate(DENSITIES):
        ax = axes[i // ncols][i % ncols]
        for j, m in enumerate(METHODS):
            instances = data.get(dkey, {}).get(m)
            if not instances:
                continue
            pct = np.array([r["pct"] for r in instances], dtype=float)
            median = np.median(pct)
            p25, p75 = np.percentile(pct, 25), np.percentile(pct, 75)
            err = np.array([[median - p25], [p75 - median]])
            ax.errorbar(j, median, yerr=err, fmt="s", ms=11, color=METHOD_COLOR[m],
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)
        ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in METHODS], fontsize=9.5, rotation=12)
        ax.tick_params(axis="y", labelsize=10)
        panel_letter = chr(ord("a") + i)
        ax.set_xlabel(f"({panel_letter}) $\\delta = {dkey}$", fontsize=11, labelpad=10)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("iteration reduction vs. plain DR (%)", fontsize=11)

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=METHOD_COLOR[m],
                          markeredgecolor="white", markersize=10, label=METHOD_LABEL[m]) for m in METHODS]
    fig.legend(handles=handles, frameon=False, ncol=len(METHODS), loc="upper center",
               bbox_to_anchor=(0.5, 1.04), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(_rc_old)
    print(f"wrote {out_path}")


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "fig_methods_comparison_id_paperstyle_with05.png"
    data, warnings = load_data()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    print("\nSummary (median % reduction, IQR):")
    for dkey in DENSITIES:
        for m in METHODS:
            instances = data.get(dkey, {}).get(m)
            if not instances:
                print(f"  density={dkey:4s} {m:9s}  MISSING")
                continue
            pct = np.array([r["pct"] for r in instances], dtype=float)
            sp = np.array([r["speedup"] for r in instances], dtype=float)
            win = (sp >= 1.0).mean()
            print(f"  density={dkey:4s} {m:9s} n={len(instances):2d}  "
                  f"median_pct={np.median(pct):6.1f}%  IQR=[{np.percentile(pct,25):5.1f}, {np.percentile(pct,75):5.1f}]  "
                  f"win_rate={win:.0%}")
    plot_methods_comparison_paperstyle(data, out_name)


if __name__ == "__main__":
    main()
