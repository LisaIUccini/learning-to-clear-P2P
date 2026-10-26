"""
build_methods_comparison_with_distributed_plot.py
==========================================

Same structure/style as build_methods_comparison_plot.py (median+IQR block
markers, Computer Modern serif, faceted by density, IN-DISTRIBUTION ONLY),
extended with all 3 DISTRIBUTED L2WS variants (message-passing GNN warm-
start, distributed_model.py) alongside their centralized counterparts --
8 categories per panel instead of 5:

    DR+GNN, A2DR, L2WS-reg (centralized), L2WS-reg (distributed),
    L2WS-asym (centralized), L2WS-asym (distributed),
    L2WS-fpres (centralized), L2WS-fpres (distributed)

Color encodes LOSS-MODE IDENTITY (one hue per {phiw, a2dr, regression,
asymmetric, fp-residual} -- same 5 Okabe-Ito hues build_methods_comparison_
plot.py already uses, unchanged); centralized-vs-distributed is a SECOND,
independent encoding (filled square = centralized, hollow square =
distributed) rather than 3 more hues -- keeps the categorical palette at
its original validated 5 colors (adding 3 more would leave no safety
margin in an 8-way Okabe-Ito split) and visually groups each loss mode's
two variants together, which is the actual comparison this figure exists
to show.

Usage: python3 build_methods_comparison_with_distributed_plot.py [out_name]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import plot_style as ps

OUT_DIR = "runs/paper_plots"
DENSITIES = ["0.2", "0.4", "0.5", "0.6", "0.8", "1.0"]
METHODS = ["phiw", "a2dr", "l2ws_reg", "l2ws_reg_dist", "l2ws_asym", "l2ws_asym_dist",
           "l2ws_fpres", "l2ws_fpres_dist"]
METHOD_LABEL = {
    "phiw": "DR+GNN", "a2dr": "A2DR (no training)",
    "l2ws_reg": "L2WS-reg (centralized)", "l2ws_reg_dist": "L2WS-reg (distributed)",
    "l2ws_asym": "L2WS-asym (centralized)", "l2ws_asym_dist": "L2WS-asym (distributed)",
    "l2ws_fpres": "L2WS-fpres (centralized)", "l2ws_fpres_dist": "L2WS-fpres (distributed)",
}
METHOD_SHORT = {
    "phiw": "DR+GNN", "a2dr": "A2DR",
    "l2ws_reg": "L2WS-reg (cent.)", "l2ws_reg_dist": "L2WS-reg (dist.)",
    "l2ws_asym": "L2WS-asym (cent.)", "l2ws_asym_dist": "L2WS-asym (dist.)",
    "l2ws_fpres": "L2WS-fpres (cent.)", "l2ws_fpres_dist": "L2WS-fpres (dist.)",
}
# color = loss-mode identity (same 5 Okabe-Ito hues as build_methods_comparison_plot.py);
# centralized/distributed is encoded by marker fill (filled vs hollow), not color.
METHOD_COLOR = {
    "phiw": "#0072B2", "a2dr": "#CC79A7",
    "l2ws_reg": "#E69F00", "l2ws_reg_dist": "#E69F00",
    "l2ws_asym": "#009E73", "l2ws_asym_dist": "#009E73",
    "l2ws_fpres": "#D55E00", "l2ws_fpres_dist": "#D55E00",
}
DISTRIBUTED = {"l2ws_reg_dist", "l2ws_asym_dist", "l2ws_fpres_dist"}

A2DR_JSON = {d: f"a2dr_benchmark/runs/a2dr_vs_official_d{d}/results.json" for d in DENSITIES}

PHIW_LOG = {
    "0.2": "runs/official/official_N10_d0.2_seed3_replicate/train.log",
    "0.4": "runs/official/official_N10_d0.4_seed3_replicate/train.log",
    "0.5": "runs/official/official_N10_d0.5_seed1_replicate/train.log",
    "0.6": "runs/official/official_N10_d0.6_seed3_replicate/train.log",
    "0.8": "runs/official/official_N10_d0.8_seed0_replicate_fixed/train.log",
    "1.0": "runs/official/official_N10_d1.0_seed2_replicate/train.log",
}

SEED = {"0.2": 3, "0.4": 3, "0.5": 1, "0.6": 3, "0.8": 0, "1.0": 2}
FPRES_DIR = {"0.2": "l2ws_N10_d0.2_seed3_fpresidual_FAILED"}  # only d0.2's centralized fpres has this suffix


def centralized_log(dkey, loss):
    seed = SEED[dkey]
    if loss == "fpresidual" and dkey in FPRES_DIR:
        name = FPRES_DIR[dkey]
    else:
        name = f"l2ws_N10_d{dkey}_seed{seed}_{loss}"
    return f"l2ws_benchmark/runs/l2ws/centralized/{name}/train.log"


def distributed_log(dkey, loss):
    seed = SEED[dkey]
    return f"l2ws_benchmark/runs/l2ws/distributed/l2ws_N10_d{dkey}_seed{seed}_distributed_{loss}/train.log"


LINE_RE = re.compile(
    r"DR\s+(\d+)\s*it\s*\|\s*(?:learned|L2WS)\s+(\d+)\s*it\s*\|\s*x\s*([\d.]+)\s*\(([+-][\d.]+)%\)"
)


def parse_last_n(log_path, n=30):
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
        return matches, f"WARNING: expected >= {n}, parsed {len(matches)} from {log_path}"
    return matches[-n:], None


def load_a2dr_json(json_path, n=30):
    if not os.path.exists(json_path):
        return None, f"missing: {json_path}"
    import json
    with open(json_path) as f:
        d = json.load(f)
    dr_iters, a2_iters = d["dr_iters"], d["a2dr_iters"]
    if len(dr_iters) != n:
        return dr_iters, f"WARNING: expected {n}, found {len(dr_iters)} in {json_path}"
    matches = []
    for dr_it, a2_it in zip(dr_iters, a2_iters):
        matches.append({"dr": dr_it, "learned": a2_it, "speedup": dr_it / max(a2_it, 1),
                         "pct": (dr_it - a2_it) / max(dr_it, 1) * 100.0})
    return matches, None


def load_data():
    data = {}
    warnings = []
    for dkey in DENSITIES:
        data[dkey] = {}
        for m in METHODS:
            if m == "phiw":
                path = PHIW_LOG[dkey]
                instances, warn = parse_last_n(path, n=30)
            elif m == "a2dr":
                path = A2DR_JSON[dkey]
                instances, warn = load_a2dr_json(path, n=30)
            else:
                base = m[:-5] if m in DISTRIBUTED else m          # strip "_dist"
                loss = {"l2ws_reg": "regression", "l2ws_asym": "asymmetric",
                        "l2ws_fpres": "fpresidual"}[base]
                path = distributed_log(dkey, loss) if m in DISTRIBUTED else centralized_log(dkey, loss)
                instances, warn = parse_last_n(path, n=30)
            if warn:
                warnings.append(warn)
            if instances and len(instances) == 30:
                data[dkey][m] = instances
    return data, warnings


def plot_methods_comparison_with_distributed(data, out_name):
    n = len(DENSITIES)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    _rc_old = {k: plt.rcParams[k] for k in ("font.family", "mathtext.fontset")}
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.1 * nrows), sharey=True, squeeze=False)

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
            is_dist = m in DISTRIBUTED
            ax.errorbar(j, median, yerr=err, fmt="s", ms=11,
                        color=METHOD_COLOR[m],
                        mfc=("none" if is_dist else METHOD_COLOR[m]),
                        mec=(METHOD_COLOR[m] if is_dist else "white"),
                        mew=(1.8 if is_dist else 0.6),
                        capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)
        ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_SHORT[m] for m in METHODS], fontsize=9, rotation=32, ha="right")
        ax.tick_params(axis="y", labelsize=10)
        panel_letter = chr(ord("a") + i)
        ax.set_xlabel(f"({panel_letter}) $\\delta = {dkey}$", fontsize=11, labelpad=14)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("iteration reduction vs. plain DR (%)", fontsize=11)

    color_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=METHOD_COLOR[m],
                                markeredgecolor="white", markersize=10,
                                label={"phiw": "DR+GNN", "a2dr": "A2DR", "l2ws_reg": "L2WS-regression",
                                       "l2ws_asym": "L2WS-asymmetric", "l2ws_fpres": "L2WS-fp-residual"}[m])
                     for m in ("phiw", "a2dr", "l2ws_reg", "l2ws_asym", "l2ws_fpres")]
    style_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=ps.INK, markeredgecolor="white",
                   markersize=10, label="centralized"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor=ps.INK,
                   markeredgewidth=1.8, markersize=10, label="distributed (GNN)"),
    ]
    leg1 = fig.legend(handles=color_handles, frameon=False, ncol=len(color_handles), loc="upper center",
                       bbox_to_anchor=(0.5, 1.05), fontsize=10)
    fig.add_artist(leg1)
    fig.legend(handles=style_handles, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.00), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(_rc_old)
    print(f"wrote {out_path}")


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "fig_methods_comparison_id_paperstyle_with_distributed.png"
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
                print(f"  density={dkey:4s} {m:15s}  MISSING")
                continue
            pct = np.array([r["pct"] for r in instances], dtype=float)
            sp = np.array([r["speedup"] for r in instances], dtype=float)
            win = (sp >= 1.0).mean()
            print(f"  density={dkey:4s} {m:15s} n={len(instances):2d}  "
                  f"median_pct={np.median(pct):6.1f}%  IQR=[{np.percentile(pct,25):5.1f}, {np.percentile(pct,75):5.1f}]  "
                  f"win_rate={win:.0%}")
    plot_methods_comparison_with_distributed(data, out_name)


if __name__ == "__main__":
    main()
