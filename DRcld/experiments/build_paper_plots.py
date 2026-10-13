"""
build_paper_plots.py
==========================================

Reads runs/cross_density/paper_plots/data.json (built by build_paper_data.py)
and produces the paper's main density-sweep figures + table:

  fig1_curves.png   -- single panel, line/marker style: in-distribution (home)
                       + OOD curves (level 20/30/40%) together vs density,
                       one line each, shaded 95% CI band per curve
  fig2_bars.png     -- grouped bar chart, same data, one figure:
                       5 density groups x bars per condition, 95% CI error bars
  table.md          -- markdown table, same numbers + communications/iteration column
                       (= 2 x edge count, since each prosumer only talks to its direct
                       neighbors each DR iteration -- a structural fact, not a trained result)

Colorblind-safe palette: plasma (perceptually uniform sequential colormap),
sampled across the conditions' natural severity ordering (home -> level50).

Usage: python3 build_paper_plots.py [data_path] [out_suffix] [full] [nodes]
       data_path: defaults to runs/cross_density/paper_plots/data.json
       out_suffix: appended to every output filename before the extension
       (e.g. "_v17" -> fig1_curves_v17.png), default "" (overwrites the
       standard files). Never overwrites a different suffix's files.
       full: pass the literal word "full" to include ALL 5 severity
       conditions (home + level20/30/40/50) instead of dropping +/-50%,
       on both the curve plots and the bar/block charts.
       nodes: pass the literal word "nodes" to ALSO add the node-removal
       topology-generalization conditions (remove1/remove2) as extra bars/
       blocks on fig2_bars/fig2_blocks only (never on the line plots,
       where they don't belong on the severity axis).
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

DATA_PATH = "runs/cross_density/paper_plots/data.json"
OUT_DIR = "runs/cross_density/paper_plots"
SUFFIX = ""

DENSITIES = ["0.2", "0.4", "0.6", "0.8", "1.0"]  # overwritten in main() from the loaded data's own keys
CONDITIONS = ["home", "level20", "level30", "level40", "level50", "remove1", "remove2", "pm1", "rewire"]
# figures currently drop the +/-50% OOD series (too noisy / wide CI at n=30);
# the table keeps all of CONDITIONS regardless.
PLOT_CONDITIONS = ["home", "level20", "level30", "level40"]
# bar/block charts (density-grouped, not a continuous line) can also show
# the node-removal topology-generalization conditions, which don't belong
# on the severity-ordered line plots. Overridable via CLI ("nodes").
BAR_CONDITIONS = ["home", "level20", "level30", "level40"]
COND_LABEL = {
    "home": "in-distribution (train)",
    "level20": "OOD $\\pm$20%",
    "level30": "OOD $\\pm$30%",
    "level40": "OOD $\\pm$40%",
    "level50": "OOD $\\pm$50%",
    "remove1": "remove 1 node",
    "remove2": "remove 2 nodes",
    "pm1": "$\\pm$1 node",
    "rewire": "rewire (same degree)",
}
# plasma colormap, sampled at 5 evenly spaced points (perceptually uniform,
# colorblind-safe sequential palette -- fits the natural ordering of these
# conditions: home (mildest, +/-10%) -> level20 -> level30 -> level40 ->
# level50 (most severe)). Stop short of pure yellow (poor contrast on white).
_PLASMA = plt.colormaps["plasma"]
_PLASMA_SAMPLES = [_PLASMA(x) for x in np.linspace(0.0, 0.82, 5)]
COND_COLOR = {
    "home": _PLASMA_SAMPLES[0],
    "level20": _PLASMA_SAMPLES[1],
    "level30": _PLASMA_SAMPLES[2],
    "level40": _PLASMA_SAMPLES[3],
    "level50": _PLASMA_SAMPLES[4],
    # node-removal (topology generalization) -- a different kind of
    # perturbation, not part of the level-severity scale, so deliberately
    # NOT sampled from the plasma sequence: distinct teal/grey instead.
    "remove1": "#009E73",
    "remove2": "#555555",
    "pm1": "#0072B2",
    "rewire": "#CC79A7",
}
# Okabe-Ito colorblind-safe qualitative palette, reserved for coloring BY
# DENSITY (a different semantic axis than COND_COLOR's severity-ordered
# plasma scale, used for the scatter plot pooling instances across densities)
_OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2", "#F0E442"]
# edges per base market (N=10 nodes, fixed) at each density -- structural fact,
# not a trained quantity: communications/iteration = 2 x edges (each neighbor
# pair exchanges a message in each direction every DR iteration)
EDGES = {"0.2": 16, "0.4": 23, "0.5": 27, "0.6": 31, "0.8": 38, "1.0": 45}
# per-node degree (= messages one single agent sends, and separately receives,
# per iteration -- the distributed system's actual per-agent communication
# burden, as opposed to the network-wide total above). Not a regular graph,
# so min/max matter: e.g. density=0.2 ranges from 1 to 5 neighbors depending
# on the node. Computed once via build_realistic_base_market per density's
# chosen base_seed (see inline degree-check script).
DEGREE = {
    "0.2": {"mean": 3.2, "min": 1, "max": 5},
    "0.4": {"mean": 4.6, "min": 3, "max": 7},
    "0.5": {"mean": 5.4, "min": 3, "max": 7},
    "0.6": {"mean": 6.2, "min": 4, "max": 8},
    "0.8": {"mean": 7.6, "min": 5, "max": 9},
    "1.0": {"mean": 9.0, "min": 9, "max": 9},
}


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def outp(name: str) -> str:
    base, ext = os.path.splitext(name)
    return os.path.join(OUT_DIR, f"{base}{SUFFIX}{ext}")


def get_series(data, condition):
    """Returns (densities_present, means, ci_lo, ci_hi, ns) for one condition
    across all 5 densities, skipping densities where that condition is missing."""
    xs, means, los, his, ns = [], [], [], [], []
    for dkey in DENSITIES:
        cond = data["densities"].get(dkey, {}).get("conditions", {}).get(condition)
        if cond is None:
            continue
        s = cond["summary"]
        xs.append(float(dkey))
        means.append(s["pct_mean"])
        los.append(s["pct_ci95_lo"])
        his.append(s["pct_ci95_hi"])
        ns.append(s["n"])
    return np.array(xs), np.array(means), np.array(los), np.array(his), ns


def get_series_median_iqr(data, condition):
    """Same as get_series but median + IQR (25th/75th percentile) computed
    directly from the per-instance pct values, instead of mean + 95% CI."""
    xs, medians, p25s, p75s = [], [], [], []
    for dkey in DENSITIES:
        cond = data["densities"].get(dkey, {}).get("conditions", {}).get(condition)
        if cond is None:
            continue
        pct = np.array([r["pct"] for r in cond["per_instance"]], dtype=float)
        xs.append(float(dkey))
        medians.append(np.median(pct))
        p25s.append(np.percentile(pct, 25))
        p75s.append(np.percentile(pct, 75))
    return np.array(xs), np.array(medians), np.array(p25s), np.array(p75s)


def plot_curves_median_iqr(data):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    handles, labels = [], []
    for cond in PLOT_CONDITIONS:
        xs, medians, p25s, p75s = get_series_median_iqr(data, cond)
        if len(xs) == 0:
            continue
        h, = ax.plot(xs, medians, "-o", color=COND_COLOR[cond], lw=2.2, ms=6.5, zorder=3)
        ax.fill_between(xs, p25s, p75s, color=COND_COLOR[cond], alpha=0.18, zorder=2, linewidth=0)
        handles.append(h); labels.append(COND_LABEL[cond])

    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("density")
    ax.set_ylabel("iteration reduction vs. plain DR (%)")
    ps.clean_axes(ax)

    fig.legend(handles, labels, frameon=False, ncol=len(PLOT_CONDITIONS), loc="upper center",
               bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = outp("fig1_curves_median_iqr.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_scatter_difficulty(data):
    """Per-instance scatter, pooled across ALL densities (in-distribution
    'home' condition only -- mixing in OOD severity too would conflate two
    different axes of variation). x = plain DR iterations (baseline
    difficulty), y = % iteration reduction. Tests visually whether the
    learned model's gain grows with how hard the baseline instance already
    is, within each density's own cloud of instances."""
    fig, ax = plt.subplots(figsize=(7.8, 5.6))

    for i, dkey in enumerate(DENSITIES):
        cond = data["densities"].get(dkey, {}).get("conditions", {}).get("home")
        if cond is None:
            continue
        dr = np.array([r["dr"] for r in cond["per_instance"]], dtype=float)
        pct = np.array([r["pct"] for r in cond["per_instance"]], dtype=float)
        ax.scatter(dr, pct, color=_OKABE_ITO[i % len(_OKABE_ITO)], s=32, alpha=0.8,
                   edgecolors="white", linewidths=0.5, zorder=3, label=f"density={dkey}")

    ax.set_xscale("log")
    ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("plain DR iterations (baseline difficulty, log scale)")
    ax.set_ylabel("iteration reduction vs. plain DR (%)")
    ps.clean_axes(ax)
    ax.legend(frameon=False, loc="best", fontsize=9, title="density", title_fontsize=9)

    fig.tight_layout()
    out_path = outp("fig3_scatter_difficulty.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_curves(data):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    handles, labels = [], []
    for cond in PLOT_CONDITIONS:
        xs, means, los, his, ns = get_series(data, cond)
        if len(xs) == 0:
            continue
        h, = ax.plot(xs, means, "-o", color=COND_COLOR[cond], lw=2.2, ms=6.5, zorder=3)
        ax.fill_between(xs, los, his, color=COND_COLOR[cond], alpha=0.18, zorder=2, linewidth=0)
        handles.append(h); labels.append(COND_LABEL[cond])

    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("density")
    ax.set_ylabel("iteration reduction vs. plain DR (%)")
    ps.clean_axes(ax)

    fig.legend(handles, labels, frameon=False, ncol=len(PLOT_CONDITIONS), loc="upper center",
               bbox_to_anchor=(0.5, 1.02), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = outp("fig1_curves.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_bars(data):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    n_cond = len(BAR_CONDITIONS)
    bar_w = 0.15
    group_x = np.arange(len(DENSITIES))

    for j, cond in enumerate(BAR_CONDITIONS):
        offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * bar_w
        means, errs = [], []
        for dkey in DENSITIES:
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                means.append(np.nan); errs.append(0.0)
                continue
            s = c["summary"]
            means.append(s["pct_mean"])
            errs.append((s["pct_ci95_hi"] - s["pct_ci95_lo"]) / 2.0)
        ax.bar(group_x + offsets[j], means, width=bar_w, color=COND_COLOR[cond],
               label=COND_LABEL[cond], yerr=errs, capsize=2.5,
               error_kw={"elinewidth": 1.0, "ecolor": ps.INK_MUTED}, zorder=3)

    ax.set_xticks(group_x)
    ax.set_xticklabels(DENSITIES)
    ax.set_xlabel("density")
    ax.set_ylabel("iteration reduction vs. plain DR (%)")
    ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
    ax.legend(frameon=False, ncol=min(len(BAR_CONDITIONS), 5), loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8.5)
    ps.clean_axes(ax)
    fig.tight_layout()
    out_path = outp("fig2_bars.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_blocks(data):
    """Same grouped layout as plot_bars, but each condition is a small
    BLOCK (square marker) sitting at its value, not a bar filled from
    zero -- removes the "histogram" look while keeping the same grouping
    and error bars."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    n_cond = len(BAR_CONDITIONS)
    bar_w = 0.15
    group_x = np.arange(len(DENSITIES))

    for j, cond in enumerate(BAR_CONDITIONS):
        offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * bar_w
        means, errs = [], []
        for dkey in DENSITIES:
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                means.append(np.nan); errs.append(0.0)
                continue
            s = c["summary"]
            means.append(s["pct_mean"])
            errs.append((s["pct_ci95_hi"] - s["pct_ci95_lo"]) / 2.0)
        ax.errorbar(group_x + offsets[j], means, yerr=errs, fmt="s", ms=9,
                     color=COND_COLOR[cond], mec="white", mew=0.6, capsize=2.5,
                     elinewidth=1.2, ecolor=ps.INK_MUTED, label=COND_LABEL[cond], zorder=3)

    ax.set_xticks(group_x)
    ax.set_xticklabels(DENSITIES)
    ax.set_xlabel("density")
    ax.set_ylabel("iteration reduction vs. plain DR (%)")
    ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
    for x in group_x[:-1]:
        ax.axvline(x + 0.5, color=ps.GRID, lw=0.8, zorder=1)
    ax.legend(frameon=False, ncol=min(len(BAR_CONDITIONS), 5), loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8.5)
    ps.clean_axes(ax)
    fig.tight_layout()
    out_path = outp("fig2_blocks.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_blocks_faceted(data):
    """Small-multiples version of plot_blocks: one small SQUARE panel per
    density, all sharing the same y-axis, each with 4 blocks (home ->
    level20 -> level30 -> level40, darkest to lightest plasma)."""
    conds = ["home", "level20", "level30", "level40"]
    n = len(DENSITIES)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), sharey=True, squeeze=False)

    x = np.arange(len(conds))
    for i, dkey in enumerate(DENSITIES):
        ax = axes[i // ncols][i % ncols]
        for j, cond in enumerate(conds):
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            s = c["summary"]
            err = (s["pct_ci95_hi"] - s["pct_ci95_lo"]) / 2.0
            ax.errorbar(j, s["pct_mean"], yerr=err, fmt="s", ms=11, color=COND_COLOR[cond],
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)
        ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_title(f"density={dkey}", loc="left", fontsize=10)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("iteration reduction vs. plain DR (%)", fontsize=9)

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COND_COLOR[c],
                          markeredgecolor="white", markersize=10, label=COND_LABEL[c]) for c in conds]
    fig.legend(handles=handles, frameon=False, ncol=len(conds), loc="upper center",
               bbox_to_anchor=(0.5, 1.03), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = outp("fig4_blocks_faceted.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_blocks_faceted_separate(data):
    """Same panels as plot_blocks_faceted, but each density saved as its
    own separate square image (fig4_block_density<dkey>.png) sharing a
    common y-axis range across all of them, for manual composition."""
    conds = ["home", "level20", "level30", "level40"]
    short_label = {"home": "in-dist.", "level20": "±20%", "level30": "±30%", "level40": "±40%"}
    x = np.arange(len(conds))

    # shared y-limits across every panel, computed once
    y_lo, y_hi = np.inf, -np.inf
    for dkey in DENSITIES:
        for cond in conds:
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            s = c["summary"]
            y_lo = min(y_lo, s["pct_ci95_lo"]); y_hi = max(y_hi, s["pct_ci95_hi"])
    pad = 0.08 * (y_hi - y_lo)
    y_lo, y_hi = y_lo - pad, y_hi + pad

    for dkey in DENSITIES:
        fig, ax = plt.subplots(figsize=(3.3, 3.3))
        for j, cond in enumerate(conds):
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            s = c["summary"]
            err = (s["pct_ci95_hi"] - s["pct_ci95_lo"]) / 2.0
            ax.errorbar(j, s["pct_mean"], yerr=err, fmt="s", ms=12, color=COND_COLOR[cond],
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)
        ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label[c] for c in conds], fontsize=8)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("iteration reduction vs. plain DR (%)", fontsize=8.5)
        ps.clean_axes(ax)
        fig.tight_layout()
        out_path = outp(f"fig4_block_density{dkey.replace('.', '')}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


def plot_boxplots_faceted(data, conds, out_name):
    """Single combined small-multiples figure, one panel per density, but
    with REAL boxplots (median line, IQR box, whiskers to 1.5xIQR, outlier
    points) instead of a single median+IQR marker -- shows the full
    per-instance distribution shape, not just its summary. Larger tick
    labels than plot_blocks_faceted, per request."""
    short_label = {"home": "in-dist.", "level20": "±20%", "level30": "±30%",
                   "level40": "±40%", "level50": "±50%", "pm1": "±1 node",
                   "rewire": "rewire"}
    n = len(DENSITIES)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    # match the paper's own LaTeX (Computer Modern) typesetting for this
    # figure specifically -- restored at the end so every other figure in
    # this file keeps the clean sans-serif house style
    _rc_old = {k: plt.rcParams[k] for k in ("font.family", "mathtext.fontset")}
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), sharey=True, squeeze=False)

    # y-range from WHISKER extents (Tukey: Q1-1.5*IQR / Q3+1.5*IQR, clipped
    # to the data), not the raw min/max -- a few extreme outliers (e.g.
    # -540% on one node-removal instance) would otherwise squash every real
    # box into an unreadable sliver on a shared axis. Outlier points outside
    # this range are simply not drawn (still present in the underlying data/
    # table, just off-canvas here).
    y_lo, y_hi = np.inf, -np.inf
    for dkey in DENSITIES:
        for cond in conds:
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            pct = np.array([r["pct"] for r in c["per_instance"]], dtype=float)
            q1, q3 = np.percentile(pct, 25), np.percentile(pct, 75)
            iqr = q3 - q1
            whisk_lo = max(pct.min(), q1 - 1.5 * iqr)
            whisk_hi = min(pct.max(), q3 + 1.5 * iqr)
            y_lo = min(y_lo, whisk_lo); y_hi = max(y_hi, whisk_hi)
    pad = 0.12 * (y_hi - y_lo)
    y_lo, y_hi = y_lo - pad, y_hi + pad

    x = np.arange(len(conds))
    for i, dkey in enumerate(DENSITIES):
        ax = axes[i // ncols][i % ncols]
        box_data, positions, colors = [], [], []
        for j, cond in enumerate(conds):
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            pct = [r["pct"] for r in c["per_instance"]]
            box_data.append(pct); positions.append(j); colors.append(COND_COLOR[cond])
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                        medianprops={"color": "white", "linewidth": 1.8},
                        whiskerprops={"color": ps.INK_MUTED, "linewidth": 1.1},
                        capprops={"color": ps.INK_MUTED, "linewidth": 1.1},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": ps.INK_MUTED,
                                    "markeredgecolor": "none", "alpha": 0.6},
                        zorder=3)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_edgecolor("white"); patch.set_linewidth(0.8)
        ax.axhline(0, color=ps.INK_MUTED, lw=0.9, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label[c] for c in conds], fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        panel_letter = chr(ord("a") + i)
        ax.set_xlabel(f"({panel_letter}) $\\delta = {dkey}$", fontsize=12, labelpad=10)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("iteration reduction vs. plain DR (%)", fontsize=12)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COND_COLOR[c], edgecolor="white", label=COND_LABEL[c])
               for c in conds]
    fig.legend(handles=handles, frameon=False, ncol=len(conds), loc="upper center",
               bbox_to_anchor=(0.5, 1.04), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = outp(out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(_rc_old)
    print(f"wrote {out_path}")


def plot_blocks_faceted_median_iqr_separate(data, conds, name_suffix):
    """Separate-image version of plot_blocks_faceted_median_iqr: one
    square image per density (fig4_block_median_iqr_density<dkey><name_suffix>.png),
    median + IQR, shared y-axis range across all of them."""
    short_label = {"home": "in-dist.", "level20": "±20%", "level30": "±30%",
                   "level40": "±40%", "level50": "±50%", "pm1": "±1 node",
                   "rewire": "rewire"}
    x = np.arange(len(conds))

    y_lo, y_hi = np.inf, -np.inf
    for dkey in DENSITIES:
        for cond in conds:
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            pct = np.array([r["pct"] for r in c["per_instance"]], dtype=float)
            y_lo = min(y_lo, np.percentile(pct, 25)); y_hi = max(y_hi, np.percentile(pct, 75))
    pad = 0.08 * (y_hi - y_lo)
    y_lo, y_hi = y_lo - pad, y_hi + pad

    for dkey in DENSITIES:
        fig, ax = plt.subplots(figsize=(3.3, 3.3))
        for j, cond in enumerate(conds):
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            pct = np.array([r["pct"] for r in c["per_instance"]], dtype=float)
            median = np.median(pct)
            p25, p75 = np.percentile(pct, 25), np.percentile(pct, 75)
            err = np.array([[median - p25], [p75 - median]])
            ax.errorbar(j, median, yerr=err, fmt="s", ms=12, color=COND_COLOR[cond],
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)
        ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label[c] for c in conds], fontsize=8)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("iteration reduction vs. plain DR (%)", fontsize=8.5)
        ps.clean_axes(ax)
        fig.tight_layout()
        out_path = outp(f"fig4_block_median_iqr_density{dkey.replace('.', '')}{name_suffix}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


def plot_blocks_faceted_median_iqr(data, conds, out_name):
    """Same small-multiples layout as plot_blocks_faceted, but median +
    IQR (25th/75th percentile, from the per-instance pct values) instead
    of mean + 95% CI. `conds` and `out_name` let this serve several
    variants (with/without the +/-1 node condition) without overwriting
    each other."""
    n = len(DENSITIES)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.1 * nrows), sharey=True, squeeze=False)

    x = np.arange(len(conds))
    for i, dkey in enumerate(DENSITIES):
        ax = axes[i // ncols][i % ncols]
        for j, cond in enumerate(conds):
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            pct = np.array([r["pct"] for r in c["per_instance"]], dtype=float)
            median = np.median(pct)
            p25, p75 = np.percentile(pct, 25), np.percentile(pct, 75)
            err = np.array([[median - p25], [p75 - median]])
            ax.errorbar(j, median, yerr=err, fmt="s", ms=11, color=COND_COLOR[cond],
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)
        ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_title(f"density={dkey}", loc="left", fontsize=10)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("iteration reduction vs. plain DR (%)", fontsize=9)

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COND_COLOR[c],
                          markeredgecolor="white", markersize=10, label=COND_LABEL[c]) for c in conds]
    fig.legend(handles=handles, frameon=False, ncol=len(conds), loc="upper center",
               bbox_to_anchor=(0.5, 1.03), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = outp(out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_blocks_faceted_median_iqr_paperstyle(data, conds, out_name):
    """Same data/markers as plot_blocks_faceted_median_iqr, but typeset to
    match the paper's own LaTeX figure: Computer Modern serif font, panel
    labels "(a) $\\delta = 0.2$" centered below each panel (condition tick
    labels shown) instead of a bold top-left "density=0.2" title."""
    short_label = {"home": "in-dist.", "level20": "±20%", "level30": "±30%",
                   "level40": "±40%", "level50": "±50%", "pm1": "±1 node",
                   "rewire": "rewire"}
    n = len(DENSITIES)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    _rc_old = {k: plt.rcParams[k] for k in ("font.family", "mathtext.fontset")}
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows), sharey=True, squeeze=False)

    x = np.arange(len(conds))
    for i, dkey in enumerate(DENSITIES):
        ax = axes[i // ncols][i % ncols]
        for j, cond in enumerate(conds):
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            pct = np.array([r["pct"] for r in c["per_instance"]], dtype=float)
            median = np.median(pct)
            p25, p75 = np.percentile(pct, 25), np.percentile(pct, 75)
            err = np.array([[median - p25], [p75 - median]])
            ax.errorbar(j, median, yerr=err, fmt="s", ms=11, color=COND_COLOR[cond],
                        mec="white", mew=0.6, capsize=2.5, elinewidth=1.2, ecolor=ps.INK_MUTED, zorder=3)
        ax.axhline(0, color=ps.INK_MUTED, lw=0.8, linestyle="--", zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label[c] for c in conds], fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        panel_letter = chr(ord("a") + i)
        ax.set_xlabel(f"({panel_letter}) $\\delta = {dkey}$", fontsize=11, labelpad=10)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ps.clean_axes(ax)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    for row in range(nrows):
        axes[row][0].set_ylabel("iteration reduction vs. plain DR (%)", fontsize=11)

    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=COND_COLOR[c],
                          markeredgecolor="white", markersize=10, label=COND_LABEL[c]) for c in conds]
    fig.legend(handles=handles, frameon=False, ncol=len(conds), loc="upper center",
               bbox_to_anchor=(0.5, 1.04), fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = outp(out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(_rc_old)
    print(f"wrote {out_path}")


def write_latex_table(data, conds=None, stat="mean_std", out_name="table.tex"):
    """LaTeX version of the 'iterations to convergence' table: one
    multirow block per condition (DR row + learned row), one column per
    density. Matches the paper's own table style (booktabs, \\multirow).

    stat: "mean_std" (mean $\\pm$ std dev, default) or "median_iqr"
    (median [25th-75th percentile] -- consistent with the *_median_iqr
    figures, so this table can sit next to them)."""
    conds = conds or PLOT_CONDITIONS
    ncols = len(DENSITIES)
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("  \\centering")
    cap = ("Iterations to convergence: plain DR vs.\\ DR + learned perturbation (mean $\\pm$ std dev)."
           if stat == "mean_std" else
           "Iterations to convergence: plain DR vs.\\ DR + learned perturbation (median [IQR]).")
    lines.append(f"  \\caption{{{cap}}}")
    lines.append("  \\label{tab:iterations_std}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{ll " + "c" * ncols + "}")
    lines.append("    \\toprule")
    lines.append(f"    & & \\multicolumn{{{ncols}}}{{c}}{{density}} \\\\")
    lines.append(f"    \\cmidrule(lr){{3-{2 + ncols}}}")
    lines.append("    test & method & " + " & ".join(DENSITIES) + " \\\\")
    lines.append("    \\midrule")
    for idx, cond in enumerate(conds):
        label = COND_LABEL[cond].replace("%", "\\%")
        dr_vals, learned_vals = [], []
        for dkey in DENSITIES:
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                dr_vals.append("--"); learned_vals.append("--")
                continue
            dr = np.array([r["dr"] for r in c["per_instance"]], dtype=float)
            learned = np.array([r["learned"] for r in c["per_instance"]], dtype=float)
            if stat == "mean_std":
                dr_vals.append(f"{dr.mean():.0f} $\\pm$ {dr.std(ddof=1):.0f}")
                learned_vals.append(f"{learned.mean():.0f} $\\pm$ {learned.std(ddof=1):.0f}")
            else:
                dr_p25, dr_p75 = np.percentile(dr, 25), np.percentile(dr, 75)
                le_p25, le_p75 = np.percentile(learned, 25), np.percentile(learned, 75)
                dr_vals.append(f"{np.median(dr):.0f} [{dr_p25:.0f}--{dr_p75:.0f}]")
                learned_vals.append(f"{np.median(learned):.0f} [{le_p25:.0f}--{le_p75:.0f}]")
        lines.append(f"    \\multirow{{2}}{{*}}{{{label}}}")
        lines.append(f"      & DR      & " + " & ".join(dr_vals) + " \\\\")
        lines.append(f"      & learned & " + " & ".join(learned_vals) + " \\\\")
        lines.append("    \\bottomrule" if idx == len(conds) - 1 else "    \\midrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    out_path = outp(out_name)
    with open(out_path, "w") as f:
        f.write(tex + "\n")
    print(f"wrote {out_path}")


def write_table(data):
    lines = []

    # Table 0: DR / GNN hyperparameters actually used per density's champion
    # checkpoint. lambda (DR relaxation) is a single global constant (never
    # varied); gamma (DR scale) and rho (GNN decay) are chosen per density
    # (freshly calibrated per the project's standing rule: unroll=mean_DR/2,
    # rho=1-10/unroll, measured fresh per base market).
    lines.append("### DR / GNN hyperparameters per density\n")
    lines.append("| density | lambda (DR relax.) | gamma (DR scale) | rho (GNN decay) | unroll | base_seed |")
    lines.append("|---|---|---|---|---|---|")
    for dkey in DENSITIES:
        hp = data["densities"].get(dkey, {}).get("hyperparams", {})
        lines.append(f"| {dkey} | 1.0 | {hp.get('dr_gamma', '--')} | {hp.get('gnn_rho', '--')} | "
                      f"{hp.get('unroll', '--')} | {hp.get('base_seed', '--')} |")
    lines.append("")

    # Table A: structural fact (density -> edges -> communications/iteration).
    # Not a trained result -- each prosumer only talks to its direct network
    # neighbors every DR iteration (one message each way per edge), so this
    # follows purely from the graph, independent of iteration count.
    lines.append("### Communications per iteration (structural, not learned)\n")
    lines.append("| density | edges | network-wide msgs/iter (=2 x edges) | "
                  "per-agent msgs/iter: mean | per-agent msgs/iter: min | per-agent msgs/iter: max |")
    lines.append("|---|---|---|---|---|---|")
    for dkey in DENSITIES:
        deg = DEGREE[dkey]
        lines.append(f"| {dkey} | {EDGES[dkey]} | {2 * EDGES[dkey]} | "
                      f"{deg['mean']:.1f} | {deg['min']} | {deg['max']} |")
    lines.append("")
    lines.append("*Per-agent = each prosumer's own degree (number of direct neighbors); "
                  "it sends one message to, and receives one from, each neighbor every DR "
                  "iteration -- this is the actual communication burden on a single node in "
                  "the decentralized algorithm, as opposed to the network-wide total.*\n")

    # Table B: iteration counts (plain DR vs learned), mean/min/max, per
    # density x condition -- the actual result the % plots are derived from.
    lines.append("### Iterations to convergence: plain DR vs. DR + learned perturbation\n")
    lines.append("| density | condition | DR mean | DR std dev | learned mean | learned std dev |")
    lines.append("|---|---|---|---|---|---|")
    for dkey in DENSITIES:
        for cond in CONDITIONS:
            c = data["densities"].get(dkey, {}).get("conditions", {}).get(cond)
            if c is None:
                continue
            dr = np.array([r["dr"] for r in c["per_instance"]], dtype=float)
            learned = np.array([r["learned"] for r in c["per_instance"]], dtype=float)
            lines.append(f"| {dkey} | {COND_LABEL[cond]} | "
                          f"{dr.mean():.0f} | ±{dr.std(ddof=1):.0f} | "
                          f"{learned.mean():.0f} | ±{learned.std(ddof=1):.0f} |")

    table_md = "\n".join(lines)
    out_path = outp("table.md")
    with open(out_path, "w") as f:
        f.write(table_md + "\n")
    print(f"wrote {out_path}\n")
    print(table_md)


def main():
    global DATA_PATH, SUFFIX, DENSITIES, PLOT_CONDITIONS, BAR_CONDITIONS
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        SUFFIX = sys.argv[2]
    extra = sys.argv[3:]
    if "full" in extra:
        PLOT_CONDITIONS = CONDITIONS[:5]  # include +/-50% too, instead of dropping it (not remove1/2)
        BAR_CONDITIONS = CONDITIONS[:5]
    if "nodes" in extra:
        BAR_CONDITIONS = BAR_CONDITIONS + ["remove1", "remove2"]

    data = load_data()
    DENSITIES = sorted(data["densities"].keys(), key=float)

    plot_curves(data)
    plot_curves_median_iqr(data)
    plot_bars(data)
    plot_blocks(data)
    plot_blocks_faceted(data)
    plot_blocks_faceted_separate(data)
    plot_blocks_faceted_median_iqr(data, ["home", "level20", "level30", "level40"],
                                    "fig4_blocks_faceted_median_iqr.png")
    plot_blocks_faceted_median_iqr(data, ["home", "level20", "level30", "level40", "pm1"],
                                    "fig4_blocks_faceted_median_iqr_pm1.png")
    plot_blocks_faceted_median_iqr_separate(data, ["home", "level20", "level30", "level40", "pm1"], "_pm1")
    plot_boxplots_faceted(data, ["home", "level20", "level30", "level40", "pm1"], "fig4_boxplots_faceted_pm1.png")
    plot_blocks_faceted_median_iqr_paperstyle(data, ["home", "level20", "level30", "level40", "pm1"],
                                               "fig4_blocks_faceted_median_iqr_pm1_paperstyle.png")
    plot_scatter_difficulty(data)
    write_table(data)
    latex_conds = PLOT_CONDITIONS + ["pm1"]
    write_latex_table(data, conds=latex_conds)
    write_latex_table(data, conds=latex_conds, stat="median_iqr", out_name="table_median_iqr.tex")


if __name__ == "__main__":
    main()
