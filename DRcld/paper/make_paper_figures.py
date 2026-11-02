#!/usr/bin/env python3
"""
make_paper_figures.py
======================
Single reproducible entry point for every figure/table of Section 5
(Numerical Results) and the appendix, per prompt_figures_tables.md.

Reads only SAVED results (no training/experiments are run here). Where a
required input does not exist on disk, the corresponding output is skipped
and listed in the final MISSING report instead of being invented.

Usage:
    python make_paper_figures.py --results <experiments_dir> --out <out_dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as sstats

# ----------------------------------------------------------------------
# 1. Global conventions
# ----------------------------------------------------------------------
DENSITIES = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
BASE_SEED = {0.2: 3, 0.4: 3, 0.5: 1, 0.6: 3, 0.8: 0, 1.0: 2}
L_DIST = 3            # message-passing rounds for the distributed L2WS warm start (verified, see report)
A2DR_TOL = 1e-4        # local-constraint-violation threshold used alongside gap levels
CAP = 4000             # DR_MAX_ITER

# NOT verified by compiling the actual l4dc2026 class (no LaTeX installation on
# this machine -- `which pdflatex/latex/xelatex` all empty). This is the
# standard PMLR/JMLR-W&CP single-column text width (10pt, Times) and is used
# as a documented ASSUMPTION; flagged again in the final report.
TEXTWIDTH_IN = 6.00

METHODS = {
    "dr":              dict(name="DR",                        color="#000000", marker="o", filled=True,  kind="baseline"),
    "learned":         dict(name="Learned (ours)",             color="#0072B2", marker="o", filled=True,  kind="distributed"),
    "l2ws_reg_dist":   dict(name="L2WS-reg (dist.)",           color="#E69F00", marker="s", filled=False, kind="distributed"),
    "l2ws_asym_dist":  dict(name="L2WS-asym (dist.)",          color="#009E73", marker="^", filled=False, kind="distributed"),
    "l2ws_fpres_dist": dict(name="L2WS-fp-residual (dist.)",   color="#D55E00", marker="v", filled=False, kind="distributed"),
    "l2ws_reg_cent":   dict(name="L2WS-reg (cent.)$^\\dagger$",  color="#E69F00", marker="s", filled=True,  kind="centralized"),
    "l2ws_asym_cent":  dict(name="L2WS-asym (cent.)$^\\dagger$", color="#009E73", marker="^", filled=True,  kind="centralized"),
    "l2ws_fpres_cent": dict(name="L2WS-fp-residual (cent.)$^\\dagger$", color="#D55E00", marker="v", filled=True, kind="centralized"),
    "a2dr":            dict(name="A2DR$^\\dagger$",             color="#CC79A7", marker="D", filled=True,  kind="centralized"),
}
DIST_ORDER = ["learned", "l2ws_reg_dist", "l2ws_asym_dist"]
DIST_ORDER_FULL = ["learned", "l2ws_reg_dist", "l2ws_asym_dist", "l2ws_fpres_dist"]
CENT_ORDER = ["l2ws_reg_cent", "l2ws_asym_cent", "a2dr"]
CENT_ORDER_FULL = ["l2ws_reg_cent", "l2ws_asym_cent", "l2ws_fpres_cent", "a2dr"]

MISSING = []   # (tag, message) pairs collected for the final report
NUMBERS = {}   # paper_numbers.json content


def flag_missing(tag, msg):
    MISSING.append((tag, msg))
    print(f"  [MISSING] {tag}: {msg}")


# ----------------------------------------------------------------------
# 2. Plot style (no usetex -- no LaTeX install on this machine; mathtext
#    with the 'cm' fontset is used as the closest available approximation
#    to the paper's Computer-Modern/Times math font. Flagged in the report.)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "savefig.dpi": 200,
})


def save_fig(fig, out_figs, name):
    fig.savefig(os.path.join(out_figs, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(os.path.join(out_figs, f"{name}.png"), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"  wrote {name}.pdf (+ .png)")


# ----------------------------------------------------------------------
# 3. Stats helpers (never round before computing; round only when printing)
# ----------------------------------------------------------------------
def delta_pct(dr_iters, m_iters):
    dr = np.asarray(dr_iters, dtype=float)
    m = np.asarray(m_iters, dtype=float)
    return (dr - m) / dr * 100.0


def median_iqr(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))


def win_rate(d):
    d = np.asarray(d, dtype=float)
    return float(np.mean(d > 0.0))


def worst_case(d):
    return float(np.min(np.asarray(d, dtype=float)))


def fmt_cell(d, with_win=True):
    med, q1, q3 = median_iqr(d)
    s = f"{med:.1f}\\% [{q1:.1f}, {q3:.1f}]"
    if with_win:
        s += f" ({win_rate(d) * 100:.0f}\\%)"
    return s


# ----------------------------------------------------------------------
# 4. Data loaders -- read saved results only
# ----------------------------------------------------------------------
def load_official(results_dir, density):
    seed = BASE_SEED[density]
    pat = os.path.join(results_dir, "runs", "official", f"official_N10_d{density}_seed{seed}_replicate*")
    dirs = glob.glob(pat)
    if not dirs:
        return None
    run_dir = sorted(dirs)[0]
    files = sorted(glob.glob(os.path.join(run_dir, "instance_plots", "*.json")))
    insts = [json.load(open(f)) for f in files]
    log_lines = open(os.path.join(run_dir, "train.log")).readlines()
    header = log_lines[0]
    slow_line = next((l for l in log_lines if "slow channel" in l), "")
    return {"dir": run_dir, "instances": insts, "header": header, "slow_line": slow_line}


def load_l2ws_cent(results_dir, density, loss):
    seed = BASE_SEED[density]
    dirname_loss = loss.replace("_", "")  # on-disk dirs use "fpresidual", not "fp_residual"
    run_dir = os.path.join(results_dir, "l2ws_benchmark", "runs", "l2ws", "centralized",
                            f"l2ws_N10_d{density}_seed{seed}_{dirname_loss}")
    tfile = os.path.join(run_dir, "timing_results.json")
    if not os.path.isdir(run_dir) or not os.path.exists(tfile):
        return None
    timing = json.load(open(tfile))
    files = sorted(glob.glob(os.path.join(run_dir, "instance_plots", "*.json")))
    insts = [json.load(open(f)) for f in files]
    return {"dir": run_dir, "timing": timing, "instances": insts}


_LOG_RE_DIST = re.compile(r"DR\s+(\d+)\s+it\s+\|\s+L2WS\s+(\d+)\s+it")


def load_l2ws_dist(results_dir, density, loss):
    seed = BASE_SEED[density]
    dirname_loss = loss.replace("_", "")  # on-disk dirs use "fpresidual", not "fp_residual"
    run_dir = os.path.join(results_dir, "l2ws_benchmark", "runs", "l2ws", "distributed",
                            f"l2ws_N10_d{density}_seed{seed}_distributed_{dirname_loss}")
    log_path = os.path.join(run_dir, "train.log")
    if not os.path.exists(log_path):
        return None
    text = open(log_path).read()
    header = text.splitlines()[0]
    n_rounds_m = re.search(r"n_rounds=(\d+)", header)
    L = int(n_rounds_m.group(1)) if n_rounds_m else L_DIST
    matches = _LOG_RE_DIST.findall(text)
    if not matches:
        return None
    dr_iters = [int(a) for a, b in matches]
    l2ws_post = [int(b) for a, b in matches]
    return {"dir": run_dir, "dr_iters": dr_iters, "l2ws_post_iters": l2ws_post, "L": L}


def load_a2dr(results_dir, density):
    run_dir = os.path.join(results_dir, "a2dr_benchmark", "runs", f"a2dr_vs_official_d{density}")
    rfile = os.path.join(run_dir, "results.json")
    if not os.path.exists(rfile):
        return None
    return {"dir": run_dir, "data": json.load(open(rfile))}


_LOG_RE_OOD30 = re.compile(r"\[ood\]\s*(\d+)/30\s+DR\s+(\d+)\s+it\s+\|\s+learned\s+(\d+)\s+it")


def load_ood_level(results_dir, density, level):
    run_dir = os.path.join(results_dir, "runs", "official_ood", f"zeroshot_d{density}_{level}")
    log_path = os.path.join(run_dir, "live.log")
    if not os.path.exists(log_path):
        return None
    text = open(log_path).read()
    matches = _LOG_RE_OOD30.findall(text)
    if not matches:
        return None
    dr_iters = [int(b) for a, b, c in matches]
    learned_iters = [int(c) for a, b, c in matches]
    return {"dir": run_dir, "dr_iters": dr_iters, "learned_iters": learned_iters}


_LOG_RE_PM1 = re.compile(r"\[(add|remove)\s*\]\s+N=(\d+)\s+DR\s+(\d+)\s+it\s+\|\s+learned\s+(\d+)\s+it")


def load_ood_pm1(results_dir, density):
    run_dir = os.path.join(results_dir, "runs", "official_ood", f"pm1_d{density}")
    log_path = os.path.join(run_dir, "live.log")
    if not os.path.exists(log_path):
        return None
    text = open(log_path).read()
    matches = _LOG_RE_PM1.findall(text)
    if not matches:
        return None
    dr_iters = [int(c) for _, _, c, _ in matches]
    learned_iters = [int(d) for _, _, _, d in matches]
    kinds = [k for k, _, _, _ in matches]   # "add" (+1 prosumer) / "remove" (-1 prosumer)
    return {"dir": run_dir, "dr_iters": dr_iters, "learned_iters": learned_iters, "kinds": kinds}


# ----------------------------------------------------------------------
# 5. First-hitting-time helper (for F-M / T-gap): first k with
#    relative_gap[k] <= eps AND violation[k] <= A2DR_TOL
# ----------------------------------------------------------------------
def first_hitting_time(rel_gap, violation, eps):
    rel_gap = np.asarray(rel_gap)
    violation = np.asarray(violation) if violation is not None else np.zeros_like(rel_gap)
    ok = (rel_gap <= eps) & (violation <= A2DR_TOL)
    idx = np.flatnonzero(ok)
    return int(idx[0]) if idx.size else None


# ----------------------------------------------------------------------
# 6. Figure / table builders
# ----------------------------------------------------------------------
def build_T_A(results_dir, out_tables, official, l2ws_cent, l2ws_dist, a2dr):
    print("\n[T-A] comparison table")
    cols = DENSITIES
    rows_dist = {}
    rows_cent = {}
    dr_median_row = []
    for density in cols:
        off = official[density]
        dr_median_row.append(np.median([i["k_DR"] for i in off["instances"]]) if off else None)

    def row_for(method_key, density):
        off = official.get(density)
        if method_key == "learned":
            if off is None:
                return None
            d = delta_pct([i["k_DR"] for i in off["instances"]], [i["k_learned"] for i in off["instances"]])
            return d, "0"
        if method_key in ("l2ws_reg_dist", "l2ws_asym_dist"):
            loss = "regression" if method_key == "l2ws_reg_dist" else "asymmetric"
            ld = l2ws_dist.get((density, loss))
            if ld is None:
                return None
            post = np.array(ld["l2ws_post_iters"]) + ld["L"]
            d = delta_pct(ld["dr_iters"], post)
            return d, f"L={ld['L']}"
        if method_key in ("l2ws_reg_cent", "l2ws_asym_cent"):
            loss = "regression" if method_key == "l2ws_reg_cent" else "asymmetric"
            lc = l2ws_cent.get((density, loss))
            if lc is None:
                return None
            d = delta_pct(lc["timing"]["dr_iters"], lc["timing"]["l2ws_iters"])
            return d, "global"
        if method_key == "a2dr":
            ad = a2dr.get(density)
            if ad is None:
                return None
            d = delta_pct(ad["data"]["dr_iters"], ad["data"]["a2dr_iters"])
            return d, "global"
        return None

    for density in cols:
        for mk in DIST_ORDER:
            rows_dist.setdefault(mk, {})[density] = row_for(mk, density)
        for mk in CENT_ORDER:
            rows_cent.setdefault(mk, {})[density] = row_for(mk, density)

    lines = []
    lines.append("% DRAFT CAPTION: In-distribution iteration reduction of each method relative to the\n"
                  "% DR baseline, median [IQR] over 30 test instances (win-rate in parentheses).\n"
                  "% 'extra rounds' = additional communication beyond the reported iteration count.\n")
    lines.append("\\begin{tabular}{l" + "c" * len(cols) + "c}")
    lines.append("\\toprule")
    header = " & ".join([f"$\\delta={d}$" for d in cols])
    lines.append("method & " + header + " & extra rounds \\\\")
    lines.append("\\midrule")
    dr_row = " & ".join(f"{int(v)}" if v is not None else "--" for v in dr_median_row)
    lines.append("DR iterations & " + dr_row + " & -- \\\\")
    lines.append("\\midrule")

    def best_per_col(mk_list, density):
        meds = {}
        for mk in mk_list:
            r = rows_dist.get(mk, {}).get(density) or rows_cent.get(mk, {}).get(density)
            if r is not None:
                meds[mk] = np.median(r[0])
        if not meds:
            return None
        return max(meds, key=meds.get)

    for mk in DIST_ORDER:
        cells = []
        for density in cols:
            r = rows_dist[mk][density]
            if r is None:
                cells.append("--")
                continue
            d, extra = r
            med = np.median(d)
            best = best_per_col(DIST_ORDER, density)
            cell = fmt_cell(d)
            if best == mk:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        extra_label = next((rows_dist[mk][d][1] for d in cols if rows_dist[mk][d] is not None), "--")
        lines.append(f"{METHODS[mk]['name']} & " + " & ".join(cells) + f" & {extra_label} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(cols) + 2) + "}{l}{Centralized references} \\\\")
    for mk in CENT_ORDER:
        cells = []
        for density in cols:
            r = rows_cent[mk][density]
            if r is None:
                cells.append("--")
                continue
            d, extra = r
            cells.append(fmt_cell(d))
        extra_label = next((rows_cent[mk][d][1] for d in cols if rows_cent[mk][d] is not None), "--")
        lines.append(f"{METHODS[mk]['name']} & " + " & ".join(cells) + f" & {extra_label} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    with open(os.path.join(out_tables, "tabA_comparison.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tabA_comparison.tex")

    NUMBERS["T-A"] = {
        "dr_median_iterations": {str(d): v for d, v in zip(cols, dr_median_row)},
        "distributed": {mk: {str(d): (list(rows_dist[mk][d][0]) if rows_dist[mk][d] else None) for d in cols}
                         for mk in DIST_ORDER},
        "centralized": {mk: {str(d): (list(rows_cent[mk][d][0]) if rows_cent[mk][d] else None) for d in cols}
                         for mk in CENT_ORDER},
    }


def build_F_A(out_figs, official):
    print("\n[F-A] convergence curves (DR, Learned only -- L2WS-dist trajectory unavailable, see report)")
    panel_densities = [0.2, 0.8]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN * 1.08, TEXTWIDTH_IN * 0.48))
    K = {0.2: 426, 0.8: 1159}
    for ax, density in zip(axes, panel_densities):
        off = official.get(density)
        if off is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            continue
        for key, label, color in [("dr", "DR", METHODS["dr"]["color"]), ("learned", "Learned (ours)", METHODS["learned"]["color"])]:
            traces = [np.asarray(i[key]["relative_gap"], dtype=float) for i in off["instances"]]
            maxlen = max(len(t) for t in traces)
            padded = np.array([np.pad(t, (0, maxlen - len(t)), mode="edge") for t in traces])
            med = np.median(padded, axis=0)
            q1 = np.percentile(padded, 25, axis=0)
            q3 = np.percentile(padded, 75, axis=0)
            k = np.arange(maxlen)
            ax.plot(k, med, color=color, label=label, lw=1.3)
            ax.fill_between(k, q1, q3, color=color, alpha=0.2, lw=0)
        ax.set_yscale("log")
        ax.axvline(K[density], color="gray", ls="--", lw=0.8)
        ax.text(K[density], ax.get_ylim()[1], " $k=K$", va="top", ha="left", fontsize=7, color="gray")
        for lvl in (1e-2, 1e-3, 1e-4):
            ax.axhline(lvl, color="gray", ls=":", lw=0.6)
        ax.set_xlabel("iteration $k$ (= communication rounds)")
    axes[0].set_ylabel("relative optimality gap")
    axes[0].legend(loc="upper right", frameon=False)
    fig.tight_layout()
    axes[0].annotate("(a) $\\delta = 0.2$", xy=(0.5, 0), xycoords="axes fraction",
                      xytext=(0, -46), textcoords="offset points", ha="center", fontsize=9)
    axes[1].annotate("(b) $\\delta = 0.8$", xy=(0.5, 0), xycoords="axes fraction",
                      xytext=(0, -46), textcoords="offset points", ha="center", fontsize=9)
    save_fig(fig, out_figs, "figA_convergence")
    flag_missing("F-A", "L2WS-reg (dist.) curve omitted: no per-instance residual/gap trajectory is saved for "
                         "the distributed L2WS runs (only final iteration counts in train.log). Only DR and "
                         "Learned (ours) are plotted, both using the saved 'relative_gap' field as-is.")


def build_F_M(out_figs, official, l2ws_cent, appendix=False):
    tag = "F-M (appendix 2x3)" if appendix else "F-M (main, 2 panels)"
    print(f"\n[{tag}]")
    densities = DENSITIES if appendix else [0.2, 0.8]
    eps_grid = np.logspace(0, -5, 30)
    ncols = 3 if appendix else 2
    nrows = 2 if appendix else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXTWIDTH_IN * (1.0 if appendix else 1.08),
                                                     TEXTWIDTH_IN * (0.9 if appendix else 0.48)),
                              gridspec_kw={"hspace": 0.7} if appendix else {})
    axes = np.atleast_1d(axes).ravel()
    for ax, density in zip(axes, densities):
        off = official.get(density)
        if off is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            continue
        for key, label, color, ls in [("dr", "DR", METHODS["dr"]["color"], "-"),
                                       ("learned", "Learned (ours)", METHODS["learned"]["color"], "-")]:
            hits = []
            for i in off["instances"]:
                rg, vi = i[key]["relative_gap"], i[key]["violation"]
                hits.append([first_hitting_time(rg, vi, e) for e in eps_grid])
            hits = np.array(hits, dtype=object)
            meds, q1s, q3s, xs = [], [], [], []
            for j, e in enumerate(eps_grid):
                col = hits[:, j]
                n_reached = sum(v is not None for v in col)
                if n_reached < len(col) / 2:
                    continue
                vals = [v for v in col if v is not None]
                meds.append(np.median(vals)); q1s.append(np.percentile(vals, 25)); q3s.append(np.percentile(vals, 75))
                xs.append(e)
            if xs:
                ax.plot(xs, meds, color=color, ls=ls, label=label, lw=1.3)
                ax.fill_between(xs, q1s, q3s, color=color, alpha=0.2, lw=0)
        lc = l2ws_cent.get((density, "regression"))
        if lc is not None:
            hits = []
            for i in lc["instances"]:
                rg, vi = i["learned"]["relative_gap"], i["learned"]["violation"]
                hits.append([first_hitting_time(rg, vi, e) for e in eps_grid])
            hits = np.array(hits, dtype=object)
            meds, xs = [], []
            for j, e in enumerate(eps_grid):
                col = hits[:, j]
                n_reached = sum(v is not None for v in col)
                if n_reached < len(col) / 2:
                    continue
                vals = [v for v in col if v is not None]
                meds.append(np.median(vals)); xs.append(e)
            if xs:
                ax.plot(xs, meds, color=METHODS["l2ws_reg_cent"]["color"], ls="--", label="L2WS-reg (cent.)", lw=1.1)
        ax.set_xscale("log")
        ax.set_xlim(1, 1e-5)
        ax.set_xlabel("optimality gap $\\varepsilon$")
    fig.supylabel("iterations to reach gap $\\leq \\varepsilon$", fontsize=10)
    fig.tight_layout()
    for ax, density in zip(axes, densities):
        ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                     xytext=(0, -42), textcoords="offset points", ha="center", fontsize=9)
    axes[0].legend(loc="upper right", frameon=False, fontsize=7)
    save_fig(fig, out_figs, "figApp_gap_levels" if appendix else "figM_gap_levels")
    flag_missing(tag, "A2DR and L2WS-reg (dist.) curves omitted: no residual/gap trajectory is saved for "
                       "either (A2DR's results.json has only final iteration counts; L2WS-distributed's "
                       "train.log has only final iteration counts). L2WS-reg (cent.) is shown as an optional "
                       "dashed reference where available, per the spec's own allowance.")


def build_F_B(out_figs, official):
    print("\n[F-B] difficulty vs. gain")
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.48, TEXTWIDTH_IN * 0.58))
    cmap = plt.get_cmap("viridis")
    all_x, all_y, per_density = [], [], {}
    for idx, density in enumerate(DENSITIES):
        off = official.get(density)
        if off is None:
            continue
        x = np.array([i["k_DR"] for i in off["instances"]], dtype=float)
        y = delta_pct(x, [i["k_learned"] for i in off["instances"]])
        ax.scatter(x, y, color=cmap(idx / (len(DENSITIES) - 1)), s=14, label=f"$\\delta = {density}$")
        all_x.extend(x); all_y.extend(y)
        rho, p = sstats.spearmanr(x, y)
        per_density[density] = {"rho": float(rho), "p": float(p)}
    rho_all, p_all = sstats.spearmanr(all_x, all_y)
    ax.set_xscale("log")
    ax.set_xlabel("DR iteration count (log scale)")
    ax.set_ylabel("iteration reduction vs. DR (%)")
    ax.legend(loc="best", frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    save_fig(fig, out_figs, "figB_difficulty")
    NUMBERS["F-B"] = {"spearman_overall": {"rho": float(rho_all), "p": float(p_all)}, "spearman_per_density": per_density}
    print(f"  Spearman overall: rho={rho_all:.3f} p={p_all:.2e}")
    for d, v in per_density.items():
        print(f"  Spearman delta={d}: rho={v['rho']:.3f} p={v['p']:.2e}")


def build_F_C_T_C(out_figs, out_tables, official, ood):
    print("\n[F-C / T-C] robustness to distribution shift")
    cats = [("ID", "id"), ("$\\pm20\\%$", "level20"), ("$\\pm30\\%$", "level30"), ("$\\pm40\\%$", "level40"), ("$\\pm1$ node", "pm1")]
    colors = {"id": "#000000", "level20": "#56B4E9", "level30": "#E69F00", "level40": "#CC79A7", "pm1": "#009E73"}
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85),
                              gridspec_kw={"hspace": 0.75})
    axes = axes.ravel()
    table_rows = defaultdict(dict)
    for ax, density in zip(axes, DENSITIES):
        off = official.get(density)
        box_data, box_colors, box_labels = [], [], []
        for label, key in cats:
            if key == "id":
                d = delta_pct([i["k_DR"] for i in off["instances"]], [i["k_learned"] for i in off["instances"]]) if off else None
            else:
                rec = ood.get((density, key))
                d = delta_pct(rec["dr_iters"], rec["learned_iters"]) if rec else None
            if d is None:
                continue
            box_data.append(d); box_colors.append(colors[key]); box_labels.append(label)
            med, q1, q3 = median_iqr(d)
            table_rows[label][density] = (med, q1, q3, win_rate(d), worst_case(d))
        bp = ax.boxplot(box_data, labels=box_labels, showfliers=True, patch_artist=True,
                         flierprops=dict(markersize=2, marker="o"))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
        for med_line in bp["medians"]:
            med_line.set_color("black")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.supylabel("iteration reduction vs. DR (%)", fontsize=10)
    ylims = [ax.get_ylim() for ax in axes]
    ymin, ymax = min(l[0] for l in ylims), max(l[1] for l in ylims)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    for ax, density in zip(axes, DENSITIES):
        ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                     xytext=(0, -52), textcoords="offset points", ha="center", fontsize=9)
    save_fig(fig, out_figs, "figC_ood")

    lines = ["% DRAFT CAPTION: OOD robustness of Learned (ours): median [IQR] iteration reduction (\\%), "
             "win-rate, and worst case, per density and test set.\n",
             "\\begin{tabular}{l" + "c" * len(DENSITIES) + "}", "\\toprule",
             "test set & " + " & ".join(f"$\\delta={d}$" for d in DENSITIES) + " \\\\", "\\midrule"]
    for label, _ in cats:
        cells = []
        for density in DENSITIES:
            if density not in table_rows[label]:
                cells.append("--"); continue
            med, q1, q3, wr, worst = table_rows[label][density]
            cells.append(f"{med:.1f} [{q1:.1f},{q3:.1f}] / {wr*100:.0f}\\% / {worst:.1f}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_tables, "tabC_ood.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tabC_ood.tex")
    NUMBERS["T-C"] = {label: {str(d): table_rows[label].get(d) for d in DENSITIES} for label, _ in cats}


def build_F_full_methods(out_figs, official, l2ws_cent, l2ws_dist, a2dr, with_fpres, name):
    print(f"\n[{name}]")
    dist_keys = DIST_ORDER_FULL if with_fpres else DIST_ORDER
    cent_keys = CENT_ORDER_FULL if with_fpres else CENT_ORDER
    width = TEXTWIDTH_IN if not with_fpres else TEXTWIDTH_IN
    fig, axes = plt.subplots(2, 3, figsize=(width, width * 1.05),
                              gridspec_kw={"hspace": 1.5})
    axes = axes.ravel()
    for ax, density in zip(axes, DENSITIES):
        off = official.get(density)
        data, colors, filled, positions, labels = [], [], [], [], []
        pos = 0
        for mk in dist_keys:
            if mk == "learned":
                d = delta_pct([i["k_DR"] for i in off["instances"]], [i["k_learned"] for i in off["instances"]]) if off else None
            elif mk in ("l2ws_reg_dist", "l2ws_asym_dist", "l2ws_fpres_dist"):
                loss = {"l2ws_reg_dist": "regression", "l2ws_asym_dist": "asymmetric", "l2ws_fpres_dist": "fp_residual"}[mk]
                ld = l2ws_dist.get((density, loss))
                d = delta_pct(ld["dr_iters"], np.array(ld["l2ws_post_iters"]) + ld["L"]) if ld else None
            else:
                d = None
            if d is None:
                continue
            data.append(d); colors.append(METHODS[mk]["color"]); filled.append(False); labels.append(METHODS[mk]["name"])
            positions.append(pos); pos += 1
        pos += 0.6
        sep = pos - 0.3
        for mk in cent_keys:
            if mk in ("l2ws_reg_cent", "l2ws_asym_cent", "l2ws_fpres_cent"):
                loss = {"l2ws_reg_cent": "regression", "l2ws_asym_cent": "asymmetric", "l2ws_fpres_cent": "fp_residual"}[mk]
                lc = l2ws_cent.get((density, loss))
                d = delta_pct(lc["timing"]["dr_iters"], lc["timing"]["l2ws_iters"]) if lc else None
            elif mk == "a2dr":
                ad = a2dr.get(density)
                d = delta_pct(ad["data"]["dr_iters"], ad["data"]["a2dr_iters"]) if ad else None
            else:
                d = None
            if d is None:
                continue
            data.append(d); colors.append(METHODS[mk]["color"]); filled.append(True); labels.append(METHODS[mk]["name"])
            positions.append(pos); pos += 1
        bp = ax.boxplot(data, positions=positions, widths=0.6, showfliers=True, patch_artist=True,
                         flierprops=dict(markersize=2, marker="o"))
        for patch, c, f in zip(bp["boxes"], colors, filled):
            patch.set_edgecolor(c)
            patch.set_facecolor(c if f else "white")
            patch.set_alpha(0.55 if f else 1.0)
        for med_line in bp["medians"]:
            med_line.set_color("black")
        ax.axvline(sep, color="gray", lw=0.6, ls="-")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
    fig.supylabel("iteration reduction vs. DR (%)", fontsize=10)
    ylims = [ax.get_ylim() for ax in axes]
    ymin, ymax = min(l[0] for l in ylims), max(l[1] for l in ylims)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    for ax, density in zip(axes, DENSITIES):
        ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                     xytext=(0, -98), textcoords="offset points", ha="center", fontsize=9)
    save_fig(fig, out_figs, name)


def build_T_full(out_tables, official, l2ws_cent, l2ws_dist, a2dr, ood):
    print("\n[T-full] full results table")
    method_series = {}
    for density in DENSITIES:
        off = official.get(density)
        if off:
            method_series[("learned", density, "ID")] = delta_pct([i["k_DR"] for i in off["instances"]], [i["k_learned"] for i in off["instances"]])
        for key in ("level20", "level30", "level40", "pm1"):
            rec = ood.get((density, key))
            if rec:
                method_series[("learned", density, key)] = delta_pct(rec["dr_iters"], rec["learned_iters"])
        for mk, loss in [("l2ws_reg_dist", "regression"), ("l2ws_asym_dist", "asymmetric"), ("l2ws_fpres_dist", "fp_residual")]:
            ld = l2ws_dist.get((density, loss))
            if ld:
                method_series[(mk, density, "ID")] = delta_pct(ld["dr_iters"], np.array(ld["l2ws_post_iters"]) + ld["L"])
        for mk, loss in [("l2ws_reg_cent", "regression"), ("l2ws_asym_cent", "asymmetric"), ("l2ws_fpres_cent", "fp_residual")]:
            lc = l2ws_cent.get((density, loss))
            if lc:
                method_series[(mk, density, "ID")] = delta_pct(lc["timing"]["dr_iters"], lc["timing"]["l2ws_iters"])
        ad = a2dr.get(density)
        if ad:
            method_series[("a2dr", density, "ID")] = delta_pct(ad["data"]["dr_iters"], ad["data"]["a2dr_iters"])

    lines = ["% DRAFT CAPTION: Full results -- mean, median, IQR, win-rate, worst case, capped instances, "
             "per density x method x test set.\n",
             "\\begin{tabular}{llcccccc}", "\\toprule",
             "test set & method & mean & median & Q1 & Q3 & win-rate & worst \\\\", "\\midrule"]
    n_entries = 0
    for (mk, density, testset), d in sorted(method_series.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        med, q1, q3 = median_iqr(d)
        lines.append(f"{testset} $\\delta={density}$ & {METHODS[mk]['name']} & {np.mean(d):.1f} & {med:.1f} & "
                     f"{q1:.1f} & {q3:.1f} & {win_rate(d)*100:.0f}\\% & {worst_case(d):.1f} \\\\")
        n_entries += 1
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_tables, "tabApp_full.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote tabApp_full.tex ({n_entries} rows)")


def build_T_gap(out_tables, official, l2ws_cent):
    print("\n[T-gap] iterations to optimality-gap levels (DR, Learned, L2WS-reg cent. only -- see report)")
    levels = [1e-2, 1e-3, 1e-4]
    lines = ["% DRAFT CAPTION: Median [IQR] iterations to reach optimality-gap level eps (with local-constraint\n"
             "% violation <= 1e-4). A2DR and L2WS-reg (dist.) are omitted: no gap trajectory is saved for\n"
             "% either (see report).\n",
             "\\begin{tabular}{lc" + "c" * len(levels) + "c}", "\\toprule",
             "density & method & " + " & ".join(f"$\\varepsilon={e:g}$" for e in levels) + " & never reached \\\\", "\\midrule"]
    for density in DENSITIES:
        off = official.get(density)
        lc = l2ws_cent.get((density, "regression"))
        for label, src_insts, field in [("DR", off["instances"] if off else None, "dr"),
                                         ("Learned", off["instances"] if off else None, "learned"),
                                         ("L2WS-reg (cent.)", lc["instances"] if lc else None, "learned")]:
            if src_insts is None:
                lines.append(f"$\\delta={density}$ & {label} & -- & -- & -- & -- \\\\")
                continue
            cells = []
            never = 0
            for e in levels:
                hits = [first_hitting_time(i[field]["relative_gap"], i[field]["violation"], e) for i in src_insts]
                reached = [h for h in hits if h is not None]
                never = sum(h is None for h in hits)
                if reached:
                    med, q1, q3 = median_iqr(reached)
                    cells.append(f"{med:.0f} [{q1:.0f},{q3:.0f}]")
                else:
                    cells.append("--")
            lines.append(f"$\\delta={density}$ & {label} & " + " & ".join(cells) + f" & {never} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_tables, "tab_gap_levels.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tab_gap_levels.tex")
    flag_missing("T-gap", "A2DR and L2WS-reg (dist.) rows omitted: no gap trajectory saved for either.")


def build_T_hyper(out_tables, official, l2ws_dist, a2dr):
    print("\n[T-hyper] hyperparameters actually used in the final runs")
    rows = []
    rows.append(("$d_h$ (GNN hidden dim)", "32 (all densities)"))
    rows.append(("MLP layers per head", "1 hidden layer (Linear--Tanh--Linear)"))
    rows.append(("$\\rho_{\\mathrm f}$ rule", "$1-10/K$ (verified exact for all 6 densities)"))
    rows.append(("$\\rho_{\\mathrm s}$ (slow channel)", "0.99 (fixed, all densities)"))
    for density in DENSITIES:
        off = official.get(density)
        if off is None:
            continue
        h = off["header"]
        sl = off["slow_line"]
        def grab(key, text):
            m = re.search(rf"{key}=([^\s)]+)", text)
            return m.group(1) if m else "?"
        rows.append((f"$\\delta={density}$: $K$, $\\gamma^\\star$, scale$_{{\\mathrm s}}$",
                     f"K={grab('unroll', h)}, $\\gamma^\\star$={grab('gamma.', h)}, scale$_s$={grab('scale_slow', sl)}"))
    rows.append(("loss weights ($\\alpha_{\\text{behind}},\\alpha_{\\text{ahead}},\\beta$)", "2.0, 1.0, 2.0 (gamma_mono)"))
    rows.append(("optimizer", "Adam, lr$=3\\times10^{-3}$, cosine decay to $1\\%$"))
    rows.append(("gradient clipping", "$\\|\\nabla\\|\\le 1.0$"))
    rows.append(("epochs", "25"))
    rows.append(("L2WS (distributed): $L$, arch., lr, epochs", "$L=3$, shared weights + residual update, lr$=3\\times10^{-3}$, 50 epochs"))
    for density in DENSITIES:
        ad = a2dr.get(density)
        if ad:
            rows.append((f"A2DR $\\delta={density}$: memory, $\\eta$",
                         f"$m={ad['data']['a2dr_mmax']}$, $\\eta={ad['data']['a2dr_eta']:g}$"))
    rows.append(("DR: $\\lambda$, bisection steps, tol., cap", "$\\lambda=1.0$, 30 steps, $10^{-5}$, 4000"))
    lines = ["% DRAFT CAPTION: Hyperparameters of the final runs, read from logs/config (not defaults).\n",
             "\\begin{tabular}{ll}", "\\toprule", "parameter & value \\\\", "\\midrule"]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_tables, "tabApp_hyper.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  wrote tabApp_hyper.tex")


# ----------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/home/lisa/energy_market_fixed_point/DRcld/experiments")
    ap.add_argument("--out", default="/home/lisa/energy_market_fixed_point/DRcld/paper")
    args = ap.parse_args()

    out_figs = os.path.join(args.out, "paper_figs")
    out_tables = os.path.join(args.out, "paper_tables")
    os.makedirs(out_figs, exist_ok=True)
    os.makedirs(out_tables, exist_ok=True)

    print("=" * 70)
    print("Loading saved results (no experiments are run)")
    print("=" * 70)
    official = {d: load_official(args.results, d) for d in DENSITIES}
    for d in DENSITIES:
        if official[d] is None:
            flag_missing("official", f"delta={d}: no official run found")
    l2ws_cent = {}
    l2ws_dist = {}
    for d in DENSITIES:
        for loss in ("regression", "asymmetric", "fp_residual"):
            lc = load_l2ws_cent(args.results, d, loss)
            if lc is not None:
                l2ws_cent[(d, loss)] = lc
            ld = load_l2ws_dist(args.results, d, loss)
            if ld is not None:
                l2ws_dist[(d, loss)] = ld
    a2dr = {d: load_a2dr(args.results, d) for d in DENSITIES}
    a2dr = {d: v for d, v in a2dr.items() if v is not None}
    ood = {}
    for d in DENSITIES:
        for level in ("level20", "level30", "level40"):
            rec = load_ood_level(args.results, d, level)
            if rec is not None:
                ood[(d, level)] = rec
        rec = load_ood_pm1(args.results, d)
        if rec is not None:
            ood[(d, "pm1")] = rec

    print(f"official: {sum(v is not None for v in official.values())}/6 densities")
    print(f"l2ws centralized: {len(l2ws_cent)}/18 (density x loss)")
    print(f"l2ws distributed: {len(l2ws_dist)}/18 (density x loss)")
    print(f"a2dr: {len(a2dr)}/6 densities")
    print(f"ood: {len(ood)}/24 (density x test-set)")

    print("\n" + "=" * 70)
    print("Main-text outputs")
    print("=" * 70)
    build_T_A(args.results, out_tables, official, l2ws_cent, l2ws_dist, a2dr)
    build_F_A(out_figs, official)
    build_F_M(out_figs, official, l2ws_cent, appendix=False)
    build_F_B(out_figs, official)
    build_F_C_T_C(out_figs, out_tables, official, ood)

    print("\n" + "=" * 70)
    print("Appendix outputs")
    print("=" * 70)
    build_T_gap(out_tables, official, l2ws_cent)
    build_F_full_methods(out_figs, official, l2ws_cent, l2ws_dist, a2dr, with_fpres=True, name="figApp_methods")
    build_F_full_methods(out_figs, official, l2ws_cent, l2ws_dist, a2dr, with_fpres=False, name="figB2_methods")
    build_F_M(out_figs, official, l2ws_cent, appendix=True)
    build_T_full(out_tables, official, l2ws_cent, l2ws_dist, a2dr, ood)
    build_T_hyper(out_tables, l2ws_dist=l2ws_dist, a2dr=a2dr, official=official)

    flag_missing("F-gamma", "no DR step-size grid-search log is saved on disk (figApp_gamma.pdf not produced).")
    flag_missing("T-calib", "no sanity-check/calibration output for the current generator (c~U(0.05,0.18)) is "
                             "saved on disk (tabApp_calibration.tex not produced).")
    flag_missing("T-seeds", "only one base market (one seed) exists per density -- not applicable.")
    flag_missing("Ablations", "no saved rho-variant or slow-channel on/off runs besides the single frozen "
                               "config per density -- not produced.")
    flag_missing("T4", "no timing logs saved; measuring it requires running a new (cheap) script -- not run, "
                        "per instruction to ask first.")

    with open(os.path.join(args.out, "paper_numbers.json"), "w") as f:
        json.dump(NUMBERS, f, indent=2)
    print("\nwrote paper_numbers.json")

    print("\n" + "=" * 70)
    print(f"DONE. {len(MISSING)} missing/skipped items (see above).")
    print("=" * 70)


if __name__ == "__main__":
    main()
