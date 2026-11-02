"""Analysis of the rollouts saved by unperturbed_stop.py: per-instance and
per-density CSVs, figures (suffix _unperturbed_stop), a LaTeX table in the
format of Table 5 and the gap diagnostic at k = 0, 1, 2, 5, 10 for density
0.8. Never overwrites (files opened with "x").

    python3 -m scripts.analysis.unperturbed_stop_report <out_dir>
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from l2p2p import constants as C
from l2p2p import gnn as gnn_mod
from l2p2p import market, official, paths
from scripts.analysis import unperturbed_stop as R

sys.path.insert(0, str(paths.PAPER))
import common          # noqa: E402  (paper/common.py: paper figure conventions)
import paper_style     # noqa: E402

paper_style.apply()
TEXTWIDTH_IN = common.TEXTWIDTH_IN
TAU_GRID = np.logspace(-1, -5, 25)      # same grid as fig2b_stopping_tolerance.py
STYLE = {
    "dr":              dict(name="DR", color="#888888"),
    "gnn_perturbed":   dict(name="DR+GNN (perturbed stop)", color="#0072B2"),
    "gnn_unperturbed": dict(name="DR+GNN (unperturbed stop)", color="#D55E00"),
}
LEARNED = ["gnn_perturbed", "gnn_unperturbed"]


def _open_new(out_dir, name):
    return open(os.path.join(out_dir, name), "x")


def stop_series(r):
    """The residual the run actually stopped on."""
    return r["dr_residual"] if r["stop_on"] == "unperturbed" else r["residual"]


def first_hit(series, tau):
    idx = np.flatnonzero(np.asarray(series) <= tau)
    return int(idx[0]) + 1 if idx.size else None


def per_instance_rows(densities, out_dir):
    rows = []
    for d in densities:
        by, off = R.load_traj(out_dir, d), R.load_official(d)
        for i in sorted(by):
            k_dr = by[i]["dr"]["k"]
            for m in R.METHODS:
                r = by[i][m]
                row = dict(density=d, instance=i, method=m, gamma=r["gamma"], tol=r["tol"],
                           k=r["k"], converged=r["converged"], k_DR=k_dr,
                           delta=(k_dr - r["k"]) / k_dr,
                           residual_perturbed_final=r["residual"][-1],
                           residual_unperturbed_final=r["dr_residual"][-1],
                           rel_gap_final=abs(r["signed_gap"][-1]),
                           signed_rel_gap_final=r["signed_gap"][-1],
                           violation_final=r["violation"][-1])
                if m in ("dr", "gnn_perturbed"):
                    key, kk = ("dr", "k_DR") if m == "dr" else ("learned", "k_learned")
                    o = off[i]
                    row["k_official"] = o[kk]
                    row["reproduces_official"] = bool(
                        o[kk] == r["k"] and o[key]["residual"] == r["residual"]
                        and np.allclose(o[key]["violation"], r["violation"], rtol=1e-6, atol=1e-12)
                        and np.allclose(o[key]["relative_gap"], np.abs(r["signed_gap"]), rtol=1e-6, atol=1e-12))
                else:
                    row["k_official"] = ""
                    row["reproduces_official"] = ""
                rows.append(row)
    return rows


def summarize(rows, densities):
    out = []
    for d in densities:
        for m in R.METHODS:
            rr = [r for r in rows if r["density"] == d and r["method"] == m]
            k = np.array([r["k"] for r in rr]); dl = np.array([r["delta"] for r in rr])
            out.append(dict(
                density=d, method=m, n=len(rr), n_hit_cap=int(sum(not r["converged"] for r in rr)),
                k_median=np.median(k), k_q1=np.percentile(k, 25), k_q3=np.percentile(k, 75),
                delta_median=np.median(dl), delta_q1=np.percentile(dl, 25), delta_q3=np.percentile(dl, 75),
                rel_gap_final_median=np.median([r["rel_gap_final"] for r in rr]),
                violation_final_median=np.median([r["violation_final"] for r in rr]),
                residual_perturbed_final_median=np.median([r["residual_perturbed_final"] for r in rr]),
                residual_unperturbed_final_median=np.median([r["residual_unperturbed_final"] for r in rr]),
                n_reproduces_official=(sum(r["reproduces_official"] is True for r in rr)
                                       if m != "gnn_unperturbed" else "")))
    return out


def write_csv(out_dir, name, rows):
    with _open_new(out_dir, name) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def fig_reduction(out_dir, summary, densities):
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.60),
                              sharey=True, gridspec_kw={"hspace": 0.33, "wspace": 0.12})
    axes = axes.ravel()
    for ax, d in zip(axes, R.DENSITIES):
        for pos, m in enumerate(LEARNED):
            s = next((s for s in summary if s["density"] == d and s["method"] == m), None)
            if s is None:
                continue
            med, q1, q3 = 100 * s["delta_median"], 100 * s["delta_q1"], 100 * s["delta_q3"]
            ax.errorbar([pos], [med], yerr=[[med - q1], [q3 - med]], fmt="s", ecolor="black",
                        markerfacecolor=STYLE[m]["color"], markeredgecolor="black", markeredgewidth=0.8,
                        markersize=4.0, elinewidth=1.1, capsize=2.2, capthick=1.1, zorder=3)
        ax.axhline(0, color="gray", ls="--", lw=1.0, zorder=0)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xticks([]); ax.set_xlim(-0.7, len(LEARNED) - 0.3)
    ymin, ymax = axes[0].get_ylim()
    axes[0].set_yticks(np.arange(25 * np.ceil(ymin / 25), ymax + 1e-9, 25))
    fig.supylabel("iteration reduction vs. DR (%)", fontsize=9)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.07, top=0.86)
    for ax, d in zip(axes, R.DENSITIES):
        ax.annotate(f"$\\delta = {d}$", xy=(0.5, 0), xycoords="axes fraction",
                    xytext=(0, -10), textcoords="offset points", ha="center", fontsize=8.5)
    handles = [Patch(facecolor=STYLE[m]["color"], edgecolor="black", linewidth=0.8, label=STYLE[m]["name"])
               for m in LEARNED]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=2, frameon=False,
               handlelength=1.3, handleheight=1.1, columnspacing=1.1, fontsize=8)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        p = os.path.join(out_dir, f"reduction_comparison_unperturbed_stop.{ext}")
        assert not os.path.exists(p), p
        fig.savefig(p, bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)


def fig_tolerance_d08(out_dir):
    by = R.load_traj(out_dir, 0.8)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.5, TEXTWIDTH_IN * 0.38))
    for m in R.METHODS:
        hits = [[first_hit(stop_series(by[i][m]), t) for t in TAU_GRID] for i in sorted(by)]
        xs, med, q1, q3 = [], [], [], []
        for j, t in enumerate(TAU_GRID):
            vals = [h[j] for h in hits if h[j] is not None]
            if len(vals) < len(hits) / 2:
                continue
            xs.append(t); med.append(np.median(vals))
            q1.append(np.percentile(vals, 25)); q3.append(np.percentile(vals, 75))
        c = STYLE[m]["color"]
        ax.plot(xs, med, color=c, lw=1.15, label=STYLE[m]["name"], zorder=3)
        ax.fill_between(xs, q1, q3, color=c, alpha=0.18, lw=0, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(1e-1, 1e-5)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel("stopping tolerance $\\tau$"); ax.set_ylabel("iterations (= communication rounds)")
    ax.text(0.03, 0.95, "$\\delta = 0.8$", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=1, frameon=False, fontsize=7.5)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        p = os.path.join(out_dir, f"tolerance_d08_unperturbed_stop.{ext}")
        assert not os.path.exists(p), p
        fig.savefig(p, bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)


def table_tex(out_dir, summary, densities):
    g = {(s["density"], s["method"]): s for s in summary}
    lines = ["% DRAFT CAPTION: Solution quality at termination (same definitions as Table 5 /",
             "% tabF_stopping_quality.py: relative gap |J(xbar)-J*|/|J*| and dist(xbar, X_loc) of the shadow",
             "% iterate), median over the 30 official test instances. P = DR+GNN stopped on the perturbed",
             "% residual ||x^{k+1}-x^k|| < 1e-5 (official); U = DR+GNN stopped on the unperturbed residual",
             "% ||lam(z^k - xtilde^k)|| < 1e-5 (same quantity DR stops on).", "",
             "\\begin{tabular}{lcccccc}", "\\toprule",
             "$\\delta$ & gap DR & gap DR+GNN (P) & gap DR+GNN (U) & viol. DR & viol. DR+GNN (P) & viol. DR+GNN (U) \\\\",
             "\\midrule"]
    for d in densities:
        gaps = " & ".join(f"{g[(d, m)]['rel_gap_final_median']:.2e}" for m in R.METHODS)
        viols = " & ".join(f"{g[(d, m)]['violation_final_median']:.2e}" for m in R.METHODS)
        lines.append(f"$\\delta={d}$ & {gaps} & {viols} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with _open_new(out_dir, "accuracy_termination_unperturbed_stop.tex") as f:
        f.write("\n".join(lines) + "\n")


def _balance_at_k(density, i, method, ks):
    """Short re-run (max(ks) iterations, same x0/model) to get max|nodal balance| of xbar^k --
    not stored by run_dr and not part of dist_to_Xloc."""
    _, _, test_specs = official.datasets(density, n_train=1, n_test=i)
    spec = test_specs[i - 1]
    gen = torch.Generator(); gen.manual_seed(C.TEST_SEED * 7919 + i - 1)
    net = market.build_network(spec, rng=gen)
    if method != "dr":
        x0 = net.flatten().clone()
        net = market.build_network(spec); net.unflatten(x0)
        gnn_mod.deploy(official.load_official_model(density), net)
    out = {0: float(net.balance_at(net.project_reciprocity(net.flatten())).abs().max())}
    operator = official.dr_config(density).operator()
    net.reset()
    with torch.no_grad():
        for k in range(1, max(ks) + 1):
            net.communicate()
            for p in net.prosumers:
                p.step(operator)
            if k in ks:
                out[k] = float(net.balance_at(net.project_reciprocity(net.flatten())).abs().max())
    return out


def gap_diagnostic(out_dir):
    by = R.load_traj(out_dir, 0.8)
    inst = sorted(by)
    rows, lines = [], []
    same_x0 = all(by[i]["dr"]["x0"] == by[i]["gnn_perturbed"]["x0"] == by[i]["gnn_unperturbed"]["x0"]
                  for i in inst)
    lines.append(f"delta=0.8, 30 instances. (a) x0 identical for DR / DR+GNN(P) / DR+GNN(U) on every "
                 f"instance: {same_x0}")
    lines.append("Indexing: k=0 is xbar^0 = proj_Xrec(x0) (before any iteration); k>=1 is after k iterations "
                 "(= relative_gap[k-1] in the official instance JSONs).")
    lines.append("Columns: median signed rel. gap (J-J*)/|J*| [min,max], median |gap| (Table 5 / Fig. gap def.), "
                 "#instances with J<J*, median dist(xbar,X_loc), median max|nodal balance| of xbar.")
    for m in ("dr", "gnn_perturbed"):
        bal = {i: _balance_at_k(0.8, i, m, R.DIAG_K) for i in inst}
        lines.append(f"\n[{STYLE[m]['name']}]")
        for k in R.DIAG_K:
            sg = np.array([by[i][m]["gap0_signed"] if k == 0 else by[i][m]["signed_gap"][k - 1] for i in inst])
            vi = np.array([by[i][m]["viol0"] if k == 0 else by[i][m]["violation"][k - 1] for i in inst])
            bl = np.array([bal[i][k] for i in inst])
            lines.append(f"  k={k:2d}  signed gap {np.median(sg):+.3e} [{sg.min():+.2e},{sg.max():+.2e}]  "
                         f"|gap| {np.median(np.abs(sg)):.3e}  #J<J* {int((sg < 0).sum()):2d}  "
                         f"viol {np.median(vi):.3e}  balance {np.median(bl):.3e}")
            for i in inst:
                rows.append(dict(method=m, instance=i, k=k, signed_rel_gap=sg[inst.index(i)],
                                 abs_rel_gap=abs(sg[inst.index(i)]), violation=vi[inst.index(i)],
                                 max_abs_balance=bal[i][k]))
        fh, rebound, neg = [], 0, 0
        for i in inst:
            a = np.abs(by[i][m]["signed_gap"])
            f = first_hit(a, 0.1); fh.append(f)
            rebound += int(np.any(a[f:] > 0.1))
            neg += int(by[i][m]["signed_gap"][f - 1] < 0)
        lines.append(f"  first k with |gap|<=0.1 (fig2_gap_levels.first_hit): median {np.median(fh):.0f}, "
                     f"min {min(fh)}, max {max(fh)};  J<J* at that k: {neg}/30;  "
                     f"|gap| later back above 0.1: {rebound}/30")
    with _open_new(out_dir, "gap_diagnostic_d0.8_unperturbed_stop.txt") as f:
        f.write("\n".join(lines) + "\n")
    write_csv(out_dir, "gap_diagnostic_d0.8_unperturbed_stop.csv", rows)
    print("\n".join(lines))


def main():
    out_dir = sys.argv[1]
    densities = [d for d in R.DENSITIES if os.path.exists(os.path.join(out_dir, f"trajectories_d{d}.json"))]
    rows = per_instance_rows(densities, out_dir)
    write_csv(out_dir, "per_instance_metrics_unperturbed_stop.csv", rows)
    summary = summarize(rows, densities)
    write_csv(out_dir, "summary_by_density_unperturbed_stop.csv", summary)
    fig_reduction(out_dir, summary, densities)
    if 0.8 in densities:
        fig_tolerance_d08(out_dir)
        gap_diagnostic(out_dir)
    table_tex(out_dir, summary, densities)
    for s in summary:
        print(f"d={s['density']} {s['method']:16s} k_med={s['k_median']:7.1f} "
              f"delta_med={100 * s['delta_median']:6.1f}% [{100 * s['delta_q1']:.1f},{100 * s['delta_q3']:.1f}] "
              f"cap={s['n_hit_cap']} gap={s['rel_gap_final_median']:.2e} viol={s['violation_final_median']:.2e} "
              f"repro={s['n_reproduces_official']}")


if __name__ == "__main__":
    main()
