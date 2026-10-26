#!/usr/bin/env python3
"""
fig2_gap_levels.py
====================
Figura 2 "ufficiale" -- stile Moret et al. 2018 Fig. 1, versione completa:
DR (grigio), DR+GNN (blu), A2DR (rosa), L2WS-reg (arancione: tratteggio =
distribuito, continuo = centralizzato), L2WS-asym (verde: stessa
convenzione) -- 7 curve, SOLA serie del gap (niente violazione). UN'UNICA
immagine con 6 sottopannelli (uno per densita'), stesso schema di
fig1/fig4. Banda IQR leggera solo su DR e DR+GNN (per non affollare troppo
con 7 bande). Conteggio: L2WS (dist.) somma gli L round; A2DR conta tutte
le applicazioni di T_DR; L2WS (cent.)/A2DR richiedono un coordinatore
(comunicazione globale, non quantificata qui, vedi Tabella A).

DR, DR+GNN, L2WS-reg/asym (cent.): traiettorie dalla pipeline ufficiale
(nessun calcolo extra). A2DR, L2WS-reg/asym (dist.): da fig2_data/ (vedi
gen_fig2_trajectories.py, valutazione extra, nessun training).

Stile: dimensioni reali, font 8-9/7-8pt, linee 1-1.2pt, legenda sopra,
palette Okabe-Ito coerente con fig1/fig4, griglia leggera solo
orizzontale, no bordo top/right, no titolo interno, PNG + PDF vettoriale.
"""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_paper_figures as mpf

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figs_official")
FIG2_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_data")
os.makedirs(OUT_DIR, exist_ok=True)

TEXTWIDTH_IN = mpf.TEXTWIDTH_IN
RESULTS_DIR = "/home/lisa/energy_market_fixed_point/DRcld/experiments"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

EPS_GRID = np.logspace(0, -5, 30)

# (display name, color, linestyle, show_iqr)
METHODS = {
    "dr":             dict(name="DR", color="#888888", ls="-", iqr=True),
    "learned":        dict(name="DR+GNN", color="#0072B2", ls="-", iqr=True),
    "a2dr":           dict(name="A2DR", color="#CC79A7", ls="-", iqr=False),
    "l2ws_reg_cent":  dict(name="L2WS-reg (cent.)", color="#E69F00", ls="-", iqr=False),
    "l2ws_reg_dist":  dict(name="L2WS-reg (dist.)", color="#E69F00", ls="--", iqr=False),
    "l2ws_asym_cent": dict(name="L2WS-asym (cent.)", color="#009E73", ls="-", iqr=False),
    "l2ws_asym_dist": dict(name="L2WS-asym (dist.)", color="#009E73", ls="--", iqr=False),
}
ORDER = ["dr", "learned", "a2dr", "l2ws_reg_cent", "l2ws_reg_dist", "l2ws_asym_cent", "l2ws_asym_dist"]


def first_hit(series, eps):
    series = np.asarray(series)
    idx = np.flatnonzero(series <= eps)
    return int(idx[0]) + 1 if idx.size else None


def median_iqr_curve(hits_per_instance, eps_grid):
    n = len(hits_per_instance)
    xs, med, q1, q3 = [], [], [], []
    for j, e in enumerate(eps_grid):
        vals = [hits_per_instance[i][j] for i in range(n) if hits_per_instance[i][j] is not None]
        if len(vals) < n / 2:
            continue
        xs.append(e)
        med.append(np.median(vals)); q1.append(np.percentile(vals, 25)); q3.append(np.percentile(vals, 75))
    return xs, med, q1, q3


def plot_method(ax, mk, gap_arrs, offset=0):
    m = METHODS[mk]
    hits = [[first_hit(a, e) for e in EPS_GRID] for a in gap_arrs]
    xs, med, q1, q3 = median_iqr_curve(hits, EPS_GRID)
    if xs:
        med = [v + offset for v in med]
        ax.plot(xs, med, color=m["color"], ls=m["ls"], lw=1.15, label=m["name"], zorder=3)
        if m["iqr"]:
            q1 = [v + offset for v in q1]; q3 = [v + offset for v in q3]
            ax.fill_between(xs, q1, q3, color=m["color"], alpha=0.15, lw=0, zorder=2)


def main():
    official = {d: mpf.load_official(RESULTS_DIR, d) for d in mpf.DENSITIES}
    l2ws_cent = {}
    for d in mpf.DENSITIES:
        for loss in ("regression", "asymmetric"):
            lc = mpf.load_l2ws_cent(RESULTS_DIR, d, loss)
            if lc is not None:
                l2ws_cent[(d, loss)] = lc
    extra = {}
    for d in mpf.DENSITIES:
        p = os.path.join(FIG2_DATA_DIR, f"fig2_extra_trajectories_d{d}.json")
        if os.path.exists(p):
            extra[d] = json.load(open(p))

    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.68),
                              gridspec_kw={"hspace": 0.6, "wspace": 0.2})
    axes = axes.ravel()

    missing = []
    for ax, density in zip(axes, mpf.DENSITIES):
        off = official.get(density)
        if off is not None:
            plot_method(ax, "dr", [i["dr"]["relative_gap"] for i in off["instances"]])
            plot_method(ax, "learned", [i["learned"]["relative_gap"] for i in off["instances"]])

        for mk, loss in [("l2ws_reg_cent", "regression"), ("l2ws_asym_cent", "asymmetric")]:
            lc = l2ws_cent.get((density, loss))
            if lc is not None:
                plot_method(ax, mk, [i["learned"]["relative_gap"] for i in lc["instances"]])
            else:
                missing.append((mk, density))

        ex = extra.get(density)
        if ex is not None:
            a2dr_d = ex.get("a2dr", {})
            if a2dr_d.get("instances"):
                plot_method(ax, "a2dr", [i["relative_gap"] for i in a2dr_d["instances"]])
            else:
                missing.append(("a2dr", density))
            l2dist = ex.get("l2ws_dist", {})
            for mk, loss in [("l2ws_reg_dist", "regression"), ("l2ws_asym_dist", "asymmetric")]:
                d_ = l2dist.get(loss)
                if d_ and d_.get("instances"):
                    plot_method(ax, mk, [i["relative_gap"] for i in d_["instances"]], offset=d_["L"])
                else:
                    missing.append((mk, density))
        else:
            missing += [("a2dr", density), ("l2ws_reg_dist", density), ("l2ws_asym_dist", density)]

        ax.set_xscale("log")
        ax.set_xlim(1, 1e-5)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_xlabel("optimality gap $\\varepsilon$")

    fig.supylabel("iterations (= communication rounds)", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.0, 1, 0.84))
    for ax, density in zip(axes, mpf.DENSITIES):
        ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                     xytext=(0, -42), textcoords="offset points", ha="center", fontsize=8.5)

    seen, handles, labels = set(), [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen.add(ll); handles.append(hh); labels.append(ll)
    ordered = [(h, l) for mk in ORDER for h, l in zip(handles, labels) if l == METHODS[mk]["name"]]
    handles2, labels2 = ([h for h, _ in ordered], [l for _, l in ordered]) if ordered else (handles, labels)
    fig.legend(handles2, labels2, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=4, frameon=False, fontsize=7.5, handlelength=1.8, columnspacing=1.0)

    out_pdf = os.path.join(OUT_DIR, "fig2_gap_levels.pdf")
    out_png = os.path.join(OUT_DIR, "fig2_gap_levels.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING:", missing)


if __name__ == "__main__":
    main()
