#!/usr/bin/env python3
"""
fig1_methods_comparison.py
===========================
Figura 1 "ufficiale": confronto fra metodi, UN'UNICA immagine con 6
sottopannelli (uno per densita), ciascuno con 6 punti (mediana + barra
IQR [Q1,Q3]), colore = metodo, nessuna etichetta ruotata (identita del
metodo solo via colore + legenda su una riga in alto).

Regole di stile (istruzioni utente 2026-10-20): dimensioni reali
(larghezza = \\textwidth, altezza totale ~0.5x larghezza), font 8-9pt
(etichette)/7-8pt (numeri), linee 1-1.2pt, marker 3-4pt, palette
Okabe-Ito coerente con paper_plots, legenda su una riga sopra la figura,
griglia leggera solo orizzontale, no bordo top/right, tratteggiata a 0,
stessi limiti y fra pannelli, no titolo interno, no pallini (outlier),
salvataggio PNG + PDF vettoriale.

Riusa i loader e le convenzioni di colore/metodo di make_paper_figures.py.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import make_paper_figures as mpf

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figs_official")
os.makedirs(OUT_DIR, exist_ok=True)

TEXTWIDTH_IN = mpf.TEXTWIDTH_IN  # 6.00in, stesso valore (non verificato, vedi report precedente)
RESULTS_DIR = "/home/lisa/energy_market_fixed_point/DRcld/experiments"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

METHOD_ORDER = ["learned", "a2dr", "l2ws_reg_cent", "l2ws_reg_dist", "l2ws_asym_cent", "l2ws_asym_dist"]
DISPLAY_NAME = {"learned": "DR+GNN"}


def load_all():
    official = {d: mpf.load_official(RESULTS_DIR, d) for d in mpf.DENSITIES}
    l2ws_cent, l2ws_dist = {}, {}
    for d in mpf.DENSITIES:
        for loss in ("regression", "asymmetric"):
            lc = mpf.load_l2ws_cent(RESULTS_DIR, d, loss)
            if lc is not None:
                l2ws_cent[(d, loss)] = lc
            ld = mpf.load_l2ws_dist(RESULTS_DIR, d, loss)
            if ld is not None:
                l2ws_dist[(d, loss)] = ld
    a2dr = {d: v for d, v in ((d, mpf.load_a2dr(RESULTS_DIR, d)) for d in mpf.DENSITIES) if v is not None}
    return official, l2ws_cent, l2ws_dist, a2dr


def delta_for(mk, density, official, l2ws_cent, l2ws_dist, a2dr):
    if mk == "learned":
        off = official.get(density)
        if off is None:
            return None
        return mpf.delta_pct([i["k_DR"] for i in off["instances"]], [i["k_learned"] for i in off["instances"]])
    if mk == "a2dr":
        ad = a2dr.get(density)
        if ad is None:
            return None
        return mpf.delta_pct(ad["data"]["dr_iters"], ad["data"]["a2dr_iters"])
    if mk in ("l2ws_reg_cent", "l2ws_asym_cent"):
        loss = "regression" if mk == "l2ws_reg_cent" else "asymmetric"
        lc = l2ws_cent.get((density, loss))
        if lc is None:
            return None
        return mpf.delta_pct(lc["timing"]["dr_iters"], lc["timing"]["l2ws_iters"])
    if mk in ("l2ws_reg_dist", "l2ws_asym_dist"):
        loss = "regression" if mk == "l2ws_reg_dist" else "asymmetric"
        ld = l2ws_dist.get((density, loss))
        if ld is None:
            return None
        post = np.array(ld["l2ws_post_iters"]) + ld["L"]
        return mpf.delta_pct(ld["dr_iters"], post)
    return None


def main():
    official, l2ws_cent, l2ws_dist, a2dr = load_all()
    n_methods = len(METHOD_ORDER)

    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.60),
                              sharey=True, gridspec_kw={"hspace": 0.33, "wspace": 0.12})
    axes = axes.ravel()

    missing = []
    panel_data = {}  # density -> list of (pos, med, q1, q3, color, filled)
    for density in mpf.DENSITIES:
        rows = []
        for mi, mk in enumerate(METHOD_ORDER):
            d = delta_for(mk, density, official, l2ws_cent, l2ws_dist, a2dr)
            if d is None:
                missing.append((mk, density))
                continue
            med = np.median(d)
            q1, q3 = np.percentile(d, 25), np.percentile(d, 75)
            rows.append((mi, med, q1, q3, mpf.METHODS[mk]["color"], mpf.METHODS[mk]["filled"]))
        panel_data[density] = rows

    for ax, density in zip(axes, mpf.DENSITIES):
        for pos, med, q1, q3, c, filled in panel_data[density]:
            # bordi del marker e baffi neri per contrasto -- eccetto i marker
            # "hollow" (distribuito), dove il bordo colorato e' l'unico modo
            # di identificare il metodo (niente riempimento su cui appoggiarsi)
            ax.errorbar([pos], [med], yerr=[[med - q1], [q3 - med]],
                        fmt="s", ecolor="black", markerfacecolor=c if filled else "white",
                        markeredgecolor="black" if filled else c, markeredgewidth=0.8, markersize=4.0,
                        elinewidth=1.1, capsize=2.2, capthick=1.1, zorder=3)
        ax.axhline(0, color="gray", ls="--", lw=1.0, zorder=0)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_xticks([])
        ax.set_xlim(-0.7, n_methods - 0.3)

    # asse y piu' dettagliato sopra lo 0 (step 25) che sotto (step 50)
    ymin, ymax = axes[0].get_ylim()
    ticks_below = np.arange(0, ymin - 1e-9, -50)
    ticks_above = np.arange(0, ymax + 1e-9, 25)
    yticks = np.unique(np.concatenate([ticks_below, ticks_above]))
    axes[0].set_yticks(yticks)

    fig.supylabel("iteration reduction vs. DR (%)", fontsize=9)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.07, top=0.82)
    for ax, density in zip(axes, mpf.DENSITIES):
        ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                     xytext=(0, -10), textcoords="offset points", ha="center", fontsize=8.5)

    # alpha=1.0 (colore pieno): i marker nei pannelli sono gia' pieni
    # (errorbar non applica alpha), la legenda deve corrispondere esattamente
    legend_handles = [Patch(facecolor=mpf.METHODS[mk]["color"] if mpf.METHODS[mk]["filled"] else "white",
                             edgecolor="black" if mpf.METHODS[mk]["filled"] else mpf.METHODS[mk]["color"],
                             alpha=1.0, linewidth=0.8,
                             label=DISPLAY_NAME.get(mk, mpf.METHODS[mk]["name"].replace("$^\\dagger$", "")))
                      for mk in METHOD_ORDER]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=3, frameon=False, handlelength=1.3, handleheight=1.1,
               columnspacing=1.1, labelspacing=0.5, fontsize=8)

    out_pdf = os.path.join(OUT_DIR, "fig1_methods_comparison.pdf")
    out_png = os.path.join(OUT_DIR, "fig1_methods_comparison.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING (not plotted):", missing)


if __name__ == "__main__":
    main()
