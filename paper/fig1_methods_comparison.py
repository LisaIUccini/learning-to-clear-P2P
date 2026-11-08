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

Riusa i loader e le convenzioni di colore/metodo di common.py.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import common
import paper_style

OUT_DIR = common.FIGURES
os.makedirs(OUT_DIR, exist_ok=True)

TEXTWIDTH_IN = common.TEXTWIDTH_IN  # 6.00in, stesso valore (non verificato, vedi report precedente)

paper_style.apply()  # font/colori come il testo del paper (Times)

METHOD_ORDER = ["learned", "a2dr", "l2ws_reg_cent", "l2ws_reg_dist", "l2ws_asym_cent", "l2ws_asym_dist"]
DISPLAY_NAME = {"learned": "DR+GNN"}


def load_all():
    official = {d: common.load_official(d) for d in common.DENSITIES}
    l2ws_cent, l2ws_dist = {}, {}
    for d in common.DENSITIES:
        for loss in ("regression", "asymmetric"):
            lc = common.load_l2ws_cent(d, loss)
            if lc is not None:
                l2ws_cent[(d, loss)] = lc
            ld = common.load_l2ws_dist(d, loss)
            if ld is not None:
                l2ws_dist[(d, loss)] = ld
    a2dr = {d: v for d, v in ((d, common.load_a2dr(d)) for d in common.DENSITIES) if v is not None}
    return official, l2ws_cent, l2ws_dist, a2dr


def delta_for(mk, density, official, l2ws_cent, l2ws_dist, a2dr):
    if mk == "learned":
        off = official.get(density)
        if off is None:
            return None
        return common.delta_pct([i["k_DR"] for i in off["instances"]], [i["k_learned"] for i in off["instances"]])
    if mk == "a2dr":
        ad = a2dr.get(density)
        if ad is None:
            return None
        return common.delta_pct(ad["data"]["dr_iters"], ad["data"]["a2dr_iters"])
    if mk in ("l2ws_reg_cent", "l2ws_asym_cent"):
        loss = "regression" if mk == "l2ws_reg_cent" else "asymmetric"
        lc = l2ws_cent.get((density, loss))
        if lc is None:
            return None
        return common.delta_pct(lc["timing"]["dr_iters"], lc["timing"]["l2ws_iters"])
    if mk in ("l2ws_reg_dist", "l2ws_asym_dist"):
        loss = "regression" if mk == "l2ws_reg_dist" else "asymmetric"
        ld = l2ws_dist.get((density, loss))
        if ld is None:
            return None
        post = np.array(ld["l2ws_post_iters"]) + ld["L"]
        return common.delta_pct(ld["dr_iters"], post)
    return None


def main(method_order=METHOD_ORDER, suffix="", display_name=DISPLAY_NAME, compact=False):
    official, l2ws_cent, l2ws_dist, a2dr = load_all()
    n_methods = len(method_order)

    if compact:  # layout comune compatto (paper_style.grid_2x3), coerente con fig2b/fig4
        fig, axes = paper_style.grid_2x3(TEXTWIDTH_IN, sharey=True)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.60),
                                  sharey=True, gridspec_kw={"hspace": 0.33, "wspace": 0.12})
        axes = axes.ravel()

    missing = []
    panel_data = {}  # density -> list of (pos, med, q1, q3, color, filled)
    for density in common.DENSITIES:
        rows = []
        for mi, mk in enumerate(method_order):
            d = delta_for(mk, density, official, l2ws_cent, l2ws_dist, a2dr)
            if d is None:
                missing.append((mk, density))
                continue
            med = np.median(d)
            q1, q3 = np.percentile(d, 25), np.percentile(d, 75)
            rows.append((mi, med, q1, q3, common.METHODS[mk]["color"], common.METHODS[mk]["filled"]))
        panel_data[density] = rows

    for ax, density in zip(axes, common.DENSITIES):
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

    if compact:
        paper_style.supylabel(fig, "iteration reduction vs. DR (%)")
    else:
        fig.supylabel("iteration reduction vs. DR (%)", fontsize=9)
    if not compact:
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.07, top=0.82 if n_methods > 4 else 0.90)
    for ax, density in zip(axes, common.DENSITIES):
        if compact:
            paper_style.delta_label_below(ax, density)
        else:
            ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                         xytext=(0, -10), textcoords="offset points", ha="center", fontsize=8.5)

    # alpha=1.0 (colore pieno): i marker nei pannelli sono gia' pieni
    # (errorbar non applica alpha), la legenda deve corrispondere esattamente
    legend_handles = [Patch(facecolor=common.METHODS[mk]["color"] if common.METHODS[mk]["filled"] else "white",
                             edgecolor="black" if common.METHODS[mk]["filled"] else common.METHODS[mk]["color"],
                             alpha=1.0, linewidth=0.8,
                             label=display_name.get(mk, common.METHODS[mk]["name"].replace("$^\\dagger$", "")))
                      for mk in method_order]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=3 if n_methods > 4 else n_methods, frameon=False, handlelength=1.3, handleheight=1.1,
               columnspacing=1.1, labelspacing=0.5, fontsize=8)

    out_pdf = os.path.join(OUT_DIR, f"fig1_methods_comparison{suffix}.pdf")
    out_png = os.path.join(OUT_DIR, f"fig1_methods_comparison{suffix}.png")
    if compact:
        paper_style.save_fixed(fig, out_pdf, out_png)  # dimensione fissa, niente ritaglio
    else:
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING (not plotted):", missing)


if __name__ == "__main__":
    main()
