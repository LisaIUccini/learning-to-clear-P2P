#!/usr/bin/env python3
"""
fig4_ood_robustness.py
========================
Figura 4 "ufficiale": robustezza fuori distribuzione di DR+GNN. UN'UNICA
immagine con 6 sottopannelli (uno per densita), stile IDENTICO a
fig1_methods_comparison.py: un marker alla mediana + barra d'errore IQR
[Q1,Q3] per categoria (ID, +-20%, +-30%, +-40%, +-1 prosumer), niente box,
niente outlier, niente etichette ruotate (colore + legenda su una riga
sopra). Scala di blu (stesso blu di "DR+GNN" in fig1, #0072B2) progressiva
per i tre shift operativi, nero per ID (baseline, come DR altrove),
vermiglio per +-1 prosumer (shift di natura diversa).

Stile: dimensioni reali, font 8-9/7-8pt, linee 1-1.2pt, marker 3-4pt,
bordi 0.8pt, palette Okabe-Ito coerente con fig1, griglia leggera solo
orizzontale, no bordo top/right, tratteggiata a 0, stessi limiti y fra
pannelli, no titolo interno, PNG + PDF vettoriale.
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

TEXTWIDTH_IN = common.TEXTWIDTH_IN

paper_style.apply()  # font/colori come il testo del paper (Times)

# stesso blu di "DR+GNN" in fig1 (#0072B2) come tonalita' piena/scura della
# rampa (Figura 4 e' interamente su DR+GNN); nero per ID come baseline;
# vermiglio per +-1 nodo (shift di natura diversa dai tre operativi)
CATS = [
    ("id", "ID", "#000000"),
    ("level20", "$\\pm20\\%$", "#A5CDE4"),
    ("level30", "$\\pm30\\%$", "#59A3CD"),
    ("level40", "$\\pm40\\%$", "#0072B2"),
    ("pm1", "$\\pm1$ prosumer", "#D55E00"),
]


def load_all():
    official = {d: common.load_official(d) for d in common.DENSITIES}
    ood = {}
    for d in common.DENSITIES:
        for level in ("level20", "level30", "level40"):
            rec = common.load_ood_level(d, level)
            if rec is not None:
                ood[(d, level)] = rec
        rec = common.load_ood_pm1(d)
        if rec is not None:
            ood[(d, "pm1")] = rec
    return official, ood


def delta_for(key, density, official, ood):
    if key == "id":
        off = official.get(density)
        if off is None:
            return None
        return common.delta_pct([i["k_DR"] for i in off["instances"]], [i["k_learned"] for i in off["instances"]])
    rec = ood.get((density, key))
    if rec is None:
        return None
    return common.delta_pct(rec["dr_iters"], rec["learned_iters"])


def main():
    official, ood = load_all()
    n_cats = len(CATS)

    # layout comune compatto (paper_style.grid_2x3), coerente con fig1/fig2b
    fig, axes = paper_style.grid_2x3(TEXTWIDTH_IN, sharey=True)

    missing = []
    panel_data = {}
    for density in common.DENSITIES:
        rows = []
        for mi, (key, label, color) in enumerate(CATS):
            d = delta_for(key, density, official, ood)
            if d is None:
                missing.append((key, density)); continue
            rows.append((mi, np.median(d), np.percentile(d, 25), np.percentile(d, 75), color))
        panel_data[density] = rows

    for ax, density in zip(axes, common.DENSITIES):
        for pos, med, q1, q3, c in panel_data[density]:
            ax.errorbar([pos], [med], yerr=[[med - q1], [q3 - med]],
                        fmt="s", ecolor="black", markerfacecolor=c,
                        markeredgecolor="black", markeredgewidth=0.8, markersize=4.0,
                        elinewidth=1.1, capsize=2.2, capthick=1.1, zorder=3)
        ax.axhline(0, color="gray", ls="--", lw=1.0, zorder=0)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_xticks([])
        ax.set_xlim(-0.7, n_cats - 0.3)
        paper_style.delta_label_below(ax, density)

    # asse y piu' dettagliato (step 25 invece di 50), stessa convenzione di fig1
    ymin, ymax = axes[0].get_ylim()
    yticks = np.arange(np.floor(ymin / 25) * 25, np.ceil(ymax / 25) * 25 + 1, 25)
    axes[0].set_yticks(yticks)

    paper_style.supylabel(fig, "iteration reduction vs. DR (%)")

    legend_handles = [Patch(facecolor=c, edgecolor="black", alpha=1.0, linewidth=0.8, label=label)
                      for _, label, c in CATS]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=n_cats, frameon=False, handlelength=1.3, handleheight=1.1,
               columnspacing=1.1, fontsize=8)

    out_pdf = os.path.join(OUT_DIR, "fig4_ood_robustness.pdf")
    out_png = os.path.join(OUT_DIR, "fig4_ood_robustness.png")
    paper_style.save_fixed(fig, out_pdf, out_png)  # dimensione fissa, niente ritaglio
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING (not plotted):", missing)


if __name__ == "__main__":
    main()
