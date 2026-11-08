#!/usr/bin/env python3
"""
fig4_ood_robustness_3dens.py
===============================
Variante di fig4_ood_robustness.py: SOLO 3 densita' (0.2, 0.6, 1.0),
UN'UNICA riga di 3 sottopannelli (1x3) invece di 2x3. Stessi dati, stesso
stile (riusa load_all/delta_for/CATS dal file originale, nessun dato
ricalcolato, nessun file originale sovrascritto -- output con nome
diverso).
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
from fig4_ood_robustness import load_all, delta_for, CATS, OUT_DIR, TEXTWIDTH_IN

DENSITIES_SUBSET = [0.2, 0.6, 1.0]


def main():
    official, ood = load_all()
    n_cats = len(CATS)

    # layout comune compatto (paper_style.grid_2x3 con una riga), coerente con fig1/fig2b
    fig, axes = paper_style.grid_2x3(TEXTWIDTH_IN, sharey=True, nrows=1)

    missing = []
    panel_data = {}
    for density in DENSITIES_SUBSET:
        rows = []
        for mi, (key, label, color) in enumerate(CATS):
            d = delta_for(key, density, official, ood)
            if d is None:
                missing.append((key, density)); continue
            rows.append((mi, np.median(d), np.percentile(d, 25), np.percentile(d, 75), color))
        panel_data[density] = rows

    for ax, density in zip(axes, DENSITIES_SUBSET):
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

    ymin, ymax = axes[0].get_ylim()
    yticks = np.arange(np.floor(ymin / 25) * 25, np.ceil(ymax / 25) * 25 + 1, 25)
    axes[0].set_yticks(yticks)

    paper_style.supylabel(fig, "iteration reduction\nvs. DR (%)")  # 2 righe: una sola riga di pannelli

    legend_handles = [Patch(facecolor=c, edgecolor="black", alpha=1.0, linewidth=0.8, label=label)
                      for _, label, c in CATS]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=n_cats, frameon=False, handlelength=1.3, handleheight=1.1,
               columnspacing=1.1, fontsize=8)

    out_pdf = os.path.join(OUT_DIR, "fig4_ood_robustness_3dens.pdf")
    out_png = os.path.join(OUT_DIR, "fig4_ood_robustness_3dens.png")
    paper_style.save_fixed(fig, out_pdf, out_png)  # dimensione fissa, niente ritaglio
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING (not plotted):", missing)


if __name__ == "__main__":
    main()
