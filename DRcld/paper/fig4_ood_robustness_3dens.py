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

import make_paper_figures as mpf
from fig4_ood_robustness import load_all, delta_for, CATS, OUT_DIR, TEXTWIDTH_IN

DENSITIES_SUBSET = [0.2, 0.6, 1.0]


def main():
    official, ood = load_all()
    n_cats = len(CATS)

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.35),
                              sharey=True, gridspec_kw={"wspace": 0.12})

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
        ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                     xytext=(0, -18), textcoords="offset points", ha="center", fontsize=8.5)

    ymin, ymax = axes[0].get_ylim()
    yticks = np.arange(np.floor(ymin / 25) * 25, np.ceil(ymax / 25) * 25 + 1, 25)
    axes[0].set_yticks(yticks)

    fig.supylabel("iteration reduction vs. DR (%)", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.78)

    legend_handles = [Patch(facecolor=c, edgecolor="black", alpha=1.0, linewidth=0.8, label=label)
                      for _, label, c in CATS]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=n_cats, frameon=False, handlelength=1.3, handleheight=1.1,
               columnspacing=1.1, fontsize=8)

    out_pdf = os.path.join(OUT_DIR, "fig4_ood_robustness_3dens.pdf")
    out_png = os.path.join(OUT_DIR, "fig4_ood_robustness_3dens.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING (not plotted):", missing)


if __name__ == "__main__":
    main()
