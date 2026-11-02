#!/usr/bin/env python3
"""
fig1_methods_comparison_3dens.py
===================================
Variante di fig1_methods_comparison.py: SOLO 3 densita' (0.2, 0.6, 1.0),
UN'UNICA riga di 3 sottopannelli (1x3) invece di 2x3. Stessi dati, stesso
stile (riusa load_all/delta_for dal file originale, nessun dato
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
from fig1_methods_comparison import load_all, delta_for, METHOD_ORDER, DISPLAY_NAME, OUT_DIR, TEXTWIDTH_IN

DENSITIES_SUBSET = [0.2, 0.6, 1.0]


def main(method_order=METHOD_ORDER, suffix=""):
    official, l2ws_cent, l2ws_dist, a2dr = load_all()
    n_methods = len(method_order)

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.35),
                              sharey=True, gridspec_kw={"wspace": 0.12})

    missing = []
    panel_data = {}
    for density in DENSITIES_SUBSET:
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

    for ax, density in zip(axes, DENSITIES_SUBSET):
        for pos, med, q1, q3, c, filled in panel_data[density]:
            ax.errorbar([pos], [med], yerr=[[med - q1], [q3 - med]],
                        fmt="s", ecolor="black", markerfacecolor=c if filled else "white",
                        markeredgecolor="black" if filled else c, markeredgewidth=0.8, markersize=4.0,
                        elinewidth=1.1, capsize=2.2, capthick=1.1, zorder=3)
        ax.axhline(0, color="gray", ls="--", lw=1.0, zorder=0)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_xticks([])
        ax.set_xlim(-0.7, n_methods - 0.3)

    ymin, ymax = axes[0].get_ylim()
    ticks_below = np.arange(0, ymin - 1e-9, -50)
    ticks_above = np.arange(0, ymax + 1e-9, 25)
    yticks = np.unique(np.concatenate([ticks_below, ticks_above]))
    axes[0].set_yticks(yticks)

    fig.supylabel("iteration reduction vs. DR (%)", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.72 if n_methods > 4 else 0.84)
    for ax, density in zip(axes, DENSITIES_SUBSET):
        ax.annotate(f"$\\delta = {density}$", xy=(0.5, 0), xycoords="axes fraction",
                     xytext=(0, -18), textcoords="offset points", ha="center", fontsize=8.5)

    legend_handles = [Patch(facecolor=common.METHODS[mk]["color"] if common.METHODS[mk]["filled"] else "white",
                             edgecolor="black" if common.METHODS[mk]["filled"] else common.METHODS[mk]["color"],
                             alpha=1.0, linewidth=0.8,
                             label=DISPLAY_NAME.get(mk, common.METHODS[mk]["name"].replace("$^\\dagger$", "")))
                      for mk in method_order]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=3 if n_methods > 4 else n_methods, frameon=False, handlelength=1.3, handleheight=1.1,
               columnspacing=1.1, labelspacing=0.5, fontsize=8)

    out_pdf = os.path.join(OUT_DIR, f"fig1_methods_comparison_3dens{suffix}.pdf")
    out_png = os.path.join(OUT_DIR, f"fig1_methods_comparison_3dens{suffix}.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING (not plotted):", missing)


if __name__ == "__main__":
    main()
