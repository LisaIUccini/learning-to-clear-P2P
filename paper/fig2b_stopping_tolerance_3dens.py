#!/usr/bin/env python3
"""
fig2b_stopping_tolerance_3dens.py
===================================
Variante di fig2b_stopping_tolerance.py: SOLO 3 densita' (0.2, 0.6, 1.0),
UN'UNICA riga di 3 sottopannelli (1x3) invece di 2x3. Stessi dati, stesso
stile (riusa plot_method dal file originale, nessun dato ricalcolato,
nessun file originale sovrascritto -- output con nome diverso).
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import common
import paper_style
from fig2b_stopping_tolerance import plot_method, OUT_DIR, TEXTWIDTH_IN

DENSITIES_SUBSET = [0.2, 0.6, 1.0]


def main():
    official = {d: common.load_official(d) for d in DENSITIES_SUBSET}

    # layout comune compatto (paper_style.grid_2x3 con una riga), coerente con fig1/fig4
    fig, axes = paper_style.grid_2x3(TEXTWIDTH_IN, nrows=1, bottom_in=paper_style.BOTTOM_TICKS_IN, wspace=0.30)

    missing = []
    for ax, density in zip(axes, DENSITIES_SUBSET):
        off = official.get(density)
        if off is not None:
            plot_method(ax, "dr", [i["dr"]["residual"] for i in off["instances"]])
            plot_method(ax, "learned", [i["learned"]["residual"] for i in off["instances"]])
        else:
            missing.append(density)

        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e-5)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10]))
        ax.text(0.03, 0.95, f"$\\delta = {density}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5)

    fig.supxlabel("stopping tolerance $\\tau$", fontsize=9, y=0.0, va="bottom")
    paper_style.supylabel(fig, "iterations")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=2, frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.2)

    out_pdf = os.path.join(OUT_DIR, "fig2b_stopping_tolerance_3dens.pdf")
    out_png = os.path.join(OUT_DIR, "fig2b_stopping_tolerance_3dens.png")
    paper_style.save_fixed(fig, out_pdf, out_png)  # dimensione fissa, niente ritaglio
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING official results for densities:", missing)


if __name__ == "__main__":
    main()
