#!/usr/bin/env python3
"""
fig2_gap_levels_2methods.py
=============================
Variante di fig2_gap_levels.py: SOLO DR e DR+GNN (2 curve con banda IQR),
stesso layout e stile di fig2b_stopping_tolerance.py. Stessi dati (riusa
plot_method dal file originale, nessun dato ricalcolato, nessun file
originale sovrascritto -- output con nome diverso).
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import make_paper_figures as mpf
from fig2_gap_levels import plot_method, OUT_DIR, RESULTS_DIR, TEXTWIDTH_IN
import paper_style

paper_style.apply()  # font/colori come il testo del paper (Times)


def main():
    official = {d: mpf.load_official(RESULTS_DIR, d) for d in mpf.DENSITIES}

    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.56),
                              gridspec_kw={"hspace": 0.28, "wspace": 0.30})
    axes = axes.ravel()

    missing = []
    for ax, density in zip(axes, mpf.DENSITIES):
        off = official.get(density)
        if off is not None:
            plot_method(ax, "dr", [i["dr"]["relative_gap"] for i in off["instances"]])
            plot_method(ax, "learned", [i["learned"]["relative_gap"] for i in off["instances"]])
        else:
            missing.append(density)

        ax.set_xscale("log")
        ax.set_xlim(1, 1e-5)
        ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10]))
        ax.text(0.03, 0.95, f"$\\delta = {density}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5)

    fig.supxlabel("optimality gap $\\varepsilon$", fontsize=9, y=0.015)
    fig.supylabel("iterations (= communication rounds)", fontsize=9)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.11, top=0.86)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=2, frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.2)

    out_pdf = os.path.join(OUT_DIR, "fig2_gap_levels_2methods.pdf")
    out_png = os.path.join(OUT_DIR, "fig2_gap_levels_2methods.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING:", missing)


if __name__ == "__main__":
    main()
