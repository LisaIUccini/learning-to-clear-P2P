#!/usr/bin/env python3
"""
fig2b_stopping_tolerance.py
=============================
Variante di fig2_gap_levels.py: invece di valutare livelli di GAP lungo
UNA traiettoria gia' fermata a tolleranza 1e-5, l'asse x e' la TOLLERANZA
DI ARRESTO stessa (residual < tau), da 1e-1 a 1e-5 -- cioe' "quante
iterazioni servirebbero se il criterio di arresto standard usasse quella
tolleranza invece di 1e-5?". Calcolato come primo k con residual[k] < tau
sulla traiettoria del residual GIA' REGISTRATA (quella del criterio
standard, che e' un sovrainsieme di qualunque tau >= 1e-5 piu' lasco).

3 metodi (DR grigio, DR+GNN blu, L2WS-asym (dist.) arancione), un'unica
immagine con 6 sottopannelli (uno per densita'), stesso schema di fig2.
Banda IQR leggera. Conteggio: L2WS somma gli L round.

Stile: dimensioni reali, font 8-9/7-8pt, linee 1-1.2pt, legenda su una
riga sopra, palette Okabe-Ito coerente, griglia leggera solo orizzontale,
no bordo top/right, no titolo interno, PNG + PDF vettoriale.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import common
import paper_style

OUT_DIR = common.FIGURES
FIG2_DATA_DIR = os.path.join(common.PAPER_DATA, "fig2_data")
os.makedirs(OUT_DIR, exist_ok=True)

TEXTWIDTH_IN = common.TEXTWIDTH_IN

paper_style.apply()  # font/colori come il testo del paper (Times)

TAU_GRID = np.logspace(-1, -5, 25)

METHODS = {
    "dr":      dict(name="DR", color="#888888"),
    "learned": dict(name="DR+GNN", color="#0072B2"),
}


def first_hit(series, tau):
    series = np.asarray(series)
    idx = np.flatnonzero(series <= tau)
    return int(idx[0]) + 1 if idx.size else None


def median_iqr_curve(hits_per_instance, grid):
    n = len(hits_per_instance)
    xs, med, q1, q3 = [], [], [], []
    for j, e in enumerate(grid):
        vals = [hits_per_instance[i][j] for i in range(n) if hits_per_instance[i][j] is not None]
        if len(vals) < n / 2:
            continue
        xs.append(e)
        med.append(np.median(vals)); q1.append(np.percentile(vals, 25)); q3.append(np.percentile(vals, 75))
    return xs, med, q1, q3


def plot_method(ax, mk, res_arrs, offset=0):
    c = METHODS[mk]["color"]
    hits = [[first_hit(a, tau) for tau in TAU_GRID] for a in res_arrs]
    xs, med, q1, q3 = median_iqr_curve(hits, TAU_GRID)
    if xs:
        med = [m + offset for m in med]; q1 = [v + offset for v in q1]; q3 = [v + offset for v in q3]
        ax.plot(xs, med, color=c, ls="-", lw=1.15, label=METHODS[mk]["name"], zorder=3)
        ax.fill_between(xs, q1, q3, color=c, alpha=0.18, lw=0, zorder=2)


def main():
    official = {d: common.load_official(d) for d in common.DENSITIES}

    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.56),
                              gridspec_kw={"hspace": 0.28, "wspace": 0.30})
    axes = axes.ravel()

    missing = []
    for ax, density in zip(axes, common.DENSITIES):
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
        # piu' tacche sull'asse y (numero di iterazioni): fino a 10 invece
        # delle ~5 di default, matplotlib sceglie comunque passi "tondi"
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10]))
        ax.text(0.03, 0.95, f"$\\delta = {density}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=8.5)

    # un solo xlabel condiviso (l'asse x e' "tolleranza di arresto" ovunque,
    # non serve ripeterlo sotto ogni pannello) -- posizionato esplicitamente
    # per evitare che si sovrapponga ai numeri dei tick del pannello sotto
    fig.supxlabel("stopping tolerance $\\tau$", fontsize=9, y=0.015)
    fig.supylabel("iterations (= communication rounds)", fontsize=9)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.11, top=0.86)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.0),
               ncol=2, frameon=False, fontsize=8, handlelength=1.6, columnspacing=1.2)

    out_pdf = os.path.join(OUT_DIR, "fig2b_stopping_tolerance.pdf")
    out_png = os.path.join(OUT_DIR, "fig2b_stopping_tolerance.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    if missing:
        print("MISSING L2WS-dist residual trajectory for densities:", missing)


if __name__ == "__main__":
    main()
