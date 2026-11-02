#!/usr/bin/env python3
"""
fig3_difficulty_gain.py
=========================
Figura 3 "ufficiale": difficolta' dell'istanza vs. guadagno. Un solo
pannello piccolo (mezza larghezza). Un punto per istanza di test
(in-distribuzione, tutte le densita' insieme), colorato per densita',
piccolo e leggermente trasparente. x = iterazioni DR (log), y = riduzione
% di DR+GNN. Spearman riportato SOLO in testo/caption, non nella figura.

Stile: dimensioni reali (0.48*textwidth, altezza ~0.4x larghezza), font
8-9/7-8pt, marker 3-4pt, legenda su una riga sopra la figura, palette
sequenziale (la densita' e' una grandezza ordinale), griglia leggera solo
orizzontale, no bordo top/right, tratteggiata a 0, no titolo interno,
PNG + PDF vettoriale.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

import common

OUT_DIR = common.FIGURES
os.makedirs(OUT_DIR, exist_ok=True)

TEXTWIDTH_IN = common.TEXTWIDTH_IN

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


def main():
    official = {d: common.load_official(d) for d in common.DENSITIES}

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.48, TEXTWIDTH_IN * 0.48 * 0.75))
    cmap = plt.get_cmap("viridis")

    all_x, all_y = [], []
    per_density = {}
    for idx, density in enumerate(common.DENSITIES):
        off = official.get(density)
        if off is None:
            continue
        x = np.array([i["k_DR"] for i in off["instances"]], dtype=float)
        y = common.delta_pct(x, [i["k_learned"] for i in off["instances"]])
        ax.scatter(x, y, s=10, color=cmap(idx / (len(common.DENSITIES) - 1)),
                   alpha=0.6, edgecolors="none", label=f"$\\delta={density}$", zorder=3)
        all_x.extend(x); all_y.extend(y)
        rho, p = sstats.spearmanr(x, y)
        per_density[density] = (rho, p)

    rho_all, p_all = sstats.spearmanr(all_x, all_y)

    ax.axhline(0, color="gray", ls="--", lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_xlabel("DR iteration count (log scale)")
    ax.set_ylabel("iteration reduction vs. DR (%)")

    fig.legend(loc="upper center", bbox_to_anchor=(0.56, 1.14), ncol=3,
               frameon=False, fontsize=7, handletextpad=0.3, columnspacing=0.9,
               markerscale=1.6)

    fig.tight_layout()
    out_pdf = os.path.join(OUT_DIR, "fig3_difficulty_gain.pdf")
    out_png = os.path.join(OUT_DIR, "fig3_difficulty_gain.png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"\nSpearman overall: rho={rho_all:.3f}  p={p_all:.2e}")
    for d, (rho, p) in per_density.items():
        print(f"  delta={d}: rho={rho:.3f}  p={p:.2e}")


if __name__ == "__main__":
    main()
