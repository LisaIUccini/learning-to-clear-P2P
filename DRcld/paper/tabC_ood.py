#!/usr/bin/env python3
"""
tabC_ood.py
============
Tabella C "ufficiale": stessi dati della Figura 4, in forma numerica.
Righe = test set (ID, +-20%, +-30%, +-40%, +-1 prosumer), ciascuno con
DUE righe (DR, DR+GNN) raggruppate con \\multirow; colonne = densita'.
Cella: iterazioni, mediana $\\pm$ meta' IQR (stesso formato di
tabA_comparison.py) -- niente percentuali/win-rate (si leggono dalla
Figura 4). Solo booktabs, niente griglia verticale.
"""
from __future__ import annotations

import os
import numpy as np

import make_paper_figures as mpf

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_tables_official")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_DIR = "/home/lisa/energy_market_fixed_point/DRcld/experiments"

CATS = [("id", "ID"), ("level20", "$\\pm20\\%$"), ("level30", "$\\pm30\\%$"),
        ("level40", "$\\pm40\\%$"), ("pm1", "$\\pm1$ prosumer")]


def load_all():
    official = {d: mpf.load_official(RESULTS_DIR, d) for d in mpf.DENSITIES}
    ood = {}
    for d in mpf.DENSITIES:
        for level in ("level20", "level30", "level40"):
            rec = mpf.load_ood_level(RESULTS_DIR, d, level)
            if rec is not None:
                ood[(d, level)] = rec
        rec = mpf.load_ood_pm1(RESULTS_DIR, d)
        if rec is not None:
            ood[(d, "pm1")] = rec
    return official, ood


def iters(key, density, official, ood, field):
    """field: 'k_DR'/'dr_iters' vs 'k_learned'/'learned_iters' depending on source."""
    if key == "id":
        off = official.get(density)
        if off is None:
            return None
        return [i["k_DR"] for i in off["instances"]] if field == "dr" else [i["k_learned"] for i in off["instances"]]
    rec = ood.get((density, key))
    if rec is None:
        return None
    return rec["dr_iters"] if field == "dr" else rec["learned_iters"]


def fmt_cell(d):
    med = np.median(d)
    half_iqr = (np.percentile(d, 75) - np.percentile(d, 25)) / 2
    return f"{med:.0f} $\\pm$ {half_iqr:.0f}"


def main():
    official, ood = load_all()
    cols = mpf.DENSITIES

    lines = []
    lines.append("% DRAFT CAPTION: Iterations under distribution shift, DR vs. DR+GNN (median $\\pm$\n"
                  "% half-IQR over 30 test instances per cell); percentages and win-rate are in Figure 4.\n")
    lines.append("\\begin{tabular}{ll" + "c" * len(cols) + "}")
    lines.append("\\toprule")
    header = " & ".join(f"$\\delta={d}$" for d in cols)
    lines.append("test set & method & " + header + " \\\\")
    lines.append("\\midrule")

    for ci, (key, label) in enumerate(CATS):
        for ri, (field, mname) in enumerate([("dr", "DR"), ("learned", "DR+GNN")]):
            cells = []
            for density in cols:
                d = iters(key, density, official, ood, field)
                cells.append(fmt_cell(d) if d is not None else "--")
            row_label = "\\multirow{2}{*}{" + label + "}" if ri == 0 else ""
            lines.append(f"{row_label} & {mname} & " + " & ".join(cells) + " \\\\")
        if ci < len(CATS) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path = os.path.join(OUT_DIR, "tabC_ood.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
