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

import common

OUT_DIR = common.TABLES
os.makedirs(OUT_DIR, exist_ok=True)

CATS = [("id", "ID"), ("level20", "$\\pm20\\%$"), ("level30", "$\\pm30\\%$"),
        ("level40", "$\\pm40\\%$"), ("plus1", "$+1$ prosumer"), ("minus1", "$-1$ prosumer")]
# +1/-1 prosumer: sottoinsiemi del test pm1 (stesso log), separati per tipo
PM1_KIND = {"plus1": "add", "minus1": "remove"}


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


def iters(key, density, official, ood, field):
    """field: 'k_DR'/'dr_iters' vs 'k_learned'/'learned_iters' depending on source."""
    if key == "id":
        off = official.get(density)
        if off is None:
            return None
        return [i["k_DR"] for i in off["instances"]] if field == "dr" else [i["k_learned"] for i in off["instances"]]
    if key in PM1_KIND:
        rec = ood.get((density, "pm1"))
        if rec is None:
            return None
        vals = rec["dr_iters"] if field == "dr" else rec["learned_iters"]
        sel = [v for v, k in zip(vals, rec["kinds"]) if k == PM1_KIND[key]]
        return sel or None
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
    cols = common.DENSITIES

    lines = []
    n_pm = {key: [len(iters(key, d, official, ood, "dr") or []) for d in cols] for key in PM1_KIND}
    lines.append("% DRAFT CAPTION: Iterations under distribution shift, DR vs. DR+GNN (median $\\pm$\n"
                  "% half-IQR over 30 test instances per cell; the 30 +/-1 prosumer instances are split\n"
                  "% into +1 / -1, n per density = " +
                  ", ".join(f"{d}: {a}/{b}" for d, a, b in zip(cols, n_pm["plus1"], n_pm["minus1"])) +
                  "); percentages and win-rate are in Figure 4.\n")
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
