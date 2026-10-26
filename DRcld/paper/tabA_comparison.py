#!/usr/bin/env python3
"""
tabA_comparison.py
====================
Tabella A "ufficiale": confronto numerico fra metodi, stessi dati della
Figura 1. Righe = metodi in due blocchi (distribuiti / riferimenti
centralizzati, separati da una riga orizzontale); colonne = densita'.
Prima riga: mediana delle iterazioni di DR. Grassetto sul miglior metodo
DISTRIBUITO per colonna; (dagger) sui centralizzati. Solo booktabs
(toprule/midrule/bottomrule), niente griglia verticale.

Riusa i loader di make_paper_figures.py.
"""
from __future__ import annotations

import os
import numpy as np

import make_paper_figures as mpf

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_tables_official")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_DIR = "/home/lisa/energy_market_fixed_point/DRcld/experiments"

DIST_ORDER = ["learned", "l2ws_reg_dist", "l2ws_asym_dist"]
CENT_ORDER = ["l2ws_reg_cent", "l2ws_asym_cent", "a2dr"]
DISPLAY_NAME = {"learned": "DR+GNN"}


def load_all():
    official = {d: mpf.load_official(RESULTS_DIR, d) for d in mpf.DENSITIES}
    l2ws_cent, l2ws_dist = {}, {}
    for d in mpf.DENSITIES:
        for loss in ("regression", "asymmetric"):
            lc = mpf.load_l2ws_cent(RESULTS_DIR, d, loss)
            if lc is not None:
                l2ws_cent[(d, loss)] = lc
            ld = mpf.load_l2ws_dist(RESULTS_DIR, d, loss)
            if ld is not None:
                l2ws_dist[(d, loss)] = ld
    a2dr = {d: v for d, v in ((d, mpf.load_a2dr(RESULTS_DIR, d)) for d in mpf.DENSITIES) if v is not None}
    return official, l2ws_cent, l2ws_dist, a2dr


def row_for(mk, density, official, l2ws_cent, l2ws_dist, a2dr):
    """Ritorna (array di CONTEGGI DI ITERAZIONE grezzi del metodo, extra-rounds label)
    -- non piu' la riduzione percentuale: quella si legge dalla Figura 1, qui restano
    i numeri esatti (mediana [IQR]) da citare nel testo."""
    if mk == "learned":
        off = official.get(density)
        if off is None:
            return None
        return [i["k_learned"] for i in off["instances"]], "0"
    if mk in ("l2ws_reg_dist", "l2ws_asym_dist"):
        loss = "regression" if mk == "l2ws_reg_dist" else "asymmetric"
        ld = l2ws_dist.get((density, loss))
        if ld is None:
            return None
        post = np.array(ld["l2ws_post_iters"]) + ld["L"]
        return post, f"L={ld['L']}"
    if mk in ("l2ws_reg_cent", "l2ws_asym_cent"):
        loss = "regression" if mk == "l2ws_reg_cent" else "asymmetric"
        lc = l2ws_cent.get((density, loss))
        if lc is None:
            return None
        return lc["timing"]["l2ws_iters"], "global"
    if mk == "a2dr":
        ad = a2dr.get(density)
        if ad is None:
            return None
        return ad["data"]["a2dr_iters"], "global"
    return None


def fmt_cell(d):
    med = np.median(d)
    half_iqr = (np.percentile(d, 75) - np.percentile(d, 25)) / 2
    return f"{med:.0f} $\\pm$ {half_iqr:.0f}"


def main():
    official, l2ws_cent, l2ws_dist, a2dr = load_all()
    cols = mpf.DENSITIES

    dr_median_row = []
    for density in cols:
        off = official.get(density)
        dr_median_row.append(fmt_cell([i["k_DR"] for i in off["instances"]]) if off else None)

    rows_dist = {mk: {density: row_for(mk, density, official, l2ws_cent, l2ws_dist, a2dr) for density in cols}
                 for mk in DIST_ORDER}
    rows_cent = {mk: {density: row_for(mk, density, official, l2ws_cent, l2ws_dist, a2dr) for density in cols}
                 for mk in CENT_ORDER}

    def best_dist_per_col(density):
        # meno iterazioni = meglio (non piu' % di riduzione)
        meds = {mk: np.median(rows_dist[mk][density][0]) for mk in DIST_ORDER if rows_dist[mk][density] is not None}
        return min(meds, key=meds.get) if meds else None

    # extra-rounds e' costante per metodo (non varia per densita'): si mette
    # come suffisso del nome del metodo invece che come colonna a parte --
    # dimezza la larghezza della tabella. Niente piu' IQR nelle celle (solo
    # mediana): lo spread si legge dalla Figura 1.
    EXTRA_SUFFIX = {"0": "", "global": " (global)"}

    def extra_suffix(label):
        if label.startswith("L="):
            return f" ({label})"
        return EXTRA_SUFFIX.get(label, f" ({label})")

    lines = []
    lines.append("% DRAFT CAPTION: Iterations to convergence per method and density (median $\\pm$ half-IQR\n"
                  "% over 30 test instances); percentages and win-rate are in Figure 1. Bold = best (fewest\n"
                  "% iterations) distributed method per column. (dagger) = centralized reference; '(global)'/\n"
                  "% '(L=k)' = additional communication beyond the reported iteration count (see Sec. 5.2).\n")
    lines.append("\\begin{tabular}{l" + "c" * len(cols) + "}")
    lines.append("\\toprule")
    header = " & ".join(f"$\\delta={d}$" for d in cols)
    lines.append("method & " + header + " \\\\")
    lines.append("\\midrule")
    dr_row = " & ".join(f"{v}" if v is not None else "--" for v in dr_median_row)
    lines.append("DR & " + dr_row + " \\\\")
    lines.append("\\midrule")

    for mk in DIST_ORDER:
        cells = []
        for density in cols:
            r = rows_dist[mk][density]
            if r is None:
                cells.append("--"); continue
            cell = fmt_cell(r[0])
            if best_dist_per_col(density) == mk:
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        extra = next((rows_dist[mk][d][1] for d in cols if rows_dist[mk][d] is not None), "--")
        name = DISPLAY_NAME.get(mk, mpf.METHODS[mk]["name"]) + extra_suffix(extra)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    for mk in CENT_ORDER:
        cells = []
        for density in cols:
            r = rows_cent[mk][density]
            cells.append(fmt_cell(r[0]) if r is not None else "--")
        extra = next((rows_cent[mk][d][1] for d in cols if rows_cent[mk][d] is not None), "--")
        name = mpf.METHODS[mk]["name"] + extra_suffix(extra)  # name already carries $^\dagger$
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path = os.path.join(OUT_DIR, "tabA_comparison.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
