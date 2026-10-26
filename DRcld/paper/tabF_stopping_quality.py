#!/usr/bin/env python3
"""
tabF_stopping_quality.py
==========================
Tabella "qualita' all'arresto": quando DR e DR+GNN si fermano (stesso
criterio, residual < tol), hanno risolto il mercato con la stessa
qualita'? Per ogni densita' (mediana sulle 30 istanze ufficiali di test):
gap di ottimalita' relativo, violazione dei vincoli locali (letti dai
dati salvati), e ||z^k - x_tilde^k|| (da table_ek_comparison.py --
valutazione extra, colonna aggiunta quando pronta, vedi quella nota
li' sul perche' non coincide col residual salvato per DR+GNN).
Solo booktabs, niente griglia verticale.
"""
from __future__ import annotations

import os
import json
import numpy as np

import make_paper_figures as mpf

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_tables_official")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_DIR = "/home/lisa/energy_market_fixed_point/DRcld/experiments"
EK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_ek_data", "table_ek_comparison.json")


def main():
    ek_data = json.load(open(EK_PATH)) if os.path.exists(EK_PATH) else None

    lines = []
    lines.append("% DRAFT CAPTION: Solution quality when DR and DR+GNN stop (same criterion,\n"
                  "% residual < 1e-5), median over 30 official test instances per density: relative\n"
                  "% optimality gap, local-constraint violation, and ||z^k - x_tilde^k|| (the DR-only\n"
                  "% part of the step; for DR+GNN this differs from the stored residual, which also\n"
                  "% includes the GNN perturbation v^k added on top -- see Sec. 5.2 discussion).\n")
    has_ek = ek_data is not None
    ncols = 6 if has_ek else 4
    lines.append("\\begin{tabular}{l" + "c" * ncols + "}")
    lines.append("\\toprule")
    header = "$\\delta$ & gap DR & gap DR+GNN & viol. DR & viol. DR+GNN"
    if has_ek:
        header += " & $e^k$ DR & $e^k$ DR+GNN"
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    for d in mpf.DENSITIES:
        off = mpf.load_official(RESULTS_DIR, d)
        gap_dr = np.median([i["dr"]["relative_gap"][-1] for i in off["instances"]])
        gap_l = np.median([i["learned"]["relative_gap"][-1] for i in off["instances"]])
        viol_dr = np.median([i["dr"]["violation"][-1] for i in off["instances"]])
        viol_l = np.median([i["learned"]["violation"][-1] for i in off["instances"]])
        row = f"$\\delta={d}$ & {gap_dr:.2e} & {gap_l:.2e} & {viol_dr:.2e} & {viol_l:.2e}"
        if has_ek:
            r = ek_data.get(str(d))
            if r:
                row += f" & {r['ek_dr_median']:.2e} & {r['ek_learned_median']:.2e}"
            else:
                row += " & -- & --"
        lines.append(row + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path = os.path.join(OUT_DIR, "tabF_stopping_quality.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    if not has_ek:
        print("NOTE: e^k columns omitted -- table_ek_comparison.json not found yet (still running).")


if __name__ == "__main__":
    main()
