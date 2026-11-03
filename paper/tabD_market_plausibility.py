#!/usr/bin/env python3
"""
tabD_market_plausibility.py
=============================
Tabella D "ufficiale": parametri del generatore e check di plausibilita'
(feasibility, compratori/venditori, curtailment, dumping, linee congestionate),
sul generatore e sui base_seed REALMENTE usati dalle run ufficiali (vedi
gen_table_d_market_plausibility.py). Righe = metrica, colonne = densita'.
Solo booktabs, niente griglia verticale.
"""
from __future__ import annotations

import os
import json

import common

DATA_PATH = os.path.join(common.PAPER_DATA, "table_d_data", "table_d_plausibility.json")
OUT_DIR = common.TABLES
os.makedirs(OUT_DIR, exist_ok=True)

DENSITIES = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def main():
    data = {float(k): v for k, v in json.load(open(DATA_PATH)).items()}

    lines = []
    lines.append("% DRAFT CAPTION: Plausibility of the generated markets (market.MarketConfig, c in\n"
                  "% U(0.05,0.18), the generator actually used for the reported results), 200\n"
                  "% level+width-perturbed instances per density, solved exactly (equilibrium QP).\n"
                  "% 'at capacity' is solver-dependent at tau=0 when the trading graph has cycles\n"
                  "% (individual edge flows are non-unique there -- not a real diagnostic, see\n"
                  "% equilibrium.py); kept only for completeness.\n")
    lines.append("\\begin{tabular}{l" + "c" * len(DENSITIES) + "}")
    lines.append("\\toprule")
    header = " & ".join(f"$\\delta={d}$" for d in DENSITIES)
    lines.append("metric & " + header + " \\\\")
    lines.append("\\midrule")

    def row(label, fmt, key, suffix=""):
        cells = [fmt(data[d][key]) + suffix if data[d][key] is not None else "--" for d in DENSITIES]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")

    row("feasible instances", lambda v: f"{v*100:.0f}\\%", "feasible_rate")
    lines.append("buyers / sellers (avg. of 10)" + " & " +
                 " & ".join(f"{data[d]['buyers_mean']:.1f} / {data[d]['sellers_mean']:.1f}" for d in DENSITIES) + " \\\\")
    row("demand curtailment (median)", lambda v: f"{v*100:.0f}\\%", "curtailment_median_median")
    row("dumping (unabsorbed export)", lambda v: f"{v*100:.1f}\\%", "frac_dumping_mean")
    row("nodal price $\\lambda$ (median)", lambda v: f"{v:.3f}", "lam_median")
    row("trades at capacity$^{*}$", lambda v: f"{v*100:.0f}\\%", "frac_at_capacity_mean")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path = os.path.join(OUT_DIR, "tabD_market_plausibility.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
