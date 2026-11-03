#!/usr/bin/env python3
"""
tabT4_timing.py
=================
Tabella T4 "ufficiale": tempo della prox locale e di una valutazione di
Phi_w, per agente per iterazione (media su ~5 istanze di test per
densita', su tutte le iterazioni e tutti gli agenti), overhead %, e tempo
di training (colonna aggiunta quando la misura isolata e' pronta). Solo
booktabs, niente griglia verticale. Caption: hardware, precisione,
numero di thread (dai metadati salvati dagli script di misura).
"""
from __future__ import annotations

import os
import json

import common

TIMING_PATH = os.path.join(common.PAPER_DATA, "table_t4_data", "table_t4_timing.json")
TRAIN_PATH = os.path.join(common.PAPER_DATA, "table_t4_data", "table_t4_train_time.json")
OUT_DIR = common.TABLES
os.makedirs(OUT_DIR, exist_ok=True)

DENSITIES = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


def main():
    timing = json.load(open(TIMING_PATH))
    train = json.load(open(TRAIN_PATH)) if os.path.exists(TRAIN_PATH) else None

    lines = []
    cap = (f"% DRAFT CAPTION: Per-agent, per-iteration wall-clock time of the local prox step\n"
           f"% (bisection) and of one Phi_w evaluation (messages, aggregation, GRU, output heads),\n"
           f"% mean over ~5 test instances per density, all iterations and agents. Overhead =\n"
           f"% Phi_w / prox. Hardware: {timing['hardware']} ({timing['platform']}), "
           f"{timing['n_threads']} thread, {timing['dtype']}.\n")
    lines.append(cap)
    has_train = train is not None
    ncols = 5 if has_train else 4
    lines.append("\\begin{tabular}{l" + "c" * ncols + "}")
    lines.append("\\toprule")
    header = "$\\delta$ & local prox [ms/agent/iter.] & $\\Phi_w$ [ms/agent/iter.] & overhead"
    if has_train:
        header += " & training [min]"
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    for d in DENSITIES:
        r = timing["results"][str(d)]
        row = (f"$\\delta={d}$ & {r['prox_ms_per_agent_iter']:.3f} & {r['gnn_ms_per_agent_iter']:.3f} & "
               f"{r['overhead_pct']:.0f}\\%")
        if has_train:
            tr = train["results"].get(str(d))
            row += f" & {tr['extrapolated_25ep_minutes']:.1f}" if tr else " & --"
        lines.append(row + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path = os.path.join(OUT_DIR, "tabT4_timing.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    if not has_train:
        print("NOTE: training-time column omitted -- table_t4_train_time.json not found yet.")


if __name__ == "__main__":
    main()
