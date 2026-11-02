#!/usr/bin/env python3
"""
tabE_hyperparameters.py
=========================
Tabella E "ufficiale": tutti gli iperparametri di DR+GNN, L2WS, gamma (per
densita') e A2DR, con il metodo di selezione. Letti dai log/config reali
delle run finali (non dai default). Due parti: (E1) parametri fissi /
condivisi fra densita' (parametro, valore, metodo di selezione); (E2) i
pochi parametri che VARIANO per densita' (gamma, K, scale del canale
lento, memoria/regolarizzazione di A2DR), in una sotto-tabella compatta
con le densita' come colonne (stesso schema di Tabella A/D) -- altrimenti
le liste "per densita'" in una singola cella sforano la larghezza di
pagina, come successo con Tabella A. Solo booktabs, niente griglia verticale.
"""
from __future__ import annotations

import os
import re

import common

OUT_DIR = common.TABLES
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    official = {d: common.load_official(d) for d in common.DENSITIES}
    a2dr = {d: v for d, v in ((d, common.load_a2dr(d)) for d in common.DENSITIES) if v is not None}

    def grab(text, key):
        m = re.search(rf"{key}=([^\s)]+)", text)
        return m.group(1) if m else "?"

    gamma_per_d, K_per_d, scale_s_per_d = {}, {}, {}
    for d in common.DENSITIES:
        off = official[d]
        h, sl = off["header"], off["slow_line"]
        gamma_per_d[d] = grab(h, "gamma.")
        K_per_d[d] = grab(h, "unroll")
        scale_s_per_d[d] = grab(sl, "scale_slow")

    # ---------------- E1: parametri fissi / condivisi ----------------
    rows = [
        ("\\multicolumn{3}{l}{\\textit{DR (baseline, shared by all methods)}}", "", ""),
        ("step size $\\gamma$ / relaxation $\\lambda$", "see Table~\\ref{tab:hyper2} / 1.0", "grid search (train. set) / fixed"),
        ("bisection steps / tol. / cap", "30 / $10^{-5}$ / 4000", "fixed"),
        ("\\multicolumn{3}{l}{\\textit{DR+GNN ($\\Phi_w$) architecture}}", "", ""),
        ("hidden dim $d_h$ / MLP depth", "32 / 1 hidden layer", "fixed, all densities"),
        ("fast-channel decay $\\rho_{\\mathrm f}$", "$1-10/K$", "formula, verified exact all 6 densities"),
        ("slow-channel decay $\\rho_{\\mathrm s}$", "0.99", "fixed, all densities"),
        ("slow-channel scale, unroll $K$", "see Table~\\ref{tab:hyper2}", "validation set / half mean DR iters."),
        ("\\multicolumn{3}{l}{\\textit{DR+GNN training}}", "", ""),
        ("loss weights $\\alpha_{\\text{behind}},\\alpha_{\\text{ahead}},\\beta$", "2.0, 1.0, 2.0", "fixed"),
        ("optimizer / LR schedule", "Adam lr$=3\\times10^{-3}$ / cosine to 1\\%", "fixed"),
        ("gradient clipping / epochs", "$\\|\\nabla\\|\\le 1.0$ / 25", "fixed"),
        ("\\multicolumn{3}{l}{\\textit{L2WS (distributed)}}", "", ""),
        ("message-passing rounds $L$", "3", "ablation: $L\\in\\{1..5\\}$, validation loss"),
        ("architecture", "shared weights, residual update", "ablation: shared/separate $\\times$ GRU/residual"),
        ("optimizer / epochs", "Adam lr$=3\\times10^{-3}$ / 50", "ablation: lr $\\times$ epochs grid"),
        ("\\multicolumn{3}{l}{\\textit{A2DR}}", "", ""),
        ("memory $m$ / regularization $\\eta$", "see Table~\\ref{tab:hyper2}", "grid search, validation set"),
    ]

    lines = []
    lines.append("% DRAFT CAPTION (Table E1): Hyperparameters of the final runs shared across\n"
                  "% densities (read from logs/config, not defaults), and how each was selected.\n"
                  "% Parameters that vary by density are in Table E2.\n")
    lines.append("\\begin{tabular}{llp{4.2cm}}")
    lines.append("\\toprule")
    lines.append("parameter & value(s) & selection \\\\")
    lines.append("\\midrule")
    for name, value, method in rows:
        if value == "" and method == "":
            lines.append(name + " \\\\")
        else:
            lines.append(f"{name} & {value} & {method} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    out_path = os.path.join(OUT_DIR, "tabE_hyperparameters.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")

    # ---------------- E2: parametri che variano per densita' ----------------
    lines2 = []
    lines2.append("% DRAFT CAPTION (Table E2, \\label{tab:hyper2}): the hyperparameters of Table E1\n"
                  "% that vary by density (DR step size, unroll horizon, DR+GNN slow-channel scale,\n"
                  "% A2DR memory/regularization), all selected on the validation set (K via the\n"
                  "% training set, see Table E1).\n")
    lines2.append("\\begin{tabular}{l" + "c" * len(common.DENSITIES) + "}")
    lines2.append("\\toprule")
    header = " & ".join(f"$\\delta={d}$" for d in common.DENSITIES)
    lines2.append("parameter & " + header + " \\\\")
    lines2.append("\\midrule")
    lines2.append("DR step size $\\gamma$ & " + " & ".join(gamma_per_d[d] for d in common.DENSITIES) + " \\\\")
    lines2.append("unroll horizon $K$ & " + " & ".join(K_per_d[d] for d in common.DENSITIES) + " \\\\")
    lines2.append("slow-channel scale $s_{\\mathrm s}$ & " + " & ".join(scale_s_per_d[d] for d in common.DENSITIES) + " \\\\")
    lines2.append("A2DR memory $m$ & " +
                  " & ".join(str(a2dr[d]["data"]["a2dr_mmax"]) if d in a2dr else "--" for d in common.DENSITIES) + " \\\\")
    lines2.append("A2DR reg. $\\eta$ & " +
                  " & ".join(f"{a2dr[d]['data']['a2dr_eta']:g}" if d in a2dr else "--" for d in common.DENSITIES) + " \\\\")
    lines2.append("\\bottomrule")
    lines2.append("\\end{tabular}")

    out_path2 = os.path.join(OUT_DIR, "tabE2_hyperparameters_per_density.tex")
    with open(out_path2, "w") as f:
        f.write("\n".join(lines2) + "\n")
    print(f"wrote {out_path2}")


if __name__ == "__main__":
    main()
