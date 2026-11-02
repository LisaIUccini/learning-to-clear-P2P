#!/usr/bin/env python3
"""
fig1_noreg.py
==============
Figura 1 (2x3 e 3 densita') senza L2WS-reg: stessi dati e stile, output con
suffisso "_noreg", nessun file originale sovrascritto.
"""
import fig1_methods_comparison as fig1
import fig1_methods_comparison_3dens as fig1_3dens

METHODS_NOREG = [mk for mk in fig1.METHOD_ORDER if not mk.startswith("l2ws_reg")]

if __name__ == "__main__":
    fig1.main(METHODS_NOREG, "_noreg")
    fig1_3dens.main(METHODS_NOREG, "_noreg")
