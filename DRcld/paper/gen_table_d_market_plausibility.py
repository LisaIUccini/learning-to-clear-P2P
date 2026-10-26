#!/usr/bin/env python3
"""
gen_table_d_market_plausibility.py
=====================================
Valutazione extra (NON training) per la Tabella D: le stesse 6 metriche
di plausibilita' di sanity_check_calibrated_market.py, ma sul generatore
e sui base_seed REALMENTE usati dalle run ufficiali (market.MarketConfig,
c in U(0.05,0.18) -- confermato in ogni NOTES.md come "generator=old"),
non sul generatore "ricalibrato" (build_calibrated_market.py) a cui
puntava lo script originale.

Solo QP solve (equilibrium.solve_equilibrium_diagnostics, CLARABEL), nessun
training, nessun DR -- economico, <1 minuto totale per le 6 densita'.

Salva un JSON in paper/table_d_data/.
"""
from __future__ import annotations

import os
import sys
import json

_ROOT = "/home/lisa/energy_market_fixed_point/DRcld"
_EXPERIMENTS = os.path.join(_ROOT, "experiments")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EXPERIMENTS)

import numpy as np
import torch
torch.set_num_threads(1)

import config
import market
import equilibrium
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_d_data")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_SEED = {0.2: 3, 0.4: 3, 0.5: 1, 0.6: 3, 0.8: 0, 1.0: 2}  # stessi delle run ufficiali
N_INSTANCES = 200
PCT_LEVEL, PCT_WIDTH = 0.10, 0.04


def run_density(density):
    base_seed = BASE_SEED[density]
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=(0.05, 0.18))
    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    specs, _, n_infeasible_primitive = sample_level_width_perturbed_dataset(
        base_spec, n_instances=N_INSTANCES - 1, seed=777, pct_level=PCT_LEVEL, pct_width=PCT_WIDTH)
    specs = [base_spec] + specs

    n_qp_infeasible = 0
    frac_at_cap, frac_dump, frac_surplus, curtail_med = [], [], [], []
    lam_all = []
    n_buy, n_sell, n_neu = [], [], []
    for spec in specs:
        net = market.build_network(spec, device=torch.device("cpu"))
        diag = equilibrium.solve_equilibrium_diagnostics(net)
        if not diag["feasible"]:
            n_qp_infeasible += 1
            continue
        frac_at_cap.append(diag["frac_at_capacity"])
        frac_dump.append(diag["frac_dumping"])
        frac_surplus.append(diag["frac_surplus_res"])
        curtail_med.append(float(np.median(diag["curtailment"])))
        lam_all.extend(diag["lam"].tolist())
        n_buy.append(diag["n_buyers"]); n_sell.append(diag["n_sellers"]); n_neu.append(diag["n_neutral"])

    n_ok = len(frac_at_cap)
    lam_arr = np.array(lam_all)
    result = {
        "density": density, "base_seed": base_seed, "n_instances": N_INSTANCES,
        "n_infeasible_primitive": n_infeasible_primitive, "n_qp_infeasible": n_qp_infeasible,
        "feasible_rate": n_ok / N_INSTANCES,
        "frac_at_capacity_mean": float(np.mean(frac_at_cap)) if n_ok else None,
        "frac_dumping_mean": float(np.mean(frac_dump)) if n_ok else None,
        "frac_surplus_res_mean": float(np.mean(frac_surplus)) if n_ok else None,
        "lam_mean": float(lam_arr.mean()) if n_ok else None,
        "lam_median": float(np.median(lam_arr)) if n_ok else None,
        "buyers_mean": float(np.mean(n_buy)) if n_ok else None,
        "sellers_mean": float(np.mean(n_sell)) if n_ok else None,
        "neutral_mean": float(np.mean(n_neu)) if n_ok else None,
        "curtailment_median_mean": float(np.mean(curtail_med)) if n_ok else None,
        "curtailment_median_median": float(np.median(curtail_med)) if n_ok else None,
    }
    print(f"density={density}: feasible={n_ok}/{N_INSTANCES} ({n_ok/N_INSTANCES:.1%})  "
          f"buy/sell/neu={np.mean(n_buy):.1f}/{np.mean(n_sell):.1f}/{np.mean(n_neu):.1f}  "
          f"curtail_med={np.median(curtail_med):.1%}  dump={np.mean(frac_dump):.1%}  "
          f"at_cap={np.mean(frac_at_cap):.1%}", flush=True)
    return result


def main():
    results = {}
    for density in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        results[density] = run_density(density)
    out_path = os.path.join(OUT_DIR, "table_d_plausibility.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
