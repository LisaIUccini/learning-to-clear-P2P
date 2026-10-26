"""
sanity_check_calibrated_market.py
==========================================

Pre-training sanity check for the recalibrated market generator
(`build_calibrated_market.py`): for each density, build ONE calibrated
base market, draw ~200 level+width-perturbed instances from it (same
+/-10%/+/-4% convention as training), solve the EXACT centralised QP for
each (`equilibrium.solve_equilibrium_diagnostics`), and report the 6
equilibrium statistics the calibration is checked against:

  1. feasible-instance rate (Assumption 1)         target: >95%
  2. fraction of trades at full capacity            target: <10-20%
  3. "dumping" (unabsorbed export S_n>1e-4 while
     G_n also above its own floor)                  target: ~0%
     vs. "RES surplus" (same S_n>1e-4 but
     G_n ~ G_min_n -- legitimate, not economic waste)
  4. nodal price (dual lambda_n) range               target: ~0.05-0.3 EUR/kWh
  5. buyer/seller mix                                target: >=1/3 each type
  6. demand curtailment (D*-D)/D*                    target: moderate (median <~20%)

No training. Only the QP (fast: a 10-node convex QP solves in
milliseconds). Pure diagnostic -- prints a table, does not change
`b_tilde`/class ranges itself (re-run by hand after adjusting
`build_calibrated_market.py`'s constants if a density fails a target).

Usage: python3 sanity_check_calibrated_market.py [n_instances=200]
       [densities=0.2,0.4,0.5,0.6,0.8,1.0] [base_seed=0] [pct_level=0.10]
       [pct_width=0.04]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

import numpy as np
import torch

import market
import equilibrium
from build_calibrated_market import sample_calibrated_base_market, CalibratedConfig
from build_realistic_training_dataset import sample_level_width_perturbed_dataset


def run_density(density: float, base_seed: int, n_instances: int, pct_level: float, pct_width: float):
    cfg = CalibratedConfig(density=density)
    neighbors, base_spec, tier_names = sample_calibrated_base_market(base_seed, cfg)
    specs, log_rows, n_infeasible_primitive = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_instances - 1, seed=777, pct_level=pct_level, pct_width=pct_width)
    specs = [base_spec] + specs

    n_qp_infeasible = 0
    frac_at_cap, frac_dump, frac_surplus, curtail_med = [], [], [], []
    lam_all = []
    n_buy, n_sell, n_neu = [], [], []
    for spec in specs:
        net = market.build_network(spec, device=torch.device("cpu"))
        # uses equilibrium.py's own default solver (CLARABEL as of
        # 2026-10-14, see that module's docstring for why) -- no override
        # needed here anymore.
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
    n_total_attempted = n_instances + n_infeasible_primitive  # primitive-level discards already excluded from `specs`
    feasible_rate = n_ok / n_instances if n_instances else float("nan")

    print(f"\n=== density={density} ===  (base tiers: "
          f"low={tier_names.count('low')} medium={tier_names.count('medium')} high={tier_names.count('high')})")
    print(f"  instances requested={n_instances}  primitive-level infeasible discarded={n_infeasible_primitive}  "
          f"QP-infeasible among the rest={n_qp_infeasible}")
    print(f"  1. feasible-instance rate: {n_ok}/{n_instances} = {n_ok/n_instances:.1%}  (target >95%)")
    if n_ok == 0:
        print("  -- no feasible instances, cannot report further stats --")
        return
    print(f"  2. frac trades at full capacity: mean={np.mean(frac_at_cap):.1%}  "
          f"(target <10-20% -- UNRELIABLE at tau=0 when the graph has cycles: individual "
          f"edge Q's are non-unique, this solver-dependent number is NOT a real diagnostic, "
          f"see equilibrium.py's solve_equilibrium_diagnostics docstring. Kept tau=0 project-wide "
          f"per paper Sec.4/Thm.6, which don't require equilibrium uniqueness -- 2026-10-14 decision.)")
    print(f"  3. frac 'dumping' (unabsorbed export S_n>1e-4 AND G_n>G_min_n+1e-4): "
          f"mean={np.mean(frac_dump):.1%}  (target ~0%)")
    print(f"     frac 'RES surplus' (same S_n>1e-4 but G_n~G_min_n, legitimate): "
          f"mean={np.mean(frac_surplus):.1%}")
    lam_arr = np.array(lam_all)
    print(f"  4. nodal price lambda_n: mean={lam_arr.mean():.4f}  median={np.median(lam_arr):.4f}  "
          f"p5={np.percentile(lam_arr,5):.4f}  p95={np.percentile(lam_arr,95):.4f}  "
          f"(target ~0.05-0.3 EUR/kWh)")
    print(f"  5. buyer/seller mix (avg per instance, N=10): buyers={np.mean(n_buy):.1f}  "
          f"sellers={np.mean(n_sell):.1f}  neutral={np.mean(n_neu):.1f}  "
          f"(target >=1/3 each ~ >=3.3)")
    print(f"  6. demand curtailment (D*-D)/D*, per-instance median: "
          f"mean={np.mean(curtail_med):.1%}  median={np.median(curtail_med):.1%}  "
          f"(target median <~20%)")


def main():
    n_instances = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    densities = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    base_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    pct_level = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
    pct_width = float(sys.argv[5]) if len(sys.argv) > 5 else 0.04

    print(f"sanity check  n_instances={n_instances}  densities={densities}  base_seed={base_seed}  "
          f"pct_level={pct_level}  pct_width={pct_width}", flush=True)
    for d in densities:
        run_density(d, base_seed, n_instances, pct_level, pct_width)


if __name__ == "__main__":
    main()
