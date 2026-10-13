"""
build_paper_data.py
==========================================

Parses the live.log files of the per-density "best checkpoint" training runs
and their zero-shot OOD (level+width) test runs, and consolidates everything
into ONE structured JSON file (runs/cross_density/paper_plots/data.json).

Why: the paper plots (% improvement vs density, in-distribution + OOD curves
at several perturbation intensities, with 95% CI bands) may be re-styled
several times. Re-parsing scattered live.log files by hand each time is slow
and error-prone. This script is the single source of truth: it keeps the
FULL per-instance (dr_iters, learned_iters, speedup, pct_reduction) arrays
(not just summary stats), so any future plotting/CI choice can be recomputed
from the JSON without re-running experiments.

Usage: python3 build_paper_data.py [density10_variant] [out_path]
       density10_variant: "v21" (default, current champion, seed=2) or
       "v17" (the original seed=0 checkpoint) -- selects which density=1.0
       checkpoint+logs to use; all other densities are unaffected.
       out_path: defaults to runs/cross_density/paper_plots/data.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import statistics as stats

OUT_DIR = "runs/cross_density/paper_plots"
OUT_PATH = os.path.join(OUT_DIR, "data.json")

# per-instance line format (both demo_train_* and demo_zeroshot_levelwidth_v4):
#   [label]  i/N  DR  XXX it  |  learned  YYY it  |  xS.SS (+PP.P%)
LINE_RE = re.compile(
    r"DR\s+(\d+)\s*it\s*\|\s*learned\s+(\d+)\s*it\s*\|\s*x\s*([\d.]+)\s*\(([+-][\d.]+)%\)"
)

# density=1.0 has two checkpoint variants -- selected via CLI arg (default v21).
DENSITY10_VARIANTS = {
    "v21": {
        "checkpoint": "runs/cross_density/combined_density10_v21_seed2_wbehind10/phi_w.pt",
        "home_log": "runs/cross_density/combined_density10_v21_seed2_wbehind10/live.log",
        "hyperparams": {"dr_gamma": 4.0, "unroll": 680, "gnn_rho": 0.9853, "gnn_scale": 1.8,
                         "gnn_rho_slow": 0.99, "gnn_scale_slow": 0.4, "w_ahead": 1.6, "w_behind": 10.0,
                         "omega_mode": "dr_slope", "base_seed": 2},
        "ood_logs": {
            "level20": ("runs/cross_density/zeroshot_density10_v21_level20_resample/live.log", 0.20, 0.04),
            "level30": ("runs/cross_density/zeroshot_density10_v21_level30_resample/live.log", 0.30, 0.04),
            "level40": ("runs/cross_density/zeroshot_density10_v21_level40_resample/live.log", 0.40, 0.04),
            "level50": ("runs/cross_density/zeroshot_density10_v21_level50_resample/live.log", 0.50, 0.04),
        },
    },
    "v17": {
        "checkpoint": "runs/cross_density/combined_v17_wahead16_s18/phi_w.pt",
        "home_log": "runs/cross_density/combined_v17_wahead16_s18/live.log",
        "hyperparams": {"dr_gamma": 4.0, "unroll": 386, "gnn_rho": 0.9741, "gnn_scale": 1.8,
                         "gnn_rho_slow": 0.99, "gnn_scale_slow": 0.4, "w_ahead": 1.6, "w_behind": 6.0,
                         "omega_mode": "dr_slope", "base_seed": 0},
        "ood_logs": {
            "level20": ("runs/cross_density/zeroshot_density10_level20/live.log", 0.20, 0.04),
            "level30": ("runs/cross_density/zeroshot_density10_level30/live.log", 0.30, 0.04),
            "level40": ("runs/cross_density/zeroshot_density10_level40/live.log", 0.40, 0.04),
            "level50": ("runs/cross_density/zeroshot_density10_level50/live.log", 0.50, 0.04),
        },
    },
}

# (density_key) -> checkpoint path + hyperparams + home (in-distribution) training log
CHECKPOINTS = {
    "1.0": {},  # filled in main() from DENSITY10_VARIANTS[variant]
    "0.5": {
        # supervisor-driven pivot dropped further 0.5 work mid-sweep (see memory); this is just
        # the best candidate found before that pivot -- home (in-distribution) only, no OOD runs.
        "checkpoint": "runs/cross_density/combined_density05_v25_seed1/phi_w.pt",
        "home_log": "runs/cross_density/combined_density05_v25_seed1/live.log",
        "hyperparams": {"dr_gamma": 6.0, "unroll": 395, "gnn_rho": 0.9747, "gnn_scale": 1.0,
                         "gnn_rho_slow": 0.99, "gnn_scale_slow": 0.0, "w_ahead": 1.0, "w_behind": 2.0,
                         "omega_mode": "fixed", "omega_power": 1.5, "base_seed": 1},
    },
    "0.8": {
        "checkpoint": "runs/cross_density/combined_density08_v1/phi_w.pt",
        "home_log": "runs/cross_density/combined_density08_v1/live.log",
        "hyperparams": {"dr_gamma": 4.0, "unroll": 685, "gnn_rho": 0.9854, "gnn_scale": 1.8,
                         "gnn_rho_slow": 0.99, "gnn_scale_slow": 0.4, "w_ahead": 1.6, "w_behind": 6.0,
                         "omega_mode": "dr_slope", "base_seed": 0},
    },
    "0.6": {
        "checkpoint": "runs/cross_density/combined_density06_v9_mild/phi_w.pt",
        "home_log": "runs/cross_density/combined_density06_v9_mild/live.log",
        "hyperparams": {"dr_gamma": 4.0, "unroll": 321, "gnn_rho": 0.9688, "gnn_scale": 1.0,
                         "gnn_rho_slow": 0.99, "gnn_scale_slow": 0.0, "w_ahead": 1.0, "w_behind": 2.0,
                         "omega_mode": "fixed", "omega_power": 1.5, "base_seed": 3},
    },
    "0.4": {
        "checkpoint": "runs/cross_density/combined_density04_v2_omegapower15/phi_w.pt",
        "home_log": "runs/cross_density/combined_density04_v2_omegapower15/live.log",
        "hyperparams": {"dr_gamma": 4.0, "unroll": 318, "gnn_rho": 0.9686, "gnn_scale": 1.8,
                         "gnn_rho_slow": 0.99, "gnn_scale_slow": 0.4, "w_ahead": 1.6, "w_behind": 6.0,
                         "omega_mode": "fixed", "omega_power": 1.5, "base_seed": 3},
    },
    "0.2": {
        "checkpoint": "runs/cross_density/combined_density02_v2_mild/phi_w.pt",
        "home_log": "runs/cross_density/combined_density02_v2_mild/live.log",
        "hyperparams": {"dr_gamma": 4.0, "unroll": 426, "gnn_rho": 0.9765, "gnn_scale": 1.0,
                         "gnn_rho_slow": 0.99, "gnn_scale_slow": 0.0, "w_ahead": 1.0, "w_behind": 2.0,
                         "omega_mode": "fixed", "omega_power": 1.5, "base_seed": 3},
    },
}

# (density_key, condition_key) -> (zero-shot test log, pct_level, pct_width)
# "home" is NOT listed here -- it comes from each checkpoint's own training log (last 30 lines).
OOD_LOGS = {
    "1.0": {},  # filled in main() from DENSITY10_VARIANTS[variant]["ood_logs"]
    "0.8": {
        "level20": ("runs/cross_density/zeroshot_density08_level20/live.log", 0.20, 0.04),
        "level30": ("runs/cross_density/zeroshot_density08_level30/live.log", 0.30, 0.04),
        "level40": ("runs/cross_density/zeroshot_density08_level40/live.log", 0.40, 0.04),
        "level50": ("runs/cross_density/zeroshot_density08_level50/live.log", 0.50, 0.04),
    },
    "0.6": {
        "level20": ("runs/cross_density/zeroshot_density06_v2_level20/live.log", 0.20, 0.04),
        "level30": ("runs/cross_density/zeroshot_density06_v2_level30/live.log", 0.30, 0.04),
        "level40": ("runs/cross_density/zeroshot_density06_v2_level40/live.log", 0.40, 0.04),
        "level50": ("runs/cross_density/zeroshot_density06_v2_level50/live.log", 0.50, 0.04),
    },
    "0.4": {
        "level20": ("runs/cross_density/zeroshot_density04_level20/live.log", 0.20, 0.04),
        "level30": ("runs/cross_density/zeroshot_density04_level30/live.log", 0.30, 0.04),
        "level40": ("runs/cross_density/zeroshot_density04_level40/live.log", 0.40, 0.04),
        "level50": ("runs/cross_density/zeroshot_density04_level50/live.log", 0.50, 0.04),
    },
    "0.5": {
        "level20": ("runs/cross_density/zeroshot_density05_v25_level20_resample/live.log", 0.20, 0.04),
        "level30": ("runs/cross_density/zeroshot_density05_v25_level30_resample/live.log", 0.30, 0.04),
        "level40": ("runs/cross_density/zeroshot_density05_v25_level40_resample/live.log", 0.40, 0.04),
        "level50": ("runs/cross_density/zeroshot_density05_v25_level50_resample/live.log", 0.50, 0.04),
    },
    "0.2": {
        "level20": ("runs/cross_density/zeroshot_density02_level20/live.log", 0.20, 0.04),
        "level30": ("runs/cross_density/zeroshot_density02_level30/live.log", 0.30, 0.04),
        "level40": ("runs/cross_density/zeroshot_density02_level40/live.log", 0.40, 0.04),
        "level50": ("runs/cross_density/zeroshot_density02_level50/live.log", 0.50, 0.04),
    },
}


# SUPPORTING diagnostic, not the headline result -- see NODE_PM1_LOGS below
# for the primary topology-generalization condition used in the paper.
# (density_key) -> {"remove1": log_path, "remove2": log_path} -- node-removal
# zero-shot tests (topology generalization: same base market, 1 or 2 nodes
# deleted, no retraining, n=10 each -- see demo_zeroshot_node_removal.py).
# Runs archived to runs/_archive/session_topology_generalization_support/.
NODE_REMOVAL_LOGS = {
    "1.0": {"remove1": "runs/_archive/session_topology_generalization_support/node_removal/density10_remove1/live.log",
            "remove2": "runs/_archive/session_topology_generalization_support/node_removal/density10_remove2/live.log"},
    "0.8": {"remove1": "runs/_archive/session_topology_generalization_support/node_removal/density08_remove1/live.log",
            "remove2": "runs/_archive/session_topology_generalization_support/node_removal/density08_remove2/live.log"},
    "0.6": {"remove1": "runs/_archive/session_topology_generalization_support/node_removal/density06_remove1/live.log",
            "remove2": "runs/_archive/session_topology_generalization_support/node_removal/density06_remove2/live.log"},
    "0.5": {"remove1": "runs/_archive/session_topology_generalization_support/node_removal/density05_remove1/live.log",
            "remove2": "runs/_archive/session_topology_generalization_support/node_removal/density05_remove2/live.log"},
    "0.4": {"remove1": "runs/_archive/session_topology_generalization_support/node_removal/density04_remove1/live.log",
            "remove2": "runs/_archive/session_topology_generalization_support/node_removal/density04_remove2/live.log"},
    "0.2": {"remove1": "runs/_archive/session_topology_generalization_support/node_removal/density02_remove1/live.log",
            "remove2": "runs/_archive/session_topology_generalization_support/node_removal/density02_remove2/live.log"},
}

# PRIMARY topology-generalization result for the paper (most realistic
# scenario -- a prosumer joining/leaving the market -- largest n, and the
# mean/median gap itself is informative: typical-case robustness is decent,
# a tail of occasional severe failures drags the mean down).
# (density_key) -> log_path -- +/-1 node zero-shot test (each of n=30
# instances independently adds OR removes one node, pooled -- see
# demo_zeroshot_node_pm1.py).
NODE_PM1_LOGS = {
    "1.0": "runs/cross_density/node_pm1/density10/live.log",
    "0.8": "runs/cross_density/node_pm1/density08/live.log",
    "0.6": "runs/cross_density/node_pm1/density06/live.log",
    "0.5": "runs/cross_density/node_pm1/density05/live.log",
    "0.4": "runs/cross_density/node_pm1/density04/live.log",
    "0.2": "runs/cross_density/node_pm1/density02/live.log",
}

# SUPPORTING diagnostic, not the headline result -- see NODE_PM1_LOGS above.
# (density_key) -> log_path -- degree-preserving edge-rewire zero-shot test
# (N and every node's degree unchanged, only the specific wiring differs,
# n=15 -- see demo_zeroshot_node_rewire.py). Not applicable to density=1.0
# (the complete graph has nothing to rewire).
# Runs archived to runs/_archive/session_topology_generalization_support/.
NODE_REWIRE_LOGS = {
    "0.8": "runs/_archive/session_topology_generalization_support/node_rewire/density08/live.log",
    "0.6": "runs/_archive/session_topology_generalization_support/node_rewire/density06/live.log",
    "0.5": "runs/_archive/session_topology_generalization_support/node_rewire/density05/live.log",
    "0.4": "runs/_archive/session_topology_generalization_support/node_rewire/density04/live.log",
    "0.2": "runs/_archive/session_topology_generalization_support/node_rewire/density02/live.log",
}


def parse_instances(log_path: str, n_expected: int | None = None, tail_only: bool = False):
    """Returns a list of {dr, learned, speedup, pct} dicts parsed from a live.log.

    If tail_only, only the LAST contiguous run of matching lines is kept (used
    for training logs, where the final test-set evaluation block is what we
    want, not any earlier per-epoch validation prints -- though in practice
    these training scripts only print per-instance lines once, at final test).
    """
    if not os.path.exists(log_path):
        return None, f"missing: {log_path}"
    with open(log_path) as f:
        lines = f.readlines()
    matches = []
    for ln in lines:
        m = LINE_RE.search(ln)
        if m:
            dr_it, learned_it, sp, pct = m.groups()
            matches.append({"dr": int(dr_it), "learned": int(learned_it),
                             "speedup": float(sp), "pct": float(pct)})
    if tail_only and matches:
        pass  # only one block expected in these logs; keep all
    if n_expected is not None and len(matches) != n_expected:
        return matches, f"WARNING: expected {n_expected} instances, parsed {len(matches)} from {log_path}"
    return matches, None


def summarize(per_instance: list[dict]) -> dict:
    sp = [r["speedup"] for r in per_instance]
    pct = [r["pct"] for r in per_instance]
    n = len(pct)
    mean_pct = stats.mean(pct)
    # 95% CI via normal approximation on the per-instance pct values
    if n > 1:
        sd = stats.stdev(pct)
        half_width = 1.96 * sd / (n ** 0.5)
    else:
        half_width = 0.0
    return {
        "n": n,
        "speedup_mean": stats.mean(sp), "speedup_median": stats.median(sp),
        "speedup_min": min(sp), "speedup_max": max(sp),
        "pct_mean": mean_pct, "pct_median": stats.median(pct),
        "pct_min": min(pct), "pct_max": max(pct),
        "pct_ci95_lo": mean_pct - half_width, "pct_ci95_hi": mean_pct + half_width,
        "win_rate": sum(s >= 1.0 for s in sp) / n,
        "n_win": sum(s >= 1.0 for s in sp),
    }


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "v21"
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUT_PATH
    v = DENSITY10_VARIANTS[variant]
    CHECKPOINTS["1.0"] = {"checkpoint": v["checkpoint"], "home_log": v["home_log"], "hyperparams": v["hyperparams"]}
    OOD_LOGS["1.0"] = v["ood_logs"]

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "generated_by": "build_paper_data.py",
        "notes": ("Per-instance DR vs learned iteration counts, parsed from each run's live.log. "
                  "'home' = in-distribution test set (+/-10% level, +/-4% width, same base market "
                  "the checkpoint was trained on). 'levelNN' = zero-shot OOD test, same base market, "
                  "perturbation widened to +/-NN% level (width held at the training convention, "
                  "+/-4%), fresh random x0 per instance, n=30, inference only (no retraining)."),
        "densities": {},
        "warnings": [],
    }

    for dkey, info in CHECKPOINTS.items():
        entry = {
            "checkpoint": info["checkpoint"],
            "hyperparams": info["hyperparams"],
            "conditions": {},
        }

        # home (in-distribution): last 30 per-instance lines of the training log
        matches, warn = parse_instances(info["home_log"], n_expected=30)
        if warn:
            out["warnings"].append(warn)
        if matches:
            home_instances = matches[-30:]
            entry["conditions"]["home"] = {
                "pct_level": 0.10, "pct_width": 0.04,
                "source_log": info["home_log"],
                "per_instance": home_instances,
                "summary": summarize(home_instances),
            }

        # OOD zero-shot tests
        for ckey, (log_path, pct_level, pct_width) in OOD_LOGS.get(dkey, {}).items():
            matches, warn = parse_instances(log_path, n_expected=30)
            if warn:
                out["warnings"].append(warn)
            if matches:
                entry["conditions"][ckey] = {
                    "pct_level": pct_level, "pct_width": pct_width,
                    "source_log": log_path,
                    "per_instance": matches,
                    "summary": summarize(matches),
                }

        # node-removal zero-shot tests (topology generalization, n=10)
        for ckey, log_path in NODE_REMOVAL_LOGS.get(dkey, {}).items():
            matches, warn = parse_instances(log_path, n_expected=10)
            if warn:
                out["warnings"].append(warn)
            if matches:
                entry["conditions"][ckey] = {
                    "source_log": log_path,
                    "per_instance": matches,
                    "summary": summarize(matches),
                }

        # +/-1 node zero-shot test (n=30, pooled add+remove)
        if dkey in NODE_PM1_LOGS:
            matches, warn = parse_instances(NODE_PM1_LOGS[dkey], n_expected=30)
            if warn:
                out["warnings"].append(warn)
            if matches:
                entry["conditions"]["pm1"] = {
                    "source_log": NODE_PM1_LOGS[dkey],
                    "per_instance": matches,
                    "summary": summarize(matches),
                }

        # degree-preserving edge-rewire zero-shot test (n=15, not density=1.0)
        if dkey in NODE_REWIRE_LOGS:
            matches, warn = parse_instances(NODE_REWIRE_LOGS[dkey], n_expected=15)
            if warn:
                out["warnings"].append(warn)
            if matches:
                entry["conditions"]["rewire"] = {
                    "source_log": NODE_REWIRE_LOGS[dkey],
                    "per_instance": matches,
                    "summary": summarize(matches),
                }

        out["densities"][dkey] = entry

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {out_path}")
    if out["warnings"]:
        print("\nWARNINGS:")
        for w in out["warnings"]:
            print(f"  {w}")
    print("\nSummary table:")
    for dkey, entry in out["densities"].items():
        for ckey, cond in entry["conditions"].items():
            s = cond["summary"]
            print(f"  density={dkey:4s} {ckey:7s} n={s['n']:3d}  "
                  f"speedup={s['speedup_mean']:.2f}x  pct_mean={s['pct_mean']:6.1f}%  "
                  f"pct_median={s['pct_median']:6.1f}%  CI95=[{s['pct_ci95_lo']:.1f}, {s['pct_ci95_hi']:.1f}]  "
                  f"win_rate={s['win_rate']:.0%}")


if __name__ == "__main__":
    main()
