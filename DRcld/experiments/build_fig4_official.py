"""
build_fig4_official.py
==========================================

Recreates build_paper_plots.py's `fig4_blocks_faceted_median_iqr_pm1_
paperstyle_with05.png` (faceted by density, median+IQR block markers per
condition, Computer Modern paper style) but sourced from THIS session's
own "official" replicate pipeline (train_official_pilot.py + the
official_ood zero-shot runs) instead of the older runs/cross_density/
checkpoints that figure was originally built from -- same visual style,
reused directly from build_paper_plots.py (`COND_COLOR`, `COND_LABEL`,
`plot_blocks_faceted_median_iqr_paperstyle`), new data.

Five conditions per panel, one panel per density (0.2/0.4/0.5/0.6/0.8/1.0):
  home     -- in-distribution, last 30 "DR XXX it | learned YYY it | ..."
              lines of runs/official/official_N10_d<density>_*/train.log
  level20/30/40 -- runs/official_ood/zeroshot_d<density>_level<NN>/live.log
  pm1      -- +/-1 node topology generalization,
              runs/official_ood/pm1_d<density>/live.log

Usage: python3 build_fig4_official.py [out_name]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import re
import sys

import build_paper_plots as bpp

LINE_RE = re.compile(
    r"DR\s+(\d+)\s*it\s*\|\s*learned\s+(\d+)\s*it\s*\|\s*x\s*([\d.]+)\s*\(([+-][\d.]+)%\)"
)

DENSITIES = ["0.2", "0.4", "0.5", "0.6", "0.8", "1.0"]

HOME_LOG = {
    "0.2": "runs/official/official_N10_d0.2_seed3_replicate/train.log",
    "0.4": "runs/official/official_N10_d0.4_seed3_replicate/train.log",
    "0.5": "runs/official/official_N10_d0.5_seed1_replicate/train.log",
    "0.6": "runs/official/official_N10_d0.6_seed3_replicate/train.log",
    "0.8": "runs/official/official_N10_d0.8_seed0_replicate_fixed/train.log",
    "1.0": "runs/official/official_N10_d1.0_seed2_replicate/train.log",
}


def ood_log(dkey, cond):
    if cond == "pm1":
        return f"runs/official_ood/pm1_d{dkey}/live.log"
    return f"runs/official_ood/zeroshot_d{dkey}_{cond}/live.log"


def parse_last_n(log_path, n=30):
    if not _os.path.exists(log_path):
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
    if len(matches) < n:
        return matches, f"WARNING: expected >= {n}, parsed {len(matches)} from {log_path}"
    return matches[-n:], None


def build_data():
    data = {"densities": {}}
    warnings = []
    for dkey in DENSITIES:
        conditions = {}
        home, warn = parse_last_n(HOME_LOG[dkey], n=30)
        if warn:
            warnings.append(warn)
        if home and len(home) == 30:
            conditions["home"] = {"per_instance": home}

        for cond in ("level20", "level30", "level40", "pm1"):
            instances, warn = parse_last_n(ood_log(dkey, cond), n=30)
            if warn:
                warnings.append(warn)
            if instances and len(instances) == 30:
                conditions[cond] = {"per_instance": instances}

        data["densities"][dkey] = {"conditions": conditions}
    return data, warnings


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "fig4_blocks_faceted_median_iqr_pm1_paperstyle_official.png"
    data, warnings = build_data()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")

    bpp.DENSITIES = DENSITIES
    bpp.OUT_DIR = "runs/paper_plots"
    bpp.SUFFIX = ""

    print("\nSummary (median % reduction, IQR):")
    for dkey in DENSITIES:
        for cond in ("home", "level20", "level30", "level40", "pm1"):
            c = data["densities"][dkey]["conditions"].get(cond)
            if not c:
                print(f"  density={dkey:4s} {cond:8s}  MISSING")
                continue
            import numpy as np
            pct = np.array([r["pct"] for r in c["per_instance"]], dtype=float)
            print(f"  density={dkey:4s} {cond:8s} n={len(pct):2d}  median={np.median(pct):6.1f}%  "
                  f"IQR=[{np.percentile(pct,25):5.1f}, {np.percentile(pct,75):5.1f}]")

    bpp.plot_blocks_faceted_median_iqr_paperstyle(data, ["home", "level20", "level30", "level40", "pm1"], out_name)


if __name__ == "__main__":
    main()
