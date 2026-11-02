"""Regenerate the official results, stage by stage.

    python3 -m scripts.reproduce paper                 # figures and tables from the saved results (~1 min)
    python3 -m scripts.reproduce inference             # re-evaluate every saved model (hours)
    python3 -m scripts.reproduce training              # retrain everything, then evaluate (days)
    python3 -m scripts.reproduce paper-data            # extra evaluations read by some figures/tables
    python3 -m scripts.reproduce training --dry-run    # only list the commands

Each stage overwrites the corresponding folders of results/official/.
Commands run one after the other; the list printed by --dry-run can also be
run in parallel by hand.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from l2p2p import official, paths
from l2p2p.l2ws.training import LOSSES
from scripts.eval_ood import SHIFTS
from scripts.train_l2ws import VARIANTS


def commands(stage: str):
    py = [sys.executable, "-m"]
    densities = [str(d) for d in official.DENSITIES]
    if stage == "paper":
        return [[sys.executable, str(paths.PAPER / "make_all.py")]]
    if stage == "paper-data":
        return [py + ["scripts.paper_data", c]
                for c in ("trajectories", "plausibility", "timing", "train-time", "stopping-quality")]
    eval_only = ["--eval-only"] if stage == "inference" else []
    cmds = [py + ["scripts.train_dr_gnn", "--density", d] + eval_only for d in densities]
    cmds += [py + ["scripts.eval_ood", "--density", d, "--shift", s] for d in densities for s in SHIFTS]
    cmds += [py + ["scripts.train_l2ws", "--density", d, "--variant", v, "--loss", l] + eval_only
             for d in densities for v in VARIANTS for l in LOSSES]
    if stage == "training":
        cmds += [py + ["scripts.run_a2dr", "calibrate", "--density", d] for d in densities]
    cmds += [py + ["scripts.run_a2dr", "compare", "--density", d] for d in densities]
    return cmds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("paper", "inference", "training", "paper-data"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for cmd in commands(args.stage):
        print("python3 " + " ".join(cmd[1:]), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=paths.ROOT)


if __name__ == "__main__":
    main()
