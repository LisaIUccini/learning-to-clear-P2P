"""Shared by the paper figure/table scripts: constants, method styles and
loaders of the saved results (results/official/; nothing is recomputed).

Importing this module also sets the base matplotlib style; scripts that
call `style.apply()` afterwards override part of it.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from l2p2p import paths                                     # noqa: E402
from l2p2p.official import DENSITIES                        # noqa: E402

# L2P2P_PAPER_OUT redirects the outputs (figures/ and tables/ inside it), e.g. for tests.
_OUT = os.environ.get("L2P2P_PAPER_OUT")
FIGURES = os.path.join(_OUT, "figures") if _OUT else str(paths.PAPER_FIGURES)
TABLES = os.path.join(_OUT, "tables") if _OUT else str(paths.PAPER_TABLES)
PAPER_DATA = str(paths.PAPER_DATA)

# Assumed PMLR text width (no LaTeX installation here to measure the class).
TEXTWIDTH_IN = 6.00

METHODS = {
    "dr":              dict(name="DR",                        color="#000000", marker="o", filled=True,  kind="baseline"),
    "learned":         dict(name="Learned (ours)",             color="#0072B2", marker="o", filled=True,  kind="distributed"),
    "l2ws_reg_dist":   dict(name="L2WS-reg (dist.)",           color="#E69F00", marker="s", filled=False, kind="distributed"),
    "l2ws_asym_dist":  dict(name="L2WS-asym (dist.)",          color="#009E73", marker="^", filled=False, kind="distributed"),
    "l2ws_reg_cent":   dict(name="L2WS-reg (cent.)$^\\dagger$",  color="#E69F00", marker="s", filled=True,  kind="centralized"),
    "l2ws_asym_cent":  dict(name="L2WS-asym (cent.)$^\\dagger$", color="#009E73", marker="^", filled=True,  kind="centralized"),
    "a2dr":            dict(name="A2DR$^\\dagger$",             color="#CC79A7", marker="D", filled=True,  kind="centralized"),
}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "savefig.dpi": 200,
})


def delta_pct(dr_iters, m_iters):
    """Per-instance iteration reduction w.r.t. DR, in %."""
    dr = np.asarray(dr_iters, dtype=float)
    m = np.asarray(m_iters, dtype=float)
    return (dr - m) / dr * 100.0


def _instance_records(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "instance_plots", "*.json")))
    return [json.load(open(f)) for f in files]


def load_official(density):
    """DR+GNN run: per-instance records, first line of train.log and its slow-channel line."""
    run_dir = str(paths.DR_GNN / paths.density_tag(density))
    if not os.path.isdir(run_dir):
        return None
    log_lines = open(os.path.join(run_dir, "train.log")).readlines()
    slow_line = next((l for l in log_lines if "slow channel" in l), "")
    return {"dir": run_dir, "instances": _instance_records(run_dir), "header": log_lines[0],
            "slow_line": slow_line}


def load_l2ws_cent(density, loss):
    """Centralized L2WS: per-instance records; "timing" holds the iteration counts."""
    run_dir = str(paths.L2WS_CENTRALIZED / f"{paths.density_tag(density)}_{loss}")
    if not os.path.isdir(run_dir):
        return None
    insts = _instance_records(run_dir)
    timing = {"dr_iters": [i["k_DR"] for i in insts], "l2ws_iters": [i["k_learned"] for i in insts]}
    return {"dir": run_dir, "timing": timing, "instances": insts}


_LOG_RE_DIST = re.compile(r"DR\s+(\d+)\s+it\s+\|\s+L2WS\s+(\d+)\s+it")


def load_l2ws_dist(density, loss):
    """Distributed L2WS: iteration counts parsed from train.log, plus its L rounds."""
    run_dir = str(paths.L2WS_DISTRIBUTED / f"{paths.density_tag(density)}_{loss}")
    log_path = os.path.join(run_dir, "train.log")
    if not os.path.exists(log_path):
        return None
    text = open(log_path).read()
    n_rounds = re.search(r"n_rounds=(\d+)", text.splitlines()[0])
    matches = _LOG_RE_DIST.findall(text)
    if not (matches and n_rounds):
        return None
    return {"dir": run_dir, "dr_iters": [int(a) for a, _ in matches],
            "l2ws_post_iters": [int(b) for _, b in matches], "L": int(n_rounds.group(1))}


def load_a2dr(density):
    rfile = paths.A2DR / paths.density_tag(density) / "results.json"
    if not rfile.exists():
        return None
    return {"dir": str(rfile.parent), "data": json.load(open(rfile))}


_LOG_RE_OOD30 = re.compile(r"\[ood\]\s*(\d+)/30\s+DR\s+(\d+)\s+it\s+\|\s+learned\s+(\d+)\s+it")
_LOG_RE_PM1 = re.compile(r"\[(add|remove)\s*\]\s+N=(\d+)\s+DR\s+(\d+)\s+it\s+\|\s+learned\s+(\d+)\s+it")


def _ood_log(density, shift):
    log_path = paths.OOD / f"{paths.density_tag(density)}_{shift}" / "live.log"
    return open(log_path).read() if log_path.exists() else None


def load_ood_level(density, level):
    """Level shift ("level20", "level30", "level40"): iteration counts from live.log."""
    text = _ood_log(density, level)
    matches = _LOG_RE_OOD30.findall(text) if text else []
    if not matches:
        return None
    return {"dr_iters": [int(b) for _, b, _ in matches], "learned_iters": [int(c) for _, _, c in matches]}


def load_ood_pm1(density):
    """One prosumer more or less: iteration counts and "add"/"remove" per instance."""
    text = _ood_log(density, "pm1")
    matches = _LOG_RE_PM1.findall(text) if text else []
    if not matches:
        return None
    return {"dr_iters": [int(c) for _, _, c, _ in matches], "learned_iters": [int(d) for _, _, _, d in matches],
            "kinds": [k for k, _, _, _ in matches]}
