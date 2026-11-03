"""Test-time evaluation shared by DR+GNN, its OOD tests and the benchmarks.

A *pair runner* is `run_pair(spec, method, dr_cfg, device, seed, record)
-> (res_dr, res_method)`: plain DR and the evaluated method on one
instance. `run_pair` below is the DR+GNN one (both start from the same
random x^0); the L2WS benchmarks have their own (warm start instead of x^0).
"""

from __future__ import annotations

import json
import os
import statistics as stats

import numpy as np
import torch

from . import gnn as gnn_mod
from . import market
from .equilibrium import solve_equilibrium_qp
from .fixed_point import run_dr
from .plotting import plot_instance


def run_pair(spec, model, dr_cfg, device, seed, record=False):
    """Plain DR and DR+GNN from the same random initial state x^0 (drawn with
    `seed`)."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = market.build_network(spec, device=device, rng=gen)
    x0 = net.flatten().clone()

    gnn_mod.undeploy(net)
    res_dr = run_dr(net, dr_cfg, record=record)

    net2 = market.build_network(spec, device=device)
    net2.unflatten(x0)
    gnn_mod.deploy(model, net2)
    res_learned = run_dr(net2, dr_cfg, record=record)

    return res_dr, res_learned


def summarize(test_res: dict) -> dict:
    """Speed-up and per-instance iteration reduction (%) statistics."""
    sp = np.asarray(test_res["speedup"])
    dr = np.asarray(test_res["dr"]); learned = np.asarray(test_res["learned"])
    pct = (dr - learned) / dr * 100.0
    return {
        "mean": sp.mean(), "median": np.median(sp), "min": sp.min(), "max": sp.max(),
        "pct_mean": pct.mean(), "pct_median": np.median(pct), "pct_min": pct.min(), "pct_max": pct.max(),
        "win_rate": (sp >= 1.0).mean(), "n_win": int((sp >= 1.0).sum()), "n": len(sp),
    }


def evaluate(specs, method, dr_cfg, device, seed, label, pair_runner=run_pair, method_name="learned",
             records_dir=None):
    """Run `pair_runner` on every spec (instance i uses seed * 7919 + i) and
    print one line per instance plus a summary; the log format is parsed by
    the paper loaders, keep it stable. With `records_dir`, also save the
    per-instance records (`save_instance_record`)."""
    if records_dir is not None:
        os.makedirs(records_dir, exist_ok=True)
    dr_iters, learned_iters, speedups, curves_dr, curves_learned = [], [], [], [], []
    for i, spec in enumerate(specs):
        res_dr, res_learned = pair_runner(spec, method, dr_cfg, device, seed=seed * 7919 + i, record=True)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter); speedups.append(sp)
        curves_dr.append(np.asarray(res_dr.residual)); curves_learned.append(np.asarray(res_learned.residual))
        print(f"  [{label}] {i + 1:2d}/{len(specs)}  DR {res_dr.n_iter:5d} it  |  "
              f"{method_name} {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)
        if records_dir is not None:
            save_instance_record(i, spec, res_dr, res_learned, dr_cfg.tol, device, records_dir)
    print_speedup_summary(label, speedups)
    if records_dir is not None:
        print(f"saved {len(specs)} per-instance diagnostic plots + raw-trajectory JSON files to {records_dir}/")
    return {"dr": dr_iters, "learned": learned_iters, "speedup": speedups,
            "curves_dr": curves_dr, "curves_learned": curves_learned}


def print_speedup_summary(label, speedups):
    n_win = sum(s >= 1.0 for s in speedups)
    print(f"  [{label}] mean speed-up {stats.mean(speedups):.2f}x   median {stats.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {n_win / len(speedups):.0%} "
          f"({n_win}/{len(speedups)})", flush=True)


def relative_gap(cost, j_star):
    """|J(xbar^k) - J*| / |J*| along a cost trajectory (Table 5 definition)."""
    j_denom = max(abs(j_star), 1e-9)
    return (np.abs(np.asarray(cost) - j_star) / j_denom).tolist()


def save_instance_record(i, spec, res_dr, res_learned, tol, device, out_dir):
    """Instance i of a test set: a 4-panel PNG and a JSON with k_DR, k_learned
    and the full residual / shadow-residual / cost / relative-gap / violation
    trajectories of both methods (the raw data behind the paper figures)."""
    net = market.build_network(spec, device=device)
    j_star = solve_equilibrium_qp(net)
    sp = res_dr.n_iter / max(res_learned.n_iter, 1)
    pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
    title = (f"instance {i + 1}  N={spec.N}  DR {res_dr.n_iter} it | learned {res_learned.n_iter} it "
             f"| x{sp:.2f} ({pct:+.1f}%)")
    base = os.path.join(out_dir, f"instance_{i + 1:02d}_N{spec.N}")
    plot_instance(res_dr, res_learned, j_star, tol, title, base + ".png")

    data = {
        "instance": i + 1, "N": spec.N, "j_star": j_star, "tol": tol,
        "k_DR": res_dr.n_iter, "k_learned": res_learned.n_iter,
        "speedup": sp, "pct_reduction": pct,
        "dr": {"residual": res_dr.residual, "shadow_residual": res_dr.shadow_residual,
               "cost": res_dr.cost, "relative_gap": relative_gap(res_dr.cost, j_star),
               "violation": res_dr.violation},
        "learned": {"residual": res_learned.residual, "shadow_residual": res_learned.shadow_residual,
                    "cost": res_learned.cost, "relative_gap": relative_gap(res_learned.cost, j_star),
                    "violation": res_learned.violation},
    }
    with open(base + ".json", "w") as f:
        json.dump(data, f)


def write_notes(out_dir: str, text: str) -> None:
    """NOTES.md describing what a run folder contains."""
    path = os.path.join(out_dir, "NOTES.md")
    with open(path, "w") as f:
        f.write(text.strip() + "\n")
    print(f"saved {path}")
