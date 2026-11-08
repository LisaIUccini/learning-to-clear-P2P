"""Re-evaluation of DR+GNN with two stopping rules (no training: official
checkpoints, same 30 test instances and initial points as the official runs):

    perturbed   : ||x^{k+1} - x^k|| < tol         (official rule, includes v^k)
    unperturbed : ||x_hat^k - x^k|| < tol         (the DR part only, as for plain DR)

Same norm, tolerance and iteration cap as the official evaluation.

    python3 -m scripts.analysis.unperturbed_stop run <out_dir> <density> [<density> ...]
    python3 -m scripts.analysis.unperturbed_stop_report <out_dir>

Results of 2026-10-28 (folder removed): git show ffcb491:results/reeval_unperturbed_stop/20261028_2006/REPORT.md
"""
from __future__ import annotations

import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

from l2p2p import constants as C
from l2p2p import gnn as gnn_mod
from l2p2p import market, official
from l2p2p.equilibrium import solve_equilibrium_qp
from l2p2p.fixed_point import run_dr

DENSITIES = list(official.DENSITIES)
N_TEST = official.N_TEST
METHODS = ["dr", "gnn_perturbed", "gnn_unperturbed"]
DIAG_K = [0, 1, 2, 5, 10]


def _task(args):
    """One (density, instance, method) rollout, from the official x^0 of that instance."""
    density, i, method = args
    torch.set_num_threads(1)
    device = torch.device("cpu")
    _, _, test_specs = official.datasets(density, n_train=1, n_test=i + 1)
    spec = test_specs[i]
    gen = torch.Generator(device=device); gen.manual_seed(C.TEST_SEED * 7919 + i)
    net = market.build_network(spec, device=device, rng=gen)
    x0 = net.flatten().clone()
    if method != "dr":
        net = market.build_network(spec, device=device)
        net.unflatten(x0)
        gnn_mod.deploy(official.load_official_model(density), net)
    x0_used = net.flatten().clone()

    stop_on = "unperturbed" if method == "gnn_unperturbed" else "perturbed"
    cfg = official.dr_config(density, stop_on=stop_on)
    res = run_dr(net, cfg, record=True)

    ref = market.build_network(spec, device=device)
    j_star = float(solve_equilibrium_qp(ref))
    j_den = max(abs(j_star), 1e-9)
    shadow0 = ref.project_reciprocity(x0_used)
    signed = ((np.asarray(res.cost) - j_star) / j_den).tolist()
    return {
        "density": density, "instance": i + 1, "method": method, "gamma": cfg.gamma,
        "lam": cfg.lam, "tol": cfg.tol, "stop_on": stop_on,
        "k": res.n_iter, "converged": res.converged, "j_star": j_star,
        "x0_sum": float(x0_used.double().sum()), "x0_norm": float(torch.linalg.vector_norm(x0_used)),
        "x0": x0_used.tolist(),
        "gap0_signed": float((ref.cost_at(shadow0) - j_star) / j_den),
        "viol0": float(ref.dist_to_Xloc(shadow0)),
        "residual": res.residual, "dr_residual": res.base_residual,
        "signed_gap": signed, "violation": res.violation,
    }


def cmd_run(out_dir, densities):
    os.makedirs(out_dir, exist_ok=True)
    for d in densities:
        path = os.path.join(out_dir, f"trajectories_d{d}.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} already exists -- refusing to overwrite")
    tasks = [(d, i, m) for d in densities for i in range(N_TEST) for m in METHODS]
    # longest first (plain DR) for better load balance
    tasks.sort(key=lambda t: t[2] != "dr")
    with Pool(min(len(tasks), os.cpu_count() - 2)) as pool:
        out = pool.map(_task, tasks, chunksize=1)
    for d in densities:
        rows = sorted([r for r in out if r["density"] == d], key=lambda r: (r["instance"], r["method"]))
        with open(os.path.join(out_dir, f"trajectories_d{d}.json"), "x") as f:
            json.dump(rows, f)
        print(f"saved trajectories_d{d}.json ({len(rows)} rollouts)", flush=True)


# ----------------------------------------------------------------------
# analysis / report
# ----------------------------------------------------------------------
def load_traj(out_dir, d):
    rows = json.load(open(os.path.join(out_dir, f"trajectories_d{d}.json")))
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["method"]] = r
    return by


def load_official(d):
    run_dir = str(official.dr_gnn_dir(d))
    files = sorted(glob.glob(os.path.join(run_dir, "instance_plots", "*.json")))
    return {json.load(open(f))["instance"]: json.load(open(f)) for f in files}


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2], [float(x) for x in sys.argv[3:]])
    else:
        raise SystemExit("usage: run <out_dir> <density> ...")
