#!/usr/bin/env python3
"""
reeval_unperturbed_stop.py
============================
Rivalutazione (NESSUN training: checkpoint ufficiali in sola lettura) di
DR+GNN con due criteri di arresto, sulle stesse 30 istanze di test e gli
stessi x0 della valutazione ufficiale (train_official_pilot.py ->
diagnostics.save_instance_plots -> demo.run_pair):

    perturbed   : ||x^{k+1}-x^k|| < tol        (criterio ufficiale, include v^k)
    unperturbed : ||lam*(z^k - xtilde^k)|| < tol (solo parte DR, come per DR puro)

Stessa norma (euclidea, non normalizzata, somma dei contributi locali),
tol = config.DR_TOLERANCE, cap = config.DR_MAX_ITER, gamma letto dal
train.log del run ufficiale, lam = config.DR_LAMBDA.

Usage:
    python3 reeval_unperturbed_stop.py run <out_dir> <density> [<density> ...]
    python3 reeval_unperturbed_stop.py report <out_dir>
"""
from __future__ import annotations

import os
import sys
import csv
import glob
import json
import re
from multiprocessing import Pool

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPERIMENTS = os.path.join(_ROOT, "experiments")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EXPERIMENTS)

import numpy as np
import torch
torch.set_num_threads(1)

import config
import market
import algorithm
import gnn as gnn_mod
from equilibrium import solve_equilibrium_qp
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

RUN_DIR = {
    0.2: "official_N10_d0.2_seed3_replicate",
    0.4: "official_N10_d0.4_seed3_replicate",
    0.5: "official_N10_d0.5_seed1_replicate",
    0.6: "official_N10_d0.6_seed3_replicate",
    0.8: "official_N10_d0.8_seed0_replicate_fixed",
    1.0: "official_N10_d1.0_seed2_replicate",
}
DENSITIES = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
N_TEST = 30
C_OLD_RANGE = (0.05, 0.18)     # train_official_pilot.py default, generator="old"
METHODS = ["dr", "gnn_perturbed", "gnn_unperturbed"]
DIAG_K = [0, 1, 2, 5, 10]


def run_info(density):
    run_dir = os.path.join(_EXPERIMENTS, "runs", "official", RUN_DIR[density])
    header = open(os.path.join(run_dir, "train.log")).readline()
    gamma = float(re.search(r"gamma\*=([\d.]+)", header).group(1))
    base_seed = int(re.search(r"base_seed=(\d+)", header).group(1))
    assert "n_test=30" in header
    return run_dir, gamma, base_seed


def test_specs_for(density, base_seed):
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=C_OLD_RANGE)
    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    specs, _, _ = sample_level_width_perturbed_dataset(
        base_spec, n_instances=N_TEST - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
    return [base_spec] + specs


def load_model(run_dir):
    blob = torch.load(os.path.join(run_dir, "phi_w.pt"), map_location="cpu", weights_only=False)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig(**blob["cfg"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def _task(args):
    """One (density, instance, method) rollout -- same x0 construction as demo.run_pair."""
    density, i, method = args
    device = torch.device("cpu")
    run_dir, gamma, base_seed = run_info(density)
    spec = test_specs_for(density, base_seed)[i]
    seed = config.TEST_SEED * 7919 + i
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = market.build_network(spec, device=device, rng=gen)
    x0 = net.flatten().clone()
    if method != "dr":
        net = market.build_network(spec, device=device)
        net.unflatten(x0)
        gnn_mod.deploy(load_model(run_dir), net)
    else:
        gnn_mod.undeploy(net)
    x0_used = net.flatten().clone()

    stop_on = "unperturbed" if method == "gnn_unperturbed" else "perturbed"
    cfg = algorithm.DRConfig(gamma=gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                              tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS, stop_on=stop_on)
    res = algorithm.run_dr(net, cfg, record=True)

    ref = market.build_network(spec, device=device)
    j_star = float(solve_equilibrium_qp(ref))
    j_den = max(abs(j_star), 1e-9)
    shadow0 = ref.project_reciprocity(x0_used)
    signed = ((np.asarray(res.cost) - j_star) / j_den).tolist()
    return {
        "density": density, "instance": i + 1, "method": method, "gamma": gamma,
        "lam": config.DR_LAMBDA, "tol": cfg.tol, "stop_on": stop_on,
        "k": res.n_iter, "converged": res.converged, "j_star": j_star,
        "x0_sum": float(x0_used.double().sum()), "x0_norm": float(torch.linalg.vector_norm(x0_used)),
        "x0": x0_used.tolist(),
        "gap0_signed": float((ref.cost_at(shadow0) - j_star) / j_den),
        "viol0": float(ref.dist_to_Xloc(shadow0)),
        "residual": res.residual, "dr_residual": res.dr_residual,
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
    run_dir = os.path.join(_EXPERIMENTS, "runs", "official", RUN_DIR[d])
    files = sorted(glob.glob(os.path.join(run_dir, "instance_plots", "*.json")))
    return {json.load(open(f))["instance"]: json.load(open(f)) for f in files}


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2], [float(x) for x in sys.argv[3:]])
    else:
        raise SystemExit("usage: run <out_dir> <density> ...")
