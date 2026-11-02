#!/usr/bin/env python3
"""
reeval_f64_exact_prox.py
==========================
Rivalutazione (NESSUN training: checkpoint ufficiali in sola lettura, pesi
float32 convertiti a double) di DR e DR+GNN con tre configurazioni numeriche,
sulle stesse 30 istanze e gli stessi x0 della valutazione ufficiale:

    f32_bis30 : float32, bisezione 30 passi (ufficiale, controllo)
    f64_bis30 : float64, bisezione 30 passi
    f64_exact : float64, prox esatto (radice della funzione di bilancio lineare
                a tratti, trovata sui breakpoint, senza iterazioni)

x0 e' sempre quello float32 ufficiale (cast esatto a double). Nelle
configurazioni f64 i parametri sono i double originali di MarketSpec, che
market.build_network invece tronca a float32.

Usage:
    python3 reeval_f64_exact_prox.py selftest
    python3 reeval_f64_exact_prox.py run <out_dir> <density> [<density> ...]
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reeval_unperturbed_stop import (N_TEST, config, gnn_mod, load_model, market, run_info,
                                     solve_equilibrium_qp, test_specs_for)

import numpy as np
import torch

import algorithm
import network as network_mod

CONFIGS = {"f32_bis30": (torch.float32, False), "f64_bis30": (torch.float64, False),
           "f64_exact": (torch.float64, True)}
METHODS = ["dr", "gnn_perturbed", "gnn_unperturbed"]
N_WORKERS = 8
_BISECTION_PROX = network_mod.Prosumer.local_prox


def exact_local_prox(self, gamma, rD, rG, rq, n_bisect=0, tau=0.0):
    """Stesso problema di Prosumer.local_prox (con tau=0): la funzione
    bal(nu) = D(nu) - G(nu) - G_res - sum q(nu) e' lineare a tratti e non
    decrescente, quindi la radice si ottiene valutandola sui breakpoint e
    interpolando sul segmento che la contiene."""
    assert tau == 0.0
    f = lambda v: float(v)
    g = float(gamma)
    denomD, denomG = 2.0 * g * f(self.a_tilde) + 1.0, g * f(self.a) + 1.0
    numD, numG = 2.0 * g * f(self.a_tilde) * f(self.D_star) + f(rD), f(rG) - g * f(self.b)
    Dmin, Dmax, Gmin, Gmax = f(self.D_min), f(self.D_max), f(self.G_min), f(self.G_max)
    q_const = np.array([f(rq[m]) - g * f(self.c[m]) for m in self.neighbors])
    kappa = np.array([f(self.k[m]) for m in self.neighbors])

    def primal(nu):
        D = np.clip((numD + nu) / denomD, Dmin, Dmax)
        G = np.clip((numG - nu) / denomG, Gmin, Gmax)
        return D, G, np.minimum(q_const - nu, kappa)

    def bal(nu):
        D, G, q = primal(nu)
        return D - G - f(self.G_res) - q.sum()

    bp = np.sort(np.concatenate([[Dmin * denomD - numD, Dmax * denomD - numD,
                                  numG - Gmax * denomG, numG - Gmin * denomG], q_const - kappa]))
    vals = np.array([bal(nu) for nu in bp])
    i = int(np.argmax(vals >= 0.0)) if (vals >= 0.0).any() else len(bp)
    if i == len(bp):              # oltre l'ultimo breakpoint: pendenza = |Omega_n|
        nu = bp[-1] - vals[-1] / len(self.neighbors)
    elif i == 0 or vals[i] == vals[i - 1]:
        nu = bp[i]
    else:
        nu = bp[i - 1] - vals[i - 1] * (bp[i] - bp[i - 1]) / (vals[i] - vals[i - 1])
    D, G, q = primal(nu)
    t = lambda v: torch.tensor(float(v), dtype=self.D.dtype, device=self.device)
    return t(D), t(G), {m: t(q[j]) for j, m in enumerate(self.neighbors)}


def build_network_f64(spec, device):
    """market.build_network senza il troncamento a float32 (x0 lo imposta il chiamante)."""
    t = lambda v: torch.tensor(float(v), dtype=torch.float64, device=device)
    prosumers = [network_mod.Prosumer(
        node_id=n, neighbors=spec.neighbors[n], device=device,
        D_min=t(spec.D_min[n]), D_max=t(spec.D_max[n]), G_min=t(spec.G_min[n]), G_max=t(spec.G_max[n]),
        G_res=t(spec.G_res[n]), a_tilde=t(spec.a_tilde[n]), a=t(spec.a[n]), b=t(spec.b[n]),
        D_star=t(spec.D_star[n]), c={m: t(spec.c[(m, n)]) for m in spec.neighbors[n]},
        k={m: t(spec.kappa[(m, n)]) for m in spec.neighbors[n]})
        for n in range(spec.N)]
    return network_mod.Network(prosumers, spec.neighbors, device=device)


def selftest():
    """Prox esatto vs bisezione a 60 passi su punti casuali di tutte le istanze d0.2."""
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    _, gamma, base_seed = run_info(0.2)
    rng = np.random.default_rng(0)
    worst_x, worst_bal, n = 0.0, 0.0, 0
    for spec in test_specs_for(0.2, base_seed):
        net = build_network_f64(spec, device)
        for p in net.prosumers:
            for _ in range(200):
                scale = rng.choice([0.5, 5.0, 50.0])
                rD = torch.tensor(rng.normal(0, scale)); rG = torch.tensor(rng.normal(0, scale))
                rq = {m: torch.tensor(rng.normal(0, scale)) for m in p.neighbors}
                a = exact_local_prox(p, gamma, rD, rG, rq)
                b = _BISECTION_PROX(p, gamma, rD, rG, rq, n_bisect=60)
                diff = max(abs(float(a[0] - b[0])), abs(float(a[1] - b[1])),
                           max(abs(float(a[2][m] - b[2][m])) for m in p.neighbors))
                resid = abs(float(a[0] - a[1] - p.G_res - sum(a[2].values())))
                worst_x, worst_bal, n = max(worst_x, diff), max(worst_bal, resid), n + 1
    print(f"{n} prox: max|esatto-bisezione60|={worst_x:.2e}  max|bilancio esatto|={worst_bal:.2e}")
    assert worst_x < 1e-8 and worst_bal < 1e-10


def _task(args):
    density, i, method, cname = args
    dtype, exact = CONFIGS[cname]
    device = torch.device("cpu")
    run_dir, gamma, base_seed = run_info(density)
    spec = test_specs_for(density, base_seed)[i]

    # x0 ufficiale: stessa costruzione di demo.run_pair, sempre in float32
    torch.set_default_dtype(torch.float32)
    gen = torch.Generator(device=device); gen.manual_seed(config.TEST_SEED * 7919 + i)
    x0 = market.build_network(spec, device=device, rng=gen).flatten().clone()

    torch.set_default_dtype(dtype)
    network_mod.Prosumer.local_prox = exact_local_prox if exact else _BISECTION_PROX
    build = (lambda: build_network_f64(spec, device)) if dtype == torch.float64 else (
        lambda: market.build_network(spec, device=device))
    net = build()
    net.unflatten(x0.to(dtype))
    if method == "dr":
        gnn_mod.undeploy(net)
    else:
        gnn_mod.deploy(load_model(run_dir).to(dtype), net)

    stop_on = "unperturbed" if method == "gnn_unperturbed" else "perturbed"
    cfg = algorithm.DRConfig(gamma=gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                              tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS, stop_on=stop_on)
    res = algorithm.run_dr(net, cfg, record=True)

    ref = build()
    j_star = float(solve_equilibrium_qp(ref))
    gap = (np.abs(np.asarray(res.cost) - j_star) / max(abs(j_star), 1e-9)).tolist()
    return {"density": density, "instance": i + 1, "method": method, "config": cname,
            "k": res.n_iter, "converged": res.converged, "j_star": j_star, "residual": res.residual,
            "dr_residual": res.dr_residual, "gap": gap, "violation": res.violation}


def cmd_run(out_dir, densities):
    os.makedirs(out_dir, exist_ok=True)
    for d in densities:
        path = os.path.join(out_dir, f"trajectories_d{d}.json")
        if os.path.exists(path):
            raise SystemExit(f"{path} already exists -- refusing to overwrite")
    tasks = [(d, i, m, c) for d in densities for i in range(N_TEST) for m in METHODS for c in CONFIGS]
    tasks.sort(key=lambda t: t[2] != "dr")
    with Pool(N_WORKERS, maxtasksperchild=1) as pool:
        out = pool.map(_task, tasks, chunksize=1)
    for d in densities:
        rows = sorted([r for r in out if r["density"] == d],
                      key=lambda r: (r["instance"], r["method"], r["config"]))
        with open(os.path.join(out_dir, f"trajectories_d{d}.json"), "x") as f:
            json.dump(rows, f)
        print(f"saved trajectories_d{d}.json ({len(rows)} rollouts)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "run":
        cmd_run(sys.argv[2], [float(x) for x in sys.argv[3:]])
    else:
        raise SystemExit("usage: selftest | run <out_dir> <density> ...")
