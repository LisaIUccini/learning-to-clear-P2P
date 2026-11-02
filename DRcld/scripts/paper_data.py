"""Extra evaluations read by some paper figures/tables (no training except
`train-time`). Each command writes one or more JSON files into
results/official/paper_data/ (or --out).

    python3 -m scripts.paper_data trajectories      # fig2_data/: A2DR and distributed-L2WS trajectories (gap figures)
    python3 -m scripts.paper_data plausibility      # table_d_data/: market plausibility statistics (Table D)
    python3 -m scripts.paper_data timing            # table_t4_data/: ms per agent and iteration (Table 2)
    python3 -m scripts.paper_data train-time        # table_t4_data/: training time per base market (Table 2)
    python3 -m scripts.paper_data stopping-quality  # table_ek_data/: ||z^k - xtilde^k|| at the stop (Table 5)

Timings are wall-clock measurements (single thread) and are not reproducible
bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time

import numpy as np
import torch

from l2p2p import constants as C
from l2p2p import equilibrium, gnn as gnn_mod, market, official, paths
from l2p2p.a2dr import run_a2dr
from l2p2p.evaluation import relative_gap, run_pair
from l2p2p.fixed_point import run_dr
from l2p2p.l2ws.distributed_model import DistributedPredictor
from l2p2p.scenarios import sample_level_width_dataset
from scripts.run_a2dr import a2dr_config
from scripts.train_dr_gnn import train_model
from scripts.train_l2ws import CHECKPOINT, output_dir

DEVICE = torch.device("cpu")


def _write(out_dir, name, data, indent=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
    print(f"saved {path}", flush=True)


def trajectories(out_dir, densities, n_test):
    """Per test instance: A2DR from the official x^0, and plain DR warm-started
    by the distributed L2WS (regression and asymmetric losses)."""
    losses = ("regression", "asymmetric")
    for density in densities:
        d = official.DENSITY[density]
        _, _, test_specs = official.datasets(density, n_train=1, n_test=n_test)
        dr_cfg = official.dr_config(density)
        a2_cfg = a2dr_config(density, d.a2dr_eta, d.a2dr_mmax)
        predictors = {loss: DistributedPredictor.load(output_dir("distributed", density, loss)
                                                      / CHECKPOINT["distributed"]) for loss in losses}
        a2dr_insts, l2ws_insts = [], {loss: [] for loss in losses}
        for i, spec in enumerate(test_specs):
            j_star = equilibrium.solve_equilibrium_qp(market.build_network(spec, device=DEVICE))
            gen = torch.Generator(device=DEVICE); gen.manual_seed(C.TEST_SEED * 7919 + i)
            x0 = market.build_network(spec, device=DEVICE, rng=gen).flatten().clone()
            net_a2 = market.build_network(spec, device=DEVICE)
            net_a2.unflatten(x0)
            res = run_a2dr(net_a2, a2_cfg, record=True)
            a2dr_insts.append({"j_star": j_star, "n_iter": res.n_iter, "converged": res.converged,
                               "cost": res.cost, "violation": res.violation, "residual": res.residual,
                               "relative_gap": relative_gap(res.cost, j_star)})
            msg = f"  d{density} {i + 1:2d}/{len(test_specs)}  A2DR {res.n_iter:5d} it  |"
            for loss, predictor in predictors.items():
                net_l = market.build_network(spec, device=DEVICE)
                net_l.unflatten(predictor.predict(spec, differentiable=False))
                res = run_dr(net_l, dr_cfg, record=True)
                L = predictor.gnn.cfg.n_rounds
                l2ws_insts[loss].append({"j_star": j_star, "n_iter": res.n_iter, "converged": res.converged,
                                         "L": L, "cost": res.cost, "violation": res.violation,
                                         "residual": res.residual, "relative_gap": relative_gap(res.cost, j_star)})
                msg += f"  L2WS-{loss[:3]}-dist(+L) {res.n_iter + L:5d} it"
            print(msg, flush=True)
        out = {"density": density, "base_seed": d.base_seed, "n_test": len(test_specs),
               "l2ws_dist": {loss: {"loss": loss, "L": predictors[loss].gnn.cfg.n_rounds,
                                    "instances": l2ws_insts[loss]} for loss in losses},
               "a2dr": {"eta": d.a2dr_eta, "mmax": d.a2dr_mmax, "gamma": d.gamma, "instances": a2dr_insts}}
        _write(out_dir, f"fig2_extra_trajectories_d{density}.json", out)


def plausibility(out_dir, densities, n_instances=200, seed=777):
    """Equilibrium statistics (QP, no DR) over `n_instances` in-distribution instances."""
    results = {}
    for density in densities:
        base = official.base_market(density)
        specs, n_discarded = sample_level_width_dataset(base, n_instances, seed=seed,
                                                        pct_level=official.ID_PCT_LEVEL,
                                                        pct_width=official.ID_PCT_WIDTH)
        n_qp_infeasible = 0
        frac_at_cap, frac_dump, frac_surplus, curtail_med, lam_all = [], [], [], [], []
        n_buy, n_sell, n_neu = [], [], []
        for spec in specs:
            diag = equilibrium.solve_equilibrium_diagnostics(market.build_network(spec, device=DEVICE))
            if not diag["feasible"]:
                n_qp_infeasible += 1
                continue
            frac_at_cap.append(diag["frac_at_capacity"]); frac_dump.append(diag["frac_dumping"])
            frac_surplus.append(diag["frac_surplus_res"])
            curtail_med.append(float(np.median(diag["curtailment"])))
            lam_all.extend(diag["lam"].tolist())
            n_buy.append(diag["n_buyers"]); n_sell.append(diag["n_sellers"]); n_neu.append(diag["n_neutral"])
        n_ok = len(frac_at_cap)
        lam = np.array(lam_all)
        results[density] = {
            "density": density, "base_seed": official.DENSITY[density].base_seed, "n_instances": n_instances,
            "n_infeasible_primitive": n_discarded, "n_qp_infeasible": n_qp_infeasible,
            "feasible_rate": n_ok / n_instances,
            "frac_at_capacity_mean": float(np.mean(frac_at_cap)) if n_ok else None,
            "frac_dumping_mean": float(np.mean(frac_dump)) if n_ok else None,
            "frac_surplus_res_mean": float(np.mean(frac_surplus)) if n_ok else None,
            "lam_mean": float(lam.mean()) if n_ok else None,
            "lam_median": float(np.median(lam)) if n_ok else None,
            "buyers_mean": float(np.mean(n_buy)) if n_ok else None,
            "sellers_mean": float(np.mean(n_sell)) if n_ok else None,
            "neutral_mean": float(np.mean(n_neu)) if n_ok else None,
            "curtailment_median_mean": float(np.mean(curtail_med)) if n_ok else None,
            "curtailment_median_median": float(np.median(curtail_med)) if n_ok else None,
        }
        print(f"density={density}: feasible={n_ok}/{n_instances}", flush=True)
    _write(out_dir, "table_d_plausibility.json", results, indent=2)


def _timed(fn, sink):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        sink.append(time.perf_counter() - t0)
        return out
    return wrapper


def timing(out_dir, densities, n_instances=5):
    """Mean time per agent and iteration of the local proximal step and of
    Phi_w (state construction + forward pass), timed inside the real DR+GNN loop."""
    torch.set_num_threads(1)
    results = {}
    for density in densities:
        _, _, test_specs = official.datasets(density, n_train=1, n_test=n_instances)
        model = official.load_official_model(density)
        dr_cfg = official.dr_config(density)
        prox_t, state_t, forward_t = [], [], []
        for i, spec in enumerate(test_specs):
            gen = torch.Generator(device=DEVICE); gen.manual_seed(C.TEST_SEED * 7919 + i)
            net = market.build_network(spec, device=DEVICE, rng=gen)
            gnn_mod.deploy(model, net)
            agent = net.prosumers[0].gnn_agent
            agent.perturbation = _timed(agent.perturbation, forward_t)
            agent.gnn.local_state = _timed(model.local_state, state_t)
            for p in net.prosumers:
                p.local_prox = _timed(p.local_prox, prox_t)
            run_dr(net, dr_cfg, record=False)
            del agent.gnn.local_state     # restore the class method on the shared model
        prox_ms = float(np.mean(prox_t)) * 1000
        gnn_ms = (float(np.mean(state_t)) + float(np.mean(forward_t))) * 1000
        results[density] = {"prox_ms_per_agent_iter": prox_ms, "gnn_ms_per_agent_iter": gnn_ms,
                            "overhead_pct": gnn_ms / prox_ms * 100, "n_prox_samples": len(prox_t),
                            "n_gnn_samples": len(forward_t)}
        print(f"density={density}: prox={prox_ms:.4f}ms  gnn={gnn_ms:.4f}ms", flush=True)
    _write(out_dir, "table_t4_timing.json", {"hardware": platform.processor() or platform.machine(),
                                             "platform": platform.platform(), "n_threads": torch.get_num_threads(),
                                             "dtype": "float32", "results": results}, indent=2)


def train_time(out_dir, densities, timed_epochs=3):
    """Official training timed over `timed_epochs` epochs and extrapolated to all of them."""
    results = {}
    for density in densities:
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            train_model(density, tmp, epochs=timed_epochs)
            elapsed = time.perf_counter() - t0
        full_min = elapsed / timed_epochs * official.EPOCHS / 60.0
        results[density] = {"timed_epochs": timed_epochs, "timed_seconds": elapsed,
                            "extrapolated_25ep_minutes": full_min, "unroll_K": official.DENSITY[density].unroll}
        print(f"density={density}: {timed_epochs} epochs in {elapsed:.1f}s -> {full_min:.1f} min", flush=True)
    _write(out_dir, "table_t4_train_time.json", {"hardware": platform.processor() or platform.machine(),
                                                 "platform": platform.platform(),
                                                 "n_threads": torch.get_num_threads(),
                                                 "timed_epochs": timed_epochs, "full_epochs": official.EPOCHS,
                                                 "results": results}, indent=2)


def stopping_quality(out_dir, densities, n_test):
    """||lam (z^k - xtilde^k)|| (the DR part of the last step) when DR and
    DR+GNN stop, per test instance."""
    results = {}
    for density in densities:
        _, _, test_specs = official.datasets(density, n_train=1, n_test=n_test)
        model = official.load_official_model(density)
        ek_dr, ek_learned = [], []
        for i, spec in enumerate(test_specs):
            res_dr, res_l = run_pair(spec, model, official.dr_config(density), DEVICE,
                                     seed=C.TEST_SEED * 7919 + i, record=True)
            ek_dr.append(res_dr.base_residual[-1]); ek_learned.append(res_l.base_residual[-1])
        results[density] = {"ek_dr_median": float(np.median(ek_dr)),
                            "ek_learned_median": float(np.median(ek_learned)),
                            "ek_dr_all": ek_dr, "ek_learned_all": ek_learned}
        print(f"density={density}: median e^k DR={np.median(ek_dr):.3e}  DR+GNN={np.median(ek_learned):.3e}",
              flush=True)
    _write(out_dir, "table_ek_comparison.json", results, indent=2)


COMMANDS = {"trajectories": "fig2_data", "plausibility": "table_d_data", "timing": "table_t4_data",
            "train-time": "table_t4_data", "stopping-quality": "table_ek_data"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--density", type=float, nargs="*", default=list(official.DENSITIES))
    ap.add_argument("--n", type=int, help="instances per density (default: the official number)")
    ap.add_argument("--out", help="output folder (default: the official one)")
    args = ap.parse_args()
    out_dir = args.out or str(paths.PAPER_DATA / COMMANDS[args.command])
    if args.command == "trajectories":
        trajectories(out_dir, args.density, args.n or official.N_TEST)
    elif args.command == "plausibility":
        plausibility(out_dir, args.density, args.n or 200)
    elif args.command == "timing":
        timing(out_dir, args.density, args.n or 5)
    elif args.command == "train-time":
        train_time(out_dir, args.density)
    else:
        stopping_quality(out_dir, args.density, args.n or official.N_TEST)


if __name__ == "__main__":
    main()
