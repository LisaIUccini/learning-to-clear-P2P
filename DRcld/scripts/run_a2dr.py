"""Anderson-accelerated DR (A2DR, Fu et al. 2020) benchmark.

    python3 -m scripts.run_a2dr calibrate --density 0.8   # (eta, mmax) grid on validation instances
    python3 -m scripts.run_a2dr compare --density 0.8     # DR vs A2DR vs DR+GNN on the test set

calibrate writes results/official/a2dr/calibration/d{density}_eta{eta}_mmax{mmax}.json;
compare (with the official eta, mmax of l2p2p/official.py) writes
results/official/a2dr/d{density}/results.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as stats
import time

import torch

from l2p2p import constants as C
from l2p2p import gnn as gnn_mod
from l2p2p import market, official, paths
from l2p2p.a2dr import A2DRConfig, run_a2dr
from l2p2p.fixed_point import run_dr
from l2p2p.runlog import tee_stdout
from l2p2p.scenarios import sample_level_width_dataset

# Validation instances and initial points used only for the A2DR calibration,
# disjoint from the train/val/test seeds.
CALIBRATION_SPEC_SEED = C.TRAIN_SEED + 777
CALIBRATION_X0_SEED = C.TEST_SEED + 5000
N_CALIBRATION = 10


def eta_tag(eta: float) -> str:
    """1e-4 -> "1e-4", 3e-5 -> "3e-5" (calibration file names)."""
    mantissa, exponent = f"{eta:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def a2dr_config(density, eta, mmax) -> A2DRConfig:
    return A2DRConfig(gamma=official.DENSITY[density].gamma, lam=C.DR_LAMBDA, n_iter=C.DR_MAX_ITER,
                      tol=C.DR_TOLERANCE, bisect=C.DR_BISECT_STEPS, mmax=mmax, eta=eta)


def graph_diameter(neighbors: dict) -> int:
    diam = 0
    for src in neighbors:
        dist = {src: 0}
        q = collections.deque([src])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        diam = max(diam, max(dist.values()))
    return diam


def speedup_summary(label, sps):
    win_rate = sum(s >= 1.0 for s in sps) / len(sps)
    print(f"  [{label}] mean speed-up {stats.mean(sps):.2f}x   median {stats.median(sps):.2f}x   "
          f"min {min(sps):.2f}x   max {max(sps):.2f}x   "
          f"win-rate {win_rate:.0%} ({sum(s >= 1.0 for s in sps)}/{len(sps)})", flush=True)
    return {"mean": stats.mean(sps), "median": stats.median(sps), "min": min(sps),
            "max": max(sps), "win_rate": win_rate}


def calibrate(density, eta, mmax, out_path, n_val=N_CALIBRATION):
    """DR vs A2DR(eta, mmax) on calibration instances; saves the speed-up statistics."""
    device = torch.device("cpu")
    dr_cfg = official.dr_config(density)
    a2_cfg = a2dr_config(density, eta, mmax)
    specs, _ = sample_level_width_dataset(official.base_market(density), n_val, seed=CALIBRATION_SPEC_SEED,
                                       pct_level=official.ID_PCT_LEVEL, pct_width=official.ID_PCT_WIDTH)
    print(f"calibrating eta={eta:g} mmax={mmax} on density={density}  (n_val={n_val})", flush=True)
    dr_iters, a2_iters, converged = [], [], []
    for i, spec in enumerate(specs):
        gen = torch.Generator(device=device); gen.manual_seed(CALIBRATION_X0_SEED * 7919 + i)
        net = market.build_network(spec, device=device, rng=gen)
        x0 = net.flatten().clone()
        res_dr = run_dr(net, dr_cfg, record=False)
        net_a2 = market.build_network(spec, device=device)
        net_a2.unflatten(x0)
        res_a2 = run_a2dr(net_a2, a2_cfg, record=False)
        dr_iters.append(res_dr.n_iter); a2_iters.append(res_a2.n_iter)
        converged.append(bool(res_dr.converged and res_a2.converged))
        print(f"  {i + 1:2d}/{n_val}  DR {res_dr.n_iter:5d} it  |  A2DR {res_a2.n_iter:5d} it  "
              f"x{res_dr.n_iter / max(res_a2.n_iter, 1):.2f}"
              f"{'' if res_a2.converged else '  [A2DR DID NOT CONVERGE]'}", flush=True)
    sp = [d / max(a, 1) for d, a in zip(dr_iters, a2_iters)]
    result = {
        "density": density, "eta": eta, "mmax": mmax, "n_val": n_val,
        "mean_speedup": stats.mean(sp), "median_speedup": stats.median(sp),
        "min_speedup": min(sp), "max_speedup": max(sp), "win_rate": sum(s >= 1.0 for s in sp) / len(sp),
        "all_converged": all(converged), "n_nonconverged": sum(not c for c in converged),
        "dr_iters": dr_iters, "a2dr_iters": a2_iters,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  -> mean={result['mean_speedup']:.2f}x  median={result['median_speedup']:.2f}x  "
          f"win_rate={result['win_rate']:.0%}  all_converged={result['all_converged']}", flush=True)
    return result


def compare(density, out_dir, n_test=official.N_TEST, eta=None, mmax=None):
    """DR, A2DR and DR+GNN from the same random x^0 on every test instance;
    iteration counts, communication rounds and wall-clock times."""
    d = official.DENSITY[density]
    eta = d.a2dr_eta if eta is None else eta
    mmax = d.a2dr_mmax if mmax is None else mmax
    device = torch.device("cpu")
    dr_cfg = official.dr_config(density)
    a2_cfg = a2dr_config(density, eta, mmax)
    _, _, test_specs = official.datasets(density, n_train=1, n_test=n_test)
    diameter = graph_diameter(test_specs[0].neighbors)
    # one plain-DR round + a gather and a broadcast for the global inner products
    rounds_per_iter = 1 + 2 * diameter
    model = official.load_official_model(density)
    print(f"A2DR vs DR+GNN: density={density}  gamma={d.gamma}  eta={eta}  mmax={mmax}  "
          f"n_test={n_test}  diameter={diameter}  rounds/iteration={rounds_per_iter}", flush=True)

    dr_iters, a2_iters, l_iters, dr_time, a2_time, l_time = [], [], [], [], [], []
    for i, spec in enumerate(test_specs):
        gen = torch.Generator(device=device); gen.manual_seed(C.TEST_SEED * 7919 + i)
        net = market.build_network(spec, device=device, rng=gen)
        x0 = net.flatten().clone()
        gnn_mod.undeploy(net)
        t0 = time.perf_counter(); res_dr = run_dr(net, dr_cfg, record=False); t_dr = time.perf_counter() - t0
        net_a2 = market.build_network(spec, device=device)
        net_a2.unflatten(x0)
        t0 = time.perf_counter(); res_a2 = run_a2dr(net_a2, a2_cfg, record=False); t_a2 = time.perf_counter() - t0
        net_l = market.build_network(spec, device=device)
        net_l.unflatten(x0)
        gnn_mod.deploy(model, net_l)
        t0 = time.perf_counter(); res_l = run_dr(net_l, dr_cfg, record=False); t_l = time.perf_counter() - t0

        dr_iters.append(res_dr.n_iter); a2_iters.append(res_a2.n_iter); l_iters.append(res_l.n_iter)
        dr_time.append(t_dr); a2_time.append(t_a2); l_time.append(t_l)
        flag = "" if (res_dr.converged and res_a2.converged and res_l.converged) else "  [DID NOT CONVERGE]"
        print(f"  {i + 1:2d}/{n_test}  DR {res_dr.n_iter:5d} it ({t_dr:6.2f}s)  |  "
              f"A2DR {res_a2.n_iter:5d} it ({t_a2:6.2f}s)  |  GNN {res_l.n_iter:5d} it ({t_l:6.2f}s){flag}",
              flush=True)

    sp_a2 = [a / max(b, 1) for a, b in zip(dr_iters, a2_iters)]
    sp_l = [a / max(b, 1) for a, b in zip(dr_iters, l_iters)]
    rsp_a2 = [a / max(b * rounds_per_iter, 1) for a, b in zip(dr_iters, a2_iters)]
    tsp_a2 = [a / max(b, 1e-9) for a, b in zip(dr_time, a2_time)]
    tsp_l = [a / max(b, 1e-9) for a, b in zip(dr_time, l_time)]
    print(flush=True)
    results = {
        "n_test": n_test, "density": density, "base_seed": d.base_seed, "dr_gamma": d.gamma,
        "a2dr_eta": eta, "a2dr_mmax": mmax,
        "trading_graph_diameter": diameter, "a2dr_rounds_per_iter": rounds_per_iter,
        "dr_iters": dr_iters, "a2dr_iters": a2_iters, "learned_iters": l_iters,
        "dr_time_s": dr_time, "a2dr_time_s": a2_time, "learned_time_s": l_time,
        "dr_total_rounds": dr_iters, "a2dr_total_rounds": [a * rounds_per_iter for a in a2_iters],
        "learned_total_rounds": l_iters,
        "a2dr_speedup_iters": sp_a2, "a2dr_speedup_rounds": rsp_a2, "a2dr_speedup_time": tsp_a2,
        "learned_speedup_iters": sp_l, "learned_speedup_time": tsp_l,
        "a2dr_summary_iters": speedup_summary("A2DR -- iterations", sp_a2),
        "a2dr_summary_rounds": speedup_summary("A2DR -- communication rounds", rsp_a2),
        "a2dr_summary_time": speedup_summary("A2DR -- wall-clock time", tsp_a2),
        "learned_summary_iters": speedup_summary("DR + GNN -- iterations", sp_l),
        "learned_summary_time": speedup_summary("DR + GNN -- wall-clock time", tsp_l),
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {os.path.join(out_dir, 'results.json')}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("calibrate", "compare"))
    ap.add_argument("--density", type=float, required=True, choices=official.DENSITIES)
    ap.add_argument("--out", help="output folder (default: the official one)")
    ap.add_argument("--n", type=int, help="number of instances (default: 10 calibrate, 30 compare)")
    ap.add_argument("--eta", type=float, help="calibrate a single eta instead of the whole grid")
    ap.add_argument("--mmax", type=int, help="calibrate a single mmax instead of the whole grid")
    args = ap.parse_args()

    if args.command == "calibrate":
        out_dir = args.out or str(paths.A2DR / "calibration")
        os.makedirs(out_dir, exist_ok=True)
        etas = [args.eta] if args.eta is not None else official.A2DR_ETA_GRID
        mmaxs = [args.mmax] if args.mmax is not None else official.A2DR_MMAX_GRID
        for eta in etas:
            for mmax in mmaxs:
                name = f"{paths.density_tag(args.density)}_eta{eta_tag(eta)}_mmax{mmax}.json"
                calibrate(args.density, eta, mmax, os.path.join(out_dir, name), n_val=args.n or N_CALIBRATION)
    else:
        out_dir = args.out or str(paths.A2DR / paths.density_tag(args.density))
        os.makedirs(out_dir, exist_ok=True)
        with tee_stdout(os.path.join(out_dir, "compare.log")):
            compare(args.density, out_dir, n_test=args.n or official.N_TEST)


if __name__ == "__main__":
    main()
