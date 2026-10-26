"""
time_l2ws_parallel_equivalent.py
==========================================

Parallel-equivalent (critical-path) timing for L2WS, companion to
a2dr_benchmark/time_parallel_equivalent.py (Plain DR / DR+GNN). Same
per-prosumer dr_step() timing, max-over-nodes-per-iteration model for the
post-warm-start plain-DR phase -- but L2WS's warm-start MLP call itself is
NOT node-parallel: `build_theta`/`L2WSPredictor.predict` reads every
node's/edge's private data into ONE global vector fed through ONE
monolithic MLP (model.py), a single centralised operation with no
per-node decomposition to take a max over. So L2WS's parallel-equivalent
time is:

    t_mlp (centralised, paid in FULL, no parallel speedup available)
    + sum over post-warm-start DR iterations of max_p time(p.dr_step())

vs. its sequential-equivalent time (the plain sum, matching
time_l2ws_vs_dr.py's own dr_time_s/l2ws_time_s, used here only as a
cross-check):

    t_mlp + sum over post-warm-start DR iterations of sum_p time(p.dr_step())

Usage: python3 time_l2ws_parallel_equivalent.py <density> <base_seed>
       <dr_gamma> <checkpoint_path> [n_test=30] [c_old_range=0.05,0.18]
       [out_path]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import json
import os
import statistics as stats
import sys
import time

import torch
torch.set_num_threads(1)

import config
import market
import algorithm
from model import WarmStartMLP, WarmStartConfig, ThetaNormalizer, L2WSPredictor, theta_dim
from build_realistic_training_dataset import sample_level_width_perturbed_dataset


def run_timed(net, cfg: algorithm.DRConfig):
    """Identical to a2dr_benchmark/time_parallel_equivalent.py's run_timed
    -- duplicated locally rather than imported across the two sibling
    benchmark packages (neither currently depends on the other)."""
    net.reset()
    seq_time = 0.0
    par_time = 0.0
    converged = False
    k_run = 0
    with torch.no_grad():
        for k in range(cfg.n_iter):
            k_run = k + 1
            net.communicate()
            node_times = []
            for p in net.prosumers:
                t0 = time.perf_counter()
                p.dr_step(cfg.gamma, cfg.lam, n_bisect=cfg.bisect, tau=cfg.tau)
                node_times.append(time.perf_counter() - t0)
            seq_time += sum(node_times)
            par_time += max(node_times)
            r = net.aux_residual()
            if r < cfg.tol:
                converged = True
                break
    return k_run, converged, seq_time, par_time


def load_predictor(checkpoint_path, base_spec, out_dim, device):
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mlp = WarmStartMLP(theta_dim(base_spec), out_dim, WarmStartConfig(**blob["cfg"]))
    mlp.load_state_dict(blob["state_dict"])
    mlp.eval()
    normalizer = ThetaNormalizer.__new__(ThetaNormalizer)
    normalizer.mean = blob["normalizer_mean"]
    normalizer.std = blob["normalizer_std"]
    return L2WSPredictor(mlp, normalizer, device=device)


def main():
    density = float(sys.argv[1])
    base_seed = int(sys.argv[2])
    dr_gamma = float(sys.argv[3])
    checkpoint_path = sys.argv[4]
    n_test = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    c_old_range = (tuple(float(x) for x in sys.argv[6].split(",")) if len(sys.argv) > 6
                   else (0.05, 0.18))
    out_path = sys.argv[7] if len(sys.argv) > 7 else "l2ws_parallel_timing_results.json"

    print(f"L2WS parallel-equivalent timing (density={density}  base_seed={base_seed}  gamma={dr_gamma}  "
          f"checkpoint={checkpoint_path}  n_test={n_test})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    ref_net = market.build_network(base_spec, device=device)
    out_dim = ref_net.dim

    predictor = load_predictor(checkpoint_path, base_spec, out_dim, device)
    print(f"  loaded checkpoint", flush=True)

    test_specs, _, n_discarded = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
    test_specs = [base_spec] + test_specs
    print(f"  test set: {n_test} instances (slot 0 = original), {n_discarded} infeasible discarded", flush=True)

    dr_seq, dr_par, l_seq, l_par, mlp_times, dr_iters, l_iters = [], [], [], [], [], [], []
    for i, spec in enumerate(test_specs):
        seed = config.TEST_SEED * 7919 + i
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        net_dr = market.build_network(spec, device=device, rng=gen)
        n_dr, conv_dr, sdr, pdr = run_timed(net_dr, dr_cfg)

        net_l = market.build_network(spec, device=device)
        t0 = time.perf_counter()
        warm_start = predictor.predict(spec, differentiable=False)
        t_mlp = time.perf_counter() - t0          # centralised, no node-parallel benefit
        net_l.unflatten(warm_start)
        n_l, conv_l, sl, pl = run_timed(net_l, dr_cfg)   # post-warm-start DR: node-parallel benefit applies

        dr_iters.append(n_dr); l_iters.append(n_l)
        dr_seq.append(sdr); dr_par.append(pdr)
        l_seq.append(t_mlp + sl); l_par.append(t_mlp + pl)
        mlp_times.append(t_mlp)

        sp_par = pdr / max(t_mlp + pl, 1e-9)
        sp_seq = sdr / max(t_mlp + sl, 1e-9)
        print(f"  {i + 1:2d}/{n_test}  DR {n_dr:5d} it  par={pdr:6.2f}s  |  "
              f"L2WS {n_l:5d} it  mlp={t_mlp:.4f}s par={t_mlp + pl:6.2f}s  |  "
              f"x{sp_par:5.2f} parallel-equiv, x{sp_seq:5.2f} sequential", flush=True)

    sp_par = [d / max(l, 1e-9) for d, l in zip(dr_par, l_par)]
    sp_seq = [d / max(l, 1e-9) for d, l in zip(dr_seq, l_seq)]
    win_par = sum(s >= 1.0 for s in sp_par) / len(sp_par)
    win_seq = sum(s >= 1.0 for s in sp_seq) / len(sp_seq)
    print(flush=True)
    print(f"  [parallel-equivalent]  mean speed-up {stats.mean(sp_par):.2f}x   "
          f"median {stats.median(sp_par):.2f}x   win-rate {win_par:.0%}", flush=True)
    print(f"  [sequential (cross-check)]  mean speed-up {stats.mean(sp_seq):.2f}x   "
          f"median {stats.median(sp_seq):.2f}x   win-rate {win_seq:.0%}", flush=True)

    results = {
        "n_test": n_test, "density": density, "base_seed": base_seed, "dr_gamma": dr_gamma,
        "checkpoint_path": checkpoint_path,
        "dr_iters": dr_iters, "l2ws_iters": l_iters,
        "dr_par_time_s": dr_par, "dr_seq_time_s": dr_seq,
        "l2ws_par_time_s": l_par, "l2ws_seq_time_s": l_seq, "l2ws_mlp_time_s": mlp_times,
        "speedup_parallel": sp_par, "speedup_sequential": sp_seq,
        "summary_parallel": {"mean": stats.mean(sp_par), "median": stats.median(sp_par), "win_rate": win_par},
        "summary_sequential": {"mean": stats.mean(sp_seq), "median": stats.median(sp_seq), "win_rate": win_seq},
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
