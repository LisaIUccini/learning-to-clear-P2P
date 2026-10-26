"""
time_l2ws_vs_dr.py
==========================================

Measures REAL wall-clock time for plain DR vs. plain-DR-warm-started-by-
L2WS, on an already-trained h_w.pt checkpoint, on the SAME level+width
perturbed test set the checkpoint's own train_l2ws.py run used (same
density/base_seed/c_old_range -> same base_spec -> same
sample_level_width_perturbed_dataset(seed=config.TRAIN_SEED+1) draw) and
the SAME test set compare_a2dr_vs_official.py uses for A2DR/DR+GNN, so all
five methods (DR, A2DR, DR+GNN, and now L2WS) end up timed on identical
instances per density, feeding one shared wall-clock comparison figure.

train_l2ws.py's own train.log only ever recorded ITERATION counts, never
timing, for the "DR XXX it | L2WS YYY it" evaluation lines -- this script
exists purely to retrofit real seconds onto already-trained checkpoints
without re-training anything.

Usage: python3 time_l2ws_vs_dr.py <density> <base_seed> <dr_gamma>
       <checkpoint_path> [n_test=30] [c_old_range=0.05,0.18] [out_path]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # experiments/
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # DRcld/
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # l2ws_benchmark/ (model.py)

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


def load_predictor(checkpoint_path, base_spec, out_dim, device):
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mlp = WarmStartMLP(theta_dim(base_spec), out_dim, WarmStartConfig(**blob["cfg"]))
    mlp.load_state_dict(blob["state_dict"])
    mlp.eval()
    normalizer = ThetaNormalizer.__new__(ThetaNormalizer)
    normalizer.mean = blob["normalizer_mean"]
    normalizer.std = blob["normalizer_std"]
    return L2WSPredictor(mlp, normalizer, device=device)


def timed_pair(spec, predictor, dr_cfg, device, seed):
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net_dr = market.build_network(spec, device=device, rng=gen)
    t0 = time.perf_counter()
    res_dr = algorithm.run_dr(net_dr, dr_cfg, record=False)
    t_dr = time.perf_counter() - t0

    net_l = market.build_network(spec, device=device)
    t0 = time.perf_counter()
    warm_start = predictor.predict(spec, differentiable=False)
    net_l.unflatten(warm_start)
    res_l = algorithm.run_dr(net_l, dr_cfg, record=False)
    t_l = time.perf_counter() - t0

    return res_dr, res_l, t_dr, t_l


def main():
    density = float(sys.argv[1])
    base_seed = int(sys.argv[2])
    dr_gamma = float(sys.argv[3])
    checkpoint_path = sys.argv[4]
    n_test = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    c_old_range = (tuple(float(x) for x in sys.argv[6].split(",")) if len(sys.argv) > 6
                   else (0.05, 0.18))
    out_path = sys.argv[7] if len(sys.argv) > 7 else "l2ws_timing_results.json"

    print(f"timing L2WS vs plain DR  (density={density}  base_seed={base_seed}  gamma={dr_gamma}  "
          f"checkpoint={checkpoint_path}  n_test={n_test})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    ref_net = market.build_network(base_spec, device=device)
    out_dim = ref_net.dim
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}",
          flush=True)

    predictor = load_predictor(checkpoint_path, base_spec, out_dim, device)
    print(f"  loaded checkpoint", flush=True)

    # SAME test set as compare_a2dr_vs_official.py / train_l2ws.py's own
    # in-distribution test set (slot 0 = original, same seed convention).
    test_specs, _, n_discarded = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
    test_specs = [base_spec] + test_specs
    print(f"  test set: {n_test} instances (slot 0 = original), {n_discarded} infeasible discarded", flush=True)

    dr_iters, l_iters, dr_time, l_time = [], [], [], []
    for i, spec in enumerate(test_specs):
        seed = config.TEST_SEED * 7919 + i
        res_dr, res_l, t_dr, t_l = timed_pair(spec, predictor, dr_cfg, device, seed)
        dr_iters.append(res_dr.n_iter); l_iters.append(res_l.n_iter)
        dr_time.append(t_dr); l_time.append(t_l)
        sp_it = res_dr.n_iter / max(res_l.n_iter, 1)
        sp_t = t_dr / max(t_l, 1e-9)
        flag = "" if (res_dr.converged and res_l.converged) else "  [DID NOT CONVERGE]"
        print(f"  {i + 1:2d}/{n_test}  DR {res_dr.n_iter:5d} it ({t_dr:6.2f}s)  |  "
              f"L2WS {res_l.n_iter:5d} it ({t_l:6.2f}s)  x{sp_it:5.2f} iters, x{sp_t:5.2f} time{flag}", flush=True)

    sp_iters = [d / max(l, 1) for d, l in zip(dr_iters, l_iters)]
    sp_time = [d / max(l, 1e-9) for d, l in zip(dr_time, l_time)]
    win_it = sum(s >= 1.0 for s in sp_iters) / len(sp_iters)
    win_t = sum(s >= 1.0 for s in sp_time) / len(sp_time)
    print(flush=True)
    print(f"  [L2WS -- iterations]  mean speed-up {stats.mean(sp_iters):.2f}x   "
          f"median {stats.median(sp_iters):.2f}x   win-rate {win_it:.0%}", flush=True)
    print(f"  [L2WS -- wall-clock]  mean speed-up {stats.mean(sp_time):.2f}x   "
          f"median {stats.median(sp_time):.2f}x   win-rate {win_t:.0%}", flush=True)

    results = {
        "n_test": n_test, "density": density, "base_seed": base_seed, "dr_gamma": dr_gamma,
        "checkpoint_path": checkpoint_path,
        "dr_iters": dr_iters, "l2ws_iters": l_iters,
        "dr_time_s": dr_time, "l2ws_time_s": l_time,
        "speedup_iters": sp_iters, "speedup_time": sp_time,
        "summary_iters": {"mean": stats.mean(sp_iters), "median": stats.median(sp_iters), "win_rate": win_it},
        "summary_time": {"mean": stats.mean(sp_time), "median": stats.median(sp_time), "win_rate": win_t},
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
