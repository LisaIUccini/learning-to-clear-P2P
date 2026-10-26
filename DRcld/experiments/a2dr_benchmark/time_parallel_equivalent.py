"""
time_parallel_equivalent.py
==========================================

Answers "wouldn't DR+GNN be faster if it actually ran distributed?" with a
real measurement instead of a guess. The earlier wall-clock figure
(compare_a2dr_vs_official.py / build_methods_comparison_time_plot.py) times
this project's SEQUENTIAL simulation -- one process, one CPU core, every
prosumer's `dr_step()` (including its GNN forward pass, when deployed)
called one after another in a Python for-loop (`algorithm.run_dr`,
`network.py:248-250`'s `self.gnn_agent.perturbation(...)` is invoked once
PER PROSUMER, PER ITERATION). On a real deployment each node has its own
hardware and all N local updates happen IN PARALLEL every round -- the
per-round wall-clock there is the SLOWEST node's local compute, not the
SUM over all N nodes.

True OS-level multiprocessing (one process per prosumer, barrier-
synchronised every round) was considered and rejected for this N=10 setting:
IPC/synchronisation overhead per round would swamp the actual compute (a
handful of bisection steps or one tiny MLP forward pass), making the
measurement noise-dominated rather than informative. Instead this script
times each prosumer's `dr_step()` call INDIVIDUALLY (still one process,
still sequential execution -- the measurement itself doesn't need to run
in parallel to tell us what the parallel critical path WOULD have been) and
reports, per DR iteration:

    sequential_time   = sum_p  time(p.dr_step())      [cross-check against
                                                         the earlier figure]
    parallel_time     = max_p  time(p.dr_step())       [critical-path
                                                         estimate: what a
                                                         dedicated-hardware-
                                                         per-node deployment
                                                         would see, modulo
                                                         actual network
                                                         latency, which is
                                                         NOT modelled here --
                                                         same convention as
                                                         compare_a2dr_vs_
                                                         official.py's own
                                                         "rounds" metric]

`net.communicate()` (the one-hop message exchange itself) is excluded from
both sums -- it is pairwise data movement, not decomposable per-node
compute, and its cost is already tracked separately via communication
ROUNDS elsewhere in this benchmark suite.

Usage: python3 time_parallel_equivalent.py <density> <base_seed> <dr_gamma>
       <gnn_rho> <gnn_scale> <rho_slow> <scale_slow> <checkpoint_path>
       [n_test=30] [c_old_range=0.05,0.18] [out_path]
"""
from __future__ import annotations

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_EXPERIMENTS = _os.path.dirname(_HERE)
_ROOT = _os.path.dirname(_EXPERIMENTS)
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _EXPERIMENTS)

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
import gnn as gnn_mod
from build_realistic_training_dataset import sample_level_width_perturbed_dataset


def run_timed(net, cfg: algorithm.DRConfig):
    """Same loop/stopping-rule as algorithm.run_dr, but times every
    prosumer's dr_step() call individually each round, returning both the
    sequential (sum) and parallel-equivalent (max) total compute time,
    alongside the usual n_iter/converged."""
    net.reset()
    seq_time = 0.0
    par_time = 0.0
    converged = False
    k_run = 0

    with torch.no_grad():
        for k in range(cfg.n_iter):
            k_run = k + 1
            net.communicate()   # excluded from both sums -- see module docstring
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


def main():
    density = float(sys.argv[1])
    base_seed = int(sys.argv[2])
    dr_gamma = float(sys.argv[3])
    gnn_rho = float(sys.argv[4])
    gnn_scale = float(sys.argv[5])
    rho_slow = float(sys.argv[6])
    scale_slow = float(sys.argv[7])
    checkpoint_path = sys.argv[8]
    n_test = int(sys.argv[9]) if len(sys.argv) > 9 else 30
    c_old_range = (tuple(float(x) for x in sys.argv[10].split(",")) if len(sys.argv) > 10
                   else (0.05, 0.18))
    out_path = sys.argv[11] if len(sys.argv) > 11 else "parallel_timing_results.json"

    print(f"parallel-equivalent timing (density={density}  base_seed={base_seed}  gamma={dr_gamma}  "
          f"n_test={n_test})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}",
          flush=True)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=rho_slow, scale_slow=scale_slow,
        hidden_dim=config.GNN_HIDDEN_DIM))
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"  loaded checkpoint {checkpoint_path}", flush=True)

    test_specs, _, n_discarded = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
    test_specs = [base_spec] + test_specs
    print(f"  test set: {n_test} instances (slot 0 = original), {n_discarded} infeasible discarded", flush=True)

    dr_seq, dr_par, l_seq, l_par, dr_iters, l_iters = [], [], [], [], [], []
    for i, spec in enumerate(test_specs):
        seed = config.TEST_SEED * 7919 + i
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        net = market.build_network(spec, device=device, rng=gen)
        x0 = net.flatten().clone()

        gnn_mod.undeploy(net)
        n_dr, conv_dr, sdr, pdr = run_timed(net, dr_cfg)

        net_l = market.build_network(spec, device=device)
        net_l.unflatten(x0)
        gnn_mod.deploy(model, net_l)
        n_l, conv_l, sl, pl = run_timed(net_l, dr_cfg)

        dr_iters.append(n_dr); l_iters.append(n_l)
        dr_seq.append(sdr); dr_par.append(pdr)
        l_seq.append(sl); l_par.append(pl)

        sp_par = pdr / max(pl, 1e-9)
        sp_seq = sdr / max(sl, 1e-9)
        print(f"  {i + 1:2d}/{n_test}  DR {n_dr:5d} it  seq={sdr:6.2f}s par={pdr:6.2f}s  |  "
              f"GNN {n_l:5d} it  seq={sl:6.2f}s par={pl:6.2f}s  |  "
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
        "dr_iters": dr_iters, "learned_iters": l_iters,
        "dr_seq_time_s": dr_seq, "dr_par_time_s": dr_par,
        "learned_seq_time_s": l_seq, "learned_par_time_s": l_par,
        "speedup_parallel": sp_par, "speedup_sequential": sp_seq,
        "summary_parallel": {"mean": stats.mean(sp_par), "median": stats.median(sp_par), "win_rate": win_par},
        "summary_sequential": {"mean": stats.mean(sp_seq), "median": stats.median(sp_seq), "win_rate": win_seq},
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
