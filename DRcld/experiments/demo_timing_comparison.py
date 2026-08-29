"""
demo_timing_comparison.py
============================

NO retraining -- loads an already-trained checkpoint and compares plain DR
against DR+learned using WALL-CLOCK TIME to reach convergence, not
iteration count. Each "learned" iteration does strictly more work (it also
evaluates the GNN forward pass), so the time speed-up is expected to be
smaller than the iteration-count speed-up; this measures the gap directly.

`record=False` is used for the timed runs (matches a real deployment: only
the stopping-residual check every iteration, not the extra cost/shadow
bookkeeping `record=True` does purely for diagnostics elsewhere in this
project). Each instance is timed `n_repeats` times and the MEDIAN wall time
is used, to damp OS-scheduling noise; DR and learned are interleaved
(DR, learned, DR, learned, ...) each repeat, so neither side systematically
benefits from cache/thermal warm-up.

Usage:
  python3 demo_timing_comparison.py fixed <out_dir> <weights_path>
      <base_seed> <density> <dr_gamma> <gnn_rho> <gnn_scale> <n_test> [n_repeats]
  python3 demo_timing_comparison.py dstar_gres <out_dir> <weights_path>
      <base_seed> <density> <dr_gamma> <gnn_rho> <gnn_scale> <n_test> <pct> [n_repeats]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import statistics
import sys
import time

import torch

import config
import market
import algorithm
import gnn as gnn_mod
from demo_fixed_market_on_dstar_gres_scale import build_dstar_gres_dataset

mode = sys.argv[1]
out_dir = sys.argv[2]
weights_path = sys.argv[3]
base_seed = int(sys.argv[4])
density = float(sys.argv[5])
dr_gamma = float(sys.argv[6])
gnn_rho = float(sys.argv[7])
gnn_scale = float(sys.argv[8])
n_test = int(sys.argv[9])
if mode == "dstar_gres":
    pct = float(sys.argv[10])
    n_repeats = int(sys.argv[11]) if len(sys.argv) > 11 else 5
else:
    n_repeats = int(sys.argv[10]) if len(sys.argv) > 10 else 5

os.makedirs(out_dir, exist_ok=True)
device = torch.device("cpu")
mc = market.MarketConfig(density=density)
dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                             n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                             bisect=config.DR_BISECT_STEPS)
model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=gnn_rho, scale=gnn_scale, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
blob = torch.load(weights_path, map_location="cpu", weights_only=False)
model.load_state_dict(blob["state_dict"])
model.eval()

_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
if mode == "fixed":
    test_specs = [base_spec] * n_test
elif mode == "dstar_gres":
    test_specs, _, _ = build_dstar_gres_dataset(base_spec, n_instances=n_test, pct=pct, seed=config.TRAIN_SEED + 3)
else:
    raise ValueError(mode)


def timed_dr(spec, seed) -> tuple:
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = market.build_network(spec, device=device, rng=gen)
    gnn_mod.undeploy(net)
    t0 = time.perf_counter()
    res = algorithm.run_dr(net, dr_cfg, record=False)
    dt = time.perf_counter() - t0
    return res.n_iter, dt


def timed_learned(spec, seed) -> tuple:
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = market.build_network(spec, device=device, rng=gen)
    gnn_mod.deploy(model, net)
    t0 = time.perf_counter()
    res = algorithm.run_dr(net, dr_cfg, record=False)
    dt = time.perf_counter() - t0
    gnn_mod.undeploy(net)
    return res.n_iter, dt


print(f"timing comparison: mode={mode}  weights={weights_path}  density={density}  "
      f"base_seed={base_seed}  n_test={n_test}  n_repeats={n_repeats}")
print("-" * 100)

iter_ratios, time_ratios = [], []
for i, spec in enumerate(test_specs):
    seed = config.TEST_SEED * 7919 + i
    dr_times, learned_times = [], []
    dr_iters = learned_iters = None
    for r in range(n_repeats):
        it_dr, t_dr = timed_dr(spec, seed)
        it_l, t_l = timed_learned(spec, seed)
        dr_times.append(t_dr); learned_times.append(t_l)
        dr_iters, learned_iters = it_dr, it_l   # identical every repeat (deterministic)
    med_dr = statistics.median(dr_times)
    med_l = statistics.median(learned_times)
    iter_ratio = dr_iters / max(learned_iters, 1)
    time_ratio = med_dr / max(med_l, 1e-9)
    iter_ratios.append(iter_ratio); time_ratios.append(time_ratio)
    print(f"  {i + 1:2d}/{n_test}  DR {dr_iters:5d} it / {med_dr * 1000:7.2f} ms   |   "
          f"learned {learned_iters:5d} it / {med_l * 1000:7.2f} ms   |   "
          f"iter x{iter_ratio:5.2f}   time x{time_ratio:5.2f}")

print("-" * 100)
print(f"ITERATION speed-up:  mean {statistics.mean(iter_ratios):.2f}x  "
      f"median {statistics.median(iter_ratios):.2f}x  "
      f"min {min(iter_ratios):.2f}x  max {max(iter_ratios):.2f}x  "
      f"win-rate {sum(r >= 1.0 for r in iter_ratios) / len(iter_ratios):.0%}")
print(f"WALL-CLOCK speed-up: mean {statistics.mean(time_ratios):.2f}x  "
      f"median {statistics.median(time_ratios):.2f}x  "
      f"min {min(time_ratios):.2f}x  max {max(time_ratios):.2f}x  "
      f"win-rate {sum(r >= 1.0 for r in time_ratios) / len(time_ratios):.0%}")

out_path = os.path.join(out_dir, "timing_summary.txt")
with open(out_path, "w") as f:
    f.write(f"mode={mode} weights={weights_path} density={density} base_seed={base_seed} "
            f"n_test={n_test} n_repeats={n_repeats}\n")
    f.write(f"ITERATION speed-up: mean {statistics.mean(iter_ratios):.3f}x median "
            f"{statistics.median(iter_ratios):.3f}x min {min(iter_ratios):.3f}x max "
            f"{max(iter_ratios):.3f}x\n")
    f.write(f"WALL-CLOCK speed-up: mean {statistics.mean(time_ratios):.3f}x median "
            f"{statistics.median(time_ratios):.3f}x min {min(time_ratios):.3f}x max "
            f"{max(time_ratios):.3f}x\n")
print(f"saved {out_path}")
