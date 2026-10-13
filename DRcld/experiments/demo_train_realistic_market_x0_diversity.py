"""
demo_train_realistic_market_x0_diversity.py
==============================================

Reproduces the EXACT recipe of `seed0_density1.0_seed0_drslope` (the
project's x0-diversity reference: ONE fixed market, x0 randomized per
instance/epoch, gamma=4.0, unroll=300, GNN_RHO=0.9667, GNN_SCALE=1.0,
loss_mode='asymmetric', omega_mode='dr_slope', w_behind=2.0, w_ahead=1.0,
n_train=25, epochs=60) but built on the REALISTIC market
(`build_realistic_training_dataset.build_realistic_base_market`, G>=0)
instead of the old abstract-unit default market.

Why: a diagnostic (2026-09-09) found that `seed0_density1.0_seed0_drslope`
evaluated zero-shot on a 6-parameter level+width perturbation of the NEW
realistic market failed catastrophically (mean 0.56-0.70x, near-0% win
-rate) -- but the SAME checkpoint on the SAME 6-parameter perturbation of
ITS OWN (old) market scored mean 1.55x/100% win-rate, matching or beating
v5 (the best fixed-x0/parameter-diversity-TRAINED model, 1.25x/100%) --
purely zero-shot, no retraining. This proved the earlier catastrophic
failure was a market-mismatch artifact (wrong absolute scale/cost
coefficients), not evidence that 6-parameter perturbation or box-width
jittering is inherently hard to generalize to. This run removes that last
confound: train the x0-diversity recipe on the SAME market v5 uses, so its
zero-shot performance on v5-style perturbations, and its own home-market
performance, are both directly comparable to v5 on a true apples-to-apples
basis (same market, same graph, same a_tilde/a/b/c/kappa distributions).

Dataset: ONE fixed market (realistic, density=1.0, N0=10, base_seed=0),
identical across all "instances" -- the only source of variation is x0,
freshly randomized per (instance, epoch) during training (Trainer.fit's
default behaviour, fixed_x0=None) and per test instance at evaluation.

Usage: python3 demo_train_realistic_market_x0_diversity.py <run_name>
       [n_train] [n_test] [epochs] [dr_gamma] [unroll] [gnn_rho] [gnn_scale]
       [w_ahead] [w_behind]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import statistics as stats
import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod
import train as train_mod
from diagnostics import save_instance_plots
from demo import plot_results, plot_training_loss, write_run_notes, run_pair
from build_realistic_training_dataset import build_realistic_base_market, REALISTIC_MARKET_CONFIG


def evaluate_with_curves(specs, model, dr_cfg, device, seed, label):
    dr_iters, learned_iters, speedups, curves_dr, curves_learned = [], [], [], [], []
    for i, spec in enumerate(specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=seed * 7919 + i, record=True)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter); speedups.append(sp)
        curves_dr.append(np.asarray(res_dr.residual)); curves_learned.append(np.asarray(res_learned.residual))
        print(f"  [{label}] {i + 1:2d}/{len(specs)}  DR {res_dr.n_iter:5d} it  |  "
              f"learned {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)
    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"  [{label}] mean speed-up {stats.mean(speedups):.2f}x   median {stats.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)
    return {"dr": dr_iters, "learned": learned_iters, "speedup": speedups,
            "curves_dr": curves_dr, "curves_learned": curves_learned}


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "realistic_market_x0_diversity"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    unroll = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    gnn_rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.9667
    gnn_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    w_ahead = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
    w_behind = float(sys.argv[10]) if len(sys.argv) > 10 else 2.0
    omega_mode = sys.argv[11] if len(sys.argv) > 11 else "dr_slope"

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"w_ahead={w_ahead}  w_behind={w_behind}  omega_mode={omega_mode})", flush=True)
    assert w_ahead < w_behind

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=config.TRAIN_GAMMA_MONO,
                                       epochs=epochs, w_behind=w_behind, w_ahead=w_ahead,
                                       lr=config.TRAIN_LR, grad_clip=config.TRAIN_GRAD_CLIP,
                                       seed=config.TRAIN_SEED, loss_mode="asymmetric", omega_mode=omega_mode)

    print("=" * 70)
    print("dataset: ONE FIXED realistic market (density=1.0, N0=10, base_seed=0,")
    print("G>=0 by construction) -- graph and every one of the 6 quantities")
    print("IDENTICAL across all instances. The ONLY source of variation is x0,")
    print("freshly randomized per (instance, epoch) during training and per")
    print("test instance at evaluation -- reproduces seed0_density1.0_seed0_drslope's")
    print("exact recipe, on the realistic market instead of the old abstract one.")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=0)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    train_specs = [base_spec] * n_train
    test_specs = [base_spec] * n_test

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='{omega_mode}')", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(mc, rho=gnn_rho, scale=gnn_scale))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, FRESH RANDOM x0 per test instance", flush=True)
    print("-" * 70)
    test_res = evaluate_with_curves(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="home-market")

    subtitle = (f"trained on: 1 fixed market (realistic, density=1.0), x0-diversity only "
                f"({n_train} instances x {epochs} epochs, fresh x0 each time)  |  "
                f"tested on: n={n_test}, fresh random x0")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: reproduces `seed0_density1.0_seed0_drslope`'s exact recipe (x0-diversity, one fixed
market, gamma={dr_gamma}, unroll={unroll}, GNN_RHO={gnn_rho}, GNN_SCALE={gnn_scale}, loss=asymmetric,
w_behind={w_behind}, w_ahead={w_ahead}, omega_mode='{omega_mode}', n_train={n_train}, epochs={epochs}) on
the REALISTIC market (density=1.0, N0=10, base_seed=0, G>=0 by construction) instead of the old
abstract-unit default market -- so it is directly comparable, on the SAME market, to the
fixed-x0/parameter-diversity axis (v1-v13, best result v5: 1.25x/100%).

**Motivation**: a diagnostic found `seed0_density1.0_seed0_drslope` catastrophically fails
zero-shot on a 6-parameter perturbation of the REALISTIC market (0.56-0.70x, near-0% win-rate) but
scores 1.55x/100% zero-shot on the SAME perturbation of ITS OWN (old) market -- proving the earlier
failure was a market-mismatch artifact, not evidence against generalizing across parameter
perturbation. This run removes that confound entirely.

**Dataset**: ONE fixed market, {n_train} training "instances" (all literally the same spec), x0
freshly randomized per (instance, epoch) during training (not fixed, unlike v1-v13) and per test
instance at evaluation -- {n_test} test instances, all the same spec, fresh random x0 each.

**Results (home market, x0-diversity only)**: mean {stats.mean(test_res['speedup']):.2f}x, median
{stats.median(test_res['speedup']):.2f}x, min {min(test_res['speedup']):.2f}x,
max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).

**Question answered**: does x0-diversity training on the SAME realistic market v5 uses match
`seed0_density1.0_seed0_drslope`'s 2.09x/100% home-market result? This checkpoint is also intended
for a follow-up zero-shot test on v5-style 6-parameter perturbations, to get a true apples-to-apples
comparison against v5's dedicated parameter-diversity training (1.25x/100%).
""")


if __name__ == "__main__":
    main()
