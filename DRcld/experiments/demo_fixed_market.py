"""
demo_fixed_market.py
=====================

Diagnostic: is the mediocre win-rate/magnitude caused by having to
generalise across many DIFFERENT markets (graph + parameters), or is even
a SINGLE fixed market -- only the random initial state x^0 varying --
already hard to learn a strong, reliable correction for?

Uses the recipe in `config.py` (the validated best result so far --
92% win-rate, 1.40x/1.48x mean/median), retrained from scratch on a
dataset where the graph and every market parameter are IDENTICAL across
every "instance": one spec, built once, reused for every training epoch
and every test instance. The only source of variation, for both training
and testing, is the random draw of x^0 inside the boxes -- exactly the
same per-instance/per-epoch seeding `Trainer.fit` and `evaluate` already
use, just pointed at one spec instead of many.

If this performs MUCH better than a diverse-dataset run, cross-instance
diversity is (part of) the bottleneck. If it is still mediocre, the
difficulty is intrinsic to this market's own DR dynamics (e.g. the
plateau phenomenon documented in train.py), independent of diversity.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod
import train as train_mod
from diagnostics import save_instance_plots
from demo import run_pair, evaluate, plot_results, plot_training_loss, write_run_notes


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "fixed_market"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else config.N_TRAIN_INSTANCES
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else config.N_TEST_INSTANCES
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else config.TRAIN_EPOCHS
    density = float(sys.argv[5]) if len(sys.argv) > 5 else config.MARKET_DENSITY
    eta = float(sys.argv[6]) if len(sys.argv) > 6 else config.TRAIN_ETA
    dr_gamma = float(sys.argv[7]) if len(sys.argv) > 7 else config.DR_GAMMA
    unroll = int(sys.argv[8]) if len(sys.argv) > 8 else config.TRAIN_UNROLL
    gnn_rho = float(sys.argv[9]) if len(sys.argv) > 9 else config.GNN_RHO
    gnn_scale = float(sys.argv[10]) if len(sys.argv) > 10 else config.GNN_SCALE
    base_seed = int(sys.argv[11]) if len(sys.argv) > 11 else config.BASE_MARKET_SEED
    loss_mode = sys.argv[12] if len(sys.argv) > 12 else config.TRAIN_LOSS_MODE
    w_behind = float(sys.argv[13]) if len(sys.argv) > 13 else config.TRAIN_W_BEHIND
    w_ahead = float(sys.argv[14]) if len(sys.argv) > 14 else config.TRAIN_W_AHEAD
    gnn_rho_slow = float(sys.argv[15]) if len(sys.argv) > 15 else config.GNN_RHO_SLOW
    gnn_scale_slow = float(sys.argv[16]) if len(sys.argv) > 16 else config.GNN_SCALE_SLOW
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"density={density}  eta={eta}  gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  "
          f"gnn_scale={gnn_scale}  base_seed={base_seed}  loss_mode={loss_mode!r}  "
          f"w_behind={w_behind}  w_ahead={w_ahead}  gnn_rho_slow={gnn_rho_slow}  "
          f"gnn_scale_slow={gnn_scale_slow})")

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)

    mc = market.MarketConfig(density=density)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=config.TRAIN_GAMMA_MONO,
                                       eta=eta, epochs=epochs, w_behind=w_behind, w_ahead=w_ahead,
                                       lr=config.TRAIN_LR, grad_clip=config.TRAIN_GRAD_CLIP,
                                       seed=config.TRAIN_SEED, loss_mode=loss_mode)

    print("=" * 70)
    print("dataset: ONE fixed market (graph + every parameter identical")
    print("across all instances) -- only the random initial state x^0")
    print("varies, for both training and the held-out test set")
    print("=" * 70)
    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}")

    # n_train/n_test "instances" = the SAME spec repeated -- Trainer.fit
    # and evaluate() already draw a fresh random x0 per (instance, epoch)
    # via their own seeding, so repeating the identical spec still gives
    # genuinely different starting points each time.
    train_specs = [base_spec] * n_train
    test_specs = [base_spec] * n_test

    print("\nsanity check: plain DR converges from a few random starts")
    for i in range(3):
        gen = torch.Generator(device=device); gen.manual_seed(config.TRAIN_SEED * 7919 + i)
        net = market.build_network(base_spec, device=device, rng=gen)
        res = algorithm.run_dr(net, dr_cfg)
        print(f"  start {i}  iters={res.n_iter:5d}  converged={res.converged}  "
              f"feasible={net.is_feasible(1e-3)}")

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode={train_cfg.loss_mode!r})")
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale,
        rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    print("\nevaluating: plain DR vs DR + Phi_w, fully distributed inference")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="fixed-market")

    print("\nrecording residual curves on the test set...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    eta_note = f", eta={eta}" if eta != config.TRAIN_ETA else ""
    gamma_note = f", gamma={dr_gamma}" if dr_gamma != config.DR_GAMMA else ""
    unroll_note = f", unroll={unroll}" if unroll != config.TRAIN_UNROLL else ""
    rho_note = f", rho={gnn_rho}" if gnn_rho != config.GNN_RHO else ""
    scale_note = f", scale={gnn_scale}" if gnn_scale != config.GNN_SCALE else ""
    seed_note = f", base_seed={base_seed}" if base_seed != config.BASE_MARKET_SEED else ""
    loss_note = f", loss={loss_mode}" if loss_mode != config.TRAIN_LOSS_MODE else ""
    subtitle = (f"trained on: 1 fixed market, zero diversity, {n_train} instances, {epochs} epochs"
                f"{eta_note}{gamma_note}{unroll_note}{rho_note}{scale_note}{seed_note}{loss_note}  |  tested on: same market, n={n_test}")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"),
                        subtitle=(f"trained on: 1 fixed market, zero diversity, {n_train} instances, "
                                  f"{epochs} epochs{eta_note}{gamma_note}{unroll_note}{rho_note}{scale_note}{seed_note}{loss_note}"))

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    weights_path = os.path.join(out_dir, "phi_w.pt")
    trainer.save(weights_path)
    print(f"saved trained weights to {weights_path}")

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: full training from scratch (centralised, self-supervised,
loss_mode={train_cfg.loss_mode!r}), architecture/loss/hyperparameters from
`config.py` except epochs (overridden to {epochs} here), graph density
(overridden to {density} here), eta (overridden to {eta} here, config
default is {config.TRAIN_ETA}), DR's own gamma (overridden to {dr_gamma}
here, config default is {config.DR_GAMMA} -- train_cfg.gamma is derived from
dr_cfg.gamma, so the differentiable rollout during training automatically
matches whatever gamma is used at inference), training unroll length
(overridden to {unroll} here, config default is {config.TRAIN_UNROLL}), the
perturbation's decay rate GNN_RHO (overridden to {gnn_rho} here, config
default is {config.GNN_RHO}), the perturbation's amplitude GNN_SCALE
(overridden to {gnn_scale} here, config default is {config.GNN_SCALE}), and
the loss itself (overridden to {loss_mode!r} here, config default is
{config.TRAIN_LOSS_MODE!r}{f", w_behind={w_behind}, w_ahead={w_ahead}" if loss_mode == "asymmetric" else ""}).

**Dataset**: ONE fixed market (graph density {density}, N0={config.BASE_MARKET_N},
seed={base_seed}{" -- config default is " + str(config.BASE_MARKET_SEED) if base_seed != config.BASE_MARKET_SEED else ""}) --
graph and every parameter IDENTICAL across all
"instances". The only source of variation, for both training ({n_train}
instances) and test ({n_test} instances), is the random draw of the
initial state x^0 per (instance, epoch).

**Question answered**: does more training signal (more repeats of the same
market, more epochs) improve win-rate/magnitude further on top of the
zero-diversity baseline (`runs/fixed_market`: {config.N_TRAIN_INSTANCES} train instances,
{config.TRAIN_EPOCHS} epochs, 92% win-rate, 1.40x/1.48x mean/median on n=12), or does
the zero-diversity setting already saturate with the original budget?
""")


if __name__ == "__main__":
    main()
