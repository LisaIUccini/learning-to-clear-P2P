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
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs})")

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)

    mc = market.MarketConfig(density=config.MARKET_DENSITY)
    dr_cfg = algorithm.DRConfig(gamma=config.DR_GAMMA, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=config.TRAIN_UNROLL, gamma_mono=config.TRAIN_GAMMA_MONO,
                                       eta=config.TRAIN_ETA, epochs=epochs,
                                       lr=config.TRAIN_LR, grad_clip=config.TRAIN_GRAD_CLIP,
                                       seed=config.TRAIN_SEED, loss_mode=config.TRAIN_LOSS_MODE)

    print("=" * 70)
    print("dataset: ONE fixed market (graph + every parameter identical")
    print("across all instances) -- only the random initial state x^0")
    print("varies, for both training and the held-out test set")
    print("=" * 70)
    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=config.BASE_MARKET_SEED)
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
        mc, rho=config.GNN_RHO, scale=config.GNN_SCALE,
        rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
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

    subtitle = f"trained on: 1 fixed market, zero diversity, {n_train} instances, {epochs} epochs  |  tested on: same market, n={n_test}"
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"),
                        subtitle=f"trained on: 1 fixed market, zero diversity, {n_train} instances, {epochs} epochs")

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
`config.py` except epochs (overridden to {epochs} here).

**Dataset**: ONE fixed market (graph density {config.MARKET_DENSITY}, N0={config.BASE_MARKET_N},
seed={config.BASE_MARKET_SEED}) -- graph and every parameter IDENTICAL across all
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
