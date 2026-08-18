"""
demo_param_variation.py
=========================

Follow-up to `demo_fixed_market.py`: decomposes "market diversity" into
its two components -- topology and parameters -- to see which one is
actually driving the difficulty. The graph here is left EXACTLY fixed
(identical to the `fixed_market` reference, no topology perturbation at
all); only the market parameters (a, b, a~, D*, DeltaG, c, kappa) are
jittered by +/- 10% per instance, for both training and the held-out test
set (`market.sample_param_only_dataset`).

Uses the SAME recipe as `demo_fixed_market.py` (the best validated result:
92% win-rate, 1.40x/1.48x mean/median on zero diversity) -- architecture,
loss, and every hyperparameter unchanged. Only the dataset differs.
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
from demo import run_pair, evaluate, plot_results, plot_training_loss


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "param_variation"
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/")

    device = torch.device("cpu")
    seed = 0
    torch.manual_seed(seed)

    mc = market.MarketConfig(density=0.3)
    dr_cfg = algorithm.DRConfig(gamma=4.0, lam=1.0, n_iter=4000, tol=1e-5, bisect=30)
    # EXACT fixed_market / hinge_v5_final recipe -- only the dataset changes.
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=200, gamma_mono=2.0, eta=0.15, epochs=20, lr=3e-3,
                                       seed=seed, loss_mode="hinge")

    print("=" * 70)
    print("dataset: graph EXACTLY fixed (identical to fixed_market's base);")
    print("only market PARAMETERS (a,b,a~,D*,DeltaG,c,kappa) jitter +/-10%")
    print("per instance -- isolates parameter diversity from topology")
    print("=" * 70)
    base_nb, base_prim, base_spec = market.sample_base(10, mc, seed=seed)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}")

    train_specs = [base_spec] + market.sample_param_only_dataset(
        base_nb, base_prim, mc, n_instances=24, pct=0.10, seed=seed + 1)
    test_specs = market.sample_param_only_dataset(
        base_nb, base_prim, mc, n_instances=12, pct=0.10, seed=seed + 2)

    print("\nsanity check: plain DR converges on a few training instances")
    for spec in train_specs[:3]:
        net = market.build_network(spec, device=device)
        res = algorithm.run_dr(net, dr_cfg)
        print(f"  N={spec.N:3d}  iters={res.n_iter:5d}  converged={res.converged}  "
              f"feasible={net.is_feasible(1e-3)}")

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode={train_cfg.loss_mode!r})")
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=0.95, scale=0.4, scale_slow=0.0))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    print("\nevaluating: plain DR vs DR + Phi_w, fully distributed inference")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="param-variation")

    print("\nrecording residual curves on the test set...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = "trained on: fixed graph, params independently jittered +/-10% each  |  tested on: same distribution, n=12"
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"),
                        subtitle="trained on: fixed graph, params independently jittered +/-10% each, 25 instances")

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    weights_path = os.path.join(out_dir, "phi_w.pt")
    trainer.save(weights_path)
    print(f"saved trained weights to {weights_path}")


if __name__ == "__main__":
    main()
