"""
demo_fixed_market_longhorizon.py
==================================

Follow-up to `demo_fixed_market.py`: does the "extend the correction's
reach" idea -- which collapsed win-rate on the diverse cluster dataset
(rho 0.95->0.97, unroll 200->400: `runs/hinge_v4_longhorizon_FAILED`,
33% win-rate) -- actually work once cross-instance diversity is removed?

Same fixed market as `demo_fixed_market.py` (one spec, only x0 varies),
same architecture and loss (hinge, eta=0.15), but rho/unroll set to the
EXACT values that failed on the diverse dataset, so this is a clean,
direct test of whether diversity (not the long-horizon mechanism itself)
was the actual cause of that failure. Epochs raised too: on one market,
more training only risks overfitting to that market, which is the goal
here, not a hazard.
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
from demo import run_pair, evaluate, plot_results, plot_training_loss


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "fixed_market_longhorizon"
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/")

    device = torch.device("cpu")
    seed = 0
    torch.manual_seed(seed)

    mc = market.MarketConfig(density=0.3)
    dr_cfg = algorithm.DRConfig(gamma=4.0, lam=1.0, n_iter=4000, tol=1e-5, bisect=30)
    # rho/unroll match hinge_v4_longhorizon_FAILED exactly; epochs raised
    # 20->30 since overfitting to the one market is the goal, not a risk.
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=400, gamma_mono=2.0, eta=0.15, epochs=30, lr=3e-3,
                                       seed=seed, loss_mode="hinge")

    print("=" * 70)
    print("dataset: ONE fixed market (identical to demo_fixed_market.py);")
    print("rho=0.97, unroll=400 -- the exact config that collapsed win-rate")
    print("to 33% on the DIVERSE cluster dataset (hinge_v4_longhorizon_FAILED)")
    print("=" * 70)
    _, _, base_spec = market.sample_base(10, mc, seed=seed)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}")

    train_specs = [base_spec] * 25
    test_specs = [base_spec] * 12

    print("\nsanity check: plain DR converges from a few random starts")
    for i in range(3):
        gen = torch.Generator(device=device); gen.manual_seed(seed * 7919 + i)
        net = market.build_network(base_spec, device=device, rng=gen)
        res = algorithm.run_dr(net, dr_cfg)
        print(f"  start {i}  iters={res.n_iter:5d}  converged={res.converged}  "
              f"feasible={net.is_feasible(1e-3)}")

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode={train_cfg.loss_mode!r})")
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=0.97, scale=0.4, scale_slow=0.0))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    print("\nevaluating: plain DR vs DR + Phi_w, fully distributed inference")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="fixed-market-longhorizon")

    print("\nrecording residual curves on the test set...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = "trained on: 1 fixed market, zero diversity, rho=0.97 unroll=400  |  tested on: same market, n=12"
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"),
                        subtitle="trained on: 1 fixed market, zero diversity, rho=0.97 unroll=400, 30 epochs")

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    weights_path = os.path.join(out_dir, "phi_w.pt")
    trainer.save(weights_path)
    print(f"saved trained weights to {weights_path}")


if __name__ == "__main__":
    main()
