"""
demo_train_topology_scale.py
==============================

Follow-up to `demo_common_scale.py`, the TOPOLOGY counterpart: this time
market PARAMETERS are held EXACTLY fixed (identical to the `fixed_market`
reference for every inherited node) and the GRAPH varies -- one shared
scale factor in [1-pct, 1+pct] per instance applied to node count AND
edge count together (`market.sample_common_scale_topology_dataset`), the
topology analogue of `sample_common_scale_dataset`'s single shared
parameter factor.

Unlike `demo_fixed_market_on_topology_scale.py` (zero-shot, no retraining),
this script actually TRAINS Phi_w from scratch on this topology-varying
dataset, using the exact recipe in `config.py` (the validated best result
on zero diversity: 92% win-rate, 1.40x/1.48x mean/median) -- only the
dataset differs from `demo_fixed_market.py`.
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
    run_name = sys.argv[1] if len(sys.argv) > 1 else "topology_scale"
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else config.N_TEST_INSTANCES
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_test={n_test})")

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)

    mc = market.MarketConfig(density=config.MARKET_DENSITY)
    dr_cfg = algorithm.DRConfig(gamma=config.DR_GAMMA, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=config.TRAIN_UNROLL, gamma_mono=config.TRAIN_GAMMA_MONO,
                                       eta=config.TRAIN_ETA, epochs=config.TRAIN_EPOCHS,
                                       lr=config.TRAIN_LR, grad_clip=config.TRAIN_GRAD_CLIP,
                                       seed=config.TRAIN_SEED, loss_mode=config.TRAIN_LOSS_MODE)

    print("=" * 70)
    print("dataset: market PARAMETERS EXACTLY fixed; per instance, ONE")
    print(f"shared scale factor in [1-{config.DATASET_PCT}, 1+{config.DATASET_PCT}] is applied to")
    print("node count AND edge count together (not independent topology")
    print("jitter) -- the entire graph shifts coherently")
    print("=" * 70)
    base_nb, base_prim, base_spec = market.sample_base(config.BASE_MARKET_N, mc,
                                                        seed=config.BASE_MARKET_SEED)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}")

    train_specs = [base_spec] + market.sample_common_scale_topology_dataset(
        base_nb, base_prim, mc, n_instances=config.N_TRAIN_INSTANCES - 1,
        pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 1)
    test_specs = market.sample_common_scale_topology_dataset(
        base_nb, base_prim, mc, n_instances=n_test,
        pct=config.DATASET_PCT, seed=config.TRAIN_SEED + 2)

    print("\nsanity check: plain DR converges on a few training instances")
    for spec in train_specs[:3]:
        net = market.build_network(spec, device=device)
        res = algorithm.run_dr(net, dr_cfg)
        print(f"  N={spec.N:3d}  iters={res.n_iter:5d}  converged={res.converged}  "
              f"feasible={net.is_feasible(1e-3)}")

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode={train_cfg.loss_mode!r})")
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=config.GNN_RHO, scale=config.GNN_SCALE,
        rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    print("\nevaluating: plain DR vs DR + Phi_w, fully distributed inference")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="topology-scale")

    print("\nrecording residual curves on the test set...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = f"trained on: fixed params, graph shared +/-{config.DATASET_PCT*100:.0f}% factor, {config.N_TRAIN_INSTANCES} instances  |  tested on: same distribution, n={n_test}"
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"),
                        subtitle=f"trained on: fixed params, graph shared +/-{config.DATASET_PCT*100:.0f}% factor, {config.N_TRAIN_INSTANCES} instances, {train_cfg.epochs} epochs")

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    weights_path = os.path.join(out_dir, "phi_w.pt")
    trainer.save(weights_path)
    print(f"saved trained weights to {weights_path}")

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: full retraining (centralised, self-supervised, loss_mode={train_cfg.loss_mode!r})
from scratch, using the exact `config.py` recipe validated on zero
diversity (`fixed_market`: 92% win-rate, 1.40x/1.48x mean/median).

**Dataset**: market PARAMETERS EXACTLY fixed (identical to the `fixed_market`
reference for every inherited node) for both training and test sets. The
GRAPH varies via ONE shared random factor in [1-{config.DATASET_PCT}, 1+{config.DATASET_PCT}] applied to
node count AND edge count together (`market.sample_common_scale_topology_dataset`)
-- the topology analogue of `sample_common_scale_dataset`'s single shared
parameter factor. {config.N_TRAIN_INSTANCES} training instances, {n_test} test instances
(disjoint seed draws).

**Question answered**: how well does the network learn when it is TRAINED
(not just zero-shot evaluated) on a market whose TOPOLOGY coherently shifts
by up to {config.DATASET_PCT*100:.0f}% per instance, parameters held fixed? Directly comparable
to `fixed_market_on_topology_scale` (the zero-shot version of this same test
distribution, no retraining) and to `common_scale` (the parameter-axis
counterpart of this same retraining experiment) to see which diversity axis
benefits more from retraining vs. zero-shot transfer.
""")


if __name__ == "__main__":
    main()
