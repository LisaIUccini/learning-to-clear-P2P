"""
demo_train_realistic_dstar_gres_only.py
==========================================

The SIMPLEST variant of the fixed-x0/parameter-diversity training axis:
perturbs ONLY D_star and G_res (+/-pct, one shared factor each per
instance), leaving D_min/D_max/G_min/G_max EXACTLY fixed -- matching the
original `demo_fixed_market_on_dstar_gres_scale.py` methodology (the
design that degraded gracefully, zero-shot, on the OLD market: 51%->28%
iteration reduction as pct grew 0%->50%, no cliff) instead of v5's
level+width scheme or v7-v13's x0-anchored schemes (both of which also
vary box bounds/width).

Because the box never changes, the ONE shared fixed x0 (drawn once from
the base market) is ALWAYS inside every instance's box automatically --
no clip/adjustment ever needed (verified: 0/40 in calibration). This is
also the only variant on this axis with a properly market-specific
calibration derived BEFORE training via this project's standing rule
(unroll=mean_DR/2, rho=1-10/unroll), rather than reusing another market's
values -- measured on THIS dataset+fixed-x0: mean=572.7, cv=0.0379 (n=40),
giving unroll=286, rho=0.9650.

Recipe otherwise matches the project's established fixed-x0/parameter-
diversity axis: loss_mode='asymmetric', omega_mode='dr_slope' (the pathology
hit today was specific to x0-DIVERSITY training on this market; every
fixed-x0 design in this project, v1-v13, never showed it).

Usage: python3 demo_train_realistic_dstar_gres_only.py <run_name>
       [n_train] [n_test] [epochs] [dr_gamma] [unroll] [gnn_rho] [gnn_scale]
       [pct] [w_ahead] [w_behind] [omega_mode]
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
from demo import plot_results, plot_training_loss, write_run_notes
from build_realistic_training_dataset import (
    build_realistic_base_market, sample_dstar_gres_only_dataset, adjust_x0_to_spec,
    REALISTIC_MARKET_CONFIG,
)
from demo_train_realistic_params_fixed_x0 import (
    build_per_instance_x0, run_pair_fixed_x0, evaluate_fixed_x0,
)


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "train_realistic_dstar_gres_only"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    unroll = int(sys.argv[6]) if len(sys.argv) > 6 else 286
    gnn_rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.9650
    gnn_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    pct = float(sys.argv[9]) if len(sys.argv) > 9 else 0.20
    w_ahead = float(sys.argv[10]) if len(sys.argv) > 10 else config.TRAIN_W_AHEAD
    w_behind = float(sys.argv[11]) if len(sys.argv) > 11 else config.TRAIN_W_BEHIND
    omega_mode = sys.argv[12] if len(sys.argv) > 12 else "dr_slope"

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"pct={pct}  w_ahead={w_ahead}  w_behind={w_behind}  omega_mode={omega_mode})", flush=True)
    assert w_ahead < w_behind, "w_ahead must stay strictly below w_behind -- equal collapses the loss into stability_loss"

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=config.TRAIN_GAMMA_MONO,
                                       epochs=epochs, w_behind=w_behind,
                                       w_ahead=w_ahead, lr=config.TRAIN_LR,
                                       grad_clip=config.TRAIN_GRAD_CLIP, seed=config.TRAIN_SEED,
                                       loss_mode="asymmetric", omega_mode=omega_mode)

    print("=" * 70)
    print("dataset: fixed topology (density=1.0, realistic market, base_seed=0)")
    print("+ ONE shared x0 (drawn once -- box NEVER changes here, so x0 is")
    print("always inside every instance's box automatically, no adjustment")
    print(f"ever needed) -- diversity comes ONLY from perturbing D*/G_res by")
    print(f"+/-{pct:.0%} (one shared factor each per instance), D_min/D_max/G_min/G_max")
    print("held EXACTLY fixed (matching the original D*/G_res-only sweep design")
    print("that degraded gracefully on the old market -- no box-width jitter,")
    print("no box-collapse risk of any kind).")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=0)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    fixed_x0_gen = torch.Generator(device=device); fixed_x0_gen.manual_seed(config.TRAIN_SEED)
    fixed_x0_net = market.build_network(base_spec, device=device, rng=fixed_x0_gen)
    fixed_x0_base = fixed_x0_net.flatten().detach().clone()
    print(f"  shared x0 drawn once (seed={config.TRAIN_SEED})", flush=True)

    train_specs, train_log, n_discarded_train = sample_dstar_gres_only_dataset(
        base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct=pct)
    train_specs = [base_spec] + train_specs
    test_specs, test_log, n_discarded_test = sample_dstar_gres_only_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct=pct)
    test_specs = [base_spec] + test_specs
    print(f"  train: {n_train} FIXED instances (slot 0 = original network, "
          f"{n_train - 1} perturbed), {n_discarded_train} infeasible/negative draws discarded+resampled  |  "
          f"test: {n_test} FIXED instances (slot 0 = original), {n_discarded_test} infeasible/negative draws "
          f"discarded+resampled", flush=True)

    train_x0 = build_per_instance_x0(train_specs, fixed_x0_base, device)
    test_x0 = build_per_instance_x0(test_specs, fixed_x0_base, device)
    n_adjusted_train = sum(1 for spec, x0 in zip(train_specs, train_x0)
                            if not torch.allclose(x0, fixed_x0_base))
    n_adjusted_test = sum(1 for spec, x0 in zip(test_specs, test_x0)
                           if not torch.allclose(x0, fixed_x0_base))
    print(f"  x0 needed adjusting (sanity check -- should be ~0, box never changes): "
          f"{n_adjusted_train}/{n_train} train, {n_adjusted_test}/{n_test} test", flush=True)

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='{omega_mode}')", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device,
                           fixed_x0=train_x0, val_fixed_x0=test_x0[:5])

    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, per-instance-projected fixed x0 --", flush=True)
    print("the only evaluation mode (x0 is an input the user supplies, not something that varies)", flush=True)
    print("-" * 70)
    test_res = evaluate_fixed_x0(test_specs, test_x0, model, dr_cfg, device, label="fixed-x0")

    subtitle = (f"trained on: {n_train} instances (fixed x0+topology, D*/G_res-only +/-{pct:.0%}, "
                f"box bounds fixed, w_ahead={w_ahead}/w_behind={w_behind}), "
                f"{epochs} epochs  |  tested on: n={n_test}, SAME fixed x0")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    x0_by_id = {id(spec): x0 for spec, x0 in zip(test_specs, test_x0)}
    def _run_pair_fn(spec, model_, dr_cfg_, device_, seed, record=False):
        return run_pair_fixed_x0(spec, model_, dr_cfg_, device_, seed, record=record, x0=x0_by_id[id(spec)])
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=_run_pair_fn)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: simplest variant of the fixed-x0/parameter-diversity axis -- perturbs ONLY D_star and
G_res (+/-{pct:.0%}, one shared factor each), D_min/D_max/G_min/G_max held EXACTLY fixed (matching
the original D*/G_res-only sweep design that degraded gracefully on the old market). Realistic
market (density=1.0, base_seed=0, G>=0). Since the box never changes, the one shared fixed x0 is
always inside by construction ({n_adjusted_train}/{n_train} train, {n_adjusted_test}/{n_test} test
needed adjusting).

**Calibration**: unroll={unroll}, rho={gnn_rho} derived from this dataset's OWN plain-DR distribution
under fixed x0 (measured: mean=572.7, cv=0.0379, n=40) via the project's standing rule
(unroll=mean_DR/2, rho=1-10/unroll) -- learned this session that blindly reusing another market's
calibration causes training pathologies.

**Dataset**: {n_train} FIXED training instances (slot 0 = unperturbed original), {n_test} FIXED test
instances (slot 0 = original).

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).

**Question answered**: does the simplest, most stable parameter-diversity design (2 quantities,
box fixed, properly calibrated) produce a well-behaved (if possibly modest, given low cv=0.0379)
result, avoiding every pathology hit by the wider/more complex schemes tried this session?
""")


if __name__ == "__main__":
    main()
