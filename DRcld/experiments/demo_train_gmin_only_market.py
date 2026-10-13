"""
demo_train_gmin_only_market.py
==========================================

Same design and script structure as `demo_train_realistic_params_fixed_x0.py`
(fixed topology, ONE shared x0 used RAW/UNCLIPPED, "level+width" perturbation
of D*/G_res/D_min/D_max/G_min/G_max) -- but the BASE market is built from a
config that changes ONLY ONE thing relative to the project's original
abstract-market default (`market.MarketConfig()`): `Gmin_range` moved from
(-2.0, 0.0) to (0.0, 2.0) (same width, just shifted to be non-negative,
matching `seed0_density1.0_seed0_drslope`'s own base market as closely as
possible while enforcing G>=0). Gmax_range, G_res_range and D_spread_range
are all left at the OLD defaults -- NOT `REALISTIC_MARKET_CONFIG`'s
independently-chosen "kW-scale" ranges (Gmax_range=(2,8), G_res_range=
(0.5,2), D_spread_range=(0.4,1)).

Why: isolates whether it's specifically the G>=0 constraint (the ONE change
the user always understood to be there) that matters, separate from the
OTHER three range changes `REALISTIC_MARKET_CONFIG` also introduced for
"kW-scale realism" (never validated as necessary on their own, and flagged
this session as a possible confound). This lets the fixed-x0/parameter-
diversity axis be compared against `seed0_density1.0_seed0_drslope` on a
market as close to that reference's own as G>=0 allows.

**Calibration** (measured 2026-09-11, same methodology/standing rule as
every other market this session -- unroll=mean_DR/2, rho=1-10/unroll,
plain DR under this exact fixed x0 + level/width dataset, n=40):
mean=564.5, cv=0.1006, min=526, max=801 -> unroll=282, rho=0.9645.

Usage: python3 demo_train_gmin_only_market.py <run_name>
       [n_train] [n_test] [epochs] [dr_gamma] [unroll] [gnn_rho] [gnn_scale]
       [pct_level] [pct_width] [w_ahead] [w_behind] [per_node]
       [gnn_scale_slow] [gnn_rho_slow] [gamma_mono] [omega_mode]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import statistics as stats
import sys

import torch

import config
import market
import algorithm
import gnn as gnn_mod
import train as train_mod
from diagnostics import save_instance_plots
from demo import plot_results, plot_training_loss, write_run_notes
from build_realistic_training_dataset import (
    sample_level_width_perturbed_dataset, sample_level_width_per_node_dataset,
)
from demo_train_realistic_params_fixed_x0 import run_pair_fixed_x0, evaluate_fixed_x0

# old abstract-market defaults, EXCEPT Gmin_range shifted to be non-negative
# (same width as the old (-2,0): just relocated, not re-scaled/re-chosen)
GMIN_ONLY_MARKET_CONFIG = market.MarketConfig(
    density=1.0,
    Gmax_range=(2.0, 5.0),
    Gmin_range=(0.0, 2.0),
    G_res_range=(0.0, 2.0),
    D_spread_range=(0.5, 2.0),
)


def build_gmin_only_base_market(base_seed: int = 0):
    return market.sample_base(config.BASE_MARKET_N, GMIN_ONLY_MARKET_CONFIG, seed=base_seed)


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "train_gmin_only_market"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    unroll = int(sys.argv[6]) if len(sys.argv) > 6 else 282
    gnn_rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.9645
    gnn_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    pct_level = float(sys.argv[9]) if len(sys.argv) > 9 else 0.20
    pct_width = float(sys.argv[10]) if len(sys.argv) > 10 else 0.08
    w_ahead = float(sys.argv[11]) if len(sys.argv) > 11 else config.TRAIN_W_AHEAD
    w_behind = float(sys.argv[12]) if len(sys.argv) > 12 else config.TRAIN_W_BEHIND
    per_node = (sys.argv[13] == "1") if len(sys.argv) > 13 else False
    gnn_scale_slow = float(sys.argv[14]) if len(sys.argv) > 14 else config.GNN_SCALE_SLOW
    gnn_rho_slow = float(sys.argv[15]) if len(sys.argv) > 15 else config.GNN_RHO_SLOW
    gamma_mono = float(sys.argv[16]) if len(sys.argv) > 16 else config.TRAIN_GAMMA_MONO
    omega_mode = sys.argv[17] if len(sys.argv) > 17 else "dr_slope"

    sampler = sample_level_width_per_node_dataset if per_node else sample_level_width_perturbed_dataset

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"pct_level={pct_level}  pct_width={pct_width}  w_ahead={w_ahead}  w_behind={w_behind}  "
          f"per_node={per_node}  gnn_scale_slow={gnn_scale_slow}  gnn_rho_slow={gnn_rho_slow}  "
          f"gamma_mono={gamma_mono}  omega_mode={omega_mode})", flush=True)
    assert w_ahead < w_behind, "w_ahead must stay strictly below w_behind -- equal collapses the loss into stability_loss"

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    mc = GMIN_ONLY_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=gamma_mono,
                                       epochs=epochs, w_behind=w_behind,
                                       w_ahead=w_ahead, lr=config.TRAIN_LR,
                                       grad_clip=config.TRAIN_GRAD_CLIP, seed=config.TRAIN_SEED,
                                       loss_mode="asymmetric", omega_mode=omega_mode)

    print("=" * 70)
    print("dataset: fixed topology (density=1.0, same graph as every other")
    print("experiment) + ONE shared x0 (drawn once, used RAW/UNCLIPPED for")
    print("every instance) -- base market is the OLD abstract-market defaults")
    print("with ONLY Gmin_range changed, from (-2,0) to (0,2) (same width,")
    print("just relocated to be non-negative) -- Gmax/G_res/D_spread ranges")
    print("are NOT changed to REALISTIC_MARKET_CONFIG's kW-scale choices, to")
    print("isolate the effect of G>=0 alone from those other range changes.")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_gmin_only_base_market(base_seed=0)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    fixed_x0_gen = torch.Generator(device=device); fixed_x0_gen.manual_seed(config.TRAIN_SEED)
    fixed_x0_net = market.build_network(base_spec, device=device, rng=fixed_x0_gen)
    fixed_x0_base = fixed_x0_net.flatten().detach().clone()
    print(f"  shared x0 drawn once (seed={config.TRAIN_SEED}), used RAW/UNCLIPPED for every instance", flush=True)

    train_specs, train_log, n_discarded_train = sampler(
        base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct_level=pct_level, pct_width=pct_width)
    train_specs = [base_spec] + train_specs
    test_specs, test_log, n_discarded_test = sampler(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=pct_level, pct_width=pct_width)
    test_specs = [base_spec] + test_specs
    print(f"  train: {n_train} FIXED instances (slot 0 = original network, "
          f"{n_train - 1} perturbed), {n_discarded_train} infeasible/negative draws discarded+resampled  |  "
          f"test: {n_test} FIXED instances (slot 0 = original), {n_discarded_test} infeasible/negative draws "
          f"discarded+resampled", flush=True)

    train_x0 = [fixed_x0_base] * len(train_specs)
    test_x0 = [fixed_x0_base] * len(test_specs)

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='{omega_mode}')", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device,
                           fixed_x0=train_x0, val_fixed_x0=test_x0[:5])

    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, per-instance-projected fixed x0 --", flush=True)
    print("-" * 70)
    test_res = evaluate_fixed_x0(test_specs, test_x0, model, dr_cfg, device, label="fixed-x0")

    subtitle = (f"trained on: {n_train} instances (G>=0-only market, fixed x0, level+width "
                f"+/-{pct_level:.0%}/+/-{pct_width:.0%}, w_ahead={w_ahead}/w_behind={w_behind}), "
                f"{epochs} epochs  |  tested on: n={n_test}, SAME shared x0")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    x0_by_id = {id(spec): x0 for spec, x0 in zip(test_specs, test_x0)}
    def _run_pair_fn(spec, model_, dr_cfg_, device_, seed, record=False):
        return run_pair_fixed_x0(spec, model_, dr_cfg_, device_, seed, record=record, x0=x0_by_id[id(spec)])
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=_run_pair_fn)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: fixed-x0/parameter-diversity design (same as v6, the project's best fixed-x0 result,
1.25x/100%), but the base market changes ONLY `Gmin_range`: (-2,0) -> (0,2) (old abstract-market
defaults otherwise, NOT `REALISTIC_MARKET_CONFIG`'s independently-chosen kW-scale ranges for
Gmax/G_res/D_spread). Isolates whether G>=0 alone (the one change the user always understood to be
there, vs `seed0_density1.0_seed0_drslope`'s own base market) explains any of the gap to that
reference's 2.09x/100%, separate from the other three range changes.

**Calibration**: unroll={unroll}, gnn_rho={gnn_rho}, freshly measured on this exact dataset (fixed,
unclipped x0) via the project's standing rule: mean=564.5, cv=0.1006, n=40.

**Recipe**: identical to v6 otherwise -- w_ahead={w_ahead}, w_behind={w_behind}, gamma_mono={gamma_mono},
omega_mode={omega_mode}, gnn_scale={gnn_scale}, level+width perturbation +/-{pct_level:.0%}/+/-{pct_width:.0%}.

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).
""")


if __name__ == "__main__":
    main()
