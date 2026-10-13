"""
demo_train_combined_gmin_only.py
==========================================

Same COMBINED diversity design as `demo_train_combined_levelwidth_x0diversity.py`
(market perturbed via level+width, +/-10%/+/-4% default, AND x0 left FREE --
fresh random per instance/epoch during training) -- but the base market is
`GMIN_ONLY_MARKET_CONFIG` (old abstract-market defaults, Gmin_range shifted
to (0,2) -- ONLY G>=0 changed, none of REALISTIC_MARKET_CONFIG's other three
"kW-scale" range changes) instead of the full realistic market.

Motivation (2026-09-12): the combined design on the FULL realistic market
(`combined_levelwidth10_4_x0diversity_v1`) scored 1.53x/100% home, and
generalized gracefully up to +/-20-30% zero-shot. The user asked whether
this strong result is driven by the x0-diversity mechanism itself or by
something specific to the realistic market's own numeric ranges -- this run
holds the "market perturbation + free x0" MECHANISM fixed and swaps only
the base market (to the minimal G>=0-only ablation) to isolate that.

**Calibration** (measured 2026-09-12, standing rule, n=40 draws = one
level+width-perturbed gmin-only-market instance + one fresh random x0):
see printed banner at launch for the exact numbers used.

Usage: python3 demo_train_combined_gmin_only.py <run_name>
       [n_train] [n_test] [epochs] [dr_gamma] [unroll] [gnn_rho] [gnn_scale]
       [pct_level] [pct_width] [w_ahead] [w_behind] [gnn_rho_slow]
       [gnn_scale_slow] [gamma_mono] [omega_mode]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import statistics as stats
import sys

import torch

import config
import algorithm
import gnn as gnn_mod
import train as train_mod
from diagnostics import save_instance_plots
from demo import plot_results, plot_training_loss, write_run_notes, run_pair
from demo_train_realistic_market_x0_diversity import evaluate_with_curves
from build_realistic_training_dataset import sample_level_width_perturbed_dataset
from demo_train_gmin_only_market import GMIN_ONLY_MARKET_CONFIG, build_gmin_only_base_market


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "train_combined_gmin_only"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    unroll = int(sys.argv[6]) if len(sys.argv) > 6 else 386
    gnn_rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.9741
    gnn_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    pct_level = float(sys.argv[9]) if len(sys.argv) > 9 else 0.10
    pct_width = float(sys.argv[10]) if len(sys.argv) > 10 else 0.04
    w_ahead = float(sys.argv[11]) if len(sys.argv) > 11 else 1.4
    w_behind = float(sys.argv[12]) if len(sys.argv) > 12 else 6.0
    gnn_rho_slow = float(sys.argv[13]) if len(sys.argv) > 13 else 0.99
    gnn_scale_slow = float(sys.argv[14]) if len(sys.argv) > 14 else 0.4
    gamma_mono = float(sys.argv[15]) if len(sys.argv) > 15 else config.TRAIN_GAMMA_MONO
    omega_mode = sys.argv[16] if len(sys.argv) > 16 else "dr_slope"

    out_dir = os.path.join("runs", "cross_density", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"pct_level={pct_level}  pct_width={pct_width}  w_ahead={w_ahead}  w_behind={w_behind}  "
          f"gnn_rho_slow={gnn_rho_slow}  gnn_scale_slow={gnn_scale_slow}  gamma_mono={gamma_mono}  "
          f"omega_mode={omega_mode})", flush=True)
    assert w_ahead < w_behind, "w_ahead must stay strictly below w_behind"

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    mc = GMIN_ONLY_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=gamma_mono,
                                       epochs=epochs, w_behind=w_behind, w_ahead=w_ahead,
                                       lr=config.TRAIN_LR, grad_clip=config.TRAIN_GRAD_CLIP,
                                       seed=config.TRAIN_SEED, loss_mode="asymmetric", omega_mode=omega_mode)

    print("=" * 70)
    print("dataset: COMBINED diversity on the GMIN-ONLY market (old abstract")
    print("defaults, Gmin_range=(0,2) -- ONLY G>=0 changed, NOT the other three")
    print("REALISTIC_MARKET_CONFIG range changes) -- market instances perturbed")
    print(f"via level+width (+/-{pct_level:.0%} level, +/-{pct_width:.0%} width) AND x0 left FREE.")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_gmin_only_base_market(base_seed=0)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    train_specs, train_log, n_discarded_train = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct_level=pct_level, pct_width=pct_width)
    train_specs = [base_spec] + train_specs
    test_specs, test_log, n_discarded_test = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=pct_level, pct_width=pct_width)
    test_specs = [base_spec] + test_specs
    print(f"  train: {n_train} instances (slot 0 = original), {n_discarded_train} infeasible discarded  |  "
          f"test: {n_test} instances (slot 0 = original), {n_discarded_test} infeasible discarded", flush=True)
    print("  x0: FRESH RANDOM per (instance, epoch) during training, fresh per test instance at eval", flush=True)

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='{omega_mode}')", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, FRESH RANDOM x0 per test instance", flush=True)
    print("-" * 70)
    test_res = evaluate_with_curves(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="combined-gmin")

    subtitle = (f"trained on: {n_train} instances (GMIN-ONLY market, level+width "
                f"+/-{pct_level:.0%}/+/-{pct_width:.0%} + free x0, w_ahead={w_ahead}/w_behind={w_behind}), "
                f"{epochs} epochs  |  tested on: n={n_test}, fresh random x0")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: same COMBINED diversity design as `combined_levelwidth10_4_x0diversity_v1` (market
perturbed via level+width +/-{pct_level:.0%}/+/-{pct_width:.0%}, x0 left FREE), but on the GMIN-ONLY
market (old abstract defaults, Gmin_range=(0,2) -- ONLY G>=0 changed, not the other three
REALISTIC_MARKET_CONFIG range changes) instead of the full kW-scale realistic market.

**Motivation**: isolates whether the combined design's strong result (1.53x/100% on the full
realistic market) is driven by the free-x0 mechanism itself or by something specific to
REALISTIC_MARKET_CONFIG's own numeric ranges -- holds the market-perturbation+free-x0 MECHANISM
fixed, swaps only the base market.

**Calibration**: unroll={unroll}, gnn_rho={gnn_rho}, freshly measured on this exact combined
distribution (level+width perturbed gmin-only market + fresh random x0, n=40).

**Recipe**: w_ahead={w_ahead}, w_behind={w_behind}, gamma_mono={gamma_mono}, omega_mode={omega_mode},
gnn_scale={gnn_scale}, gnn_rho_slow={gnn_rho_slow}, gnn_scale_slow={gnn_scale_slow}.

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).

**Comparison**: combined+realistic (1.53x/100%) vs combined+gmin-only (this run) -- if similar,
confirms the free-x0 mechanism (not REALISTIC_MARKET_CONFIG's specific ranges) drives the result.
""")


if __name__ == "__main__":
    main()
