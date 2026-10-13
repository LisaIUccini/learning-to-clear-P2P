"""
demo_train_combined_levelwidth_regular.py
==========================================

DEGREE-HETEROGENEITY ISOLATION EXPERIMENT: identical recipe/training setup
to `demo_train_combined_levelwidth_x0diversity.py`, but the base market's
topology is a k-REGULAR graph (every node has EXACTLY `degree` neighbors,
zero degree heterogeneity) instead of the usual density-driven random graph
(which has meaningful degree VARIANCE across nodes even at a fixed mean).

Why: across the 5 trained densities, % improvement does NOT move smoothly
with density -- it zigzags (high at 0.2, dips at 0.4-0.6, jumps back up at
0.8, drops again at 1.0). Neither plain-DR difficulty nor degree
heterogeneity (both measured independently) reproduce that exact shape on
their own, but density=1.0 is a special case: it's the ONLY one with zero
degree heterogeneity (complete graph, every node has the same N-1
neighbors). Hypothesis: the GNN's advantage partly comes from exploiting
HOW UNEVENLY connected nodes are (richer per-node message-passing signal),
not just average connectivity -- which would predict that an artificially
regularized graph (same edge count as a real density, but zero variance)
should underperform its naturally-heterogeneous counterpart, closer to how
density=1.0 underperforms density=0.8.

degree=8 is deliberately the densest NON-complete k-regular graph possible
on N=10 nodes (an N-1=9-regular graph on N=10 nodes IS the complete graph),
matching density=0.8's mean degree (7.6) as closely as a regular graph can.

Usage: python3 demo_train_combined_levelwidth_regular.py <run_name>
       [n_train] [n_test] [epochs] [dr_gamma] [unroll] [gnn_rho] [gnn_scale]
       [pct_level] [pct_width] [w_ahead] [w_behind] [gnn_rho_slow]
       [gnn_scale_slow] [gamma_mono] [omega_mode] [hidden_dim]
       [lr_final_frac] [degree] [omega_power] [base_seed]
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
from plot_style import stats_from
from demo_train_realistic_market_x0_diversity import evaluate_with_curves
from build_realistic_training_dataset import build_realistic_base_market_regular, sample_level_width_perturbed_dataset, REALISTIC_MARKET_CONFIG


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "train_combined_levelwidth_regular"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    unroll = int(sys.argv[6]) if len(sys.argv) > 6 else 400
    gnn_rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.975
    gnn_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 1.8
    pct_level = float(sys.argv[9]) if len(sys.argv) > 9 else 0.10
    pct_width = float(sys.argv[10]) if len(sys.argv) > 10 else 0.04
    w_ahead = float(sys.argv[11]) if len(sys.argv) > 11 else 1.6
    w_behind = float(sys.argv[12]) if len(sys.argv) > 12 else 6.0
    gnn_rho_slow = float(sys.argv[13]) if len(sys.argv) > 13 else 0.99
    gnn_scale_slow = float(sys.argv[14]) if len(sys.argv) > 14 else 0.4
    gamma_mono = float(sys.argv[15]) if len(sys.argv) > 15 else config.TRAIN_GAMMA_MONO
    omega_mode = sys.argv[16] if len(sys.argv) > 16 else "dr_slope"
    hidden_dim = int(sys.argv[17]) if len(sys.argv) > 17 else config.GNN_HIDDEN_DIM
    lr_final_frac = float(sys.argv[18]) if len(sys.argv) > 18 and sys.argv[18] != "none" else None
    degree = int(sys.argv[19]) if len(sys.argv) > 19 else 8
    omega_power = float(sys.argv[20]) if len(sys.argv) > 20 else 0.0
    base_seed = int(sys.argv[21]) if len(sys.argv) > 21 else 2

    out_dir = os.path.join("runs", "cross_density", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"pct_level={pct_level}  pct_width={pct_width}  w_ahead={w_ahead}  w_behind={w_behind}  "
          f"gnn_rho_slow={gnn_rho_slow}  gnn_scale_slow={gnn_scale_slow}  gamma_mono={gamma_mono}  "
          f"omega_mode={omega_mode}  hidden_dim={hidden_dim}  lr_final_frac={lr_final_frac}  "
          f"degree={degree} (REGULAR GRAPH)  omega_power={omega_power}  base_seed={base_seed})", flush=True)
    assert w_ahead < w_behind, "w_ahead must stay strictly below w_behind"

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=gamma_mono,
                                       epochs=epochs, w_behind=w_behind, w_ahead=w_ahead,
                                       lr=config.TRAIN_LR, lr_final_frac=lr_final_frac,
                                       grad_clip=config.TRAIN_GRAD_CLIP,
                                       seed=config.TRAIN_SEED, loss_mode="asymmetric", omega_mode=omega_mode,
                                       omega_power=omega_power)

    print("=" * 70)
    print("dataset: COMBINED diversity (level+width market perturbation + free x0), SAME as the")
    print("regular density-sweep recipe, EXCEPT the base topology is k-REGULAR: every node has")
    print(f"EXACTLY degree={degree} neighbors (zero degree heterogeneity), base_seed={base_seed}.")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_realistic_base_market_regular(base_seed=base_seed, degree=degree)
    degrees = [len(v) for v in base_spec.neighbors.values()]
    print(f"  N={base_spec.N}  edges={sum(degrees) // 2}  degree_min={min(degrees)}  degree_max={max(degrees)}", flush=True)

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
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow,
        hidden_dim=hidden_dim))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, FRESH RANDOM x0 per test instance", flush=True)
    print("-" * 70)
    test_res = evaluate_with_curves(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="regular")
    st = stats_from(test_res)
    print(f"  [regular] CORRECT mean % reduction (per-instance average, not 1-1/mean_speedup): "
          f"mean={st['pct_mean']:.1f}%  median={st['pct_median']:.1f}%", flush=True)

    subtitle = (f"trained on: {n_train} instances (level+width +/-{pct_level:.0%}/+/-{pct_width:.0%} "
                f"market diversity + free x0, w_ahead={w_ahead}/w_behind={w_behind}), {epochs} epochs  |  "
                f"REGULAR GRAPH degree={degree}  |  tested on: n={n_test}, fresh random x0")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: DEGREE-HETEROGENEITY ISOLATION -- same combined (level+width + free x0) training recipe
as the density sweep, but the base topology is k-REGULAR (every node has EXACTLY degree={degree}
neighbors, zero degree heterogeneity) instead of density-driven random (which has meaningful
degree variance even at the same mean). degree={degree} is the densest non-complete k-regular
graph possible on N=10 nodes, chosen to match density=0.8's mean degree (7.6) as closely as a
regular graph can, while staying short of the complete graph (density=1.0's own degenerate case).

**Motivation**: isolates whether density=0.8 outperforming density=1.0 (and more generally, the
zigzag shape of % improvement vs density) is driven by degree HETEROGENEITY specifically, as
opposed to average connectivity/edge count alone. If this regular-graph run's % improvement is
close to density=0.8's (~67%), heterogeneity isn't the driver. If it drops toward density=1.0's
level (~38-42%) despite matching 0.8's edge count, heterogeneity is implicated.

**Calibration**: unroll={unroll}, gnn_rho={gnn_rho}, freshly measured on THIS regular topology
(base_seed={base_seed}, degree={degree}, level+width perturbed + fresh x0, n=40).
Note: 2 of 6 candidate seeds for this topology (seeds 0 and 4) showed catastrophic DR
non-convergence (39/40 and 21/40 hitting the iteration cap) -- WORSE than any random-graph density
seed tested so far -- suggesting regular/symmetric topologies may be especially prone to DR limit
cycles. base_seed={base_seed} was chosen as a clean, well-converging seed among the remainder.

**Recipe**: w_ahead={w_ahead}, w_behind={w_behind}, gamma_mono={gamma_mono}, omega_mode={omega_mode},
gnn_scale={gnn_scale}, gnn_rho_slow={gnn_rho_slow}, gnn_scale_slow={gnn_scale_slow} -- matching
density=0.8's own champion recipe as closely as possible, so the ONLY deliberate difference is
topology regularity.

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).
Mean % reduction (per-instance average, the correct definition): {st['pct_mean']:.1f}%, median {st['pct_median']:.1f}%.

**Compare to**: density=0.8 (combined_density08_v1): 67.7% mean / 69.4% median.
density=1.0 (combined_v17_wahead16_s18): 38.5% mean / 39.8% median.
""")


if __name__ == "__main__":
    main()
