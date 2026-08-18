"""
example_single_dk.py
======================

SINGLE-REFERENCE variant of example_clustered_dk.py: instead of 5
reference markets, ONE reference market (N10/density=0.7, gamma=1.0
corrected baseline), expanded into 50 train + 30 test instances all by
+/-10% perturbation of that SAME reference (market_generator.
ClusteredDatasetConfig with n_references=1) -- one step further than the
5-reference dataset on the heterogeneity spectrum (iid -> 5 references ->
1 reference). Tests the NEW architecture: PerturbationGNN now includes d_k (the
PRE-perturbation local fixed-point residual ||[zD,zG,zq]-[D,G,xbar_q]||,
see Prosumer.dr_step's docstring) as a 9th node feature, giving the gate a
same-instant "does DR still want to move here" signal it did not have
before (only the one-iteration-delayed h_prev_res).

Motivation: measured directly on the OLD architecture (no d_k), the worst
regressions cluster on instances DR ALREADY solves fast, and the
perturbation's magnitude relative to DR's own local step (||v^k||/d^k)
stayed roughly CONSTANT (~0.8-0.9) for the whole trajectory, even when
d^k was already tiny -- i.e. the gate had no way to learn "back off, I'm
not needed here". A cheap 8-epoch leading-indicator check with the new
feature showed the ratio trending down (0.98 -> 0.45) as d^k shrinks,
unlike the old architecture's flat ratio even at full training -- see
conversation history / documentation.txt for the numbers. This run tests
whether that trend, carried to full training, translates into fewer
worst-case regressions end to end.

This is a FEATURE-only change (loss_gamma_cost=0, isolating d_k's effect
from the cost-gap term tested separately in example_clustered_costgap.py)
-- one variable at a time. Evaluated on 30 fresh instances from the start
this time (not 10), given the established lesson that a 10-instance
sample is too noisy to trust for a first read.

    python example_clustered_dk.py
"""

import os

import numpy as np
import torch

from market_generator import (MarketConfig, ClusteredDatasetConfig,
                              build_clustered_datasets)
from training import (TrainConfig, Trainer, compare, Reporter,
                     histories, optimal_costs)
from plot_style import set_paper_style, show_or_save_many

from example_l2play import (
    train, loss_figure, per_instance_figure, band_figure, scatter_figure,
    improvement_histogram, z_residual_figure, cost_gap_figure,
    aux_residual_figure,
)

# ============================== CONFIG ================================
N_REFERENCES     = 1
N_TRAIN_PER_REF  = 50          # 1 x 50 = 50 train instances -- ALL from the single reference
N_TEST_PER_REF   = 30          # 1 x 30 = 30 test instances -- straight to
                               # the larger, less noisy sample; SAME single reference
PCT              = 0.10

NODES           = 10
DENSITY         = 0.7
GRAPH_MODELS    = ("sparse",)

GAMMA, LAM, BISECT = 1.0, 1.0, 25      # the corrected baseline, unchanged
TOL, MAX_ITER = 1e-6, 20000

# --- model / training -- the VALIDATED recipe, unchanged, PLUS d_k feature
# (architectural, always on in this codebase revision -- no flag needed)
HIDDEN = 32
RHO = 0.9
TRAIN_SCALE = 0.3
DEPLOY_SCALE = 1.0
UNROLL = 400
TBPTT_SEGMENT = None
EPOCHS, LR = 40, 3e-3
SEED = 42

LOSS_GAMMA_COST = 0.0   # OFF -- isolate d_k's effect from the cost-gap term

MODEL_PATH = "single_dk_phi_theta.pt"
OUT_DIR = "FixedGraphN10D07test_single_dk"
FORCE_RETRAIN = False
# =====================================================================


def make_datasets():
    market_cfg = MarketConfig(graph_model=GRAPH_MODELS[0], density=DENSITY)
    cfg = ClusteredDatasetConfig(
        n_references=N_REFERENCES, n_train_per_ref=N_TRAIN_PER_REF,
        n_test_per_ref=N_TEST_PER_REF, pct=PCT, N=NODES, density=DENSITY,
        graph_model=GRAPH_MODELS[0], seed=SEED)
    train_ds, test_ds, references = build_clustered_datasets(cfg, market_cfg)
    return (train_ds, test_ds), market_cfg, references


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rep = Reporter(True)

    rep.phase(1, 5, "dataset (single reference, +/-10%)")
    (train_ds, test_ds), market_cfg, references = make_datasets()
    rep.dataset("train", train_ds)
    rep.dataset("test", test_ds)
    rep.kv("topologies", ", ".join(GRAPH_MODELS) + " (ONE graph, shared)")
    rep.kv("density", str(DENSITY))
    rep.kv("clusters", f"{N_REFERENCES} references x (+/-{PCT*100:.0f}%) | "
                       f"train {N_TRAIN_PER_REF}/ref, test {N_TEST_PER_REF}/ref")
    rep.kv("architecture", "NEW: d_k (pre-perturbation local fixed-point "
                          "residual) added as a 9th node feature")
    rep.done()

    model_path = os.path.join(OUT_DIR, MODEL_PATH)
    if not FORCE_RETRAIN and os.path.exists(model_path):
        rep.phase(2, 5, "loading existing model")
        gnn = Trainer.load(model_path)
        cfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT,
                          rho=RHO, scale=TRAIN_SCALE,
                          loss_gamma_cost=LOSS_GAMMA_COST, seed=SEED)
        hist = None
        rep.kv("loaded", f"{model_path} (skipped training)")
        rep.done()
    else:
        rep.phase(2, 5, "centralised training")
        rep.line("")
        gnn, cfg, trainer, hist = train(
            train_ds, market_cfg, val_ds=test_ds, rep=rep,
            hidden=HIDDEN, rho=RHO, train_scale=TRAIN_SCALE,
            unroll=UNROLL, tbptt_segment=TBPTT_SEGMENT,
            gamma=GAMMA, lam=LAM, bisect=BISECT,
            epochs=EPOCHS, lr=LR, seed=SEED,
            loss_gamma_cost=LOSS_GAMMA_COST)
        trainer.save(model_path)
        rep.line("")
        rep.kv("saved", f"{model_path} (trained at scale={TRAIN_SCALE:g})")
        rep.done()

    rep.phase(3, 5, "distributed inference")
    gnn.cfg.scale = DEPLOY_SCALE
    rep.kv("deployment", f"weights copied to every prosumer "
                         f"({gnn.n_parameters()} params each), "
                         f"scale {TRAIN_SCALE:g} (train) -> {DEPLOY_SCALE:g} (deploy)")
    res = compare(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER, reporter=rep)
    speedup = np.asarray(res["speedup"], dtype=float)
    rep.line("")
    rep.kv("median DR", f"{res['median_dr']} iterations")
    rep.kv("median learned", f"{res['median_learned']} iterations")
    rep.kv("mean speed-up", f"{res['mean_speedup']:.2f}x")
    rep.kv("full stats", f"median={np.median(speedup):.3f} std={speedup.std():.3f} "
                         f"min={speedup.min():.3f} max={speedup.max():.3f} "
                         f"regressed={int((speedup < 1.0).sum())}/{len(speedup)}")
    rep.line("(compare against example_single_costgap.py -- same single reference, "
             "cost-gap instead of d_k)")
    rep.done(f"total {rep.elapsed():.1f}s")

    rep.phase(4, 5, "basic figures")
    rep.kv("curves", f"recording full histories on all {len(test_ds)} test instances")
    hist_dr = histories(test_ds, None, cfg, tol=TOL, max_iter=MAX_ITER)
    hist_learned = histories(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER)
    curves_dr = [np.asarray(h["residual"]) for h in hist_dr]
    curves_learned = [np.asarray(h["residual"]) for h in hist_learned]

    set_paper_style()
    figures = []
    if hist is not None:
        figures.append((loss_figure(hist), os.path.join(OUT_DIR, "l2play_training_loss")))
    else:
        rep.line("(no training loss to plot -- model was loaded, not trained)")
    figures.append((per_instance_figure(curves_dr, curves_learned, TOL),
                    os.path.join(OUT_DIR, "l2play_residual")))
    figures.append((band_figure(curves_dr, curves_learned, TOL, GAMMA, LAM, rf"$\rho={RHO}$"),
                    os.path.join(OUT_DIR, "l2play_residual_band")))
    figures.append((scatter_figure(res), os.path.join(OUT_DIR, "l2play_scatter")))
    figures.append((improvement_histogram(res), os.path.join(OUT_DIR, "l2play_improvement_hist")))
    show_or_save_many(figures)
    rep.done()

    rep.phase(5, 5, "extended diagnostics")
    rep.line("reusing per-iteration histories recorded in phase 4")
    rep.line("solving equilibrium QP per instance (exact)")
    J_stars = optimal_costs(test_ds)
    rep.kv("J*", ", ".join(f"{j:.3f}" for j in J_stars))

    fig_aux = aux_residual_figure(hist_dr, hist_learned)
    fig_z = z_residual_figure(hist_dr, hist_learned)
    fig_cost = cost_gap_figure(hist_dr, hist_learned, J_stars)
    show_or_save_many([(fig_aux, os.path.join(OUT_DIR, "l2play_auxresidual")),
                       (fig_z, os.path.join(OUT_DIR, "l2play_zresidual")),
                       (fig_cost, os.path.join(OUT_DIR, "l2play_costgap"))])
    rep.done(f"grand total {rep.elapsed():.1f}s")


if __name__ == "__main__":
    main()
