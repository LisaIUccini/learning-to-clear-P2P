"""
example_fixedgraph.py
======================

Controlled variant of example_l2play.py: ONE graph, drawn once, shared by
every instance (train and test alike). Only the market parameters
(a, b, c, D*, capacities, ...) and the initial state differ from instance
to instance. This removes graph-topology variation as a confound, so any
difference in per-instance convergence behaviour -- why one instance is
fast, why the learned perturbation helps or hurts -- can be attributed to
parameters/initialisation alone, on a simpler, more interpretable example.

Reuses example_l2play.py's model/training code and figure functions
as-is (same architecture, same hyperparameters); only the dataset
construction and the output paths differ.

    python example_fixedgraph.py
"""

import os

import numpy as np
import torch

from market_generator import MarketConfig, MarketDataset, DatasetConfig
from training import (TrainConfig, Trainer, compare, Reporter, residual_curves,
                     histories, optimal_costs)
from plot_style import set_paper_style, show_or_save_many

from example_l2play import (
    train, loss_figure, per_instance_figure, band_figure, scatter_figure,
    improvement_histogram, z_residual_figure, cost_gap_figure,
    aux_residual_figure, 
)

# ============================== CONFIG ================================
TRAIN_INSTANCES = 50
TEST_INSTANCES = 10

NODES           = 10
DENSITY         = 0.7          # more connected than the first fixed-graph
                               # test (0.3, avg degree 4) -- avg degree ~6.8
                               # (34 edges instead of 20) to test whether a
                               # more coupled graph is genuinely harder for
                               # plain DR (more reciprocity constraints
                               # interacting), which the 0.3 draw did not show
GRAPH_MODELS    = ("sparse",)

GAMMA, LAM, BISECT = 2.0, 1.0, 25
TOL, MAX_ITER = 1e-6, 20000

# --- model / training
# REVERTED to the known-good recipe (rho=0.9, unroll=400, no TBPTT). Every
# attempt to extend reach past this (deploy-time rho override, full
# retrain, curriculum, TBPTT -- all documented in example_l2play.py's
# "ATTEMPTS TO EXTEND REACH" note) ended up worse, down to a 0.56x mean
# with the TBPTT/rho=0.997 run on this very dataset. Going back to the
# configuration that is actually proven to work while we re-examine the
# fundamentals, rather than stacking another attempt on top.
HIDDEN = 32
RHO = 0.9
TRAIN_SCALE = 0.3
DEPLOY_SCALE = 1.0
UNROLL = 400
TBPTT_SEGMENT = None
EPOCHS, LR = 40, 3e-3
SEED = 42

MODEL_PATH = "fixedgraph_phi_theta.pt"
OUT_DIR = "FixedGraphN10D07test"
FORCE_RETRAIN = True   # True always retrains, even if MODEL_PATH exists
# =====================================================================


def make_datasets():
    """ONE graph (seed=SEED), shared by every one of the
    TRAIN_INSTANCES + TEST_INSTANCES instances -- only parameters and
    initial state vary."""
    market_cfg = MarketConfig(graph_model=GRAPH_MODELS[0], density=DENSITY)
    cfg = DatasetConfig(n_instances=TRAIN_INSTANCES + TEST_INSTANCES, N=NODES,
                        density=DENSITY, graph_models=GRAPH_MODELS, seed=SEED,
                        fixed_graph=True)
    full = MarketDataset(cfg, market_cfg)
    train_ds, test_ds = full.split(TRAIN_INSTANCES / (TRAIN_INSTANCES + TEST_INSTANCES))
    return (train_ds, test_ds), market_cfg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rep = Reporter(True)

    # ---------------- phase 1: dataset ----------------
    rep.phase(1, 5, "dataset (fixed graph)")
    (train_ds, test_ds), market_cfg = make_datasets()
    rep.dataset("train", train_ds)
    rep.dataset("test", test_ds)
    rep.kv("topologies", ", ".join(GRAPH_MODELS) + " (ONE graph, shared)")
    rep.kv("density", str(DENSITY))
    rep.done()

    # ---------------- phase 2: centralised training ----------------
    model_path = os.path.join(OUT_DIR, MODEL_PATH)
    if not FORCE_RETRAIN and os.path.exists(model_path):
        rep.phase(2, 5, "loading existing model")
        gnn = Trainer.load(model_path)
        cfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT,
                          rho=RHO, scale=TRAIN_SCALE, seed=SEED)
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
            epochs=EPOCHS, lr=LR, seed=SEED)
        trainer.save(model_path)
        rep.line("")
        rep.kv("saved", f"{model_path} (trained at scale={TRAIN_SCALE:g})")
        rep.done()

    # ---------------- phase 3: distributed inference ----------------
    rep.phase(3, 5, "distributed inference")
    gnn.cfg.scale = DEPLOY_SCALE
    rep.kv("deployment", f"weights copied to every prosumer "
                         f"({gnn.n_parameters()} params each), "
                         f"scale {TRAIN_SCALE:g} (train) -> {DEPLOY_SCALE:g} (deploy)")
    res = compare(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER, reporter=rep)
    rep.line("")
    rep.kv("median DR", f"{res['median_dr']} iterations")
    rep.kv("median learned", f"{res['median_learned']} iterations")
    rep.kv("mean speed-up", f"{res['mean_speedup']:.2f}x")
    rep.done(f"total {rep.elapsed():.1f}s")

    # ---------------- phase 4: basic figures ----------------
    rep.phase(4, 5, "basic figures")
    rep.kv("curves", f"recording residuals on all {len(test_ds)} test instances")
    curves_dr = residual_curves(test_ds, None, cfg, tol=TOL, max_iter=MAX_ITER)
    curves_learned = residual_curves(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER)

    set_paper_style()
    figures = []
    if hist is not None:
        figures.append((loss_figure(hist), os.path.join(OUT_DIR, "l2play_training_loss")))
    else:
        rep.line("(no training loss to plot -- model was loaded, not trained)")
    figures.append((per_instance_figure(curves_dr, curves_learned, TOL),
                    os.path.join(OUT_DIR, "l2play_residual")))
    figures.append((band_figure(curves_dr, curves_learned, TOL, GAMMA, LAM, RHO),
                    os.path.join(OUT_DIR, "l2play_residual_band")))
    figures.append((scatter_figure(res), os.path.join(OUT_DIR, "l2play_scatter")))
    figures.append((improvement_histogram(res), os.path.join(OUT_DIR, "l2play_improvement_hist")))
    show_or_save_many(figures)
    rep.done()

    # ---------------- phase 5: extended diagnostics (z-residual, cost gap) ----------------
    rep.phase(5, 5, "extended diagnostics")
    rep.line("recomputing full per-iteration histories (DR and DR+model)")
    hist_dr = histories(test_ds, None, cfg, tol=TOL, max_iter=MAX_ITER)
    hist_learned = histories(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER)
    rep.line("estimating J* per instance (tol=1e-7)")
    J_stars = optimal_costs(test_ds, cfg, tol=1e-7, max_iter=30000)
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
