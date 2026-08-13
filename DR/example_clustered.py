"""
example_clustered.py
=====================

More realistic dataset variant of example_fixedgraph.py's validated
gamma=1.0 recipe: instead of every instance's market parameters being an
INDEPENDENT draw over the whole MarketConfig range (the existing
DatasetConfig(fixed_graph=True) scheme), instances are grouped into a
handful of CLUSTERS, each built by perturbing ONE reference market's own
parameters by +/- `PCT` (market_generator.ClusteredDatasetConfig /
build_clustered_datasets). Same graph throughout, as before.

This is meant to resemble a real deployment more closely: one physical
grid, a handful of typical operating regimes (references), small
day-to-day variation within each (the perturbed instances) -- rather than
market conditions drawn uniformly across a wide range every single time,
which is a much more heterogeneous and less realistic instance
distribution.

Train and test share the SAME references but use disjoint perturbed
instances (see ClusteredDatasetConfig's docstring) -- this tests
generalisation to new day-to-day variation on a market the network has
already seen the character of, not generalisation to an unseen market.

Everything else (gamma=1.0 corrected baseline, geometric rho=0.9 decay,
architecture, epochs, optimiser) is UNCHANGED from example_fixedgraph.py,
so this is a clean one-variable-at-a-time comparison (dataset structure)
against that run's 1.25x mean speed-up.

    python example_clustered.py
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
N_REFERENCES     = 5
N_TRAIN_PER_REF  = 10          # 5 x 10 = 50 train instances, matching
N_TEST_PER_REF   = 2           # 5 x 2  = 10 test instances,  example_fixedgraph.py
PCT              = 0.10        # +/- 10% perturbation around each reference

NODES           = 10
DENSITY         = 0.7          # SAME dataset shape as example_fixedgraph.py's
                               # gamma=1.0 corrected baseline
GRAPH_MODELS    = ("sparse",)

# gamma=1.0 -- the CORRECTED baseline (example_fixedgraph.py / documentation.txt
# sec. 10 "CONFOUND FOUND AND RE-TESTED"). Unchanged here so this run isolates
# the dataset-structure variable alone.
GAMMA, LAM, BISECT = 1.0, 1.0, 25
TOL, MAX_ITER = 1e-6, 20000

# --- model / training -- the VALIDATED recipe, unchanged ---
HIDDEN = 32
RHO = 0.9
TRAIN_SCALE = 0.3
DEPLOY_SCALE = 1.0
UNROLL = 400
TBPTT_SEGMENT = None
EPOCHS, LR = 40, 3e-3
SEED = 42

MODEL_PATH = "clustered_phi_theta.pt"
OUT_DIR = "FixedGraphN10D07test_clustered"
FORCE_RETRAIN = False   # True always retrains, even if MODEL_PATH exists
# =====================================================================


def make_datasets():
    """N_REFERENCES reference markets (ONE shared graph, seed=SEED), each
    expanded into N_TRAIN_PER_REF / N_TEST_PER_REF instances by +/- PCT
    perturbation of the reference's own parameters (disjoint rng streams
    for train vs test, see ClusteredDatasetConfig)."""
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

    # ---------------- phase 1: dataset ----------------
    rep.phase(1, 5, "dataset (clustered, +/-10% around 5 references)")
    (train_ds, test_ds), market_cfg, references = make_datasets()
    rep.dataset("train", train_ds)
    rep.dataset("test", test_ds)
    rep.kv("topologies", ", ".join(GRAPH_MODELS) + " (ONE graph, shared)")
    rep.kv("density", str(DENSITY))
    rep.kv("clusters", f"{N_REFERENCES} references x (+/-{PCT*100:.0f}%) | "
                       f"train {N_TRAIN_PER_REF}/ref, test {N_TEST_PER_REF}/ref "
                       f"(disjoint perturbations, same references)")
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
    rep.line("(compare against example_fixedgraph.py's gamma=1.0 iid-dataset run: 1.25x mean)")
    rep.done(f"total {rep.elapsed():.1f}s")

    # ---------------- phase 4: basic figures ----------------
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

    # ---------------- phase 5: extended diagnostics (z-residual, cost gap) ----------------
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
