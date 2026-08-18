"""
example_clustered_costgap.py
=============================

Same dataset as example_clustered.py (5 reference markets, +/-10%
perturbations, N10/density=0.7, gamma=1.0 corrected baseline -- the most
realistic dataset built so far, see market_generator.ClusteredDatasetConfig),
with ONE change: the optional cost-gap-to-J* loss term is switched on
(`TrainConfig.loss_gamma_cost`, see training.stability_loss's docstring
for the full derivation -- absolute value around J*, normalised by the
rollout's own starting gap, added on TOP of the existing residual +
monotonicity terms, not replacing them).

Separate output directory (does not touch example_clustered.py's
FixedGraphN10D07test_clustered results): this is a clean
one-variable-at-a-time comparison against that run's 1.42x mean speed-up.

    python example_clustered_costgap.py
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
N_TEST_PER_REF   = 2           # 5 x 2  = 10 test instances,  example_clustered.py
PCT              = 0.10        # +/- 10% perturbation around each reference

NODES           = 10
DENSITY         = 0.7          # SAME dataset shape as example_clustered.py
GRAPH_MODELS    = ("sparse",)

# gamma=1.0 -- the CORRECTED baseline. Unchanged here so this run isolates
# the loss-term variable alone.
GAMMA, LAM, BISECT = 1.0, 1.0, 25
TOL, MAX_ITER = 1e-6, 20000

# --- model / training -- the VALIDATED recipe, unchanged, PLUS the cost term
HIDDEN = 32
RHO = 0.9
TRAIN_SCALE = 0.3
DEPLOY_SCALE = 1.0
UNROLL = 400
TBPTT_SEGMENT = None
EPOCHS, LR = 40, 3e-3
SEED = 42

# THE VARIABLE UNDER TEST. Kept <= loss_gamma_mono (2.0) by design: the
# residual term stays the dominant signal (no blind spot in q), the cost
# term is a secondary nudge (blind to WHICH point in q-space, since
# trading cost is linear -- see training.stability_loss's docstring).
LOSS_GAMMA_COST = 1.0

MODEL_PATH = "clustered_costgap_phi_theta.pt"
# NEW directory -- does not touch example_clustered.py's results.
OUT_DIR = "FixedGraphN10D07test_clustered_costgap"
FORCE_RETRAIN = False   # True always retrains, even if MODEL_PATH exists
# =====================================================================


def make_datasets():
    """Same construction as example_clustered.py -- N_REFERENCES reference
    markets (ONE shared graph, seed=SEED), each expanded by +/- PCT
    perturbation (disjoint rng streams for train vs test)."""
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
    rep.kv("loss", f"residual + monotonicity + gamma_cost={LOSS_GAMMA_COST:g} "
                   f"cost-gap-to-J* (NEW -- see training.stability_loss)")
    rep.done()

    # ---------------- phase 2: centralised training ----------------
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
    rep.line("(compare against example_clustered.py's run WITHOUT cost-gap: 1.42x mean)")
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
