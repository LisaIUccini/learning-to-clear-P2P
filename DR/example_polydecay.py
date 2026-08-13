"""
example_polydecay.py
=====================

Follow-up experiment to example_fixedgraph.py (gamma=1.0 corrected
baseline), testing whether a SLOWER but still summable decay schedule
closes the gap the "DIAGNOSTIC" note in documentation.txt sec. 10 found:
the training rollout's informative window (a small fraction of unroll=400
for every instance) never reaches the horizon plain DR actually needs on
hard instances (up to 6321 iterations at gamma=1.0, ~15x the training
unroll).

What changes vs example_fixedgraph.py
--------------------------------------
  * decay="polynomial", poly_p=1.1 instead of the geometric rho=0.9.
    Combettes' perturbed-KM result (see gnn.py's module docstring) only
    needs a FIXED, theta-independent, summable sequence a_k -- geometric
    is not privileged. a_k=(k+1)^-1.1 has total budget sum_k a_k ~= 10.58,
    matching the geometric schedule's 1/(1-0.9)=10 (so this is a change of
    SHAPE, not of total budget), but stays non-negligible (~7e-5 relative
    to physical_scale) out to k=6000, where the geometric schedule is
    already ~0 to double-precision by k~600.
  * unroll=2400 with tbptt_segment=400 (6 truncated-BPTT windows of the
    exact length already known to train well single-shot) instead of a
    single-shot unroll=400. Needed because a_k is still meaningfully large
    at k=2400 (unlike geometric, where the loss stops seeing anything past
    k~200 regardless of unroll length) -- see documentation.txt sec. 10.

Everything else (dataset, gamma=1.0 baseline, architecture, epochs,
optimiser) is UNCHANGED from example_fixedgraph.py's corrected recipe, so
this is a clean one-variable-at-a-time comparison against that run.

Cheap sanity checks already passed before committing to this run (see
conversation / commit history): summability budgets computed and
poly_p=1.1 chosen to match the geometric total; a short (3-epoch,
6-instance) gradient-health check at unroll=400 and unroll=800 showed no
NaN/Inf and comparable, well-behaved loss/gradient-norm trends to the
geometric baseline. This run is the first one that actually tests the
long-horizon TBPTT credit-assignment question -- genuinely uncertain
outcome, NOT a predicted win (the geometric rho=0.997+TBPTT attempt on
this same dataset collapsed to 0.56x mean, see failed_experiments_log.md
sec. 2.8 -- this is a different decay family, not a repeat of that exact
attempt, but the failure mode it's testing against is real).

    python example_polydecay.py
"""

import os

import numpy as np
import torch

from market_generator import MarketConfig, MarketDataset, DatasetConfig
from training import (TrainConfig, Trainer, compare, Reporter,
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
DENSITY         = 0.7          # SAME dataset as example_fixedgraph.py's
                               # gamma=1.0 corrected baseline
GRAPH_MODELS    = ("sparse",)

# gamma=1.0 -- the CORRECTED baseline (see example_fixedgraph.py and
# documentation.txt sec. 10 "CONFOUND FOUND AND RE-TESTED"). Unchanged
# here so this run isolates the decay-schedule variable alone.
GAMMA, LAM, BISECT = 1.0, 1.0, 25
TOL, MAX_ITER = 1e-6, 20000

# --- model / training ---
HIDDEN = 32
TRAIN_SCALE = 0.3
DEPLOY_SCALE = 1.0

# THE VARIABLE UNDER TEST: polynomial decay instead of geometric.
DECAY = "polynomial"
POLY_P = 1.1            # total budget ~10.58, matches geometric's ~10.0
RHO = 0.9                # unused (decay != "geometric"), kept only so
                         # band_figure's old call sites elsewhere still
                         # have a value if ever reused; not read here.

UNROLL = 2400            # 6x the validated single-shot horizon
TBPTT_SEGMENT = 400      # = the validated single-shot unroll length,
                         # reused as the per-window size

EPOCHS, LR = 40, 3e-3    # UNCHANGED -- reducing epochs/instances as a
                         # cheap proxy for THIS lever is documented as
                         # unreliable (failed_experiments_log.md sec. 2.3)
SEED = 42

MODEL_PATH = "polydecay_phi_theta.pt"
OUT_DIR = "FixedGraphN10D07test_polydecay"
FORCE_RETRAIN = False   # True always retrains, even if MODEL_PATH exists
# =====================================================================


def make_datasets():
    """Same ONE graph (seed=SEED) and same split as
    example_fixedgraph.py's gamma=1.0 run, so the two are directly
    comparable instance-for-instance."""
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
    rep.kv("decay", f"{DECAY} (poly_p={POLY_P}) -- vs example_fixedgraph.py's geometric rho=0.9")
    rep.done()

    # ---------------- phase 2: centralised training ----------------
    model_path = os.path.join(OUT_DIR, MODEL_PATH)
    if not FORCE_RETRAIN and os.path.exists(model_path):
        rep.phase(2, 5, "loading existing model")
        gnn = Trainer.load(model_path)
        cfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT,
                          scale=TRAIN_SCALE, decay=DECAY, poly_p=POLY_P,
                          unroll=UNROLL, tbptt_segment=TBPTT_SEGMENT, seed=SEED)
        hist = None
        rep.kv("loaded", f"{model_path} (skipped training)")
        rep.done()
    else:
        rep.phase(2, 5, "centralised training")
        rep.line("")
        gnn, cfg, trainer, hist = train(
            train_ds, market_cfg, val_ds=test_ds, rep=rep,
            hidden=HIDDEN, train_scale=TRAIN_SCALE,
            unroll=UNROLL, tbptt_segment=TBPTT_SEGMENT,
            gamma=GAMMA, lam=LAM, bisect=BISECT,
            epochs=EPOCHS, lr=LR, seed=SEED,
            decay=DECAY, poly_p=POLY_P)
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
    rep.line("(compare against example_fixedgraph.py's gamma=1.0 run: 1.25x mean)")
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
    figures.append((band_figure(curves_dr, curves_learned, TOL, GAMMA, LAM,
                                rf"polynomial $p={POLY_P}$"),
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
