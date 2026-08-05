"""
example_l2play.py
==================

End-to-end pipeline:

    1. draw a dataset of market instances (train / test, fixed N and
       density for this first pass -- see market_generator.DatasetConfig),
    2. train Phi_theta centrally by unrolling the DR iteration
       (training.Trainer), at TRAIN_SCALE -- SKIPPED if MODEL_PATH already
       exists and FORCE_RETRAIN is False, so re-running the script only to
       re-evaluate/re-plot does not pay for a fresh multi-hour training,
    3. switch the deployed amplitude to DEPLOY_SCALE (see the CONFIG block
       for why this differs from TRAIN_SCALE), then deploy the weights to
       every prosumer and measure, in the FULLY DISTRIBUTED solver, how
       many iterations plain DR and DR+Phi_theta need to reach the same
       tolerance,
    4. plot the comparison.

    python example_l2play.py

Only step 2 is centralised. Steps 3 and 4 run the real distributed loop:
each prosumer queries its own copy of the model after the communication
round, exactly as it would on a device (see gnn.LocalAgent).
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from market_generator import MarketConfig, MarketDataset, DatasetConfig
from gnn import PerturbationGNN, GNNConfig, deploy
from training import TrainConfig, Trainer, compare, Reporter, residual_curves
from plot_style import set_paper_style, show_or_save_many, PALETTE, plot_band

# ============================== CONFIG ================================
# --- dataset: fixed size and density for this first validation pass (see
#     training.py's note on why -- widen N/DENSITY into ranges once the
#     pipeline is confirmed to work, to test zero-shot generalisation).
#     TEST_INSTANCES: the first 7 are the diagnostic set tracked across
#     every experiment so far (see make_datasets -- growing this only
#     APPENDS instances, #1-#7 keep their identity). All 7 now beat plain
#     DR (1.12x-17.01x, mean 5.26x) with the 64-instance/unroll=400 model,
#     but 7 is too small a sample to trust "never worse" in general.
#     Bumped to 30 to validate that on a real sample before optimising
#     anything else further.
#     TRAIN_INSTANCES 32 -> 64: with unroll=400 (much richer per-instance
#     gradient signal, see training.py's note), 32 instances let the two
#     previously-bad instances (#1, #7) become good -- but two previously
#     -good ones (#5, #6) became bad instead, one severely (0.43x). More
#     training diversity is the test for whether that's overfitting to a
#     too-small training pool rather than genuine long-horizon learning.
TRAIN_INSTANCES = 100
TEST_INSTANCES  = 50
NODES           = 10
DENSITY         = 0.3
GRAPH_MODELS    = ("sparse",)

# --- solver: the TUNED baseline (accelerating a slow one proves nothing).
# gamma=1.5 was an untuned guess. A gamma x lambda sweep on plain DR (no
# GNN) over this instance distribution found: lambda >= 1.6 is almost
# always catastrophic here (near-non-convergence within 2000 iterations,
# regardless of gamma); gamma=2.5+ looks great on a small sample (8
# instances) but shows real instability once evaluated on more instances
# (12): some runs fail to converge within 2000 iterations. gamma=2.0,
# lambda=1.0 was the fastest choice with ZERO non-converging instances
# across all 12 -- mean 432 vs 514 for the old (gamma=1.5) default, i.e.
# already ~16% fewer iterations from retuning DR alone, before any
# learned perturbation.
GAMMA, LAM, BISECT = 2.0, 1.0, 25
TOL, MAX_ITER = 1e-6, 20000

# --- model / training
HIDDEN = 32
RHO = 0.9
# TRAIN_SCALE vs DEPLOY_SCALE -- these are DELIBERATELY different.
# Baking a bigger scale into TRAINING itself made things WORSE (0.6 gave
# 1.21x/0.69x vs 0.3's 1.29x/0.73x on the gate-only model): training under
# larger, noisier perturbation dynamics seems to teach a lower-quality
# direction. But with the recurrent memory + local-progress feature added
# on top of the gate (see gnn.PerturbationGNN), a model TRAINED at 0.3 and
# only DEPLOYED at a larger scale keeps improving with no downside: on the
# 16-instance test set, scale=0.3 gives 1.32x mean / 0.85x worst, and just
# turning the deployed amplitude up to 1.0 (same trained weights, no
# retraining) gives 1.85x mean / 0.82x worst -- the worst case stays flat
# across the whole 0.3-1.0 range instead of collapsing the way it did
# without memory (0.73x -> 0.54x over the same range). So: always TRAIN at
# TRAIN_SCALE, then separately choose DEPLOY_SCALE for the amplitude
# actually shipped to the prosumers.
TRAIN_SCALE = 0.3
DEPLOY_SCALE = 1.0
# rho is NOT raised here on purpose. Per-instance diagnostics showed the
# perturbation's effective contribution (scale*rho^k*|direction|) is
# already ~0 by k~150 for EVERY instance, good or bad -- including the
# best ones (4.28x, 2.76x speed-up), which proves a short active window
# CAN be enough to "aim" well. The two instances that end up worse than
# plain DR do so because of a LONG plateau that only appears at k~400-1300,
# entirely after the perturbation has already died out -- so the network
# never gets to react to it in real time either way. What raising unroll
# alone tests is a DIFFERENT, cheaper hypothesis: does the training loss
# simply not look far enough ahead to teach good aim for instances shaped
# like these two, even though rho/deployment behaviour is unaffected by
# this change (amp is still ~0 past k~150)? If this does not help, the
# next step is raising rho AND unroll together, to test the more
# expensive hypothesis (the network needs real-time ability to act deep
# into a plateau, not just better foresight from a longer training window).
UNROLL = 400
# the current loss needs only ONE rollout per instance per epoch (no
# baseline rollout, no equilibrium solve), several times cheaper than the
# earlier designs -- affording more epochs in the same time budget.
EPOCHS, LR = 40, 3e-3
SEED = 42

# --- re-running just to re-evaluate / re-plot ---
MODEL_PATH = "phi_theta.pt"
FORCE_RETRAIN = False   # True always retrains, even if MODEL_PATH exists
# =====================================================================


BASE_TRAIN_INSTANCES = 32  # the original train/test pool size (seed=SEED)


def make_datasets():
    """Test set is FIXED at the original 7 instances (seed=SEED, indices
    32..38) regardless of TRAIN_INSTANCES, so results stay comparable
    across dataset-size experiments -- growing TRAIN_INSTANCES tops up
    training with EXTRA instances drawn from a distinct seed range instead
    of shifting the train/test split point (which would silently swap in
    different test instances and break before/after comparisons)."""
    market_cfg = MarketConfig(graph_model=GRAPH_MODELS[0], density=DENSITY)
    base_cfg = DatasetConfig(n_instances=BASE_TRAIN_INSTANCES + TEST_INSTANCES,
                             N=NODES, density=DENSITY,
                             graph_models=GRAPH_MODELS, seed=SEED)
    base = MarketDataset(base_cfg, market_cfg)
    train_ds, test_ds = base.split(
        BASE_TRAIN_INSTANCES / (BASE_TRAIN_INSTANCES + TEST_INSTANCES))
    if TRAIN_INSTANCES > BASE_TRAIN_INSTANCES:
        extra_cfg = DatasetConfig(n_instances=TRAIN_INSTANCES - BASE_TRAIN_INSTANCES,
                                  N=NODES, density=DENSITY, graph_models=GRAPH_MODELS,
                                  seed=SEED + 777_777)
        train_ds.specs.extend(MarketDataset(extra_cfg, market_cfg).specs)
    return (train_ds, test_ds), market_cfg


def train(train_ds, market_cfg, val_ds=None, rep=None):
    torch.manual_seed(SEED)
    gnn = PerturbationGNN(GNNConfig.from_market_config(
        market_cfg, hidden_dim=HIDDEN, rho=RHO, scale=TRAIN_SCALE))
    cfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT,
                      unroll=UNROLL, rho=RHO, scale=TRAIN_SCALE,
                      epochs=EPOCHS, lr=LR, seed=SEED, verbose=True)
    tr = Trainer(gnn, cfg)
    hist = tr.fit(train_ds, val_dataset=val_ds, reporter=rep)
    return gnn, cfg, tr, hist


def loss_figure(hist):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    epochs = np.arange(1, len(hist["loss"]) + 1)
    ax.plot(epochs, hist["loss"], color=PALETTE[0], label="train")
    if hist["val"]:
        ax.plot(epochs, hist["val"], color=PALETTE[1], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"normalised residual + $\gamma_{\mathrm{mono}}\cdot$monotonicity")
    ax.set_title("training loss (self-referential -- no baseline, no target)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def per_instance_figure(curves_dr, curves_learned, tol, ncols=4):
    """One small subplot PER TEST INSTANCE: DR vs DR+Phi_theta, overlaid.

    Complements `band_figure`: convergence times in this test set can span
    hundreds to tens of thousands of iterations, so a single median curve
    mixes instances done in a few hundred steps with ones still running
    after ten thousand, and `plot_band`'s min_fraction cuts the x-axis
    once few instances remain -- exactly where "did it actually reach
    tol" would be visible. Showing every instance on its own axes (log-y,
    tol marked, final iteration counts in the title) makes that directly
    checkable, per instance -- this is the one to read for "why is
    instance #k different", `band_figure` is the one for "what does the
    aggregate trend look like".
    """
    n = len(curves_dr)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows),
                             squeeze=False)
    for i in range(n):
        ax = axes[i // ncols][i % ncols]
        r_dr, r_l = curves_dr[i], curves_learned[i]
        ax.semilogy(np.arange(1, len(r_dr) + 1), r_dr, color=PALETTE[0],
                   lw=1.0, label="DR")
        ax.semilogy(np.arange(1, len(r_l) + 1), r_l, color=PALETTE[1],
                   lw=1.0, label=r"DR+$\Phi_\theta$")
        ax.axhline(tol, ls="--", lw=0.6, color="0.5")
        speedup = len(r_dr) / max(len(r_l), 1)
        ax.set_title(f"#{i + 1}: DR {len(r_dr)} | learned {len(r_l)} "
                    f"(x{speedup:.2f})", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(rf"distributed DR vs DR+$\Phi_\theta$, per test instance "
                rf"(tol $={tol:g}$)", fontsize=10)
    fig.tight_layout()
    return fig


def band_figure(curves_dr, curves_learned, tol, gamma, lam, rho, min_fraction=0.2):
    """Single plot: median + shaded inter-quartile band across ALL test
    instances. Quick aggregate read of the overall trend; see
    `per_instance_figure`'s docstring for why it can be misleading in
    isolation (mixes very different convergence speeds into one curve,
    and `min_fraction` truncates the x-axis once few instances remain).
    """
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.set_yscale("log")
    plot_band(ax, curves_dr, PALETTE[0], f"DR (median of {len(curves_dr)})",
              min_fraction=min_fraction)
    plot_band(ax, curves_learned, PALETTE[1],
              f"DR + $\\Phi_\\theta$ (median of {len(curves_learned)})",
              min_fraction=min_fraction)
    ax.axhline(tol, ls="--", lw=1.0, color="0.4")
    ax.text(5, tol * 1.4, rf"tol $={tol:g}$", color="0.35", va="bottom", fontsize=8)
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"shadow residual $\|\bar x_{k+1}-\bar x_k\|_2$")
    ax.set_title(rf"distributed DR vs DR+$\Phi_\theta$ over the test set "
                rf"($\gamma={gamma}$, $\lambda={lam}$, $\rho={rho}$) "
                rf"-- median $\pm$ IQR")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def scatter_figure(res):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(res["dr"], res["learned"], s=24, color=PALETTE[0], zorder=3)
    lim = max(max(res["dr"]), max(res["learned"])) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="0.5")
    ax.set_xlabel("DR iterations")
    ax.set_ylabel(r"DR + $\Phi_\theta$ iterations")
    ax.set_title("per-instance comparison (test set)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    fig.tight_layout()
    return fig


def main():
    rep = Reporter(True)

    # ---------------- phase 1: dataset ----------------
    rep.phase(1, 4, "dataset")
    (train_ds, test_ds), market_cfg = make_datasets()
    rep.dataset("train", train_ds)
    rep.dataset("test", test_ds)
    rep.kv("topologies", ", ".join(GRAPH_MODELS))
    rep.kv("density", str(DENSITY))
    rep.done()

    # ---------------- phase 2: centralised training ----------------
    if not FORCE_RETRAIN and os.path.exists(MODEL_PATH):
        rep.phase(2, 4, "loading existing model")
        gnn = Trainer.load(MODEL_PATH)
        cfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT, unroll=UNROLL,
                          rho=RHO, scale=TRAIN_SCALE, epochs=EPOCHS, lr=LR, seed=SEED)
        hist = None
        rep.kv("loaded", f"{MODEL_PATH} (skipped training -- "
                         f"set FORCE_RETRAIN=True to train a fresh one)")
        rep.done()
    else:
        rep.phase(2, 4, "centralised training")
        rep.line("")
        gnn, cfg, trainer, hist = train(train_ds, market_cfg, val_ds=test_ds, rep=rep)
        trainer.save(MODEL_PATH)
        rep.line("")
        rep.kv("saved", f"{MODEL_PATH} (trained at scale={TRAIN_SCALE:g})")
        rep.done()

    # ---------------- phase 3: distributed inference ----------------
    rep.phase(3, 4, "distributed inference")
    gnn.cfg.scale = DEPLOY_SCALE   # amplitude actually shipped to the prosumers
    rep.kv("deployment", f"weights copied to every prosumer "
                         f"({gnn.n_parameters()} params each), "
                         f"scale {TRAIN_SCALE:g} (train) -> {DEPLOY_SCALE:g} (deploy)")
    res = compare(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER, reporter=rep)
    rep.line("")
    rep.kv("median DR", f"{res['median_dr']} iterations")
    rep.kv("median learned", f"{res['median_learned']} iterations")
    rep.kv("mean speed-up", f"{res['mean_speedup']:.2f}x")
    rep.done(f"total {rep.elapsed():.1f}s")

    # ---------------- phase 4: figures ----------------
    rep.phase(4, 4, "figures")
    rep.kv("curves", f"recording residuals on all {len(test_ds)} test instances")
    curves_dr = residual_curves(test_ds, None, cfg, tol=TOL, max_iter=MAX_ITER)
    curves_learned = residual_curves(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER)

    set_paper_style()
    figures = []
    if hist is not None:
        figures.append((loss_figure(hist), "l2play_training_loss"))
    else:
        rep.line("(no training loss to plot -- model was loaded, not trained)")
    figures.append((per_instance_figure(curves_dr, curves_learned, TOL),
                    "l2play_residual"))
    figures.append((band_figure(curves_dr, curves_learned, TOL, GAMMA, LAM, RHO),
                    "l2play_residual_band"))
    figures.append((scatter_figure(res), "l2play_scatter"))
    show_or_save_many(figures)
    rep.done()


if __name__ == "__main__":
    main()
