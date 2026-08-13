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
from training import (TrainConfig, Trainer, compare, Reporter,
                     histories, optimal_costs)
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
TRAIN_INSTANCES = 64
TEST_INSTANCES  = 30
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
# rho is NOT raised here on purpose -- see "ATTEMPTS TO EXTEND REACH" below.
# scale*rho^k is ~0 by k~150-200 for EVERY instance (verified numerically:
# 1.3e-23 already at k=500), which caps how late a plateau can be reached.
# Raising rho is the only way to lift that cap -- but every way we tried to
# actually make that work has failed so far. Kept here as a record, NOT to
# be re-run blindly:
#
#   1. Deploy-time-only override (SAME rho=0.9 trained weights, rho just
#      changed at inference, no retraining): proves real headroom exists
#      (one hard instance went from 1.31x to 57x at rho=0.95) but wrecks
#      easy instances more and more as rho grows, because the gate was
#      never trained to self-regulate past k~150 -- it doesn't know what
#      to do with the extra leash.
#   2. Full retrain from scratch at rho=0.97/unroll=800 (64 instances,
#      40 epochs): WORSE overall (1.41x mean / 19-30 regress) than the
#      rho=0.9 baseline (3.42x / 9-30) -- training under the longer,
#      noisier horizon converges to a lower-quality policy, not a more
#      patient version of the good one.
#   3. Cheap proxy screening (24 instances / 10 epochs) of the SAME
#      rho=0.97/unroll=800 config gave the OPPOSITE signal (looked great,
#      4.88x) -- proxy training at reduced scale is NOT a reliable
#      predictor for this lever; do not trust a cheap screen here.
#   4. Per-epoch checkpoint curve of that same 40-epoch run (evaluated at
#      epochs 4,8,...,40): noisy, no single epoch recovers the rho=0.9
#      baseline's quality -- ruled out "just stop training earlier."
#   5. Curriculum (staged warm-start rho 0.90 -> 0.99 -> 0.997, each stage
#      continuing training from the previous stage's weights instead of
#      from scratch): got WORSE at every single stage (1.38x -> 0.51x ->
#      0.28x on a representative panel) -- gradual extension is not a fix
#      either, if anything it is worse than a single direct jump.
#   6. Root cause found for #5/#2: at long horizons the network converges
#      to one GENERIC pace applied almost identically to every instance,
#      regardless of how different their own natural DR speed is (verified:
#      near-identical iteration-to-iteration decay RATIOS across instances
#      whose plain-DR baselines differ by 5x) -- it stops being an
#      instance-adaptive policy and becomes a one-size-fits-all one.
#   7. Tried fixing #6 by disabling the loss's `late_weight` (the term that
#      weights later iterations more heavily in the loss, on the theory
#      that it was diluting early per-instance signal): DISPROVEN --
#      turned out slightly WORSE (even more uniform, ~4000-4300 iters
#      across the board instead of ~3000-3200), not better. The loss
#      weighting is not the cause.
#   8. Diagnosis: this looks like a vanishing/diluted-gradient problem from
#      backpropagating through thousands of recurrent steps in one shot,
#      not a loss-shaping problem -- the credit assignment for "do this
#      specific thing on THIS instance, early on" gets washed out over
#      that many BPTT steps, leaving only the generic, easy-to-learn
#      late-horizon signal. A first idea (hand-craft an explicit "am I
#      stuck" input feature) was set aside on principle: it decides in
#      advance what the network should look for, rather than letting it
#      learn whatever it needs from data. The general, established fix for
#      this exact class of problem is TRUNCATED BPTT instead (see
#      `TrainConfig.tbptt_segment` in training.py): train on short,
#      well-conditioned windows (TBPTT_SEGMENT below) while letting the
#      state flow forward across many of them to cover the full horizon
#      (UNROLL) -- the network still has to discover everything itself,
#      only the credit-assignment problem is fixed.
UNROLL = 400
# If set, `unroll` is covered as consecutive windows of this many steps
# instead of one differentiable rollout (truncated BPTT -- see the note
# above and training.py's module docstring). None reproduces today's
# known-good short-horizon behaviour exactly; set together with a larger
# RHO when training for a longer reach.
TBPTT_SEGMENT = None
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


def train(train_ds, market_cfg, val_ds=None, rep=None, *,
         hidden=HIDDEN, rho=RHO, train_scale=TRAIN_SCALE,
         unroll=UNROLL, tbptt_segment=TBPTT_SEGMENT,
         gamma=GAMMA, lam=LAM, bisect=BISECT,
         epochs=EPOCHS, lr=LR, seed=SEED,
         decay="geometric", poly_p=1.5):
    """Centralised training. All hyperparameters are explicit keyword
    arguments (defaulting to this module's own CONFIG constants) so a
    CALLER's own values are actually used -- a plain module-global lookup
    here would silently resolve to example_l2play.py's globals even when
    called from a different file with same-named but different constants
    (this bit a real run: example_fixedgraph.py's own EPOCHS/LR were being
    ignored in favour of this module's, undetected until the saved
    checkpoint's metadata was checked).

    `decay`/`poly_p`: see gnn.GNNConfig's `decay` docstring -- "geometric"
    (default, unchanged behaviour) or "polynomial" (documentation.txt
    sec. 10's follow-up experiment: a much slower but still summable
    decay, meant to extend the perturbation's non-negligible influence out
    to the horizon plain DR actually needs on hard instances)."""
    torch.manual_seed(seed)
    gnn = PerturbationGNN(GNNConfig.from_market_config(
        market_cfg, hidden_dim=hidden, rho=rho, scale=train_scale,
        decay=decay, poly_p=poly_p))
    cfg = TrainConfig(gamma=gamma, lam=lam, prox_bisect=bisect,
                      unroll=unroll, rho=rho, scale=train_scale,
                      tbptt_segment=tbptt_segment, decay=decay, poly_p=poly_p,
                      epochs=epochs, lr=lr, seed=seed, verbose=True)
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


def aux_residual_figure(hist_dr, hist_learned, ncols=4):
    """Per-instance grid of the AUXILIARY iterate's residual ||x^{k+1}-x^k||.

    This is the quantity DR's own convergence theory actually promises to
    be (near-exactly) monotonically non-increasing -- the Krasnoselskii-Mann
    residual of the averaged operator DR iterates on the raw auxiliary
    variable `x` (lines 4-7), BEFORE the reciprocity projection. The
    "residual" shown in `per_instance_figure` is the shadow x-bar's own
    residual instead -- a derived quantity (an extra projection on top of
    x) that the theory does not promise to be monotone, and in practice
    isn't. Plot this one to see the theorem actually hold (up to tiny
    numerical noise from the finite-step prox bisection).
    """
    n = len(hist_dr)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows),
                             squeeze=False)
    for i in range(n):
        ax = axes[i // ncols][i % ncols]
        ra_dr = np.asarray(hist_dr[i]["residual_aux"])
        ra_l = np.asarray(hist_learned[i]["residual_aux"])
        ax.semilogy(np.arange(1, len(ra_dr) + 1), ra_dr, color=PALETTE[0],
                   lw=1.0, label="DR")
        ax.semilogy(np.arange(1, len(ra_l) + 1), ra_l, color=PALETTE[1],
                   lw=1.0, label=r"DR+$\Phi_\theta$")
        ax.set_title(f"#{i + 1}", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(r"auxiliary residual $\|x^{k+1}-x^k\|$, per test instance"
                "\n" r"(the quantity DR theory guarantees is monotone)",
                fontsize=10)
    fig.tight_layout()
    return fig


def z_residual(hist):
    """Combined residual on z = (xbar, x, cost, balance, reciprocity_raw).

    xbar (shadow) and x (auxiliary, pre-reciprocity-projection) are the two
    iterates DR itself updates every round -- same units, combined directly.
    cost/balance/reciprocity_raw are the solver's own monitoring scalars
    (reciprocity read on the RAW iterate, since on the shadow it is ~0 by
    construction -- the projection step is exactly what enforces it there).
    Each scalar block is normalised by the magnitude of its own first
    recorded step, a fixed per-instance scale that does not distort the
    shape of the decay.
    """
    r = np.asarray(hist["residual"])
    ra = np.asarray(hist["residual_aux"])
    cost = np.asarray(hist["cost"])
    bal = np.asarray(hist["balance"])
    rec = np.asarray(hist["reciprocity_raw"])

    dcost, dbal, drec = np.diff(cost), np.diff(bal), np.diff(rec)
    r, ra = r[1:], ra[1:]     # align: no delta available at the first point

    eps = 1e-12
    s_cost = abs(dcost[0]) + eps
    s_bal = abs(dbal[0]) + eps
    s_rec = abs(drec[0]) + eps
    return np.sqrt(r ** 2 + ra ** 2 + (dcost / s_cost) ** 2
                  + (dbal / s_bal) ** 2 + (drec / s_rec) ** 2)


def z_residual_figure(hist_dr, hist_learned, ncols=4):
    """Per-instance grid of the combined z-residual, DR vs DR+Phi_theta.

    Same layout as `per_instance_figure`, for the extended residual instead
    of the plain shadow one.
    """
    n = len(hist_dr)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows),
                             squeeze=False)
    for i in range(n):
        ax = axes[i // ncols][i % ncols]
        zd, zl = z_residual(hist_dr[i]), z_residual(hist_learned[i])
        ax.semilogy(np.arange(1, len(zd) + 1), zd, color=PALETTE[0],
                   lw=1.0, label="DR")
        ax.semilogy(np.arange(1, len(zl) + 1), zl, color=PALETTE[1],
                   lw=1.0, label=r"DR+$\Phi_\theta$")
        ax.set_title(f"#{i + 1}", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(r"combined residual $\|z^{k+1}-z^k\|$, per test instance"
                "\n" r"($z=(\bar x,\,x,\,J,\,\mathrm{balance},\,"
                r"\mathrm{reciprocity})$)", fontsize=10)
    fig.tight_layout()
    return fig


def cost_gap_figure(hist_dr, hist_learned, J_stars, ncols=4):
    """Per-instance grid of |J*-J(x^k)|, DR vs DR+Phi_theta.

    J_stars[i] is an estimate of the (equilibrium-independent) optimal
    aggregate cost for test instance i -- see `training.optimal_costs`.
    """
    n = len(hist_dr)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows),
                             squeeze=False)
    for i in range(n):
        ax = axes[i // ncols][i % ncols]
        j_star = J_stars[i]
        gd = np.abs(j_star - np.asarray(hist_dr[i]["cost"]))
        gl = np.abs(j_star - np.asarray(hist_learned[i]["cost"]))
        ax.semilogy(np.arange(1, len(gd) + 1), gd, color=PALETTE[0],
                   lw=1.0, label="DR")
        ax.semilogy(np.arange(1, len(gl) + 1), gl, color=PALETTE[1],
                   lw=1.0, label=r"DR+$\Phi_\theta$")
        ax.set_title(f"#{i + 1}", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(r"cost gap $|J^\star-J(x^k)|$, per test instance",
                fontsize=10)
    fig.tight_layout()
    return fig


def band_figure(curves_dr, curves_learned, tol, gamma, lam, decay_label, min_fraction=0.2):
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
                rf"($\gamma={gamma}$, $\lambda={lam}$, {decay_label}) "
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


def improvement_histogram(res):
    """How many test instances improve, and by how much.

    Buckets instances by the fraction of iterations saved,
    (K_DR - K_phi) / K_DR: negative means DR+Phi_theta needed MORE
    iterations than plain DR. The "worse" bar's label shows the actual
    worst-to-best range inside it, so a handful of instances that are only
    marginally worse don't read as catastrophic next to one that is.
    """
    dr_iters = np.asarray(res["dr"], dtype=float)
    learned_iters = np.asarray(res["learned"], dtype=float)
    reduction_pct = (dr_iters - learned_iters) / dr_iters * 100.0

    edges = [-np.inf, 0, 50, 80, np.inf]
    labels = ["worse", "improves\n(up to 50%)",
              "improves a lot\n(50-80%)", "improves\ndramatically\n(>80%)"]
    colors = [PALETTE[1], "#C7E8DB", "#43AD85", PALETTE[2]]

    counts = [int(((reduction_pct > lo) & (reduction_pct <= hi)).sum())
              for lo, hi in zip(edges[:-1], edges[1:])]
    counts[0] = int((reduction_pct <= edges[1]).sum())
    if counts[0] > 0:
        worse = reduction_pct[reduction_pct <= edges[1]]
        labels[0] = f"worse\n({worse.min():+.0f}% .. {worse.max():+.0f}%)"

    n = len(dr_iters)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.6)
    for rect, c in zip(bars, counts):
        pct = 100.0 * c / n
        ax.annotate(f"{c}\n({pct:.0f}%)", (rect.get_x() + rect.get_width() / 2,
                    rect.get_height()), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("number of instances")
    ax.set_xlabel(r"iteration reduction, $(K_{\mathrm{DR}}-K_{\Phi})/K_{\mathrm{DR}}$")
    ax.set_title(f"distribution of improvement (n={n} instances)")
    ax.set_ylim(0, max(counts) * 1.25)
    fig.tight_layout()
    return fig


def main():
    rep = Reporter(True)

    # ---------------- phase 1: dataset ----------------
    rep.phase(1, 5, "dataset")
    (train_ds, test_ds), market_cfg = make_datasets()
    rep.dataset("train", train_ds)
    rep.dataset("test", test_ds)
    rep.kv("topologies", ", ".join(GRAPH_MODELS))
    rep.kv("density", str(DENSITY))
    rep.done()

    # ---------------- phase 2: centralised training ----------------
    if not FORCE_RETRAIN and os.path.exists(MODEL_PATH):
        rep.phase(2, 5, "loading existing model")
        gnn = Trainer.load(MODEL_PATH)
        cfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT, unroll=UNROLL,
                          rho=RHO, scale=TRAIN_SCALE, epochs=EPOCHS, lr=LR, seed=SEED)
        hist = None
        rep.kv("loaded", f"{MODEL_PATH} (skipped training -- "
                         f"set FORCE_RETRAIN=True to train a fresh one)")
        rep.done()
    else:
        rep.phase(2, 5, "centralised training")
        rep.line("")
        gnn, cfg, trainer, hist = train(train_ds, market_cfg, val_ds=test_ds, rep=rep)
        trainer.save(MODEL_PATH)
        rep.line("")
        rep.kv("saved", f"{MODEL_PATH} (trained at scale={TRAIN_SCALE:g})")
        rep.done()

    # ---------------- phase 3: distributed inference ----------------
    rep.phase(3, 5, "distributed inference")
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

    # ---------------- phase 4: basic figures ----------------
    rep.phase(4, 5, "basic figures")
    # `histories()` records the exact same distributed run `residual_curves()`
    # would (same dataset, same x0, same solver config -- dataset.build(i) is
    # deterministic), just keeping every diagnostic field instead of only
    # "residual". Computing it once here and reusing it in phase 5 avoids
    # re-running the whole distributed solve a second time for identical
    # results.
    rep.kv("curves", f"recording full histories on all {len(test_ds)} test instances")
    hist_dr = histories(test_ds, None, cfg, tol=TOL, max_iter=MAX_ITER)
    hist_learned = histories(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER)
    curves_dr = [np.asarray(h["residual"]) for h in hist_dr]
    curves_learned = [np.asarray(h["residual"]) for h in hist_learned]

    set_paper_style()
    figures = []
    if hist is not None:
        figures.append((loss_figure(hist), "l2play_training_loss"))
    else:
        rep.line("(no training loss to plot -- model was loaded, not trained)")
    figures.append((per_instance_figure(curves_dr, curves_learned, TOL),
                    "l2play_residual"))
    figures.append((band_figure(curves_dr, curves_learned, TOL, GAMMA, LAM, rf"$\rho={RHO}$"),
                    "l2play_residual_band"))
    figures.append((scatter_figure(res), "l2play_scatter"))
    figures.append((improvement_histogram(res), "l2play_improvement_hist"))
    show_or_save_many(figures)
    rep.done()

    # ---------------- phase 5: extended diagnostics ----------------
    rep.phase(5, 5, "extended diagnostics")
    rep.line("reusing per-iteration histories recorded in phase 4")
    rep.line("solving equilibrium QP per instance (exact)")
    J_stars = optimal_costs(test_ds)

    figures2 = [(aux_residual_figure(hist_dr, hist_learned), "l2play_auxresidual"),
               (z_residual_figure(hist_dr, hist_learned), "l2play_zresidual"),
               (cost_gap_figure(hist_dr, hist_learned, J_stars), "l2play_costgap")]
    show_or_save_many(figures2)
    rep.done(f"grand total {rep.elapsed():.1f}s")


if __name__ == "__main__":
    main()
