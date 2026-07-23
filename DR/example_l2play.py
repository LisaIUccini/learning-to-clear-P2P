"""
example_l2play.py
=================

End-to-end pipeline:

    1. draw a dataset of market instances (train / test),
    2. train Phi_theta centrally by unrolling the DR iteration,
    3. sweep rho and keep the best,
    4. deploy the weights to every prosumer and measure, in the FULLY
       DISTRIBUTED solver, how many iterations DR and L2Play need,
    5. plot the comparison.

    python example_l2play.py

Only step 2 is centralised. Steps 4 and 5 run the real distributed loop:
each prosumer queries its own copy of the model after the communication
round, exactly as it would on a device.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from market_generator import MarketConfig, MarketDataset, DatasetConfig
from gnn import PerturbationGNN, GNNConfig, deploy, undeploy
from training import TrainConfig, Trainer, compare, Reporter, residual_curves
from dr_solver import DRSolver, DRConfig
from plot_style import set_paper_style, show_or_save_many, PALETTE, plot_band

# ============================== CONFIG ================================
# --- dataset: fixed size, different graph and parameters per instance.
#     Use N=(8, 40) and density=(0.1, 0.6) to let the size vary as well.
TRAIN_INSTANCES = 24     # how many markets to train on
TEST_INSTANCES  = 5      # held-out markets
NODES           = 10     # prosumers per market (int, or (min, max))
DENSITY       = 0.3           # float, or (min, max)
GRAPH_MODELS  = ("sparse",)

# --- solver: the TUNED baseline (accelerating a slow one proves nothing)
GAMMA, LAM, BISECT = 1.5, 1.0, 30
TOL, MAX_ITER = 1e-6, 20000

# --- model / training
HIDDEN, UNROLL, EPOCHS, LR = 32, 30, 25, 1e-3
RHO = 0.80      # rho is fixed, but tuned by this sweep
SEED = 42
# =====================================================================


def make_datasets():
    dcfg = DatasetConfig(n_instances=TRAIN_INSTANCES + TEST_INSTANCES, N=NODES,
                         density=DENSITY, graph_models=GRAPH_MODELS, seed=SEED)
    full = MarketDataset(dcfg, MarketConfig(graph_model=GRAPH_MODELS[0]))
    return full.split(TRAIN_INSTANCES / (TRAIN_INSTANCES + TEST_INSTANCES))


def train_one(rho, train_ds, val_ds, rep=None):
    torch.manual_seed(SEED)
    gnn = PerturbationGNN(GNNConfig(hidden_dim=HIDDEN, rho=rho))
    cfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT,
                      unroll=UNROLL, rho=rho, epochs=EPOCHS, lr=LR,
                      seed=SEED, verbose=False)
    tr = Trainer(gnn, cfg)
    tr.fit(train_ds, reporter=rep)
    val = tr.evaluate_loss(val_ds)
    return gnn, cfg, tr, val


def main():
    rep = Reporter(True)

    # ---------------- phase 1: dataset ----------------
    rep.phase(1, 3, "dataset")
    train_ds, test_ds = make_datasets()
    rep.dataset("train", train_ds)
    rep.dataset("test", test_ds)
    rep.kv("topologies", ", ".join(GRAPH_MODELS))
    rep.kv("density", str(DENSITY))
    rep.done()

    # ---------------- phase 2: centralised training ----------------
    rep.phase(2, 3, "centralised training")
    best = None
    rep.line("")
    rho = RHO
    rep.line(f"--- rho = {rho:.2f} ---")
    gnn, cfg, tr, val = train_one(rho, train_ds, test_ds, rep)
    rep.kv("val loss", f"{val:+.4f}")
    if best is None or val < best[0]:
        best = (val, rho, gnn, cfg, tr)
    _, rho, gnn, cfg, trainer = best
    rep.line("")
    rep.kv("selected rho", f"{rho:.2f}  (best validation loss {best[0]:+.4f})")
    trainer.save("phi_theta.pt")
    rep.kv("saved", "phi_theta.pt")
    rep.done()


    # ---------------- phase 3: distributed inference ----------------
    rep.phase(3, 3, "distributed inference")
    rep.kv("deployment", f"weights copied to every prosumer "
                         f"({gnn.n_parameters()} params each)")
    res = compare(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER, reporter=rep)
    rep.line("")
    rep.kv("median DR", f"{res['median_dr']} iterations")
    rep.kv("median L2Play", f"{res['median_l2play']} iterations")
    rep.kv("mean speed-up", f"{res['mean_speedup']:.2f}x")
    rep.done(f"total {rep.elapsed():.1f}s")
 
    # ---------- residual curves over the WHOLE test set ----------
    rep.line("")
    rep.kv("curves", f"recording residuals on all {len(test_ds)} test instances")
    curves_dr = residual_curves(test_ds, None, cfg, tol=TOL, max_iter=MAX_ITER)
    curves_l2 = residual_curves(test_ds, gnn, cfg, tol=TOL, max_iter=MAX_ITER)
 
    # ---------- figures ----------
    set_paper_style()
    fig1, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.set_yscale("log")
    plot_band(ax, curves_dr, PALETTE[0],
              f"DR (median of {len(curves_dr)})", min_fraction=BAND_MIN_FRACTION)
    plot_band(ax, curves_l2, PALETTE[1],
              f"L2Play (median of {len(curves_l2)})",
              min_fraction=BAND_MIN_FRACTION)
    ax.axhline(TOL, ls="--", lw=1.0, color="0.4")
    ax.text(5, TOL * 1.4, rf"tol $={TOL:g}$", color="0.35",
            va="bottom", fontsize=8)
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"shadow residual $\|\bar x_{k+1}-\bar x_k\|_2$")
    ax.set_title(rf"Distributed DR vs L2Play over the test set "
                 rf"($\gamma={GAMMA}$, $\lambda={LAM}$, $\rho={rho}$)")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    fig1.tight_layout()
 
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.4))
    ax2.scatter(res["dr"], res["l2play"], s=24, color=PALETTE[0], zorder=3)
    lim = max(max(res["dr"]), max(res["l2play"])) * 1.05
    ax2.plot([0, lim], [0, lim], ls="--", lw=1.0, color="0.5")
    ax2.set_xlabel("DR iterations")
    ax2.set_ylabel("L2Play iterations")
    ax2.set_title("per-instance comparison (test set)")
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    fig2.tight_layout()
 
    show_or_save_many([(fig1, "l2play_residual"),
                       (fig2, "l2play_scatter")])
 
 
if __name__ == "__main__":
    main()
 