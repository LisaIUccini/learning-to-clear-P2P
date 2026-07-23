"""
example_density_sweep.py
========================

Run the SAME Douglas-Rachford solver on market instances of increasing
graph density and overlay the convergence curves (residual vs iteration).
This visualises how sparsity/density affects convergence speed.

Shows the plot on screen if a display is available, otherwise saves it
directly (headless / SSH-safe).

    python example_density_sweep.py

Note: each density is a fresh instance (same N and seed for
reproducibility, but graph and per-edge parameters differ with density).
For a statistically isolated trend, average over several seeds.
"""

import numpy as np
import matplotlib.pyplot as plt

from market_generator import MarketConfig, MarketGenerator
from dr_solver import DRSolver, DRConfig
from plot_style import set_paper_style, show_or_save

# ============================== CONFIG ================================
N         = 10
DENSITIES = [0.1, 0.3,  0.5, 0.6]   # increasing density
GAMMA     = 1.0        # near-optimal step size (see study_gamma)
LAM       = 1.0
TOL       = 1e-6
MAX_ITER  = 20000
SEED      = 42
FIG_W, FIG_H = 9.0, 4.0
SAVE_NAME = "example_density_sweep"
# =====================================================================


def residual_curve(density):
    gen = MarketGenerator(MarketConfig(graph_model="sparse", density=density))
    net = gen.sample(N, seed=SEED)
    solver = DRSolver(DRConfig(gamma=GAMMA, lam=LAM,
                               n_iter=MAX_ITER, stop_tol=TOL))
    res = solver.solve(net, record=True, write_back=False)
    return np.asarray(res.history["residual"])


def main():
    curves = {}
    for d in DENSITIES:
        curves[d] = residual_curve(d)
        print(f"density={d:.2f}: iters={len(curves[d])}")

    set_paper_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DENSITIES)))
    for col, d in zip(colors, DENSITIES):
        r = curves[d]
        ax.semilogy(np.arange(1, len(r) + 1), r, color=col, lw=1.4,
                    label=f"{d:.1f}  ({len(r)} it)")
    ax.axhline(TOL, ls="--", lw=1.0, color="0.4")
    ax.text(5, TOL * 1.4, r"tol $=10^{-6}$", color="0.35", va="bottom", fontsize=8)
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"shadow residual $\|\bar x_{k+1}-\bar x_k\|_2$")
    ax.set_title(rf"DR convergence for increasing graph density "
                 rf"($N={N}$, $\gamma={GAMMA}$)")
    ax.set_xlim(left=0)
    ax.legend(title="density (iters)", ncol=2, fontsize=7.5,
              title_fontsize=8, loc="upper right")
    fig.tight_layout()

    show_or_save(fig, SAVE_NAME)

    densities = list(curves.keys())
    iters = [len(curves[d]) for d in densities]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(densities, iters, "-o", color="#0072B2")
    ax.set_xlabel("graph density")
    ax.set_ylabel("iterations to converge")
    ax.set_title(rf"DR iterations vs density ($N={N}$, $\gamma={GAMMA}$)")
    show_or_save(fig, "iters_vs_density")

if __name__ == "__main__":
    main()