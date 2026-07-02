"""
example_dr.py
=============

Minimal, fully-adaptable example:

    1. generate one market instance (size N, tunable graph sparsity),
    2. solve it with Douglas-Rachford, stopping at shadow residual < tol,
    3. produce TWO separate figures: the network topology and the
       residual-vs-iteration convergence curve.

Shows them on screen if a display is available, otherwise saves directly
(headless / SSH-safe). Tweak the CONFIG block to adapt.

    python example_dr.py
"""

import numpy as np
import matplotlib.pyplot as plt          # interactive backend (no Agg)

from market_generator import MarketConfig, MarketGenerator
from dr_solver import DRSolver, DRConfig
from plot_style import set_paper_style, show_or_save_many, draw_network

# ============================== CONFIG ================================
N        = 20        # number of prosumers
DENSITY  = 0.3       # graph edge density in [0,1]  (0 = tree, 1 = complete)
GAMMA    = 2.0       # DR prox step size (> 0)
LAM      = 1.0       # DR relaxation in (0, 2)
TOL      = 1e-6      # stop when shadow residual < TOL
MAX_ITER = 20000    # safety cap
SEED     = 42

RES_FIG_SIZE   = (9.0, 3.4)   # residual figure (wide)
GRAPH_FIG_SIZE = (4.0, 4.0)   # network figure (square)
SHOW_GRAPH  = True            # also produce the network figure
NAME_RESID  = "example_dr_residual"
NAME_GRAPH  = "example_dr_network"
# =====================================================================


def build_example():
    gen = MarketGenerator(MarketConfig(graph_model="random", density=DENSITY))
    return gen.sample(N, seed=SEED)


def solve(net):
    cfg_dr = DRConfig(gamma=GAMMA, lam=LAM, n_iter=MAX_ITER, stop_tol=TOL)
    return DRSolver(cfg_dr).solve(net, record=True, write_back=True)


def residual_figure(result):
    res = np.asarray(result.history["residual"])
    it = np.arange(1, len(res) + 1)
    fig, ax = plt.subplots(figsize=RES_FIG_SIZE)
    ax.semilogy(it, res, color="#0072B2", label=rf"DR ($\gamma={GAMMA:.2f}$)")
    ax.axhline(TOL, ls="--", lw=1.0, color="0.4")
    ax.text(len(res) * 0.01, TOL * 1.4, rf"tol $=10^{{{int(np.log10(TOL))}}}$",
            color="0.35", va="bottom", fontsize=8)
    ax.plot(len(res), res[-1], "o", color="#D55E00", ms=5, zorder=5)
    ax.annotate(f"{len(res)} iters", xy=(len(res), res[-1]),
                xytext=(-6, 10), textcoords="offset points",
                ha="right", fontsize=8, color="#D55E00")
    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"shadow residual $\|\bar x_{k+1}-\bar x_k\|_2$")
    ax.set_title(rf"Douglas--Rachford convergence ($N={N}$, density ${DENSITY}$)")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def network_figure(net):
    fig, ax = plt.subplots(figsize=GRAPH_FIG_SIZE)
    draw_network(ax, net.neighbors, net.N, seed=SEED)
    ax.set_title(rf"network ($N={N}$, density ${DENSITY}$)")
    fig.tight_layout()
    return fig


def main():
    net = build_example()
    print(net)
    result = solve(net)

    rep = net.feasibility_report()
    print(f"iterations           : {result.n_iter}")
    print(f"converged (< {TOL:g}) : {result.converged}")
    print(f"final shadow residual: {result.history['residual'][-1]:.3e}")
    print(f"final aux residual   : {result.history['residual_aux'][-1]:.3e}")
    print(f"final cost           : {result.history['cost'][-1]:.4f}")
    print(f"feasible solution    : {rep['feasible']} "
          f"(balance {rep['balance']:.1e}, reciprocity {rep['reciprocity']:.1e})")

    set_paper_style()
    figures = [(residual_figure(result), NAME_RESID)]
    if SHOW_GRAPH:
        figures.append((network_figure(net), NAME_GRAPH))
    show_or_save_many(figures)     # one show for all, then save each


if __name__ == "__main__":
    main()