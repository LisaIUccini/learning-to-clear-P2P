"""
example_dr.py
=============

Minimal, fully-adaptable example:

    1. generate one market instance (size N, tunable graph sparsity),
    2. solve it with Douglas-Rachford, stopping at ||x_{k+1}-x_k|| < tol,
    3. plot the residual vs iteration in a paper-ready style.

Run in an environment with torch installed:

    python example_dr.py

Everything below the CONFIG block is generic; tweak CONFIG to adapt.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from market_generator import MarketConfig, MarketGenerator
from dr_solver import DRSolver, DRConfig
from plot_style import set_paper_style, save_figure, COL_WIDTH

# ============================== CONFIG ================================
N        = 20        # number of prosumers
DENSITY  = 0.3       # graph edge density in [0,1]  (0 = tree, 1 = complete)
GAMMA    = 2.0       # DR prox step size (> 0)
LAM      = 1.0       # DR relaxation in (0, 2)
TOL      = 1e-6      # stop when fixed-point residual < TOL
MAX_ITER = 200000     # safety cap
SEED     = 42
# =====================================================================


def build_example():
    cfg_mkt = MarketConfig(graph_model="sparse", density=DENSITY)
    gen = MarketGenerator(cfg_mkt)
    net = gen.sample(N, seed=SEED)
    return net


def solve(net):
    cfg_dr = DRConfig(gamma=GAMMA, lam=LAM, n_iter=MAX_ITER, stop_tol=TOL)
    solver = DRSolver(cfg_dr)
    result = solver.solve(net, record=True, write_back=True)
    return result


def plot_residual(result):
    res = np.asarray(result.history["residual"])
    it = np.arange(1, len(res) + 1)

    set_paper_style()
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.5))

    ax.semilogy(it, res, color="#0072B2", label=rf"DR ($\gamma={GAMMA:.2f}$)")
    ax.axhline(TOL, ls="--", lw=1.0, color="0.4")
    ax.text(len(res) * 0.02, TOL * 1.4, rf"tol $=10^{{{int(np.log10(TOL))}}}$",
            color="0.35", va="bottom", fontsize=8)
    ax.plot(len(res), res[-1], "o", color="#D55E00", ms=5, zorder=5)
    ax.annotate(f"{len(res)} iters", xy=(len(res), res[-1]),
                xytext=(-4, 10), textcoords="offset points",
                ha="right", fontsize=8, color="#D55E00")

    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"residual $\|x_{k+1}-x_k\|_2$")
    ax.set_title(rf"Douglas--Rachford convergence "
                 rf"($N={N}$, density ${DENSITY}$)")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    save_figure(fig, "example_dr_residual")
    print("saved example_dr_residual.{pdf,png}")


def main():
    net = build_example()
    print(net)
    result = solve(net)

    rep = net.feasibility_report()
    print(f"iterations           : {result.n_iter}")
    print(f"converged (< {TOL:g}) : {result.converged}")
    print(f"final residual       : {result.history['residual'][-1]:.3e}")
    print(f"final cost           : {result.history['cost'][-1]:.4f}")
    print(f"feasible solution    : {rep['feasible']} "
          f"(balance {rep['balance']:.1e}, reciprocity {rep['reciprocity']:.1e})")

    plot_residual(result)


if __name__ == "__main__":
    main()