import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from energy_network import EnergyNetwork
from model import GNNUpdate
from fb import pure_FB_step, learned_FB_step


# ------------------------------------------------------------------
# Run pure FB vs FB+GNN on a single instance
# ------------------------------------------------------------------

def run_comparison(
    net_init: EnergyNetwork,
    model: GNNUpdate,
    gamma: float,
    max_iter: int,
    tol: float,
    device: torch.device,
) -> dict:
    """
    Run pure FB and learned FB+GNN from the same initial state,
    tracking the fixed-point residual at each iteration.

    Both runs start from net_init (cloned independently so they never
    share state). The learned run uses distributed inference: each
    prosumer communicates its q to neighbors, then queries the GNN
    locally (training=False in learned_FB_step).

    Residual definitions
    --------------------
    Pure FB:
        res_pure[t]    = || x^{t+1}_pure - x^t_pure ||^2
                         standard iterate-difference residual.

    Learned FB:
        res_learned[t] = || x^{t+1}_learned - T(x^t_learned) ||^2
                         distance between the learned update and the
                         unperturbed FB operator applied to the same
                         starting point. This is the fixed-point residual
                         used in the training loss, so it is the natural
                         metric to compare the two algorithms.

    Stopping criterion: pure FB converged (res_pure < tol).
    The learned run always takes the same number of steps.

    Parameters
    ----------
    net_init : EnergyNetwork — initial state (not modified)
    model    : trained GNNUpdate
    gamma    : FB step size
    max_iter : maximum number of iterations
    tol      : convergence tolerance on res_pure
    device   : torch.device

    Returns
    -------
    dict with keys:
        'residuals_pure'    : list[float]    — res_pure at each iteration
        'residuals_learned' : list[float]    — res_learned at each iteration
        'pure'              : EnergyNetwork  — final state of pure FB run
        'learned'           : EnergyNetwork  — final state of learned FB run
    """
    model.eval()

    net_pure    = net_init.clone()
    net_learned = net_init.clone()

    residuals_pure    = []
    residuals_learned = []

    with torch.no_grad():
        for it in range(max_iter):

            # ----------------------------------------------------------
            # Pure FB step
            # ----------------------------------------------------------
            net_pure_new = pure_FB_step(net_pure, gamma)

            res_pure = torch.sum(
                (net_pure_new.flatten() - net_pure.flatten()) ** 2
            ).item()
            residuals_pure.append(res_pure)

            # ----------------------------------------------------------
            # Learned FB+GNN step (distributed inference)
            # ----------------------------------------------------------
            net_learned_new = learned_FB_step(
                net_learned, model, it, gamma, training=False
            )

            # Fixed-point residual for the learned run:
            # distance between learned update and unperturbed T(x^t_learned)
            res_learned = torch.sum(
                (net_learned_new.flatten()
                 - pure_FB_step(net_learned_new, gamma).flatten()) ** 2
            ).item()
            residuals_learned.append(res_learned)

            # ----------------------------------------------------------
            # Advance states
            # ----------------------------------------------------------
            net_pure    = net_pure_new
            net_learned = net_learned_new

            # ----------------------------------------------------------
            # Stopping: pure FB converged
            # ----------------------------------------------------------
            if res_pure < tol:
                print(f"  Pure FB converged at iteration {it + 1}")
                break
            else:
                if res_learned < tol:
                    print(f"  L2FB converged at iteration {it + 1}")
                    break

    return {
        "residuals_pure":    residuals_pure,
        "residuals_learned": residuals_learned,
        "pure":    net_pure,
        "learned": net_learned,
    }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_results(results: dict, save_path: str = None):
    """
    Plot fixed-point residuals for pure FB vs learned FB+GNN.

    Parameters
    ----------
    results   : output of run_comparison (or averaged results dict)
    save_path : if given, save figure here; otherwise show interactively
    """
    plt.style.use("seaborn-v0_8-pastel")
    fig, ax = plt.subplots(figsize=(12, 5))

    iters = range(len(results["residuals_pure"]))

    ax.semilogy(iters, results["residuals_pure"],    marker="o", label="Pure FB")
    ax.semilogy(iters, results["residuals_learned"], marker="*", label="FB + GNN")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Residual VI $\|x - \Pi_\mathcal{X}(x - \gamma F(x))\|$")
    ax.set_title("Residuals")
    ax.legend()

    plt.xticks(range(len(results["residuals_pure"])))
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_results_std(results: dict, std: dict = None, save_path: str = None):
    """
    Plot residuals with optional shaded standard deviation band.

    Parameters
    ----------
    results   : dict with mean residual sequences
    std       : dict with std sequences (same keys), or None
    save_path : if given, save figure here; otherwise show interactively
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    iters = np.arange(1, len(results["residuals_pure"]) + 1)

    def _plot(key, color, label):
        mean = np.array(results[key])
        ax.semilogy(iters, mean, label=label, color=color)
        if std is not None:
            s  = np.array(std[key])
            lo = np.clip(mean - s, 1e-12, None)
            hi = mean + s
            ax.fill_between(iters, lo, hi, alpha=0.2, color=color)

    _plot("residuals_pure",    "steelblue",  "Pure FB")
    _plot("residuals_learned", "darkorange", "FB + GNN")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|x^{t+1} - x^t\|^2$")
    ax.set_title("Residuals")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()