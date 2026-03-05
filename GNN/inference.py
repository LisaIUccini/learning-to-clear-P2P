import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from energy_network import EnergyNetwork, compute_gne
from model import GNNUpdate
from train import learned_FB_step
import init_parameters


# ------------------------------------------------------------------
# Pure FB step (no learning)
# ------------------------------------------------------------------

def pure_FB_step(net: EnergyNetwork, gamma: float) -> EnergyNetwork:
    """
    One step of the pure Forward-Backward algorithm (no perturbation).

    Parameters
    ----------
    net   : current EnergyNetwork state
    gamma : FB step size

    Returns
    -------
    net_new : EnergyNetwork with updated state
    """
    net_new = net.forward_step(gamma)
    net_new.project_exact()
    return net_new


# ------------------------------------------------------------------
# Run FB pure vs FB+GNN on a single instance
# ------------------------------------------------------------------

def run_comparison(
    net_init: EnergyNetwork,
    x_star: torch.Tensor,
    model: GNNUpdate,
    gamma: float,
    max_iter: int,
    tol: float,
    device: torch.device,
):
    """
    Run pure FB and learned FB+GNN from the same initial state,
    tracking residuals and distance to GNE at each iteration.

    Parameters
    ----------
    net_init : initial EnergyNetwork state (same for both runs)
    x_star   : GNE flattened state vector (dim,)
    model    : trained GNNUpdate
    gamma    : FB step size
    max_iter : maximum iterations
    tol      : convergence tolerance on ||x_{t+1} - x_t||^2
    device   : torch.device

    Returns
    -------
    results : dict with keys:
        'residuals_pure'   : list of ||x_{t+1} - x_t||^2  (pure FB)
        'residuals_learned': list of ||x_{t+1} - x_t||^2  (learned FB)
        'dist_pure'        : list of ||x_t - x*||^2       (pure FB)
        'dist_learned'     : list of ||x_t - x*||^2       (learned FB)
    """
    model.eval()

    net_pure    = net_init.clone()
    net_learned = net_init.clone()

    residuals_pure    = []
    residuals_learned = []
    dist_pure         = []
    dist_learned      = []

    with torch.no_grad():
        for it in range(max_iter):

            # --- Pure FB ---
            net_pure_new = pure_FB_step(net_pure, gamma)

            res_pure = torch.sum((net_pure_new.flatten() - net_pure.flatten()) ** 2).item()
            d_pure   = torch.sum((net_pure.flatten() - x_star) ** 2).item()

            residuals_pure.append(res_pure)
            dist_pure.append(d_pure)

            # --- Learned FB+GNN ---
            net_learned_new = learned_FB_step(
                net_learned, model, it, gamma, training=False
            )

            res_learned = torch.sum(
                (net_learned_new.flatten() - net_learned.flatten()) ** 2
            ).item()
            d_learned = torch.sum((net_learned.flatten() - x_star) ** 2).item()

            residuals_learned.append(res_learned)
            dist_learned.append(d_learned)

            # Update states
            net_pure    = net_pure_new
            net_learned = net_learned_new

            # Check convergence of pure FB (used as stopping criterion)
            if res_pure < tol:
                print(f"  Pure FB converged at iteration {it+1}")
                break

    return {
        "residuals_pure":    residuals_pure,
        "residuals_learned": residuals_learned,
        "dist_pure":         dist_pure,
        "dist_learned":      dist_learned,
    }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_results(results: dict, save_path: str = None):
    """
    Plot residuals and distance to GNE for pure FB vs learned FB+GNN.

    Parameters
    ----------
    results   : output of run_comparison
    save_path : if given, save figure here; otherwise show interactively
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    iters = range(1, len(results["residuals_pure"]) + 1)

    # --- Left: residuals ||x_{t+1} - x_t||^2 ---
    axes[0].semilogy(iters, results["residuals_pure"],    label="Pure FB",     color="steelblue")
    axes[0].semilogy(iters, results["residuals_learned"], label="FB + GNN",    color="darkorange")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\|x^{t+1} - x^t\|^2$")
    axes[0].set_title("Residuals")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # --- Right: distance to GNE ||x_t - x*||^2 ---
    axes[1].semilogy(iters, results["dist_pure"],    label="Pure FB",  color="steelblue")
    axes[1].semilogy(iters, results["dist_learned"], label="FB + GNN", color="darkorange")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\|x^t - x^*\|^2$")
    axes[1].set_title("Distance to GNE")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # --- Device and seeds ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Problem parameters ---
    (N, gamma, max_iter, tol,
     neighbors, D_min, D_max, G_min, G_max, G_res,
     D_star, a_tilde, a, b, c, k) = init_parameters.initN3()

    # --- Build network template ---
    net_template = EnergyNetwork(
        N, neighbors, D_min, D_max, G_min, G_max, G_res,
        a_tilde, a, b, D_star, c, k, device
    )

    # --- Model hyperparameters (must match training) ---
    d_h = 32
    rho = 0.5
    T   = 15
    epochs       = 50
    num_samples  = 50
    lr           = 1e-3

    model = GNNUpdate(d_h=d_h, rho=rho).to(device)

    # --- Load trained weights ---
    save_path = (
        f"checkpoints/N{N}_T{T}_e{epochs}_s{num_samples}_learned_update"
        f"_rho{rho}_lr{lr}.pt"
    )

    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"No trained model found at {save_path}. "
            f"Run train.py first."
        )

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Loaded model from {save_path}")

    # --- Generate test instances (different seed from training) ---
    n_test = 5
    print(f"\nRunning inference on {n_test} test instances...")

    all_results = []

    for i in range(n_test):
        print(f"\nTest instance {i+1}/{n_test}:")

        # Random initial state
        net_init = net_template.clone()
        net_init.D, net_init.G, net_init.Q = net_init._init_state()

        # Compute GNE offline for this instance
        print("  Computing GNE...")
        x_star = compute_gne(net_init.clone(), gamma, 200, 1e-6)

        # Run comparison
        results = run_comparison(
            net_init = net_init,
            x_star   = x_star.flatten().detach(),
            model    = model,
            gamma    = gamma,
            max_iter = max_iter,
            tol      = tol,
            device   = device,
        )
        all_results.append(results)

        # Plot individual result
        plot_results(
            results,
            save_path=f"figures/test_instance_{i+1}.png",
        )

    # --- Aggregate: average over test instances ---
    max_len = max(len(r["dist_pure"]) for r in all_results)

    def pad_and_stack(key):
        # Pad shorter sequences with their last value
        padded = []
        for r in all_results:
            seq = r[key]
            pad = [seq[-1]] * (max_len - len(seq))
            padded.append(seq + pad)
        return np.array(padded)  # (n_test, max_len)

    avg_results = {
        key: pad_and_stack(key).mean(axis=0).tolist()
        for key in ["residuals_pure", "residuals_learned", "dist_pure", "dist_learned"]
    }

    plot_results(avg_results, save_path="figures/average_comparison.png")
    print("\nDone.")


if __name__ == "__main__":
    main()