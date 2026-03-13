import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from energy_network import EnergyNetwork

from model import GNNUpdate
import init_parameters
import admm


# ------------------------------------------------------------------
# Run FB pure vs FB+GNN on a single instance
# ------------------------------------------------------------------

def run_comparison(
    net_init: EnergyNetwork,
    model: GNNUpdate,
    gamma: float,
    max_iter: int,
    tol: float,
    device: torch.device,
):
    from fb import learned_FB_step, pure_FB_step
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

    #state, rho = admm.admm_init(net_init)
    #z_pureold = state.z.clone()
    #z_learnedold = state.z.clone()
    #rho_pure = rho
    #rho_learned = rho
    state_pure, rho_pure = admm.admm_init(net_init)
    state_learned, rho_learned = admm.admm_init(net_init)
    z_pureold = state_pure.z.clone()
    z_learnedold = state_learned.z.clone()
    
    print(f"  ADMM initial rho = {rho_pure:.4f}")
    
    residuals_primal    = []
    residuals_dual = []

    residuals_pure    = []
    residuals_learned = []


    with torch.no_grad():
        for it in range(max_iter):

            # --- Pure FB ---
            """
            net_pure_new = pure_FB_step(net_pure, gamma)

            res_pure = torch.sum((net_pure_new.flatten() - net_pure.flatten()) ** 2).item()

            residuals_pure.append(res_pure)
            """
            
 
            net_pure_new, state_pure, rho_pure = admm.admm_step(net_pure, state_pure, rho_pure, training=False, z_old=z_pureold)
 
            # Primal residual: ||Q - z||^2
            r_p = torch.sum((net_pure_new.Q - state_pure.z) ** 2).item()
 
            # Dual residual: rho^2 * ||z_new - z_old||^2
            r_d = (rho_pure ** 2) * torch.sum((state_pure.z - z_pureold) ** 2).item()

            residuals_primal.append(r_p)
            residuals_dual.append(r_d)
            res_pure = torch.sum((net_pure_new.flatten() - net_pure.flatten()) ** 2).item()
            #res_pure = r_p + r_d
            residuals_pure.append(res_pure)


            # --- Learned FB+GNN ---
            """
            net_learned_new = learned_FB_step(
                net_learned, model, it, gamma, training=False
            )
            """
            net_learned_new, state_learned, rho_learned  = admm.learned_admm_step(net_learned, state_learned, rho_learned, model, it, training=False, z_old=z_learnedold)

            #residual l2admm
            # Primal residual: ||Q - z||^2
            r_p = torch.sum((net_learned_new.Q - state_learned.z) ** 2).item()
 
            # Dual residual: rho^2 * ||z_new - z_old||^2
            r_d = (rho_learned ** 2) * torch.sum((state_learned.z - z_learnedold) ** 2).item()

            net_compared, _, _ = admm.admm_step(net_learned_new, state_learned, rho_learned, training=False, z_old=state_learned.z)
            res_learned = torch.sum(
                (net_learned_new.flatten() - net_compared.flatten()) ** 2
            ).item()
            #res_learned = r_p + r_d

             # Update states
            z_pureold = state_pure.z.clone()
            net_pure    = net_pure_new
            z_learnedold=state_learned.z.clone()
            net_learned = net_learned_new

            
            residuals_learned.append(res_learned)

            # Check convergence of pure FB (used as stopping criterion)
            if res_pure<tol: #r_d < tol or r_p<tol:# or res_learned < tol:
                print(f" Pure Converged at iteration {it+1}")
                break

    net_pure.Q = state_pure.z.clone()
    
    x_proj, success = net_pure.project_exact()
    if success:
        net_pure.unflatten(x_proj)

    return {
        "residuals_primal":    residuals_primal,
        "residuals_dual": residuals_dual,
        "residuals_pure":    residuals_pure,
        "residuals_learned": residuals_learned,
        "pure": net_pure,
        "learned": net_learned,
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
    plt.style.use("seaborn-v0_8-pastel")
    fig, axes = plt.subplots(figsize=(12, 5))

    iters = range(0, len(results["residuals_pure"]) )

    # --- Left: residuals ||x_{t+1} - x_t||^2 ---
    

    axes.semilogy(iters, results["residuals_pure"],   marker="o", label="Pure ADMM")
    axes.semilogy(iters, results["residuals_learned"], marker="*", label="FB + GNN")
    #axes.semilogy(iters, results["residuals_primal"],   marker="o", label="Primal ADMM")
    #axes.semilogy(iters, results["residuals_dual"], marker="*", label="Dual ADMM")
    
    'Residual VI ||x - Pi_X(x - γF(x))||'

    axes.set_xlabel("Iteration")
    axes.set_ylabel('Residual VI ||x - Pi_X(x - γF(x))||')
    axes.set_title("Residuals")
    axes.legend()
    
    plt.xticks(range(len(results["residuals_pure"])))
    axes.grid(True, which="both", alpha=0.3)

    
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()



def plot_results_std(results: dict, std: dict = None, save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 5))

    iters = np.arange(1, len(results["residuals_pure"]) + 1)

    def _plot(mean_key, color, label):
        mean = np.array(results[mean_key])
        ax.semilogy(iters, mean, label=label, color=color)
        if std is not None:
            s = np.array(std[mean_key])
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()

