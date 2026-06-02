import torch

from energy_network import EnergyNetwork
from model          import GNNUpdate
from admm           import pure_ADMM_step, learned_ADMM_step


# ------------------------------------------------------------------
# Run pure vs learned ADMM on a single instance
# ------------------------------------------------------------------

def run_comparison(
    net_init: EnergyNetwork,
    model: GNNUpdate,
    max_iter: int,
    tol: float,
    device: torch.device,
    rho: float,
) -> dict:
    """
    Run pure and learned ADMM from the same initial state, tracking the
    fixed-point residual at each iteration.

        residual = || sigma - T_ADMM(sigma) ||^2

    Both runs start from net_init (cloned independently). z and u are
    reset to zero before each run. The learned run uses distributed
    inference (training=False).

    Stopping criterion: both pure and learned residuals fall below `tol`.

    Parameters
    ----------
    net_init : EnergyNetwork — initial state (not modified)
    model    : trained GNNUpdate
    max_iter : maximum number of ADMM iterations
    tol      : convergence tolerance on both residuals
    device   : torch.device
    rho      : ADMM penalty parameter

    Returns
    -------
    dict with keys:
        'residuals_pure'    : list[float]
        'residuals_learned' : list[float]
        'pure'              : EnergyNetwork — final state of pure run
        'learned'           : EnergyNetwork — final state of learned run
    """
    model.eval()

    net_pure    = net_init.clone(); net_pure.admm_reset()
    net_learned = net_init.clone(); net_learned.admm_reset()

    residuals_pure    = []
    residuals_learned = []

    with torch.no_grad():
        for it in range(max_iter):

            # ----- Pure ADMM step + fixed-point residual ----------
            net_pure_new = pure_ADMM_step(net_pure, rho)
            sigma_pure   = net_pure_new.flatten_admm()
            T_sigma_pure = pure_ADMM_step(net_pure_new, rho).flatten_admm()
            res_pure     = torch.sum((sigma_pure - T_sigma_pure) ** 2).item()

            # ----- Learned ADMM step + fixed-point residual -------
            net_learned_new = learned_ADMM_step(
                net_learned, model, it, rho, training=False
            )
            sigma_learned   = net_learned_new.flatten_admm()
            T_sigma_learned = pure_ADMM_step(net_learned_new, rho).flatten_admm()
            res_learned     = torch.sum(
                (sigma_learned - T_sigma_learned) ** 2
            ).item()

            residuals_pure.append(res_pure)
            residuals_learned.append(res_learned)

            # Advance states
            net_pure    = net_pure_new
            net_learned = net_learned_new

            # Stopping condition
            if res_pure < tol and res_learned < tol:
                print(f"  ADMM converged at iteration {it + 1}")
                break

    return {
        "residuals_pure":    residuals_pure,
        "residuals_learned": residuals_learned,
        "pure":    net_pure,
        "learned": net_learned,
    }