import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from energy_network import EnergyNetwork
from model import GNNUpdate
from fb   import pure_FB_step,   learned_FB_step
from admm import pure_ADMM_step, learned_ADMM_step


# ------------------------------------------------------------------
# Run pure vs learned on a single instance
# ------------------------------------------------------------------

def run_comparison(
    net_init: EnergyNetwork,
    model: GNNUpdate,
    max_iter: int,
    tol: float,
    device: torch.device,
    algorithm: str = "fb",
    gamma: float = None,
    rho: float = None,
) -> dict:
    """
    Run pure and learned algorithms from the same initial state,
    tracking the residual at each iteration.

    Supports two algorithms selected via `algorithm`:

    algorithm="fb"
        Uses pure_FB_step and learned_FB_step.
        Requires: gamma (FB step size).
        Residual pure    : || x^{t+1} - x^t ||^2
        Residual learned : || x^{t+1}_learned - T(x^t_learned) ||^2
                           where T is the unperturbed FB operator.

    algorithm="admm"
        Uses pure_ADMM_step and learned_ADMM_step.
        Requires: rho (ADMM penalty parameter).
        Residual pure    : sum_{(m,n)} (q_{mn} - z_{mn})^2  (primal residual)
        Residual learned : same definition on the learned run.
        z and u are reset to zero before each run via net.admm_reset().

    Both runs start from net_init (cloned independently).
    The learned run uses distributed inference (training=False).
    Stopping criterion: pure algorithm converged (res_pure < tol).

    Parameters
    ----------
    net_init  : EnergyNetwork — initial state (not modified)
    model     : trained GNNUpdate
    max_iter  : maximum number of iterations
    tol       : convergence tolerance on res_pure
    device    : torch.device
    algorithm : "fb" or "admm"
    gamma     : FB step size (required if algorithm="fb")
    rho       : ADMM penalty parameter (required if algorithm="admm")

    Returns
    -------
    dict with keys:
        'residuals_pure'    : list[float]
        'residuals_learned' : list[float]
        'pure'              : EnergyNetwork — final state of pure run
        'learned'           : EnergyNetwork — final state of learned run
    """
    if algorithm not in ("fb", "admm"):
        raise ValueError(f"algorithm must be 'fb' or 'admm', got '{algorithm}'")
    if algorithm == "fb" and gamma is None:
        raise ValueError("gamma is required when algorithm='fb'")
    if algorithm == "admm" and rho is None:
        raise ValueError("rho is required when algorithm='admm'")

    model.eval()

    net_pure    = net_init.clone()
    net_learned = net_init.clone()

    # Reset ADMM duals before starting
    if algorithm == "admm":
        net_pure.admm_reset()
        net_learned.admm_reset()

    residuals_pure    = []
    residuals_learned = []

    with torch.no_grad():
        for it in range(max_iter):

            if algorithm == "fb":
                # ----------------------------------------------------------
                # FB: pure step
                # ----------------------------------------------------------
                net_pure_new = pure_FB_step(net_pure, gamma)

                res_pure = torch.sum(
                    (net_pure_new.flatten() - net_pure.flatten()) ** 2
                ).item()

                # ----------------------------------------------------------
                # FB: learned step + fixed-point residual
                # ----------------------------------------------------------
                net_learned_new = learned_FB_step(
                    net_learned, model, it, gamma, training=False
                )
                res_learned = torch.sum(
                    (net_learned_new.flatten()
                     - pure_FB_step(net_learned_new, gamma).flatten()) ** 2
                ).item()

            else:  # admm
                
                # ----------------------------------------------------------
                # ADMM: pure step
                # ----------------------------------------------------------
                net_pure_new = pure_ADMM_step(net_pure, rho)

                # Primal residual: sum_{(m,n)} (q_{mn} - z_{mn})^2
                res_pure = torch.sum(
                    (net_pure_new.flatten() - net_pure.flatten()) ** 2
                ).item()
                

                # ----------------------------------------------------------
                # ADMM: learned step + primal residual
                # ----------------------------------------------------------
                net_learned_new = learned_ADMM_step(
                    net_learned, model, it, rho, training=False
                )
                res_learned = torch.sum(
                    (net_learned_new.flatten()
                        - pure_ADMM_step(net_learned_new, rho).flatten()) ** 2
                ).item()
                """
                # ----------------------------------------------------------
                # ADMM: pure step
                # res = primal + dual  (identical to train loss)
                # ----------------------------------------------------------
                z_prev_pure = {
                    n: {m: net_pure.prosumers[n].z[m].clone()
                        for m in net_pure.prosumers[n].neighbors}
                    for n in range(net_pure.N)
                }
                net_pure_new = pure_ADMM_step(net_pure, rho)
 
                r_pure = sum(
                    (p.q[m] - p.z[m]) ** 2
                    for p in net_pure_new.prosumers
                    for m in p.neighbors
                )
                s_pure = rho ** 2 * sum(
                    (net_pure_new.prosumers[n].z[m] - z_prev_pure[n][m]) ** 2
                    for n in range(net_pure_new.N)
                    for m in net_pure_new.prosumers[n].neighbors
                )
                res_pure = float(r_pure + s_pure)
 
                # ----------------------------------------------------------
                # ADMM: learned step
                # res = primal + dual  (identical to train loss)
                # ----------------------------------------------------------
                z_prev_learned = {
                    n: {m: net_learned.prosumers[n].z[m].clone()
                        for m in net_learned.prosumers[n].neighbors}
                    for n in range(net_learned.N)
                }
                net_learned_new = learned_ADMM_step(
                    net_learned, model, it, rho, training=False
                )
 
                r_learned = sum(
                    (p.q[m] - p.z[m]) ** 2
                    for p in net_learned_new.prosumers
                    for m in p.neighbors
                )
                s_learned = rho ** 2 * sum(
                    (net_learned_new.prosumers[n].z[m] - z_prev_learned[n][m]) ** 2
                    for n in range(net_learned_new.N)
                    for m in net_learned_new.prosumers[n].neighbors
                )
                res_learned = float(r_learned + s_learned)
                """

            residuals_pure.append(res_pure)
            residuals_learned.append(res_learned)

            # Advance states
            net_pure    = net_pure_new
            net_learned = net_learned_new

            # Stopping: pure algorithm converged
            if res_pure < tol:
                algo_name = "FB" if algorithm == "fb" else "ADMM"
                print(f"  Pure {algo_name} converged at iteration {it + 1}")
                break


    return {
        "residuals_pure":    residuals_pure,
        "residuals_learned": residuals_learned,
        "pure":    net_pure,
        "learned": net_learned,
    }



    """
    if algorithm == "fb":
    # ----------------------------------------------------------
    # FB: pure step
    # ----------------------------------------------------------
    net_pure_new = pure_FB_step(net_pure, gamma)

    res_pure = torch.sum(
        (net_pure_new.flatten() - net_pure.flatten()) ** 2
    ).item()

    # ----------------------------------------------------------
    # FB: learned step + fixed-point residual
    # ----------------------------------------------------------
    net_learned_new = learned_FB_step(
        net_learned, model, it, gamma, training=False
    )
    res_learned = torch.sum(
        (net_learned_new.flatten()
            - pure_FB_step(net_learned_new, gamma).flatten()) ** 2
    ).item()

else:  # admm
    # ----------------------------------------------------------
    # ADMM: pure step
    # ----------------------------------------------------------
    net_pure_new = pure_ADMM_step(net_pure, rho)

    # Primal residual: sum_{(m,n)} (q_{mn} - z_{mn})^2
    res_pure = torch.sum(
        (net_pure_new.flatten() - net_pure.flatten()) ** 2
    ).item()
    

    # ----------------------------------------------------------
    # ADMM: learned step + primal residual
    # ----------------------------------------------------------
    net_learned_new = learned_ADMM_step(
        net_learned, model, it, rho, training=False
    )
    res_learned = torch.sum(
        (net_learned_new.flatten()
            - pure_ADMM_step(net_learned_new, rho).flatten()) ** 2
    ).item()
                """