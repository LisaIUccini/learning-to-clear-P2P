"""
dr_solver.py
============

Distributed Douglas-Rachford (DR) equilibrium seeking for the P2P market
game (Algorithm 1 of the L2Play paper, without the learned perturbation).

One relaxed-DR iteration on the auxiliary iterate x = [D, G, Q]:

    xbar = proj_{X^rec}(x)                 # shadow  (J_{gamma B})  -- line 4
    refl = 2*xbar - x                      # reflection            -- line 5
    z    = prox_{gamma J}^{X^loc}(refl)    # local prox (J_{gamma A}) line 6
    x    = x + lam * (z - xbar)            # relaxed DR update      -- line 7
    x    = x + v                           # learned perturbation   -- line 8 (optional)

Convergence (Theorem 1) is tracked on the SHADOW xbar = proj_{X^rec}(x),
the reciprocity-feasible market profile. Two residuals are recorded:

    residual      = ||xbar_{k+1} - xbar_k||   (shadow move; used for stopping)
    residual_aux  = ||x_{k+1}   - x_k||       (auxiliary-iterate fixed-point
                                               residual; the canonical DR one)

Since proj_{X^rec} is nonexpansive, residual <= residual_aux; both vanish at
a fixed point. The shadow residual is the market-meaningful one, the auxiliary
residual is the strict operator fixed-point criterion -- we log both.

Note on lambda: with lam = 1 (default) the update reproduces Algorithm 1
lines 6-7 exactly (D,G take the prox output, q is relaxed). For lam != 1 this
applies the *uniform* relaxation of Theorem 1 (eq. 17) to all blocks, which
is the variant covered by the convergence proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from energy_network import EnergyNetwork

PerturbationFn = Callable[[EnergyNetwork, torch.Tensor, int], torch.Tensor]


@dataclass
class DRConfig:
    gamma: float = 0.3          # resolvent / prox step size (> 0)
    lam: float = 1.0            # DR relaxation, lambda in (0, 2)
    n_iter: int = 2000          # max iterations
    stop_tol: Optional[float] = 1e-6   # stop when shadow residual < tol (None: never)
    prox_bisect: int = 60       # bisection steps inside the prox

    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("gamma must be > 0")
        if not (0.0 < self.lam < 2.0):
            raise ValueError("lam must lie in (0, 2)")


@dataclass
class DRResult:
    x: torch.Tensor                       # final auxiliary iterate
    shadow: torch.Tensor                  # final feasible profile proj_{X^rec}(x)
    n_iter: int                           # iterations actually run
    converged: bool
    history: Dict[str, List[float]] = field(default_factory=dict)


class DRSolver:
    """Runs relaxed Douglas-Rachford on an EnergyNetwork."""

    def __init__(self, config: Optional[DRConfig] = None):
        self.cfg = config if config is not None else DRConfig()

    def solve(
        self,
        net: EnergyNetwork,
        x0: Optional[torch.Tensor] = None,
        perturbation: Optional[PerturbationFn] = None,
        training: bool = False,
        record: bool = True,
        write_back: bool = True,
    ) -> DRResult:
        """Solve the market game.

        Parameters
        ----------
        net : EnergyNetwork
        x0  : optional (dim,) starting auxiliary iterate; defaults to the
              network's current flattened state.
        perturbation : optional hook v = perturbation(net, x, k) (line 8).
        training : if True keep the autograd graph across iterations
                   (for unrolled GNN training); else run under no_grad.
        record   : log per-iteration diagnostics into result.history.
        write_back : write the final shadow into the prosumer state.
        """
        cfg = self.cfg
        x = (net.flatten() if x0 is None else x0)
        if not training:
            x = x.detach().clone()

        history: Dict[str, List[float]] = {
            "residual": [], "residual_aux": [],
            "cost": [], "balance": [], "reciprocity": [],
        }

        context = torch.enable_grad() if training else torch.no_grad()
        converged = False
        k_run = 0
        with context:
            xbar = net.project_reciprocity(x)              # shadow of x (line 4)
            for k in range(cfg.n_iter):
                k_run = k + 1
                refl = 2.0 * xbar - x                       # line 5
                z = net.prox_local(cfg.gamma, refl, n_bisect=cfg.prox_bisect)  # line 6
                x_new = x + cfg.lam * (z - xbar)            # line 7

                if perturbation is not None:               # line 8
                    v = perturbation(net, x_new, k)
                    x_new = x_new + v

                xbar_new = net.project_reciprocity(x_new)  # shadow of x_new
                res = torch.linalg.vector_norm(xbar_new - xbar)   # shadow residual

                if record:
                    res_aux = torch.linalg.vector_norm(x_new - x)
                    self._log(history, net, xbar_new, res, res_aux)

                x, xbar = x_new, xbar_new                   # advance (reuse shadow)

                if (cfg.stop_tol is not None and perturbation is None
                        and float(res) < cfg.stop_tol):
                    converged = True
                    break

        if write_back:
            net.unflatten(xbar.detach() if not training else xbar)

        return DRResult(x=x, shadow=xbar, n_iter=k_run,
                        converged=converged, history=history)

    @staticmethod
    def _log(history, net, shadow, res, res_aux):
        with torch.no_grad():
            history["residual"].append(float(res))
            history["residual_aux"].append(float(res_aux))
            history["cost"].append(float(net.cost_at(shadow)))
            history["balance"].append(float(net.balance_at(shadow).abs().max()))
            history["reciprocity"].append(float(net.reciprocity_max_at(shadow)))


# ----------------------------------------------------------------------
# Convenience one-shot function
# ----------------------------------------------------------------------
def solve_dr(net: EnergyNetwork, gamma: float = 0.3, lam: float = 1.0,
             n_iter: int = 2000, stop_tol: Optional[float] = 1e-6,
             **kwargs) -> DRResult:
    """Shortcut: build a DRSolver and solve `net` with vanilla DR."""
    solver = DRSolver(DRConfig(gamma=gamma, lam=lam, n_iter=n_iter,
                               stop_tol=stop_tol))
    return solver.solve(net, **kwargs)