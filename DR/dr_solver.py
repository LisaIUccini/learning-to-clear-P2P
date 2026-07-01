"""
dr_solver.py
============

Distributed Douglas-Rachford (DR) equilibrium seeking for the P2P market
game (Algorithm 1 of the L2Play paper, without the learned perturbation).

Given an EnergyNetwork, one relaxed-DR iteration on the auxiliary iterate
x = [D, G, Q] is:

    xbar = proj_{X^rec}(x)                 # shadow  (J_{gamma B})  -- line 4
    refl = 2*xbar - x                      # reflection            -- line 5
    z    = prox_{gamma J}^{X^loc}(refl)    # local prox (J_{gamma A}) line 6
    x    = x + lam * (z - xbar)            # relaxed DR update      -- line 7
    x    = x + v                           # learned perturbation   -- line 8 (optional)

Convergence (Theorem 1) is tracked on the SHADOW  xbar = proj_{X^rec}(x),
which is the reciprocity-feasible market profile; the auxiliary iterate x
itself need not be feasible.

The optional `perturbation` hook has signature perturbation(net, x, k) -> v
and lets the same solver run vanilla DR (v = 0) or L2Play (v from a GNN);
it is left None here and wired up in the learning step.
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
    stop_tol: Optional[float] = 1e-6   # stop when ||x_{k+1}-x_k|| < tol (None: never)
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
        training : if True, keep the autograd graph across iterations
                   (for unrolled GNN training); if False, run under no_grad.
        record   : log per-iteration diagnostics into result.history.
        write_back : write the final shadow into the prosumer state.
        """
        cfg = self.cfg
        x = (net.flatten() if x0 is None else x0)
        if not training:
            x = x.detach().clone()

        history: Dict[str, List[float]] = {
            "residual": [], "cost": [], "balance": [], "reciprocity": [],
        }

        context = torch.enable_grad() if training else torch.no_grad()
        converged = False
        k_run = 0
        with context:
            for k in range(cfg.n_iter):
                k_run = k + 1
                xbar = net.project_reciprocity(x)          # line 4
                refl = 2.0 * xbar - x                       # line 5
                z = net.prox_local(cfg.gamma, refl, n_bisect=cfg.prox_bisect)  # line 6
                x_new = x + cfg.lam * (z - xbar)            # line 7

                if perturbation is not None:               # line 8
                    v = perturbation(net, x_new, k)
                    x_new = x_new + v

                res = torch.linalg.vector_norm(x_new - x)
                x = x_new

                if record:
                    sh = net.project_reciprocity(x)
                    self._log(history, net, sh, res)

                if (cfg.stop_tol is not None and perturbation is None
                        and float(res) < cfg.stop_tol):
                    converged = True
                    break

        shadow = net.project_reciprocity(x)
        if write_back:
            net.unflatten(shadow.detach() if not training else shadow)

        return DRResult(x=x, shadow=shadow, n_iter=k_run,
                        converged=converged, history=history)

    @staticmethod
    def _log(history, net, shadow, res):
        with torch.no_grad():
            history["residual"].append(float(res))
            history["cost"].append(float(net.cost_at(shadow)))
            history["balance"].append(float(net.balance_at(shadow).abs().max()))
            history["reciprocity"].append(float(net.reciprocity_max_at(shadow)))


# ----------------------------------------------------------------------
# Convenience one-shot function
# ----------------------------------------------------------------------
def solve_dr(net: EnergyNetwork, gamma: float = 0.3, lam: float = 1.0,
             n_iter: int = 2000, stop_tol: Optional[float] = 1e-8,
             **kwargs) -> DRResult:
    """Shortcut: build a DRSolver and solve `net` with vanilla DR."""
    solver = DRSolver(DRConfig(gamma=gamma, lam=lam, n_iter=n_iter,
                               stop_tol=stop_tol))
    return solver.solve(net, **kwargs)