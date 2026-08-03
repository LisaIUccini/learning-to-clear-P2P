"""
dr_solver.py
============

Scheduler for the distributed Douglas-Rachford market clearing
(Algorithm 1 of the L2Play paper).

This module does NOT solve anything. All the computation lives in the
prosumers: each one runs `Prosumer.dr_step` on its own data. `DRSolver`
only advances synchronous rounds, exactly as a wall clock would in a real
deployment, and collects diagnostics for plotting. It is agnostic to
whether the prosumers have a learned perturbation deployed (`gnn.deploy`)
or not: `dr_step` runs plain DR either way.

One iteration
-------------
    net.communicate()          # line 3: one-hop message exchange
    for p in net.prosumers:    # lines 4-8, each prosumer independently
        p.dr_step(gamma, lam)
        #   line 4  shadow        qbar_mn = q_mn - max(0, q_mn + q_nm)/2
        #   line 5  reflection    r = 2*qbar - q     (D, G unchanged)
        #   line 6  prox          own scalar-dual bisection over X^loc_n
        #   line 7  update        x <- x + lam (z - xbar)
        #   line 8  perturbation  its own copy of Phi_theta, if deployed
        #                         (gnn.py); no-op otherwise

Exactly ONE communication round per iteration. No prosumer ever reads a
variable belonging to a non-neighbour.

Termination
-----------
    residual = ||xbar^k - xbar^{k-1}||_2

The Euclidean move of the solution (the reciprocity-feasible market
profile). Convergence is declared when it falls below `stop_tol`, which is
therefore a plain tolerance on the norm itself: stop_tol = 1e-3 means the
feasible profile moved by less than 1e-3 between two iterations. This is
the same quantity plotted on the convergence figure, so the axis label,
the stopping threshold and the logged history all agree.

    residual_aux = ||x^{k+1} - x^k||   (move of the auxiliary iterate)

The shadow of iterate x^k is (D^k, G^k, qbar^k) and qbar needs the
neighbours' q^k, which arrive with the messages of round k. The shadow
residual is therefore known with a one-iteration lag -- and costs no extra
communication. Both residuals are recorded from the second iteration on,
so their indices line up.

Summing the per-node contributions is a global reduction: it is monitoring,
not part of the algorithm. A real system would stop on a local criterion or
agree on termination through a distributed consensus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from energy_network import EnergyNetwork


@dataclass
class DRConfig:
    gamma: float = 0.3          # prox step size (> 0)
    lam: float = 1.0            # DR relaxation, lambda in (0, 2)
    n_iter: int = 2000          # max iterations
    stop_tol: Optional[float] = 1e-3   # stop when ||xbar_k+1 - xbar_k||_2 < tol
                                #   a plain tolerance on the NORM of the move
                                #   of the feasible market profile (no square).
    prox_bisect: int = 60       # bisection steps inside each local prox

    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("gamma must be > 0")
        if not (0.0 < self.lam < 2.0):
            raise ValueError("lam must lie in (0, 2)")


@dataclass
class DRResult:
    x: torch.Tensor                       # final auxiliary iterate
    shadow: torch.Tensor                  # final feasible market profile
    n_iter: int                           # iterations actually run
    converged: bool
    n_messages: int = 0                   # total one-hop messages exchanged
    history: Dict[str, List[float]] = field(default_factory=dict)


class DRSolver:
    """Advances the synchronous rounds; the prosumers do all the work."""

    def __init__(self, config: Optional[DRConfig] = None):
        self.cfg = config if config is not None else DRConfig()

    def solve(
        self,
        net: EnergyNetwork,
        x0: Optional[torch.Tensor] = None,
        record: bool = True,
        write_back: bool = True,
    ) -> DRResult:
        """Run the distributed market clearing.

        Parameters
        ----------
        net : EnergyNetwork
            Holds the prosumers and the communication primitive.
        x0 : optional (dim,) starting point; defaults to the current state.
        record : log per-iteration diagnostics into result.history.
        write_back : leave the prosumers on their feasible shadow profile.
        """
        cfg = self.cfg
        if x0 is not None:
            net.unflatten(x0.detach().clone())
        net.reset_residuals()

        history: Dict[str, List[float]] = {
            "residual": [], "residual_aux": [],
            "cost": [], "balance": [], "reciprocity": [],
        }
        converged = False
        k_run = 0

        with torch.no_grad():
            for k in range(cfg.n_iter):
                k_run = k + 1

                # ---- line 3: one communication round ------------------
                net.communicate()

                # ---- lines 4-7: every prosumer computes on its own ----
                for p in net.prosumers:
                    p.dr_step(cfg.gamma, cfg.lam, n_bisect=cfg.prox_bisect)

                # ---- monitoring (global reduction, not algorithmic) ---
                if k == 0:
                    continue                    # shadow residual not yet defined
                # move of the solution (norm): ||xbar^k - xbar^{k-1}||_2
                res = net.shadow_residual()
                if record:
                    self._log(net, history, res)
                if cfg.stop_tol is not None and res < cfg.stop_tol:
                    converged = True
                    break

        x = net.flatten()
        shadow = net.project_reciprocity(x)
        if write_back:
            net.communicate()
            net.write_shadow()

        return DRResult(x=x, shadow=shadow, n_iter=k_run, converged=converged,
                        n_messages=k_run * net.E, history=history)

    @staticmethod
    def _log(net, history, res):
        shadow = net.project_reciprocity(net.flatten())
        history["residual"].append(res)
        history["residual_aux"].append(net.residual())
        history["cost"].append(float(net.cost_at(shadow)))
        history["balance"].append(float(net.balance_at(shadow).abs().max()))
        history["reciprocity"].append(float(net.reciprocity_max_at(shadow)))


# ----------------------------------------------------------------------
# Convenience one-shot function
# ----------------------------------------------------------------------
def solve_dr(net: EnergyNetwork, gamma: float = 0.3, lam: float = 1.0,
             n_iter: int = 2000, stop_tol: Optional[float] = 1e-3,
             **kwargs) -> DRResult:
    """Shortcut: build a DRSolver and run it on `net`."""
    return DRSolver(DRConfig(gamma=gamma, lam=lam, n_iter=n_iter,
                             stop_tol=stop_tol)).solve(net, **kwargs)