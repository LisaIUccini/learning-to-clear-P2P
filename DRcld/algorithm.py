"""
algorithm.py
============

Scheduler for the distributed Douglas-Rachford market-clearing loop
(Algorithm 1, paper Sec. III-A). This module does not compute anything
itself -- it advances synchronous rounds, exactly as a wall clock would in
a real deployment, while every prosumer runs its own share of the update
(`Prosumer.dr_step`, network.py).

One iteration:

    net.communicate()          # line 3: one-hop message exchange
    for p in net.prosumers:    # lines 4-12, each prosumer independently
        p.dr_step(gamma, lam)  # DR update, plus its own GNN correction
                                # if one has been deployed (gnn.deploy)

Exactly one communication round per iteration, whether or not a learned
model is deployed -- `dr_step` is agnostic to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class DRConfig:
    gamma: float = 1.5
    lam: float = 1.0
    n_iter: int = 5000
    tol: float = 1e-5           # stop when ||x^{k+1}-x^k|| < tol
    stop_on: str = "perturbed"  # "perturbed": ||x^{k+1}-x^k|| (incl. GNN v^k, the
                                 # original rule); "unperturbed": ||lam*(z^k-xtilde^k)||,
                                 # the DR-only part (identical for plain DR)
    bisect: int = 40
    tau: float = 0.0            # volume-dependent trading-tariff weight (quadratic term
                                 # (tau/2)*q_mn^2 added to every prosumer's trading cost --
                                 # 0.0 = unchanged original model, every experiment before
                                 # 2026-10-14 used tau=0.0 implicitly; see Le Cadre et al.
                                 # 2020 Rem. 3.4 -- removes zero-cost circulating trades /
                                 # non-unique q* by making the game strongly monotone)

    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("gamma must be > 0")
        if not (0.0 < self.lam < 2.0):
            raise ValueError("lam must lie in (0,2)")
        if self.stop_on not in ("perturbed", "unperturbed"):
            raise ValueError("stop_on must be 'perturbed' or 'unperturbed'")


@dataclass
class DRResult:
    n_iter: int
    converged: bool
    residual: List[float] = field(default_factory=list)          # ||x^{k+1}-x^k||, aux iterate
    shadow_residual: List[float] = field(default_factory=list)   # ||xbar^{k+1}-xbar^k||, FEASIBLE iterate
    cost: List[float] = field(default_factory=list)              # J(xbar^k), only if record=True
    violation: List[float] = field(default_factory=list)         # dist(xbar^k, X_loc), only if record=True
    dr_residual: List[float] = field(default_factory=list)       # ||lam*(z^k-xtilde^k)||, only if record=True


def run_dr(net, cfg: DRConfig, record: bool = True) -> DRResult:
    """Run the distributed loop to convergence (or `cfg.n_iter`).

    `net`'s prosumers may or may not have a `gnn_agent` deployed
    (gnn.deploy); this function does not care -- it only schedules rounds.

    `record=True` also tracks, per iteration:
      - `shadow_residual`, the move of the SHADOW/feasible iterate xbar^k =
        proj_X^rec(x^k) -- unlike the auxiliary iterate x^k, xbar^k is
        always a feasible market allocation, and it is xbar^k (not x^k)
        that is proven to converge to the true v-GNE (paper, Sec. III-A).
        `residual` (the auxiliary iterate's own move, what the stopping
        rule actually checks) can look converged locally without xbar
        having settled -- see e.g. a long shared plateau in both.
      - `cost`, J(xbar^k) (cheap: one elementwise pass over cached
        parameter tensors) -- the same quantity `equilibrium.
        solve_equilibrium_qp` gives the true optimum of.
      - `violation`, dist(xbar^k, X_loc) (`Network.dist_to_Xloc`) -- how
        far the shadow iterate still sits from the LOCAL box constraints
        at iteration k (X^rec-feasibility is automatic for xbar^k by
        construction; X^loc-feasibility is the other half of a true
        v-GNE, and is what this tracks).
    All three are diagnostics only, no effect on the algorithm itself.
    """
    net.reset()
    residual: List[float] = []
    shadow_residual: List[float] = []
    cost: List[float] = []
    violation: List[float] = []
    dr_residual: List[float] = []
    converged = False
    k_run = 0
    prev_shadow = net.project_reciprocity(net.flatten()) if record else None

    with torch.no_grad():
        for k in range(cfg.n_iter):
            k_run = k + 1
            net.communicate()
            for p in net.prosumers:
                p.dr_step(cfg.gamma, cfg.lam, n_bisect=cfg.bisect, tau=cfg.tau)
            r = net.aux_residual()
            r_dr = net.dr_residual()
            if record:
                residual.append(r)
                dr_residual.append(r_dr)
                shadow = net.project_reciprocity(net.flatten())
                shadow_residual.append(float(torch.linalg.vector_norm(shadow - prev_shadow)))
                cost.append(float(net.cost_at(shadow)))
                violation.append(float(net.dist_to_Xloc(shadow)))
                prev_shadow = shadow
            if (r if cfg.stop_on == "perturbed" else r_dr) < cfg.tol:
                converged = True
                break

    return DRResult(n_iter=k_run, converged=converged, residual=residual, violation=violation,
                     shadow_residual=shadow_residual, cost=cost, dr_residual=dr_residual)
