"""The distributed fixed-point iteration of Algorithm 1 and its base operator.

One iteration (every prosumer n, in parallel):

    net.communicate()                    # line 5: one-hop exchange of trades
    x_hat_n = operator.local_update(n)   # lines 6-10: base update (here DR)
    x_n <- x_hat_n + v_n                 # lines 11-12: learned correction, if deployed

The learned correction is additive and agnostic to the operator, so another
distributed fixed-point method can replace Douglas-Rachford by providing the
same two methods as `DouglasRachford`:

    local_update(prosumer) -> (D, G, {m: q_m})   one node's update, inference
    centralised_update(net, x, xbar) -> x        the same map, vectorised and
                                                 differentiable, for training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


class DouglasRachford:
    """Relaxed DR splitting of the market VI (paper App. A.2):
    x_hat = x + lam * (z - xbar), with xbar = proj_Xrec(x) and
    z = prox_{gamma (J + i_Xloc)}(2 xbar - x)."""

    def __init__(self, gamma: float, lam: float = 1.0, n_bisect: int = 30):
        self.gamma = gamma
        self.lam = lam
        self.n_bisect = n_bisect

    def local_update(self, p) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """Lines 6-10 at prosumer `p` (needs a fresh `p.q_received`).
        D and G are unchanged by the projection onto X^rec."""
        qbar = p.shadow_q()                                         # line 6
        rq = {m: 2.0 * qbar[m] - p.q[m] for m in p.neighbors}       # line 8
        zD, zG, zq = p.local_prox(self.gamma, p.D, p.G, rq, self.n_bisect)  # line 9
        lam = self.lam
        D_new = p.D + lam * (zD - p.D)                              # line 10
        G_new = p.G + lam * (zG - p.G)
        q_new = {m: p.q[m] + lam * (zq[m] - qbar[m]) for m in p.neighbors}
        return D_new, G_new, q_new

    def centralised_update(self, net, x: torch.Tensor, xbar: torch.Tensor) -> torch.Tensor:
        """Lines 6-10 for the whole network at once, `xbar = proj_Xrec(x)`."""
        z = net.prox_local(self.gamma, 2.0 * xbar - x, n_bisect=self.n_bisect)
        return x + self.lam * (z - xbar)


@dataclass
class DRConfig:
    gamma: float = 1.5
    lam: float = 1.0
    n_iter: int = 5000
    tol: float = 1e-5           # stop when the residual drops below tol
    stop_on: str = "perturbed"  # "perturbed": ||x^{k+1} - x^k|| (includes the correction v^k,
                                # the rule of every official run); "unperturbed": the base
                                # operator's own step ||x_hat^k - x^k|| (identical for plain DR)
    bisect: int = 40

    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("gamma must be > 0")
        if not (0.0 < self.lam < 2.0):
            raise ValueError("lam must lie in (0,2)")
        if self.stop_on not in ("perturbed", "unperturbed"):
            raise ValueError("stop_on must be 'perturbed' or 'unperturbed'")

    def operator(self) -> DouglasRachford:
        return DouglasRachford(self.gamma, self.lam, self.bisect)


@dataclass
class DRResult:
    n_iter: int
    converged: bool
    residual: List[float] = field(default_factory=list)          # ||x^{k+1} - x^k|| (auxiliary iterate)
    shadow_residual: List[float] = field(default_factory=list)   # ||xbar^{k+1} - xbar^k|| (shadow iterate)
    cost: List[float] = field(default_factory=list)              # J(xbar^k)
    violation: List[float] = field(default_factory=list)         # dist(xbar^k, X_loc)
    base_residual: List[float] = field(default_factory=list)     # ||x_hat^k - x^k||, base operator only


def run_fixed_point(net, operator, cfg: DRConfig, record: bool = True) -> DRResult:
    """Run Algorithm 1 from the state held by `net` until the residual drops
    below `cfg.tol` (or `cfg.n_iter` iterations). Prosumers with a deployed
    model (`gnn.deploy`) add their correction; the loop does not care.

    With `record`, also tracks per iteration the diagnostics of the shadow
    iterate xbar^k = proj_Xrec(x^k), the one that converges to a v-GNE:
    its move, its cost J(xbar^k) and its distance from the local box
    constraints. They do not affect the iteration.
    """
    net.reset()
    residual: List[float] = []
    shadow_residual: List[float] = []
    cost: List[float] = []
    violation: List[float] = []
    base_residual: List[float] = []
    converged = False
    k_run = 0
    prev_shadow = net.project_reciprocity(net.flatten()) if record else None

    with torch.no_grad():
        for k in range(cfg.n_iter):
            k_run = k + 1
            net.communicate()
            for p in net.prosumers:
                p.step(operator)
            r = net.aux_residual()
            r_base = net.base_residual()
            if record:
                residual.append(r)
                base_residual.append(r_base)
                shadow = net.project_reciprocity(net.flatten())
                shadow_residual.append(float(torch.linalg.vector_norm(shadow - prev_shadow)))
                cost.append(float(net.cost_at(shadow)))
                violation.append(float(net.dist_to_Xloc(shadow)))
                prev_shadow = shadow
            if (r if cfg.stop_on == "perturbed" else r_base) < cfg.tol:
                converged = True
                break

    return DRResult(n_iter=k_run, converged=converged, residual=residual, violation=violation,
                    shadow_residual=shadow_residual, cost=cost, base_residual=base_residual)


def run_dr(net, cfg: DRConfig, record: bool = True) -> DRResult:
    """Algorithm 1 with the Douglas-Rachford base operator of `cfg`."""
    return run_fixed_point(net, cfg.operator(), cfg, record=record)
