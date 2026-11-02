"""Anderson Accelerated Douglas-Rachford splitting (A2DR), applied to the SAME
distributed fixed-point map `fixed_point.run_dr` iterates
(`net.communicate()` + `Prosumer.step` with the DR operator for every prosumer, gamma/lam
fixed, no GNN correction). This is Algorithm 3.1 of Fu, Zhang & Boyd,
"Anderson Accelerated Douglas-Rachford Splitting" (SIAM J. Sci. Comput.
42(6), 2020) -- a classical, MODEL-FREE acceleration, meant as a baseline
to compare against the learned GNN correction (`gnn.py`/`training.py`): how
much of the GNN's speed-up over plain DR can a purely numerical technique
already buy for free, with no training at all?

Mapping onto this codebase's own state representation:

    v^k          <-> net.flatten()  (the auxiliary DR iterate x=[D,G,Q];
                     called "x" elsewhere in this project, "v" in the
                     paper -- kept as "v" here to match the paper 1:1)
    F_DRS(v^k)   <-> one plain-DR round: net.unflatten(v); net.communicate();
                     for p in net.prosumers: p.step(DouglasRachford(gamma, lam))
                     -- gnn_agent must be None on every prosumer (plain DR,
                     no learned correction) for this map to be F_DRS exactly
    g^k          <-> v^k - F_DRS(v^k)     (paper's residual function G_DRS)
    stopping rule<-> ||v^{k+1}-v^k|| < tol, same convention fixed_point.run_dr
                     uses (its own docstring: "stop when ||x^{k+1}-x^k||<tol")

Algorithm 3.1, translated line-by-line (see the module's `run_a2dr`):

    1  Input: v^0 (= net.flatten() at call time), gamma,eta,D,eps,R,Mmax
    2  Initialize nAA=0, RAA=0, Isafeguard=True
    3  v^1 = F_DRS(v^0), g^0 = v^0-v^1            [the k=0 pass below, where
                                                    Mk=min(Mmax,0)=0 so no AA
                                                    candidate exists yet --
                                                    this reduces exactly to
                                                    line 3 without a special
                                                    case]
    4  for k=1,2,...
    6      Mk = min(Mmax,k)
    7      v_DRS^{k+1}=F_DRS(v^k), g^k=v^k-v_DRS^{k+1}
    8      Yk=[y^{k-Mk}..y^{k-1}], Sk=[s^{k-Mk}..s^{k-1}]  (y^i=g^{i+1}-g^i,
                                                             s^i=v^{i+1}-v^i)
   10      gamma^k = argmin ||g^k-Yk*gamma||^2 + eta(||Sk||_F^2+||Yk||_F^2)||gamma||^2   (3.4)
   11      v_AA^{k+1} = sum_j alpha_j^k * v_DRS^{k-Mk+j+1}                                (3.3)
   13-17   safeguard: accept v_AA^{k+1} iff ||g^k||_2 <= D||g^0||_2(nAA/R+1)^-(1+eps),
           else fall back to v_DRS^{k+1}; once accepted, skip the check for
           the next R-1 iterations (RAA counter)
   18      stop if ||v^{k+1}-v^k|| < tol

Hyperparameter defaults (mmax=10, eta=1e-8, D=1e6, eps=1e-6, R=10) are the
paper's own solver defaults (Sec. 7). NOTE: on THIS project's own DR map
(every local prox solved by 20-40 bisection steps under box/capacity
constraints -- a highly kinked, nonsmooth map, unlike the paper's own
closed-form/single-Newton/equilibrated test problems), eta=1e-8 under-
regularizes badly and A2DR oscillates for 1000+ iterations before
eventually converging SLOWER than plain DR; the official (eta, mmax) per
density come from the calibration grid of scripts/run_a2dr.py. Always pass an explicit
`eta`/`mmax` validated on the map at hand rather than trusting these
class defaults blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .fixed_point import DouglasRachford


@dataclass
class A2DRConfig:
    # plain-DR parameters (same meaning/defaults as fixed_point.DRConfig --
    # F_DRS is exactly fixed_point.run_dr's own per-iteration map)
    gamma: float = 1.5
    lam: float = 1.0
    n_iter: int = 5000
    tol: float = 1e-5
    bisect: int = 40

    # Anderson acceleration (type-II, adaptively regularized, safeguarded --
    # paper Sec. 3.2/6, defaults from Sec. 7)
    mmax: int = 10          # M_max: AA memory (number of PAST DRS candidates kept)
    eta: float = 1e-8       # regularization coefficient, (3.4)
    safeguard_D: float = 1e6     # D, safeguard threshold scale
    safeguard_eps: float = 1e-6  # eps, safeguard decay exponent
    safeguard_R: int = 10         # R, iterations to skip the safeguard check after acceptance

    def __post_init__(self):
        if not (self.gamma > 0):
            raise ValueError("gamma must be > 0")
        if not (0.0 < self.lam < 2.0):
            raise ValueError("lam must lie in (0,2)")
        if not (self.mmax >= 0):
            raise ValueError("mmax must be >= 0")
        if not (self.eta > 0):
            raise ValueError("eta must be > 0")
        if not (self.safeguard_R >= 1):
            raise ValueError("safeguard_R must be >= 1")


@dataclass
class A2DRResult:
    n_iter: int
    converged: bool
    residual: List[float] = field(default_factory=list)           # ||v^{k+1}-v^k||, the step ACTUALLY taken
    shadow_residual: List[float] = field(default_factory=list)    # ||xbar^{k+1}-xbar^k||, same convention as DRResult
    cost: List[float] = field(default_factory=list)               # J(xbar^k)
    violation: List[float] = field(default_factory=list)          # dist(xbar^k, X_loc), same convention as DRResult
    aa_accepted: List[bool] = field(default_factory=list)         # was the AA candidate used at this iteration?


def _drs_step(net, operator, v: torch.Tensor) -> torch.Tensor:
    """One plain-DR round F_DRS(v): overwrite net's state with v, run the
    ordinary (no-GNN) local update on every prosumer, return the new state.
    Every prosumer's `gnn_agent` must be None -- this module never deploys
    one; call `gnn.undeploy(net)` beforehand if `net` was ever used with a
    learned correction."""
    net.unflatten(v)
    net.communicate()
    for p in net.prosumers:
        if p.gnn_agent is not None:
            raise RuntimeError(
                "a2dr.run_a2dr requires plain DR (no GNN correction) on every "
                "prosumer -- call gnn.undeploy(net) first."
            )
        p.step(operator)
    return net.flatten()


def run_a2dr(net, cfg: A2DRConfig, record: bool = True) -> A2DRResult:
    """Run A2DR to convergence (or `cfg.n_iter`) starting from whatever
    state `net` currently holds (net.flatten()) -- same calling convention
    as `fixed_point.run_dr` (build/seed `net`'s initial x^0 first, then call
    this). Leaves `net` holding the final accepted iterate.

    `record=True` tracks the same three diagnostics `fixed_point.run_dr`
    does (`shadow_residual`, `cost`, both evaluated on the SHADOW/feasible
    point project_reciprocity(v)), plus `aa_accepted` (whether the
    Anderson-extrapolated candidate was used at each iteration, vs. a
    fallback to the plain DRS candidate) as an additional diagnostic.
    """
    net.reset()
    operator = DouglasRachford(cfg.gamma, cfg.lam, cfg.bisect)
    Mmax = cfg.mmax

    v = net.flatten()  # v^0
    v_window: List[torch.Tensor] = [v]     # v^{k-Mk}..v^k (capped at Mmax+1)
    g_window: List[torch.Tensor] = []      # g^{k-Mk}..g^k
    vdrs_window: List[torch.Tensor] = []   # v_DRS^{k-Mk+1}..v_DRS^{k+1}

    n_AA = 0
    R_AA = 0
    safeguard_flag = True
    g0_norm: Optional[float] = None

    residual: List[float] = []
    shadow_residual: List[float] = []
    cost: List[float] = []
    violation: List[float] = []
    aa_accepted: List[bool] = []
    converged = False
    k_run = 0
    prev_shadow = net.project_reciprocity(v) if record else None

    with torch.no_grad():
        for k in range(cfg.n_iter):
            k_run = k + 1

            v_drs = _drs_step(net, operator, v)   # v_DRS^{k+1}
            g = v - v_drs                                              # g^k
            if g0_norm is None:
                g0_norm = float(torch.linalg.vector_norm(g))

            g_window.append(g)
            vdrs_window.append(v_drs)
            if len(g_window) > Mmax + 1:
                g_window.pop(0)
            if len(vdrs_window) > Mmax + 1:
                vdrs_window.pop(0)

            Mk = min(Mmax, k)   # AA memory available at this iteration (paper: M^k)
            aa_ok = False
            v_aa = None
            if Mk >= 1:
                V = torch.stack(v_window[-(Mk + 1):], dim=1)     # dim x (Mk+1)
                G = torch.stack(g_window[-(Mk + 1):], dim=1)     # dim x (Mk+1)
                S = V[:, 1:] - V[:, :-1]                          # dim x Mk
                Y = G[:, 1:] - G[:, :-1]                          # dim x Mk
                reg = cfg.eta * (S.pow(2).sum() + Y.pow(2).sum())  # eta(||S||_F^2+||Y||_F^2), (3.4)
                sqrt_reg = torch.sqrt(torch.clamp(reg, min=0.0))
                eye = torch.eye(Mk, device=Y.device, dtype=Y.dtype)
                A_aug = torch.cat([Y, sqrt_reg * eye], dim=0)
                b_aug = torch.cat([g, torch.zeros(Mk, device=Y.device, dtype=Y.dtype)])
                aa_gamma = torch.linalg.lstsq(A_aug, b_aug.unsqueeze(1), driver="gelsd").solution.squeeze(1)  # (3.4)

                alpha = torch.empty(Mk + 1, device=Y.device, dtype=Y.dtype)  # (3.3)
                alpha[0] = aa_gamma[0]
                if Mk >= 2:
                    alpha[1:Mk] = aa_gamma[1:Mk] - aa_gamma[:Mk - 1]
                alpha[Mk] = 1.0 - aa_gamma[Mk - 1]

                VDRS = torch.stack(vdrs_window[-(Mk + 1):], dim=1)  # v_DRS^{k-Mk+1}..v_DRS^{k+1}
                v_aa = VDRS @ alpha
                aa_ok = True

            gnorm = float(torch.linalg.vector_norm(g))
            take_aa = False
            if aa_ok:
                if safeguard_flag or R_AA >= cfg.safeguard_R:
                    thresh = cfg.safeguard_D * g0_norm * (n_AA / cfg.safeguard_R + 1.0) ** (-(1.0 + cfg.safeguard_eps))
                    if gnorm <= thresh:
                        take_aa = True
                        n_AA += 1
                        safeguard_flag = False
                        R_AA = 1
                    else:
                        R_AA = 0
                else:
                    take_aa = True
                    n_AA += 1
                    R_AA += 1

            v_next = v_aa if take_aa else v_drs
            step_residual = float(torch.linalg.vector_norm(v_next - v))

            v_window.append(v_next)
            if len(v_window) > Mmax + 1:
                v_window.pop(0)
            v = v_next

            if record:
                residual.append(step_residual)
                shadow = net.project_reciprocity(v)
                shadow_residual.append(float(torch.linalg.vector_norm(shadow - prev_shadow)))
                cost.append(float(net.cost_at(shadow)))
                violation.append(float(net.dist_to_Xloc(shadow)))
                prev_shadow = shadow
                aa_accepted.append(take_aa)

            if step_residual < cfg.tol:
                converged = True
                break

    net.unflatten(v)  # leave net's state consistent with the returned iterate
    return A2DRResult(n_iter=k_run, converged=converged, residual=residual,
                       shadow_residual=shadow_residual, cost=cost, violation=violation,
                       aa_accepted=aa_accepted)
