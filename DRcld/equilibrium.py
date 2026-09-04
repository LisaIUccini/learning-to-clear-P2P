"""
equilibrium.py
==============

Exact v-GNE via one centralised QP -- originally for VALIDATION only
(never used by `train.py`/`algorithm.py`, still true for those two).
`solve_equilibrium_point` below is additionally used as a genuine
TRAINING SIGNAL by `train_warmstart_baseline.py` (the regression loss,
Eq. 6 of warm-start.pdf) -- `solve_equilibrium_qp` itself is untouched
and still only ever used for validation/diagnostics.

Every J_n depends only on prosumer n's own (D_n, G_n, q_n): there is no
cross term between different prosumers' objectives, only through the
constraints. The paper's own proof of Theorem 1 uses exactly this fact
("J = sum_n J_n, since the individual costs are separable, F = grad J",
Sec. IV-B): the game is a POTENTIAL game with potential J = sum_n J_n, so
its v-GNE coincide with the minimisers of the single convex QP

    minimise  sum_n J_n(x_n)
    s.t.      x in X^loc(theta) cap X^rec

Solving this exactly gives an independent ground truth to check the
distributed DR loop actually reaches, without relying on DR itself.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch


def solve_equilibrium_qp(net, solver: str = "OSQP"):
    N, E = net.N, net.E
    at = net.a_tilde_vec.cpu().numpy(); a = net.a_vec.cpu().numpy(); b = net.b_vec.cpu().numpy()
    Dstar = net.D_star_vec.cpu().numpy(); Gres = net.G_res_vec.cpu().numpy()
    Dmin, Dmax = net.D_min_vec.cpu().numpy(), net.D_max_vec.cpu().numpy()
    Gmin, Gmax = net.G_min_vec.cpu().numpy(), net.G_max_vec.cpu().numpy()

    D = cp.Variable(N); G = cp.Variable(N)
    cost = (cp.sum(cp.multiply(at, cp.square(D - Dstar)))
            + 0.5 * cp.sum(cp.multiply(a, cp.square(G))) + cp.sum(cp.multiply(b, G)))
    constraints = [D >= Dmin, D <= Dmax, G >= Gmin, G <= Gmax]

    if E > 0:
        Q = cp.Variable(E)
        c_edge = net.c_edge.cpu().numpy(); k_edge = net.k_edge.cpu().numpy()
        cost = cost + cp.sum(cp.multiply(c_edge, Q))
        constraints.append(Q <= k_edge)
        owner = net.owner.tolist()
        incoming = {n: [] for n in range(N)}
        for i, n in enumerate(owner):
            incoming[n].append(i)
        for n in range(N):
            idx = incoming[n]
            bal = D[n] - G[n] - Gres[n] - (cp.sum(Q[idx]) if idx else 0.0)
            constraints.append(bal == 0)
        seen = set()
        for i, (m, n) in enumerate(net.edge_list):
            if (m, n) in seen or (n, m) in seen:
                continue
            seen.add((m, n))
            j = net.idx_map[(n, m)]
            constraints.append(Q[i] + Q[j] <= 0)
    else:
        for n in range(N):
            constraints.append(D[n] - G[n] - Gres[n] == 0)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP did not solve to optimality: status={prob.status}")
    return float(prob.value)


def solve_equilibrium_point(net, solver: str = "OSQP") -> torch.Tensor:
    """Same QP as `solve_equilibrium_qp`, but returns the PRIMAL OPTIMIZER
    z* = [D*, G*, Q*] (`Network.flatten()`'s own [D,G,Q] layout, Q ordered
    by `net.edge_list`, matching `net.owner`/`net.c_edge`/`net.k_edge`)
    instead of only the optimal value.

    This is the "known (possibly non-unique) fixed point z*(theta)" that
    Sambharya et al.'s REGRESSION loss (Eq. 6, warm-start.pdf) needs, and
    which train_warmstart_baseline.py uses (loss_mode="regression"): since
    this market game is a potential game (module docstring above), z* can
    be computed EXACTLY via one centralised QP solve, rather than
    approximated -- an advantage over the source paper's general setting,
    where a ground-truth z*(theta) is often unavailable at all (hence
    their OWN preference for the fixed-point-residual loss in practice,
    Sec. 5.2 -- not applicable here since we do not share that limitation).

    Duplicates `solve_equilibrium_qp`'s ~15 lines of QP construction
    rather than sharing it, so that function's own return contract (a
    bare float, used elsewhere e.g. diagnostics.py's cost-gap plots) is
    never at risk of being disturbed by this addition.

    IMPORTANT: z* is a FEASIBLE market allocation (a v-GNE), matching the
    SHADOW/projected DR iterate xbar^k = proj_X^rec(x^k), not the raw
    auxiliary iterate x^k itself (Sec. III-A/IV-B of the market-clearing
    paper: only xbar^k is proven to converge to a v-GNE). Callers comparing
    against a rollout's auxiliary iterate must project it first via
    `Network.project_reciprocity`.
    """
    N, E = net.N, net.E
    at = net.a_tilde_vec.cpu().numpy(); a = net.a_vec.cpu().numpy(); b = net.b_vec.cpu().numpy()
    Dstar = net.D_star_vec.cpu().numpy(); Gres = net.G_res_vec.cpu().numpy()
    Dmin, Dmax = net.D_min_vec.cpu().numpy(), net.D_max_vec.cpu().numpy()
    Gmin, Gmax = net.G_min_vec.cpu().numpy(), net.G_max_vec.cpu().numpy()

    D = cp.Variable(N); G = cp.Variable(N)
    cost = (cp.sum(cp.multiply(at, cp.square(D - Dstar)))
            + 0.5 * cp.sum(cp.multiply(a, cp.square(G))) + cp.sum(cp.multiply(b, G)))
    constraints = [D >= Dmin, D <= Dmax, G >= Gmin, G <= Gmax]

    Q = None
    if E > 0:
        Q = cp.Variable(E)
        c_edge = net.c_edge.cpu().numpy(); k_edge = net.k_edge.cpu().numpy()
        cost = cost + cp.sum(cp.multiply(c_edge, Q))
        constraints.append(Q <= k_edge)
        owner = net.owner.tolist()
        incoming = {n: [] for n in range(N)}
        for i, n in enumerate(owner):
            incoming[n].append(i)
        for n in range(N):
            idx = incoming[n]
            bal = D[n] - G[n] - Gres[n] - (cp.sum(Q[idx]) if idx else 0.0)
            constraints.append(bal == 0)
        seen = set()
        for i, (m, n) in enumerate(net.edge_list):
            if (m, n) in seen or (n, m) in seen:
                continue
            seen.add((m, n))
            j = net.idx_map[(n, m)]
            constraints.append(Q[i] + Q[j] <= 0)
    else:
        for n in range(N):
            constraints.append(D[n] - G[n] - Gres[n] == 0)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP did not solve to optimality: status={prob.status}")

    D_val = np.asarray(D.value, dtype=np.float32)
    G_val = np.asarray(G.value, dtype=np.float32)
    Q_val = np.asarray(Q.value, dtype=np.float32) if E > 0 else np.zeros(0, dtype=np.float32)
    z_star = np.concatenate([D_val, G_val, Q_val])
    return torch.tensor(z_star, device=net.device, dtype=torch.float32)
