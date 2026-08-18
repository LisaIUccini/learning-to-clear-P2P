"""
equilibrium.py
==============

Exact v-GNE via one centralised QP -- for VALIDATION only (never used by
`train.py` or `algorithm.py`).

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
