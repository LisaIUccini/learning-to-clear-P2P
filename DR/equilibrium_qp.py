"""
equilibrium_qp.py
==================

Exact market equilibrium via a single centralised QP -- for EVALUATION
ONLY, never used in training (self-supervision stays strict: no target
ever enters the loss; see training.py's module docstring).

Why this is exact, not a heuristic
-----------------------------------
Every prosumer's cost J_n depends only on its own (D_n, G_n, {q_mn}) --
there is no term in the objective coupling different prosumers. The ONLY
coupling in the whole GNEP is through constraints: each node's own local
feasible set X^loc_n (box bounds on D, G, and the local balance equality
D_n - G_n - G_res_n - sum_m q_mn = 0, see EnergyNetwork.prox_local), the
per-edge capacity q_mn <= kappa_mn, and the reciprocity coupling
q_mn + q_nm <= 0 (X^rec, see EnergyNetwork.project_reciprocity_Q).

A game whose players' costs are separable like this is a POTENTIAL game
with potential P(x) = sum_n J_n(x_n) -- exactly EnergyNetwork.cost_at(x)
-- so its v-GNE coincide with the minimisers of ONE centralised problem:
"minimise P(x) subject to every constraint above, all at once". That is a
convex QP (quadratic objective, all-linear constraints: a_tilde, a > 0 by
construction, see market_generator.py's ranges). Solving it exactly with
an off-the-shelf solver is orders of magnitude cheaper and more reliable
than running DR for tens of thousands of iterations hoping it gets close
(which, for the hardest instances, did not always even converge within a
20000-30000 iteration cap -- see the J* estimation notes elsewhere in
this project).
"""

from __future__ import annotations

from typing import Tuple

import cvxpy as cp
import torch

from energy_network import EnergyNetwork


def solve_equilibrium_qp(net: EnergyNetwork, solver: str = "OSQP"
                         ) -> Tuple[torch.Tensor, float]:
    """Exact v-GNE (x*, J*) via a single centralised QP.

    Returns `(x_star, J_star)`: `x_star` in the same [D, G, Q] layout as
    `net.flatten()`, `J_star` a python float equal to `net.cost_at(x_star)`
    (up to solver tolerance). Evaluation only -- do not call this from
    anywhere in the training path.
    """
    N, E = net.N, net.E

    at = net.a_tilde_vec.detach().cpu().numpy()
    a = net.a_vec.detach().cpu().numpy()
    b = net.b_vec.detach().cpu().numpy()
    Dstar = net.D_star_vec.detach().cpu().numpy()
    Gres = net.G_res_vec.detach().cpu().numpy()
    Dmin = net.D_min_vec.detach().cpu().numpy()
    Dmax = net.D_max_vec.detach().cpu().numpy()
    Gmin = net.G_min_vec.detach().cpu().numpy()
    Gmax = net.G_max_vec.detach().cpu().numpy()

    D = cp.Variable(N)
    G = cp.Variable(N)

    cost = cp.sum(cp.multiply(at, cp.square(D - Dstar))) \
         + 0.5 * cp.sum(cp.multiply(a, cp.square(G))) \
         + cp.sum(cp.multiply(b, G))
    constraints = [D >= Dmin, D <= Dmax, G >= Gmin, G <= Gmax]

    if E > 0:
        Q = cp.Variable(E)
        c_edge = net.c_edge.detach().cpu().numpy()
        k_edge = net.k_edge.detach().cpu().numpy()
        cost = cost + cp.sum(cp.multiply(c_edge, Q))
        constraints.append(Q <= k_edge)

        # local balance per node (X^loc's equality): D_n-G_n-Gres_n-sum q_mn = 0
        for n in range(N):
            idx = net.incoming_idx[n]
            bal = D[n] - G[n] - Gres[n] - (cp.sum(Q[idx]) if idx else 0.0)
            constraints.append(bal == 0)

        # reciprocity (X^rec), one constraint per UNIQUE undirected pair
        for (m, n) in net._unique_pairs:
            i, j = net._idx_map[(m, n)], net._idx_map[(n, m)]
            constraints.append(Q[i] + Q[j] <= 0)
    else:
        Q = None
        for n in range(N):
            constraints.append(D[n] - G[n] - Gres[n] == 0)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP did not solve to optimality: status={prob.status}")

    device, dtype = net.device, net.flatten().dtype
    D_star = torch.tensor(D.value, dtype=dtype, device=device)
    G_star = torch.tensor(G.value, dtype=dtype, device=device)
    Q_star = (torch.tensor(Q.value, dtype=dtype, device=device) if E > 0
             else torch.zeros(0, dtype=dtype, device=device))

    x_star = torch.cat([D_star, G_star, Q_star])
    J_star = float(net.cost_at(x_star))
    return x_star, J_star
