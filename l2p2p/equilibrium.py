"""Exact v-GNE through one centralised QP: a reference for the distributed
iteration, never used by it.

The local costs J_n are separable, so the game is a potential game with
potential J = sum_n J_n and its v-GNEs are the minimisers of

    minimise  sum_n J_n(x_n)   s.t.  x in X^loc cap X^rec.

Solved with CLARABEL, whose tight default tolerances matter for the
dual-based diagnostics (OSQP's defaults left spurious ~1% "dumping").
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def solve_equilibrium_qp(net, solver: str = "CLARABEL"):
    """J* = min J over the feasible set (equal at every v-GNE)."""
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


def solve_equilibrium_diagnostics(net, solver: str = "CLARABEL", **solver_opts) -> dict:
    """Same QP as `solve_equilibrium_qp`, returning the market statistics of
    the plausibility table (Table D): primal solution, nodal prices (duals
    of the balance constraints), trade-at-capacity fraction, demand
    curtailment, "dumping" check and buyer/seller split.

    IMPORTANT, read before using `frac_at_capacity`/gross Q from the
    returned dict: whenever the graph has a cycle, the
    INDIVIDUAL per-edge Q's are generally NON-UNIQUE (free zero-cost
    circulation along any cycle), so `frac_at_capacity` and gross |Q| are
    solver-dependent / not meaningful as reported (empirically confirmed
    varying kappa 6x left both essentially unchanged -- OSQP
    was freely picking among infinitely many equal-cost solutions, not
    reporting a genuine binding constraint). D, G, `cost` (J*), and
    `net_trade` (= D-G-Gres per node, pinned by the UNIQUE D/G even when
    the edge-level decomposition isn't) remain well-defined regardless.
    `frac_dumping`/`frac_surplus_res` DO read individual edge Q's (the
    reciprocity-slack S_n, see below) -- used only to flag a node, never
    compared as a magnitude across instances, so the non-uniqueness
    doesn't undermine them the way it does `frac_at_capacity`."""
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
    balance_constraints = None
    k_edge = np.zeros(0, dtype=np.float32)
    if E > 0:
        Q = cp.Variable(E)
        c_edge = net.c_edge.cpu().numpy(); k_edge = net.k_edge.cpu().numpy()
        cost = cost + cp.sum(cp.multiply(c_edge, Q))
        constraints.append(Q <= k_edge)
        owner = net.owner.tolist()
        incoming = {n: [] for n in range(N)}
        for i, n in enumerate(owner):
            incoming[n].append(i)
        balance_constraints = []
        for n in range(N):
            idx = incoming[n]
            bal = D[n] - G[n] - Gres[n] - (cp.sum(Q[idx]) if idx else 0.0)
            bc = (bal == 0)
            constraints.append(bc)
            balance_constraints.append(bc)
        seen = set()
        for i, (m, n) in enumerate(net.edge_list):
            if (m, n) in seen or (n, m) in seen:
                continue
            seen.add((m, n))
            j = net.idx_map[(n, m)]
            constraints.append(Q[i] + Q[j] <= 0)
    else:
        balance_constraints = []
        for n in range(N):
            bc = (D[n] - G[n] - Gres[n] == 0)
            constraints.append(bc)
            balance_constraints.append(bc)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver, **solver_opts)
    feasible = prob.status in ("optimal", "optimal_inaccurate")
    if not feasible:
        return {"feasible": False, "status": prob.status}

    D_val = np.asarray(D.value, dtype=np.float64)
    G_val = np.asarray(G.value, dtype=np.float64)
    Q_val = np.asarray(Q.value, dtype=np.float64) if E > 0 else np.zeros(0)
    # nodal price lambda_n: dual of node n's own balance equality -- the
    # shadow price of relaxing that node's power balance by one unit,
    # i.e. exactly the competitive/social-planner nodal price at the QP
    # optimum (strong duality holds, the QP is convex with a Slater point
    # whenever Assumption 1 holds).
    lam = np.array([float(bc.dual_value) for bc in balance_constraints])

    # only the IMPORT side (Q <= k_edge) is actually constrained -- the
    # model has no explicit lower bound on Q, so a node can export (very
    # negative Q) as much as the counterparty's own import limit allows,
    # unconstrained by ITS OWN k_edge; checking abs(Q) here double-counted
    # every large legitimate export as a false "at capacity" (caught via
    # the sanity check itself: varying kappa/c by 6x changed this metric
    # by <2 points, which should be impossible if it were really tracking
    # a binding constraint -- it wasn't).
    at_capacity = (Q_val >= 0.98 * k_edge) if E > 0 else np.zeros(0, dtype=bool)
    curtailment = (Dstar - D_val) / np.maximum(Dstar, 1e-9)   # >0 = under-served vs target
    net_trade = np.zeros(N)
    if E > 0:
        owner_arr = np.array(net.owner.tolist())
        for n in range(N):
            net_trade[n] = Q_val[owner_arr == n].sum()
    is_net_seller = net_trade < -1e-6

    # "dumping": per node n,
    #   S_n = sum over neighbours m with q_mn < 0 (n exporting to m) of
    #         max(0, -(q_mn + q_nm))
    # i.e. the energy n OFFERS to m that m does not end up receiving. The
    # reciprocity constraint q_mn + q_nm <= 0 is an INEQUALITY, so it can
    # be strictly slack -- that slack is exactly the unabsorbed offer. This
    # reads individual edge Q's directly (non-unique when the
    # graph has cycles, see docstring above), which is fine here since S_n
    # only FLAGS a node, it is never compared as a magnitude across
    # instances.
    #   - "true" dumping: S_n > 1e-4 AND G_n > G_min_n + 1e-4, i.e. the node
    #     is also running dispatchable generation ABOVE its own floor while
    #     some of its offered energy goes unabsorbed -- a real economic
    #     waste (paying for generation that nobody takes).
    #   - "RES surplus" (legitimate): S_n > 1e-4 but G_n ~ G_min_n -- the
    #     unabsorbed offer comes from G_res (free, already-available RES),
    #     not from extra paid-for dispatchable generation.
    S = np.zeros(N)
    if E > 0:
        for n in range(N):
            for m in net.neighbors[n]:
                q_mn = Q_val[net.idx_map[(m, n)]]
                if q_mn < 0:
                    q_nm = Q_val[net.idx_map[(n, m)]]
                    S[n] += max(0.0, -(q_mn + q_nm))
    has_surplus = S > 1e-4
    dumping_true = has_surplus & (G_val > Gmin + 1e-4)
    surplus_res = has_surplus & ~dumping_true

    return {
        "feasible": True, "status": prob.status,
        "D": D_val, "G": G_val, "Q": Q_val, "lam": lam,
        "cost": float(prob.value),
        "curtailment": curtailment,
        "frac_at_capacity": float(at_capacity.mean()) if E > 0 else 0.0,
        "frac_dumping": float(dumping_true.mean()),
        "frac_surplus_res": float(surplus_res.mean()),
        "net_trade": net_trade,
        "n_buyers": int((net_trade > 1e-6).sum()), "n_sellers": int(is_net_seller.sum()),
        "n_neutral": int(N - (net_trade > 1e-6).sum() - is_net_seller.sum()),
    }
