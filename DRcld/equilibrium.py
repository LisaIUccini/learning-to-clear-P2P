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

Default solver is CLARABEL (changed from OSQP 2026-10-14): OSQP's default
tolerances (eps_abs/eps_rel ~1e-3) left a spurious ~0.6-3% "dumping"
residual in the calibrated-market sanity check (equilibrium.solve_
equilibrium_diagnostics' S_n check, threshold 1e-4) that CLARABEL's tight
defaults resolved to exactly 0%, confirming it was solver noise, not a
real violation of Le Cadre Prop. 3 (c<b => zero dumping at the true
optimum) -- J* itself moved <0.001% between the two, so this is a pure
precision upgrade, not a behaviour change. Applied to the shared default
here (not just the one call site) so every J*/gap computation in the
project (diagnostics.py, every experiments/validate_*.py script, the
calibrated-market sanity check) reads from the same precise ground truth
instead of being fixed in only one place.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch


def solve_equilibrium_qp(net, solver: str = "CLARABEL", tau: float = 0.0):
    """`tau`: volume-dependent trading-tariff weight, (tau/2)*sum(Q^2)
    added to the trading cost (Le Cadre et al. 2020 Rem. 3.4) -- 0.0
    (default) is the original model, matching every pre-2026-10-14 use of
    this function."""
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
        cost = cost + cp.sum(cp.multiply(c_edge, Q)) + (tau / 2.0) * cp.sum(cp.square(Q))
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


def solve_equilibrium_point(net, solver: str = "CLARABEL", tau: float = 0.0) -> torch.Tensor:
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
        cost = cost + cp.sum(cp.multiply(c_edge, Q)) + (tau / 2.0) * cp.sum(cp.square(Q))
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


def solve_equilibrium_diagnostics(net, solver: str = "CLARABEL", tau: float = 0.0, **solver_opts) -> dict:
    """Same QP as `solve_equilibrium_qp`/`solve_equilibrium_point`, but
    returns a dict of everything the parameter-calibration sanity check
    needs: the primal solution, the per-node nodal price (dual value of
    that node's balance constraint -- the market-clearing price lambda_n a
    social planner / competitive equilibrium would charge node n, by
    strong duality of the potential-game QP), and the derived statistics
    the calibration loop reads directly (trade-at-capacity fraction,
    demand curtailment, dispatchable-generation "dumping" check, buyer/
    seller split). Does not touch `solve_equilibrium_qp`/`_point`'s own
    contracts (new function, nothing shared).

    `tau`: volume-dependent trading-tariff weight, (tau/2)*sum(Q^2) added
    to the trading cost (Le Cadre et al. 2020 Rem. 3.4). DEFAULT 0.0, AND
    KEPT AT 0.0 PROJECT-WIDE (user/supervisor decision, 2026-10-14): the
    paper's own Sec. 4 explicitly works "without requiring uniqueness of
    the equilibrium" and Theorem 6 guarantees DR converges to a v-GNE even
    when the solution set isn't a singleton, so forcing uniqueness via tau
    is unnecessary for anything the paper actually claims. (A tau>0 trial
    did work -- it drove `frac_at_capacity` to exactly 0 -- but moved J*
    by single-digit percent at the tau needed, confirmed to not actually
    be a good "how much did the equilibrium move" gauge either: J sums in
    constants (d_n, -b_tilde-type terms) that don't affect WHERE the
    optimum sits but do shift J*'s VALUE, so %-change in J* conflates a
    real equilibrium shift with an arbitrary constant-offset artifact. If
    a before/after comparison like this is needed again, compare D, G,
    and the per-node net trade instead -- see the next paragraph for why
    those (unlike individual edge Q's) are already well-defined at tau=0.)

    IMPORTANT, read before using `frac_at_capacity`/gross Q from the
    returned dict: at tau=0.0, whenever the graph has a cycle, the
    INDIVIDUAL per-edge Q's are generally NON-UNIQUE (free zero-cost
    circulation along any cycle), so `frac_at_capacity` and gross |Q| are
    solver-dependent / not meaningful as reported (empirically confirmed
    2026-10-14: varying kappa 6x left both essentially unchanged -- OSQP
    was freely picking among infinitely many equal-cost solutions, not
    reporting a genuine binding constraint). D, G, `cost` (J*), and
    `net_trade` (= D-G-Gres per node, pinned by the UNIQUE D/G even when
    the edge-level decomposition isn't) remain well-defined regardless.
    `frac_dumping`/`frac_surplus_res` DO read individual edge Q's (the
    reciprocity-slack S_n, see below) -- used only to flag a node, never
    compared as a magnitude across instances/tau, so the non-uniqueness
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
        cost = cost + cp.sum(cp.multiply(c_edge, Q)) + (tau / 2.0) * cp.sum(cp.square(Q))
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

    # "dumping" (user-specified formula, 2026-10-14): per node n,
    #   S_n = sum over neighbours m with q_mn < 0 (n exporting to m) of
    #         max(0, -(q_mn + q_nm))
    # i.e. the energy n OFFERS to m that m does not end up receiving. The
    # reciprocity constraint q_mn + q_nm <= 0 is an INEQUALITY, so it can
    # be strictly slack -- that slack is exactly the unabsorbed offer. This
    # reads individual edge Q's directly (non-unique at tau=0 when the
    # graph has cycles, see docstring above), which is fine here since S_n
    # only FLAGS a node, it is never compared as a magnitude across
    # instances/tau values.
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
