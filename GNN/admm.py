"""
ADMM solver for the Peer-to-Peer Energy Market GNE problem.

Splits the feasible set X = X_loc ∩ X_glob into:
    X_loc  : local constraints per node (box bounds on D, G, Q + balance equality)
    X_glob : coupling reciprocity constraints  q_{mn} + q_{nm} <= 0

Introduces auxiliary variables z (copies of the trade variables Q) subject to
the reciprocity constraints, and enforces consensus Q = z via the augmented
Lagrangian.

ADMM iteration (at iteration k):
    1) x-update  (distributed per node, differentiable via qpth):
         For each node n, solve:
           min_{D_n, G_n, q_n}  J_n(x_n) + (rho/2) * sum_{m in Omega_n} (q_{mn} - z_{mn} + u_{mn})^2
           s.t.  D_min_n <= D_n <= D_max_n
                 G_min_n <= G_n <= G_max_n
                 q_{mn}  <= kappa_{mn}        for all m in Omega_n
                 D_n = G_n + Delta_G_n + sum_m q_{mn}    (balance)

    2) z-update  (distributed per edge pair, closed-form):
         For each unordered pair {m, n}, project (q_{mn} + u_{mn}, q_{nm} + u_{nm})
         onto the half-space  z_{mn} + z_{nm} <= 0.

    3) u-update  (dual ascent):
         u_{mn} <- u_{mn} + q_{mn} - z_{mn}    for all edges (m, n).

Includes:
    - Automatic initial rho estimation from problem scale
    - Adaptive rho (residual balancing, Boyd et al. 2011 §3.4.1)

Usage
-----
    from admm import ADMMState, admm_init, admm_step, pure_ADMM_solve

    # Initialize ADMM state (auto-tunes rho from problem parameters)
    state, rho = admm_init(net)

    # Single iteration
    net_new, state, rho = admm_step(net, state, rho, adaptive=True)

    # Or run full loop (recommended: handles everything internally)
    net_final, results = pure_ADMM_solve(net, max_iter=200, tol=1e-6)
"""

import copy
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from energy_network import EnergyNetwork
from model import GNNUpdate


# ======================================================================
# ADMM State (carries z and u across iterations)
# ======================================================================

@dataclass
class ADMMState:
    """
    Persistent state for ADMM iterations.

    z : torch.Tensor (|E|,)  — auxiliary (consensus) copies of trade variables
    u : torch.Tensor (|E|,)  — scaled dual variables
    """
    z: torch.Tensor
    u: torch.Tensor


# ======================================================================
# Automatic rho estimation
# ======================================================================

def estimate_rho(net: EnergyNetwork) -> float:
    """
    Estimate a good initial rho from the problem's cost curvature.

    The idea: rho should be comparable to the "stiffness" of the cost
    on the trade variables. Since J_n has only a linear cost c_{nm}*q_{mn}
    on trades (no quadratic term), the x-update on q_{mn} is governed
    entirely by the ADMM penalty (rho/2)*q^2. If rho is much smaller
    than the quadratic cost coefficients on D and G (which are 2*a_tilde
    and a respectively), the optimizer will "lock in" D and G first and
    barely move the trades.

    Heuristic: set rho = mean of the quadratic cost coefficients.
        rho ≈ mean(2*a_tilde, a)  (matching the Hessian diagonal scale)

    Parameters
    ----------
    net : EnergyNetwork

    Returns
    -------
    rho : float (> 0)
    """
    diag_D = 2.0 * net.a_tilde  # (N,)
    diag_G = net.a              # (N,)

    all_diag = torch.cat([diag_D, diag_G])
    nonzero = all_diag[all_diag > 0]

    if len(nonzero) == 0:
        return 1.0

    rho = nonzero.mean().item()
    return rho


# ======================================================================
# Initialization
# ======================================================================

def admm_init(net: EnergyNetwork, rho: float = None) -> tuple:
    """
    Initialize ADMM state from the current network state.

    z is warm-started by projecting current Q onto reciprocity constraints
    so that z starts feasible w.r.t. the coupling.
    u is initialized to zero.

    If rho is None, automatically estimates it from problem scale.

    Parameters
    ----------
    net : EnergyNetwork with current state
    rho : ADMM penalty parameter (None = auto-estimate)

    Returns
    -------
    state : ADMMState
    rho   : float
    """
    E = len(net.edge_list)

    if rho is None:
        rho = estimate_rho(net)

    # Warm-start z: project Q onto reciprocity so z starts feasible
    z = _z_update(net, net.Q.detach().clone(), torch.zeros(E, device=net.device))
    u = torch.zeros(E, device=net.device)

    return ADMMState(z=z, u=u), rho


# ======================================================================
# x-update: local QP per node (differentiable via qpth)
# ======================================================================

def _x_update_node_qpth(
    net: EnergyNetwork,
    n: int,
    z: torch.Tensor,
    u: torch.Tensor,
    rho: float,
):
    """
    Solve the local QP for node n using qpth (differentiable).

    Decision variables:  y_n = [D_n, G_n, q_{m1,n}, ..., q_{mk,n}]

    Objective:
        J_n(y_n) + (rho/2) * sum_{m} (q_{mn} - z_{mn} + u_{mn})^2

    Returns
    -------
    D_n_new, G_n_new, q_n_new, incoming_global
    """
    from qpth.qp import QPFunction

    neighbors_n = sorted(net.neighbors[n])
    k = len(neighbors_n)
    dim_n = 2 + k
    dev = net.device

    idx_map = {edge: i for i, edge in enumerate(net.edge_list)}
    incoming_global = [idx_map[(m, n)] for m in neighbors_n]

    # ------------------------------------------------------------------
    # Build QP:  min 0.5 y^T H y + f^T y
    # ------------------------------------------------------------------
    H = torch.zeros(dim_n, dim_n, device=dev)
    f = torch.zeros(dim_n, device=dev)

    # D_n:  a_tilde_n * (D_n - D*)^2  =>  H[0,0] = 2*a_tilde, f[0] = -2*a_tilde*D*
    H[0, 0] = 2.0 * net.a_tilde[n]
    f[0]    = -2.0 * net.a_tilde[n] * net.D_star[n]

    # G_n:  0.5 * a_n * G^2 + b_n * G  =>  H[1,1] = a_n, f[1] = b_n
    H[1, 1] = net.a[n]
    f[1]    = net.b[n]

    # q_{mn}:  c_{nm}*q + (rho/2)*(q - (z-u))^2
    for local_i, m in enumerate(neighbors_n):
        qi = 2 + local_i
        g_idx = incoming_global[local_i]
        target = z[g_idx] - u[g_idx]

        H[qi, qi] = rho
        f[qi]     = net.c_edge[g_idx] - rho * target

    # ------------------------------------------------------------------
    # Equality:  D_n - G_n - sum_m q_{mn} = G_res_n
    # ------------------------------------------------------------------
    A_eq = torch.zeros(1, dim_n, device=dev)
    A_eq[0, 0] = 1.0
    A_eq[0, 1] = -1.0
    for li in range(k):
        A_eq[0, 2 + li] = -1.0
    b_eq = net.G_res[n].unsqueeze(0)

    # ------------------------------------------------------------------
    # Inequality:  G_ineq @ y <= h
    # ------------------------------------------------------------------
    n_ineq = 4 + k
    G_ineq = torch.zeros(n_ineq, dim_n, device=dev)
    h_ineq = torch.zeros(n_ineq, device=dev)

    row = 0
    G_ineq[row, 0] = -1.0; h_ineq[row] = -net.D_min[n].item(); row += 1
    G_ineq[row, 0] =  1.0; h_ineq[row] =  net.D_max[n].item(); row += 1
    G_ineq[row, 1] = -1.0; h_ineq[row] = -net.G_min[n].item(); row += 1
    G_ineq[row, 1] =  1.0; h_ineq[row] =  net.G_max[n].item(); row += 1
    for local_i in range(k):
        g_idx = incoming_global[local_i]
        G_ineq[row, 2 + local_i] = 1.0
        h_ineq[row] = net.k_edge[g_idx].item()
        row += 1

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    qp = QPFunction(verbose=False, eps=1e-12, maxIter=10000)
    y_opt = qp(
        H.unsqueeze(0), f.unsqueeze(0),
        G_ineq.unsqueeze(0), h_ineq.unsqueeze(0),
        A_eq.unsqueeze(0), b_eq.unsqueeze(0),
    ).squeeze(0)

    return y_opt[0], y_opt[1], y_opt[2:], incoming_global


def _x_update_all(
    net: EnergyNetwork,
    z: torch.Tensor,
    u: torch.Tensor,
    rho: float,
) -> EnergyNetwork:
    """
    x-update for all nodes (snapshot pattern: all read same z, u).
    """
    net_new = net.clone()
    D_new = net_new.D.clone()
    G_new = net_new.G.clone()
    Q_new = net_new.Q.clone()

    for n in range(net.N):
        D_n, G_n, q_n, incoming_global = _x_update_node_qpth(net, n, z, u, rho)
        D_new[n] = D_n
        G_new[n] = G_n
        for local_i, g_idx in enumerate(incoming_global):
            Q_new[g_idx] = q_n[local_i]

    net_new.D = D_new
    net_new.G = G_new
    net_new.Q = Q_new
    return net_new


# ======================================================================
# z-update: projection onto reciprocity (closed-form)
# ======================================================================

def _z_update(
    net: EnergyNetwork,
    Q: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """
    z-update (inference): project (Q + u) onto z_{mn} + z_{nm} <= 0.
    """
    idx_map = {edge: i for i, edge in enumerate(net.edge_list)}
    v = Q + u
    z_new = v.clone()

    visited = set()
    for (m, n_node) in net.edge_list:
        if (m, n_node) in visited:
            continue
        visited.add((m, n_node))
        visited.add((n_node, m))

        i_mn = idx_map[(m, n_node)]
        i_nm = idx_map[(n_node, m)]

        s = v[i_mn] + v[i_nm]
        if s.item() > 0:
            z_new[i_mn] = v[i_mn] - s / 2.0
            z_new[i_nm] = v[i_nm] - s / 2.0

    return z_new


def _z_update_differentiable(
    net: EnergyNetwork,
    Q: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable z-update using ReLU (for training with backprop).
    """
    idx_map = {edge: i for i, edge in enumerate(net.edge_list)}
    v = Q + u
    z_new = v.clone()

    visited = set()
    for (m, n_node) in net.edge_list:
        if (m, n_node) in visited:
            continue
        visited.add((m, n_node))
        visited.add((n_node, m))

        i_mn = idx_map[(m, n_node)]
        i_nm = idx_map[(n_node, m)]

        s = v[i_mn] + v[i_nm]
        correction = F.relu(s) / 2.0

        z_new = z_new.clone()
        z_new[i_mn] = v[i_mn] - correction
        z_new[i_nm] = v[i_nm] - correction

    return z_new


# ======================================================================
# Adaptive rho (Boyd et al. 2011, §3.4.1)
# ======================================================================

def _adapt_rho(
    rho: float,
    r_primal_norm: float,
    r_dual_norm: float,
    mu: float = 10.0,
    tau_incr: float = 2.0,
    tau_decr: float = 2.0,
) -> tuple:
    """
    Residual balancing for adaptive rho.

    If primal residual >> dual:  rho too small, increase it.
    If dual residual >> primal:  rho too large, decrease it.

    When rho changes, the scaled dual u must be rescaled to keep
    the unscaled dual (rho * u) constant.

    Returns
    -------
    rho_new  : updated penalty
    u_scale  : factor to multiply u by (1.0 if no change)
    """
    if r_primal_norm > mu * r_dual_norm and r_dual_norm > 1e-12:
        return rho * tau_incr, 1.0 / tau_incr
    elif r_dual_norm > mu * r_primal_norm and r_primal_norm > 1e-12:
        return rho / tau_decr, tau_decr
    else:
        return rho, 1.0
    
# ======================================================================
# L2 ADMM step
# ======================================================================

def learned_admm_step(
    net: EnergyNetwork,
    state: ADMMState,
    rho: float,model: GNNUpdate,t:int,
    training: bool = False,
    adaptive: bool = True,
    z_old: torch.Tensor = None,
) -> tuple:
    
    # 1) x-update
    net_new = _x_update_all(net, state.z, state.u, rho)


    # 2) z-update
    if training:
        z_new = _z_update_differentiable(net_new, net_new.Q, state.u)
    else:
        z_new = _z_update(net_new, net_new.Q, state.u)
        #z_new = _z_update_differentiable(net_new, net_new.Q, state.u)

    # 3) Adaptive rho
    rho_new = rho
    u_scale = 1.0
    if adaptive and z_old is not None:
        r_p_norm = torch.norm(net_new.Q - z_new).item()
        r_d_norm = rho * torch.norm(z_new - z_old).item()
        rho_new, u_scale = _adapt_rho(rho, r_p_norm, r_d_norm)

    # 4) u-update (with possible rescaling if rho changed)
    u_new = u_scale * state.u + (net_new.Q - z_new)

    # 5) GNN perturbation
    snapshot = net_new.clone()

    for n in range(net_new.N):
        # Node n reads from the frozen snapshot — not from net_new.
        # This ensures that n=1 sees the same D[0], G[0] as n=0 did,
        # regardless of what n=0 already wrote to net_new.
        data_n, _ = snapshot.get_local_pyg_data(n)

        # GNN returns:
        #   v_node_n : (1+k, 2)  — v_node_n[0] = (v_D_n, v_G_n) for ego node
        #   v_edge_n : (k, 1)    — v_edge_n[i] = perturbation for q_{m_i, n}
        v_node_n, v_edge_n = model(data_n, t)

        # Node n writes its perturbation to net_new (not to snapshot)
        net_new.apply_perturbation_node(n, v_node_n[0], v_edge_n[:, 0])

    # 6) projection
    if training:
        net_new.project_differentiable()
    else:
        x_proj, success = net_new.project_exact()
        if success:
            net_new.unflatten(x_proj)


    
    state_new = ADMMState(z=z_new, u=u_new)
    return net_new, state_new, rho_new
    

# ======================================================================
# Single ADMM step
# ======================================================================

def admm_step(
    net: EnergyNetwork,
    state: ADMMState,
    rho: float,
    training: bool = False,
    adaptive: bool = True,
    z_old: torch.Tensor = None,
) -> tuple:
    """
    One ADMM iteration:  x-update -> z-update -> (adapt rho) -> u-update.

    Parameters
    ----------
    net      : current EnergyNetwork state
    state    : ADMMState with current z and u
    rho      : current ADMM penalty parameter
    training : if True, use differentiable z-update
    adaptive : if True, adapt rho based on residual balancing
    z_old    : previous z (needed for dual residual). If None, skip adaptation.

    Returns
    -------
    net_new   : EnergyNetwork with updated state
    state_new : ADMMState with updated z and u
    rho_new   : (potentially updated) penalty parameter
    """
    # 1) x-update
    net_new = _x_update_all(net, state.z, state.u, rho)

    # 2) z-update
    if training:
        z_new = _z_update_differentiable(net_new, net_new.Q, state.u)
    else:
        z_new = _z_update(net_new, net_new.Q, state.u)
        #z_new = _z_update_differentiable(net_new, net_new.Q, state.u)

    # 3) Adaptive rho
    rho_new = rho
    u_scale = 1.0
    if adaptive and z_old is not None:
        r_p_norm = torch.norm(net_new.Q - z_new).item()
        r_d_norm = rho * torch.norm(z_new - z_old).item()
        rho_new, u_scale = _adapt_rho(rho, r_p_norm, r_d_norm)

    # 4) u-update (with possible rescaling if rho changed)
    u_new = u_scale * state.u + (net_new.Q - z_new)

    state_new = ADMMState(z=z_new, u=u_new)


    


    return net_new, state_new, rho_new


# ======================================================================
# Full ADMM solver
# ======================================================================

def pure_ADMM_solve(
    net_init: EnergyNetwork,
    rho: float = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    adaptive: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    Run ADMM to convergence.

    Parameters
    ----------
    net_init : initial EnergyNetwork state
    rho      : penalty parameter (None = auto-estimate)
    max_iter : maximum iterations
    tol      : convergence tolerance on both primal and dual residual
    adaptive : enable adaptive rho (recommended)
    verbose  : print progress

    Returns
    -------
    net     : converged EnergyNetwork
    results : dict with residuals and rho history
    """
    net = net_init.clone()
    state, rho = admm_init(net, rho)

    if verbose:
        print(f"  ADMM initial rho = {rho:.4f}")

    residuals_primal = []
    residuals_dual = []
    residuals = []
    rho_history = [rho]

    with torch.no_grad():
        z_old = None

        for it in range(max_iter):
            z_prev = state.z.clone()

            net, state, rho = admm_step(
                net, state, rho,
                training=False,
                adaptive=adaptive,
                z_old=z_old,
            )

            r_p = torch.sum((net.Q - state.z) ** 2).item()
            r_d = (rho ** 2) * torch.sum((state.z - z_prev) ** 2).item()

            residuals_primal.append(r_p)
            residuals_dual.append(r_d)
            residuals.append(r_p + r_d)
            rho_history.append(rho)

            z_old = state.z.clone()

            if r_p < tol and r_d < tol:
                if verbose:
                    print(f"  ADMM converged at iteration {it+1}  "
                          f"(r_p={r_p:.2e}, r_d={r_d:.2e}, rho={rho:.4f})")
                break

            if verbose and (it + 1) % 10 == 0:
                print(f"  ADMM iter {it+1:4d}  r_p={r_p:.2e}  "
                      f"r_d={r_d:.2e}  rho={rho:.4f}")

    return net, {
        "residuals_primal": residuals_primal,
        "residuals_dual": residuals_dual,
        "residuals": residuals,
        "rho_history": rho_history,
    }










