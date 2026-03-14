import torch

from energy_network import EnergyNetwork
from model import GNNUpdate


# ------------------------------------------------------------------
# ADMM communication round
# ------------------------------------------------------------------

def _admm_communicate(net: EnergyNetwork):
    """
    Communication round for ADMM: each prosumer sends q, z, u to each neighbor.

    After this call, for every directed edge (m -> n):
        prosumer[n].q_received[m] = prosumer[m].q[n]
        prosumer[n].z_received[m] = prosumer[m].z[n]
        prosumer[n].u_received[m] = prosumer[m].u[n]

    q_received and u_received are needed for the z-update.
    z_received is kept for monitoring / convergence checking.
    """
    for (m, n) in net.edge_list:
        net.prosumers[m].send_q_to(net.prosumers[n])
        net.prosumers[m].send_z_to(net.prosumers[n])
        net.prosumers[m].send_u_to(net.prosumers[n])


# ------------------------------------------------------------------
# Pure ADMM step  (no learning)
# ------------------------------------------------------------------

def pure_ADMM_step(net: EnergyNetwork, rho: float) -> EnergyNetwork:
    """
    One iteration of distributed ADMM for the energy market v-GNE problem.

    The problem decomposes across prosumers by introducing consensus variables
    z_{mn} (held by node n) that enforce the reciprocity constraint
    q_{mn} + q_{nm} <= 0 at convergence via the augmented Lagrangian.

    Iteration (scaled ADMM form):

        1. x-update (parallel, local QP per prosumer n):
               min J_n(D,G,q) + (rho/2) sum_m ||q[m] - z[m] + u[m]||^2
               s.t. local constraints (box, capacity, balance)
               -- no reciprocity here --

        2. communicate: each n sends q[m], z[m], u[m] to each neighbor m

        3. z-update (closed-form, per edge pair):
               a = q_n[m] + u_n[m]
               b = q_m[n] + u_m[n]    (received from neighbor m)
               z_n[m] = a - max(0, (a+b)/2)   (project onto a+b <= 0)

        4. u-update (local, dual ascent):
               u[m] += q[m] - z[m]

    At convergence: q[m] = z[m] and z_n[m] + z_m[n] <= 0 (reciprocity).

    Parameters
    ----------
    net : current EnergyNetwork state (not modified)
    rho : ADMM penalty parameter

    Returns
    -------
    net_new : updated EnergyNetwork
    """
    net_new = net.clone()

    # Step 1 — x-update (local QP, parallel)
    for p in net_new.prosumers:
        p.admm_x_update(rho)

    # Step 2 — communicate q, z, u to neighbors
    _admm_communicate(net_new)

    # Step 3 — z-update (closed-form projection)
    for p in net_new.prosumers:
        p.admm_z_update()

    # Step 4 — u-update (dual ascent)
    for p in net_new.prosumers:
        p.admm_u_update()

    return net_new


# ------------------------------------------------------------------
# Learned ADMM step  (with GNN perturbation)
# ------------------------------------------------------------------

def learned_ADMM_step(
    net: EnergyNetwork,
    model: GNNUpdate,
    t: int,
    rho: float,
    training: bool,
) -> EnergyNetwork:
    """
    One iteration of learned ADMM with GNN perturbation.

    The GNN perturbation is applied AFTER the full ADMM iteration
    (x-update → communicate → z-update → u-update), so that at the
    moment of perturbation q and z are fully consistent. The perturbed
    q is then seen by the x-update of the next iteration.

    This avoids the coupling constraint violation that arises when
    perturbing q before z-update: in that case q - z != 0 at convergence
    due to the balance constraint in the x-update, causing q_{mn}+q_{nm}>0.

    Structure:
        x-update → communicate → z-update → u-update → GNN(D,G,q,z,u) → perturb

    Training mode (training=True):
        - x-update: differentiable gradient step (admm_x_update_approx)
        - GNN: centralised (full graph via get_pyg_data)

    Inference mode (training=False):
        - x-update: exact cvxpy QP (admm_x_update)
        - GNN: distributed (local subgraph via get_local_pyg_data)

    Parameters
    ----------
    net      : current EnergyNetwork state (not modified)
    model    : GNNUpdate
    t        : current iteration index
    rho      : ADMM penalty parameter
    training : True -> centralised + approx x-update
               False -> distributed + exact x-update

    Returns
    -------
    net_new : updated EnergyNetwork
    """
    net_new = net.clone()

    # ------------------------------------------------------------------
    # Step 2 — communicate
    # ------------------------------------------------------------------
    _admm_communicate(net_new)
    
    # ------------------------------------------------------------------
    # Step 1 — x-update
    # ------------------------------------------------------------------
    if training:
        for p in net_new.prosumers:
            p.admm_x_update_approx(rho)
    else:
        for p in net_new.prosumers:
            p.admm_x_update(rho)

    

    # ------------------------------------------------------------------
    # Step 3 — z-update
    # ------------------------------------------------------------------
    for p in net_new.prosumers:
        p.admm_z_update()

    # ------------------------------------------------------------------
    # Step 4 — u-update
    # ------------------------------------------------------------------
    for p in net_new.prosumers:
        p.admm_u_update()

    # ------------------------------------------------------------------
    # Step 5 — GNN perturbation (after full ADMM iteration)
    # Snapshot pattern: perturbations computed from frozen post-update
    # state, written to net_new — all nodes act simultaneously.
    # Communicate first so q_received, z_received, u_received are
    # up to date for the GNN input features.
    # ------------------------------------------------------------------
    _admm_communicate(net_new)
    snapshot = net_new.clone()

    if training:
        data = snapshot.get_pyg_data(algorithm="admm")
        v_nodes, v_edges = model(data, t)
        # v_nodes: (N, 2)   — [v_D, v_G]
        # v_edges: (|E|, 1) — [v_q]
        # z and u are used as GNN input context but not perturbed

        for n in range(net_new.N):
            v_D = v_nodes[n, 0]
            v_G = v_nodes[n, 1]
            neighbors_n = sorted(net_new.neighbors[n])
            v_q = {m: v_edges[i, 0] for i, m in zip(net_new.incoming_idx[n], neighbors_n)}
            net_new.prosumers[n].apply_perturbation(v_D, v_G, v_q)

    else:
        for n in range(net_new.N):
            data_n = snapshot.prosumers[n].get_local_pyg_data(algorithm="admm")
            v_nodes_n, v_edges_n = model(data_n, t)
            # v_nodes_n: (1+k, 2), v_edges_n: (k, 1)

            v_D = v_nodes_n[0, 0]
            v_G = v_nodes_n[0, 1]
            neighbors_n = sorted(net_new.neighbors[n])
            v_q = {m: v_edges_n[i, 0] for i, m in enumerate(neighbors_n)}
            net_new.prosumers[n].apply_perturbation(v_D, v_G, v_q)

    return net_new