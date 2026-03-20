import torch

from energy_network import EnergyNetwork
from model import GNNUpdate


# ------------------------------------------------------------------
# ADMM communication round
# ------------------------------------------------------------------

def _admm_communicate(net: EnergyNetwork, training: bool = False):
    """
    Communication round for ADMM: each prosumer sends q, z, u to each neighbor.

    After this call, for every directed edge (m -> n):
        prosumer[n].q_received[m] = prosumer[m].q[n]
        prosumer[n].z_received[m] = prosumer[m].z[n]
        prosumer[n].u_received[m] = prosumer[m].u[n]

    q_received and u_received are needed for the z-update.
    z_received is kept for monitoring / convergence checking.

    If training=True, preserves gradient flow between nodes.
    """
    prosumers = net.prosumers
    for (m, n) in net.edge_list:
        prosumers[m].send_all_to(prosumers[n], training=training)


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
        p.admm_x_update_diff(rho)

    # Step 2 — communicate q, z, u to neighbors
    _admm_communicate(net_new)

    # Steps 3+4 — z-update + u-update (both local, no communication needed between them)
    for p in net_new.prosumers:
        p.admm_z_update()
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
 
    New order (correct):
        x-update → communicate → z-update → u-update
        → communicate → GNN perturba (D,G,q,z,u) → communicate
 
    The GNN sees the full post-ADMM-step state and perturbs all
    five variable types. Perturbations on z and u are applied AFTER
    the z-update and u-update, so they are not overwritten.
 
    Parameters
    ----------
    net      : current EnergyNetwork state (not modified)
    model    : GNNUpdate
    t        : current iteration index
    rho      : ADMM penalty parameter
    training : True -> differentiable x-update (SCS)
               False -> differentiable x-update (SCS)
 
    Returns
    -------
    net_new : updated EnergyNetwork
    """
    net_new = net.clone()
 
    # ------------------------------------------------------------------
    # Step 1 — x-update (local QP, parallel)
    # ------------------------------------------------------------------
    for p in net_new.prosumers:
        p.admm_x_update_diff(rho)
 
    # ------------------------------------------------------------------
    # Step 2 — communicate q, z, u to neighbors (for z-update)
    # ------------------------------------------------------------------
    _admm_communicate(net_new, training=training)
 
    # ------------------------------------------------------------------
    # Steps 3+4 — z-update + u-update (both local, merged loop)
    # ------------------------------------------------------------------
    for p in net_new.prosumers:
        p.admm_z_update()
        p.admm_u_update()
 
    # ------------------------------------------------------------------
    # Step 5 — communicate updated state (for GNN input)
    # ------------------------------------------------------------------
    #_admm_communicate(net_new, training=training)
 
    # ------------------------------------------------------------------
    # Step 6 — GNN perturbation (snapshot pattern)
    # GNN sees the full post-step state and perturbs all variables
    # ------------------------------------------------------------------
    snapshot = net_new.clone()
    neighbors = net_new.neighbors
    snapshot_prosumers = snapshot.prosumers
    net_prosumers = net_new.prosumers
 
    for n in range(net_new.N):
        data_n = snapshot_prosumers[n].get_local_pyg_data(algorithm="admm")
        v_nodes_n, v_edges_n = model(data_n, t)
 
        v_D = v_nodes_n[0, 0]
        v_G = v_nodes_n[0, 1]
        neighbors_n = neighbors[n]
        # Build all 3 dicts in a single pass over neighbors
        v_q, v_z, v_u = {}, {}, {}
        for i, m in enumerate(neighbors_n):
            v_q[m] = v_edges_n[i, 0]
            v_z[m] = v_edges_n[i, 1]
            v_u[m] = v_edges_n[i, 2]
        net_prosumers[n].apply_perturbation(v_D, v_G, v_q, v_z, v_u)
 
    # ------------------------------------------------------------------
    # Step 7 — communicate perturbed values (ready for next iteration)
    # ------------------------------------------------------------------
    #_admm_communicate(net_new, training=training)
 
    return net_new