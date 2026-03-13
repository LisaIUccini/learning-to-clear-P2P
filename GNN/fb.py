import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from energy_network import EnergyNetwork
from model import GNNUpdate

# ------------------------------------------------------------------
# Single learned FB step
# ------------------------------------------------------------------

def learned_FB_step(
    net: EnergyNetwork,
    model: GNNUpdate,
    t: int,
    gamma: float,
    training: bool,
) -> EnergyNetwork:
    """
    One step of the distributed learned Forward-Backward algorithm.

    Execution is fully distributed: each node n independently queries
    the GNN using only its own state and messages from its neighbors,
    and receives back only its own perturbations.

    Steps:
        y^t      = x^t - gamma * F(x^t)              (forward, all nodes)
        x_proj   = proj(y^t)                          (backward)
        for each n:
            data_n, idx_n = get_local_pyg_data(n)     (local subgraph of n)
            v_n, v_e_n    = GNNUpdate(data_n, t)      (local perturbation)
            x_n += v_n, Q[idx_n] += v_e_n             (apply locally)
        x^{t+1}  = proj(x_proj + perturbations)       (re-project)

    Parameters
    ----------
    net      : current EnergyNetwork state
    model    : GNNUpdate model
    t        : current iteration index
    gamma    : FB step size
    training : if True use differentiable approx projection,
               else use exact scipy projection

    Returns
    -------
    net_new : EnergyNetwork with updated state x^{t+1}
    """
    # Forward step: y = x - gamma * F(x)
    net_new = net.forward_step(gamma)

    # Backward step: project onto feasible set
    
    if training:
        net_new.project_approx(create_graph=training)
        #net_new.project_differentiable()
    else:
        
        
        x_proj, success = net_new.project_exact()
        if success:
            net_new.unflatten(x_proj)
        
        #net_new.project_differentiable()

    # --- Distributed GNN perturbation (parallel execution) ---
    # All nodes read from the same frozen snapshot of the state,
    # then each node applies its own perturbation independently.
    # This is equivalent to all nodes acting simultaneously —
    # no node sees the perturbation applied by another node.
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

    # Re-project after perturbation
    if training:
        #net_new.project_approx(create_graph=training)
        net_new.project_differentiable()
    else:
        
        
        x_proj, success = net_new.project_exact()
        if success:
            net_new.unflatten(x_proj)
        
        #net_new.project_differentiable()

    return net_new


# ------------------------------------------------------------------
# Pure FB step (no learning)
# ------------------------------------------------------------------

def pure_FB_step(net: EnergyNetwork, gamma: float) -> EnergyNetwork:
    """
    One step of the pure Forward-Backward algorithm (no perturbation).

    Parameters
    ----------
    net   : current EnergyNetwork state
    gamma : FB step size

    Returns
    -------
    net_new : EnergyNetwork with updated state
    """
    net_new = net.forward_step(gamma)
    
    
    x_proj, success = net_new.project_exact()
    if success:
        net_new.unflatten(x_proj)
    
    #net_new.project_differentiable()
    return net_new

