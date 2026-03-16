import torch

from energy_network import EnergyNetwork
from model import GNNUpdate


# ------------------------------------------------------------------
# Learned FB step
# ------------------------------------------------------------------

def learned_FB_step(
    net: EnergyNetwork,
    model: GNNUpdate,
    t: int,
    gamma: float,
    training: bool,
) -> EnergyNetwork:
    """
    One step of the learned Forward-Backward algorithm.

    Training mode  (training=True):
        The GNN receives the full global graph (centralised) — all node and
        edge features are available. The differentiable approximate projection
        is used to keep the computation graph intact for backprop.

    Inference mode (training=False):
        Execution is fully distributed. Before querying the GNN, each prosumer
        receives the q decisions of its neighbors via communicate(). Each node
        then builds its own local subgraph (ego + received messages) and queries
        the GNN independently. No global state is ever assembled at a single node.

    In both modes the snapshot pattern is preserved: perturbations are computed
    from a frozen copy of the post-projection state, then written to net_new.
    This makes all nodes act simultaneously — no node sees another node's update.

    Algorithm (both modes):
        y^t       = x^t - gamma * F(x^t)           (forward step)
        x^t_proj  = proj(y^t)                       (backward / projection)
        snapshot  = frozen copy of x^t_proj
        for each n (parallel):
            v_n   = GNN(local or global data of n, t)
            x_n  += v_n                              (write to net_new, not snapshot)
        x^{t+1}   = proj(x^t_proj + perturbations)  (re-projection)

    Parameters
    ----------
    net      : current EnergyNetwork state (not modified)
    model    : GNNUpdate model
    t        : current FB iteration index (int >= 0)
    gamma    : FB step size
    training : True  -> centralised GNN + approx differentiable projection
               False -> distributed GNN  + exact projection

    Returns
    -------
    net_new : EnergyNetwork with updated state x^{t+1}
    """
    # ------------------------------------------------------------------
    # Forward step: y = x - gamma * F(x)
    # ------------------------------------------------------------------
    net_new = net.forward_step(gamma)

    # ------------------------------------------------------------------
    # Backward step: project y onto the feasible set
    # ------------------------------------------------------------------
    
    if training:
        net_new.project_approx(create_graph=True)
    else:
        x_proj, success = net_new.project_exact()
        if success:
            net_new.unflatten(x_proj)

    # ------------------------------------------------------------------
    # Snapshot: freeze post-projection state so all nodes read the same x
    # Perturbations are computed from snapshot, written to net_new.
    # This is equivalent to all nodes acting in parallel.
    # ------------------------------------------------------------------
    snapshot = net_new.clone()

    """
    if training:
        snapshot.communicate()
        # --------------------------------------------------------------
        # Centralised: GNN sees the full graph at once.
        # Perturbations are then sliced back to each prosumer.
        # --------------------------------------------------------------
        data = snapshot.get_pyg_data()
        v_nodes, v_edges = model(data, t)
        # v_nodes : (N, 2)    — v_nodes[n] = (v_D_n, v_G_n)
        # v_edges : (|E|, 1)  — v_edges[i] = perturbation for edge_list[i] = (m, n)
        #                        i.e. for q_{mn}, owned by prosumer n

        for n in range(net_new.N):
            v_D = v_nodes[n, 0]
            v_G = v_nodes[n, 1]

            # Collect edge perturbations for prosumer n's incoming edges.
            # incoming_idx[n] = list of edge_list indices i where edge_list[i] = (m, n).
            # The neighbor order in incoming_idx matches sorted(neighbors[n]),
            # which is also the order of prosumer n's q dict.
            neighbors_n = sorted(net_new.neighbors[n])
            v_q = {
                m: v_edges[i, 0]
                for i, m in zip(net_new.incoming_idx[n], neighbors_n)
            }
            net_new.prosumers[n].apply_perturbation(v_D, v_G, v_q)
        net_new.communicate()

    else:
    """
    # --------------------------------------------------------------
    # Distributed: each prosumer queries the GNN with its local data.
    # Before building local data, run the communication round so that
    # each prosumer's q_received buffer is up to date.
    # --------------------------------------------------------------
    snapshot.communicate()

    for n in range(net_new.N):
        # Node n builds its local subgraph from its own state and the
        # q values received from neighbors — no global data required.
        data_n = snapshot.prosumers[n].get_local_pyg_data()

        # GNN returns perturbations for the local subgraph:
        #   v_nodes : (1+k, 2) — v_nodes[0] = (v_D_n, v_G_n) for ego node
        #   v_edges : (k,   1) — v_edges[i] = perturbation for q[neighbors_n[i]]
        v_nodes_n, v_edges_n = model(data_n, t)

        v_D = v_nodes_n[0, 0]
        v_G = v_nodes_n[0, 1]

        neighbors_n = sorted(net_new.neighbors[n])
        v_q = {
            m: v_edges_n[i, 0]
            for i, m in enumerate(neighbors_n)
        }

        # Write perturbation to net_new (not to snapshot)
        net_new.prosumers[n].apply_perturbation(v_D, v_G, v_q)
    net_new.communicate()

    # ------------------------------------------------------------------
    # Re-project after perturbation
    # ------------------------------------------------------------------
    if training:
        net_new.project_approx()
    else:
        x_proj, success = net_new.project_exact()
        if success:
            net_new.unflatten(x_proj)

    return net_new


# ------------------------------------------------------------------
# Pure FB step  (no learning)
# ------------------------------------------------------------------

def pure_FB_step(net: EnergyNetwork, gamma: float) -> EnergyNetwork:
    """
    One step of the pure Forward-Backward algorithm (no perturbation).

    forward step -> exact projection.
    Used both as the baseline comparator during inference and as the
    fixed-point residual target in the training loss.

    Parameters
    ----------
    net   : current EnergyNetwork state (not modified)
    gamma : FB step size

    Returns
    -------
    net_new : EnergyNetwork with updated state
    """
    net_new = net.forward_step(gamma)
    x_proj, success = net_new.project_exact()
    if success:
        net_new.unflatten(x_proj)
    return net_new