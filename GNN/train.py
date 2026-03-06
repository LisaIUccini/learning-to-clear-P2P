import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from energy_network import EnergyNetwork
from model import GNNUpdate
from dataset import EnergyDataset
from inference import pure_FB_step
import init_parameters


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
    else:
        x_proj, success = net_new.project_exact()
        if success:
            net_new.unflatten(x_proj)


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
        net_new.project_approx()
    else:
        x_proj, success = net_new.project_exact()
        if success:
            net_new.unflatten(x_proj)

    return net_new


# ------------------------------------------------------------------
# Meta-training loop
# ------------------------------------------------------------------

def meta_train(
    model: GNNUpdate,
    dataset: EnergyDataset,
    net_template: EnergyNetwork,
    gamma: float,
    T: int,
    epochs: int,
    lr: float,
    alpha: float,
    batch_size: int,
    device: torch.device,
    save_path: str = None,
) -> GNNUpdate:
    """
    Meta-train the GNNUpdate model by unrolling T steps of the learned
    FB algorithm and minimizing the discounted distance to the GNE.

    Loss for one sample d:

        L_d = sum_{t=1}^{T}  alpha^{T-t} * ||x^t_d - x*_d||^2

    Total loss = mean over batch.

    Parameters
    ----------
    model        : GNNUpdate to train
    dataset      : EnergyDataset with (x0, x_star) pairs
    net_template : EnergyNetwork used to reconstruct state from x0
    gamma        : FB step size
    T            : number of unrolled FB steps
    epochs       : number of training epochs
    lr           : Adam learning rate
    alpha        : discount factor in (0,1) for the loss
    batch_size   : samples per gradient step
    device       : torch.device
    save_path    : if given, save model weights here after training

    Returns
    -------
    model : trained GNNUpdate
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            x0_batch     = batch["x0"].to(device)      # (batch_size, dim)
            x_star_batch = batch["x_star"].to(device)  # (batch_size, dim)

            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for b in range(x0_batch.shape[0]):
                x0     = x0_batch[b]      # (dim,)
                x_star = x_star_batch[b]  # (dim,)

                # Initialize network from x0
                net = net_template.clone()
                net.unflatten(x0)

                sample_loss = torch.tensor(0.0, device=device)

                for t in range(T):
                    net = learned_FB_step(net, model, t, gamma, training=True)

                    # Discounted squared distance to GNE
                    x_t = net.flatten()
                    sample_loss = sample_loss + torch.sum(
                        (x_t - pure_FB_step(net, gamma).flatten()) ** 2
                    )
                    """
                    sample_loss = sample_loss + (alpha ** (T - t)) * torch.sum(
                        (x_t - x_star) ** 2
                    )
                    """

                batch_loss = batch_loss + sample_loss

            # Average over batch
            batch_loss = batch_loss / x0_batch.shape[0]
            batch_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += batch_loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch+1:3d}/{epochs}]  loss: {avg_loss:.6f}")

    # Save model if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model, loss_history

