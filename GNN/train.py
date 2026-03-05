import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from energy_network import EnergyNetwork
from model import GNNUpdate
from dataset import EnergyDataset
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
    One step of the learned Forward-Backward algorithm:

        y^t      = x^t - gamma * F(x^t)          (forward)
        x_proj   = proj(y^t)                      (backward, approx if training)
        data     = get_pyg_data(x_proj)           (build PyG input)
        v_n, v_e = GNNUpdate(data, t)             (GNN perturbation)
        x^{t+1}  = proj(x_proj + v_n + v_e)      (apply + re-project)

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
        net_new.project_approx()
    else:
        net_new.project_exact()

    # Build PyG Data object for GNN input
    data = net_new.get_pyg_data()

    # GNN produces perturbations v_nodes (N,2) and v_edges (|E|,1)
    v_nodes, v_edges = model(data, t)

    # Apply perturbation to state
    net_new.apply_perturbation(v_nodes, v_edges)

    # Re-project after perturbation (state may have left feasible set)
    if training:
        net_new.project_approx()
    else:
        net_new.project_exact()

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
                    sample_loss = sample_loss + (alpha ** (T - t)) * torch.sum(
                        (x_t - x_star) ** 2
                    )

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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    # --- Device and seeds ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Problem parameters --
    (N, gamma, max_iter, tol,
     neighbors, D_min, D_max, G_min, G_max, G_res,
     D_star, a_tilde, a, b, c, k) = init_parameters.initN3()

    # --- Build network template ---
    net_template = EnergyNetwork(
        N, neighbors, D_min, D_max, G_min, G_max, G_res,
        a_tilde, a, b, D_star, c, k, device
    )
    print(net_template)

    # --- Training hyperparameters ---
    num_samples  = 50
    batch_size   = 1
    T            = 15      # unrolling steps
    epochs       = 50
    rho          = 0.5    # summability decay
    lr           = 1e-3
    alpha        = 0.95    # discount factor in loss
    d_h          = 32      # GNN hidden dimension

    gamma = 0.1  # FB step size (fixed, not learned)
    max_iter = 100
    tol = 1e-6
    

    # --- Dataset (GNE computed offline) ---
    dataset = EnergyDataset(
        num_samples  = num_samples,
        net_template = net_template,
        gamma        = gamma,
        gne_max_iter = max_iter,
        gne_tol      = tol,
        device       = device,
    )

    # --- Model ---
    model = GNNUpdate(d_h=d_h, rho=rho).to(device)
    print(f"GNNUpdate parameters: {sum(p.numel() for p in model.parameters())}")

    # --- Save path ---
    save_path = (
        f"checkpoints/N{N}_T{T}_e{epochs}_s{num_samples}_learned_update"
        f"_rho{rho}_lr{lr}.pt"
    )

    load = False

    if load and os.path.exists(save_path):
        print(f"Loading model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        model, loss_history = meta_train(
            model        = model,
            dataset      = dataset,
            net_template = net_template,
            gamma        = gamma,
            T            = T,
            epochs       = epochs,
            lr           = lr,
            alpha        = alpha,
            batch_size   = batch_size,
            device       = device,
            save_path    = save_path,
        )


if __name__ == "__main__":
    main()