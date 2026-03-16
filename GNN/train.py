import os
import torch
from torch.utils.data import DataLoader

from energy_network import EnergyNetwork
from model import GNNUpdate
from dataset import EnergyDataset
from fb   import learned_FB_step, pure_FB_step
from admm import learned_ADMM_step, pure_ADMM_step


# ------------------------------------------------------------------
# Meta-training loop
# ------------------------------------------------------------------

def meta_train(
    model: GNNUpdate,
    dataset: EnergyDataset,
    net_template: EnergyNetwork,
    T: int,
    epochs: int,
    lr: float,
    alpha: float,
    batch_size: int,
    device: torch.device,
    save_path: str = None,
    algorithm: str = "fb",
    gamma: float = None,
    rho: float = None,
) -> tuple:
    """
    Meta-train the GNNUpdate model by unrolling T steps of the chosen algorithm
    and minimising the residual at each step.

    The GNN architecture and training procedure are identical for both algorithms.
    Only the unrolled step and the loss definition change.

    algorithm="fb"
        Unrolled step : learned_FB_step (centralised, training=True)
        Loss per step : || x^t - T(x^t) ||^2
                        where T(x) = proj(x - gamma * F(x))
        Requires: gamma

    algorithm="admm"
        Unrolled step : learned_ADMM_step (centralised, training=True)
        Loss per step : sum_{(m,n)} (q_{mn} - z_{mn})^2   (primal residual)
                        measures consensus violation at each iterate
        z and u are reset to zero at the start of each sample
        Requires: rho

    In both cases the GNN receives the full graph (centralised training),
    and the model weights are updated via Adam with gradient clipping.

    Parameters
    ----------
    model        : GNNUpdate to train
    dataset      : EnergyDataset (provides x0 tensors)
    net_template : EnergyNetwork for graph structure (cloned per sample)
    T            : number of unrolled steps per sample
    epochs       : number of training epochs
    lr           : Adam learning rate
    alpha        : discount factor (kept for API compatibility; loss is undiscounted)
    batch_size   : number of samples per gradient update
    device       : torch.device
    save_path    : if given, save model weights here after training
    algorithm    : "fb" or "admm"
    gamma        : FB step size (required if algorithm="fb")
    rho          : ADMM penalty parameter (required if algorithm="admm")

    Returns
    -------
    (model, loss_history)
    """
    if algorithm not in ("fb", "admm"):
        raise ValueError(f"algorithm must be 'fb' or 'admm', got '{algorithm}'")
    if algorithm == "fb" and gamma is None:
        raise ValueError("gamma is required when algorithm='fb'")
    if algorithm == "admm" and rho is None:
        raise ValueError("rho is required when algorithm='admm'")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            x0_batch = batch["x0"].to(device)

            optimizer.zero_grad()
            batch_loss = torch.zeros(1, device=device)

            for b in range(x0_batch.shape[0]):
                x0 = x0_batch[b]

                net = net_template.clone()
                net.unflatten(x0)

                # Reset ADMM duals for this sample
                if algorithm == "admm":
                    net.admm_reset()

                sample_loss = torch.zeros(1, device=device)
                

                for t in range(T):

                    if algorithm == "fb":
                        # Learned FB step (centralised, differentiable)
                        net = learned_FB_step(net, model, t, gamma, training=True)

                        # Fixed-point residual: || x^t - T(x^t) ||^2
                        x_t  = net.flatten()
                        Tx_t = pure_FB_step(net, gamma).flatten()
                        sample_loss = sample_loss + torch.sum((x_t - Tx_t) ** 2)

                    else:  # admm
                        
                        # Save z before the step for dual residual computation
                        z_prev = {
                            n: {m: net.prosumers[n].z[m].clone().detach()
                                for m in net.prosumers[n].neighbors}
                            for n in range(net.N)
                        }

                        """
                        # Learned ADMM step (centralised, differentiable)
                        net = learned_ADMM_step(net, model, t, rho, training=True)

                        # Primal residual: r^t = sum_{(m,n)} (q_{mn} - z_{mn})^2
                        # Measures consensus violation (reciprocity)
                        r = sum(
                            (p.q[m] - p.z[m]) ** 2
                            for p in net.prosumers
                            for m in p.neighbors
                        )

                        # Dual residual: s^t = rho^2 * sum_{(m,n)} (z^t - z^{t-1})^2
                        # Measures how much z is still moving (dual feasibility)
                        s = rho ** 2 * sum(
                            (net.prosumers[n].z[m] - z_prev[n][m]) ** 2
                            for n in range(net.N)
                            for m in net.prosumers[n].neighbors
                        )

                        sample_loss = sample_loss + (r + s)
                        """
                        
                        # Learned ADMM step (centralised, differentiable)
                        net = learned_ADMM_step(net, model, t, rho, training=True)

                        # Fixed-point residual: || x^t - T(x^t) ||^2
                        x_t  = net.flatten()
                        Tx_t = pure_ADMM_step(net, rho).flatten().detach()
                        sample_loss = sample_loss + torch.sum((x_t - Tx_t) ** 2)
                        

                batch_loss = batch_loss + sample_loss

            batch_loss = batch_loss / x0_batch.shape[0]
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        #scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{epochs}]  loss: {avg_loss:.6f}  lr: {current_lr:.2e}")
        
    if save_path is not None:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True
        )
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model, loss_history





"""
if algorithm == "fb":
    # Learned FB step (centralised, differentiable)
    net = learned_FB_step(net, model, t, gamma, training=True)

    # Fixed-point residual: || x^t - T(x^t) ||^2
    x_t  = net.flatten()
    Tx_t = pure_FB_step(net, gamma).flatten()
    sample_loss = sample_loss + torch.sum((x_t - Tx_t) ** 2)

else:  # admm
    # Learned ADMM step (centralised, differentiable)
    net = learned_ADMM_step(net, model, t, rho, training=True)

    # Fixed-point residual: || x^t - T(x^t) ||^2
    x_t  = net.flatten()
    Tx_t = pure_ADMM_step(net, rho).flatten()
    sample_loss = sample_loss + torch.sum((x_t - Tx_t) ** 2)

    """
                        