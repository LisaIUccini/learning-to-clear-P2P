import os
import torch
from torch.utils.data import DataLoader

from energy_network import EnergyNetwork
from model import GNNUpdate
from dataset import EnergyDataset
from fb import learned_FB_step, pure_FB_step


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
) -> tuple:
    """
    Meta-train the GNNUpdate model by unrolling T steps of the learned
    FB algorithm and minimising the fixed-point residual at each step.

    Training is centralised: the GNN receives the full graph state at
    every unrolled iteration (see learned_FB_step with training=True).

    Loss for one sample:

        L = sum_{t=1}^{T}  || x^t - T(x^t) ||^2

    where T(x) = proj(x - gamma * F(x)) is the unperturbed FB operator.
    This drives each iterate toward the v-GNE without requiring x* explicitly.

    Parameters
    ----------
    model        : GNNUpdate to train
    dataset      : EnergyDataset (provides x0 tensors)
    net_template : EnergyNetwork used to reconstruct network state from x0.
                   Its graph structure and parameters are reused for every sample;
                   only D, G, q are overwritten via unflatten(x0).
    gamma        : FB step size
    T            : number of unrolled FB steps per sample
    epochs       : number of training epochs
    lr           : Adam learning rate
    alpha        : discount factor (kept for API compatibility; loss is undiscounted)
    batch_size   : number of samples per gradient update
    device       : torch.device
    save_path    : if given, save model weights here after training

    Returns
    -------
    (model, loss_history)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            x0_batch = batch["x0"].to(device)   # (batch_size, dim)

            optimizer.zero_grad()
            batch_loss = torch.zeros(1, device=device)

            for b in range(x0_batch.shape[0]):
                x0 = x0_batch[b]   # (dim,)

                # Reconstruct network state from x0.
                # net_template provides the fixed graph and parameters;
                # unflatten overwrites D, G, q for this sample.
                net = net_template.clone()
                net.unflatten(x0)

                sample_loss = torch.zeros(1, device=device)

                for t in range(T):
                    # One learned FB step (centralised, differentiable projection)
                    net = learned_FB_step(net, model, t, gamma, training=True)

                    # Fixed-point residual: || x^t - T(x^t) ||^2
                    # T(x^t) = pure FB step from the current iterate (no grad needed)
                    x_t    = net.flatten()
                    Tx_t   = pure_FB_step(net, gamma).flatten()
                    sample_loss = sample_loss + torch.sum((x_t - Tx_t) ** 2)

                batch_loss = batch_loss + sample_loss

            # Average over batch
            batch_loss = batch_loss / x0_batch.shape[0]
            batch_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += batch_loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1:3d}/{epochs}]  loss: {avg_loss:.6f}")

    if save_path is not None:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True
        )
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model, loss_history