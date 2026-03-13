import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from energy_network import EnergyNetwork
from model import GNNUpdate
from dataset import EnergyDataset
from fb import learned_FB_step, pure_FB_step
import admm



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
            
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            for b in range(x0_batch.shape[0]):
                x0     = x0_batch[b]      # (dim,)
                
                # Initialize network from x0
                net = net_template.clone()
                net.unflatten(x0)
                state, rho = admm.admm_init(net)

                sample_loss = torch.tensor(0.0, device=device)

                z_old = state.z.clone()
                for t in range(T):
                    #net = learned_FB_step(net, model, t, gamma, training=True)
                    """ADMM"""
                    
                    net, state, rho  = admm.learned_admm_step(net, state, rho, model, t, training=True, z_old=z_old)
                    z_old = state.z.clone()
                    z_p_old = state.z.clone()
                    net_pure,_,_ = admm.admm_step(net, state, rho=rho, training=False, z_old=z_p_old)
                    
                    # Primal residual: ||Q - z||^2
                    r_p = torch.sum((net.Q - state.z) ** 2).item()
        
                    # Dual residual: rho^2 * ||z_new - z_old||^2
                    r_d = (rho ** 2) * torch.sum((state.z - z_old) ** 2).item()


                    # Discounted squared distance to GNE
                    x_t = net.flatten()
                    x_pure = net_pure.flatten().detach()
                    weight = alpha ** (T - t)
                    sample_loss = sample_loss + weight*torch.sum(
                        (x_t - x_pure) ** 2
                    ) #+ weight*(r_p + r_d) 

                    """
                    sample_loss = sample_loss + torch.sum(
                        (x_t - pure_FB_step(net, gamma).flatten().detach()) ** 2
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

