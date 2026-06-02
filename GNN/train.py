import os
import torch
from torch.utils.data import DataLoader

from energy_network import EnergyNetwork
from model          import GNNUpdate
from dataset        import EnergyDataset
from admm           import learned_ADMM_step, pure_ADMM_step


# ------------------------------------------------------------------
# Per-step losses
# ------------------------------------------------------------------

def _meta_loss(model: GNNUpdate, t: int, rho: float, net: EnergyNetwork):
    """
    One unrolled learned ADMM step + fixed-point residual.

        L_t = || sigma^t - T_ADMM(sigma^t) ||^2

    The step is centralised (training=True) so gradients flow.
    Returns (loss, advanced_net).
    """
    net    = learned_ADMM_step(net, model, t, rho, training=True)
    sigma  = net.flatten_admm()
    Tsigma = pure_ADMM_step(net, rho).flatten_admm()
    return torch.sum((sigma - Tsigma) ** 2), net


def _meta_loss_baseline(rho: float, net_b: EnergyNetwork):
    """
    Same fixed-point residual, but the advancing step is pure ADMM
    (no GNN). Used to compute the no-learning baseline.
    """
    net_b  = pure_ADMM_step(net_b, rho)
    sigma  = net_b.flatten_admm()
    Tsigma = pure_ADMM_step(net_b, rho).flatten_admm()
    return torch.sum((sigma - Tsigma) ** 2).item(), net_b


def _ensure_dir(save_path: str):
    dirname = os.path.dirname(save_path)
    os.makedirs(dirname if dirname else ".", exist_ok=True)


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
    rho: float,
    save_path: str = None,
) -> tuple:
    """
    Meta-train GNNUpdate by unrolling T ADMM steps and minimising the
    fixed-point residual at each step:

        L(theta) = (1/|D|) * sum_d sum_t || sigma^t_d - T_ADMM(sigma^t_d) ||^2

    The unrolled step is `learned_ADMM_step` (centralised, differentiable
    via cvxpylayers for the x-update and closed-form for z/u). z and u are
    reset to zero at the start of each sample.

    A no-learning baseline is computed with `pure_ADMM_step` for reference:
    the training loss should drop below it.

    Parameters
    ----------
    model        : GNNUpdate to train
    dataset      : EnergyDataset providing x0 tensors
    net_template : EnergyNetwork for graph structure (cloned per sample)
    T            : unrolled steps per sample
    epochs       : number of training epochs
    lr           : Adam learning rate
    alpha        : kept for API compatibility (loss is undiscounted)
    batch_size   : samples per gradient update
    device       : torch.device
    rho          : ADMM penalty parameter
    save_path    : if given, save model weights after training

    Returns
    -------
    (model, loss_history)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    loss_history = []

    # ----- Baseline (pure ADMM, no GNN) -------------------------------
    with torch.no_grad():
        baseline_total = 0.0
        n_baseline     = 0
        for batch in dataloader:
            x0_batch = batch["x0"].to(device)
            for b in range(x0_batch.shape[0]):
                net_b = net_template.clone()
                net_b.unflatten(x0_batch[b])
                net_b.admm_reset()

                sample_baseline = 0.0
                for t in range(T):
                    sl, net_b        = _meta_loss_baseline(rho, net_b)
                    sample_baseline += sl

                baseline_total += sample_baseline
                n_baseline     += 1

        baseline_avg = baseline_total / max(n_baseline, 1)
        print(f"\nBaseline loss (pure ADMM, no GNN): {baseline_avg:.6f}")
        print(f"Training loss should go BELOW this value.\n")

    # ----- Training loop ----------------------------------------------
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for batch in dataloader:
            x0_batch    = batch["x0"].to(device)
            actual_size = x0_batch.shape[0]

            optimizer.zero_grad()
            batch_loss = torch.zeros(1, device=device)

            for b in range(actual_size):
                net = net_template.clone()
                net.unflatten(x0_batch[b])
                net.admm_reset()

                sample_loss = torch.zeros(1, device=device)
                for t in range(T):
                    sl, net      = _meta_loss(model, t, rho, net)
                    sample_loss += sl

                batch_loss = batch_loss + sample_loss

            batch_loss = batch_loss / actual_size
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1:3d}/{epochs}]  loss: {avg_loss:.6f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

    if save_path is not None:
        _ensure_dir(save_path)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model, loss_history