import torch
from torch.utils.data import Dataset

from energy_network import EnergyNetwork, compute_gne



class EnergyDataset(Dataset):
    """
    Dataset of EnergyNetwork instances for meta-training the GNNUpdate model.

    Each sample is a dict with:
        'x0'     : torch.Tensor (dim,)  — initial flattened state (random)
        'x_star' : torch.Tensor (dim,)  — GNE computed offline via pure FB

    The GNE computation happens once at construction time and is cached.
    During training, each sample provides a starting point x0 and the
    corresponding target x_star for the loss function.

    Parameters
    ----------
    num_samples  : number of problem instances to generate
    net_template : EnergyNetwork instance used as template (parameters are shared,
                   state is re-randomized for each sample)
    gamma        : FB step size used for GNE computation
    gne_max_iter : max iterations for offline GNE solver
    gne_tol      : convergence tolerance for GNE solver
    device       : torch.device
    """

    def __init__(
        self,
        num_samples: int,
        net_template: EnergyNetwork,
        gamma: float,
        gne_max_iter: int = 200,
        gne_tol: float = 1e-6,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.num_samples = num_samples
        self.device      = device
        self.dim         = net_template.dim

        self.x0_list     = []
        self.x_star_list = []

        print(f"Generating {num_samples} dataset samples (GNE computed offline)...")

        for i in range(num_samples):
            # Re-randomize state (parameters are fixed, only D, G, Q change)
            net = net_template.clone()
            net.D, net.G, net.Q = net._init_state()

            x0 = net.flatten().detach().clone()

            print(f"  Sample {i+1}/{num_samples}: computing GNE...")
            x_star = compute_gne(net, gamma, gne_max_iter, gne_tol).flatten().detach()

            self.x0_list.append(x0)
            self.x_star_list.append(x_star)

        # Stack into tensors for efficient batch access
        self.x0_data     = torch.stack(self.x0_list).to(device)      # (num_samples, dim)
        self.x_star_data = torch.stack(self.x_star_list).to(device)  # (num_samples, dim)

        print("Dataset generation complete.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns
        -------
        dict with:
            'x0'     : torch.Tensor (dim,)
            'x_star' : torch.Tensor (dim,)
        """
        return {
            "x0":     self.x0_data[idx],
            "x_star": self.x_star_data[idx],
        }