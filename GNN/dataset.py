import torch
from torch.utils.data import Dataset

from energy_network import EnergyNetwork


class EnergyDataset(Dataset):
    """
    Dataset of initial states for meta-training the GNNUpdate model.

    Each sample is a dict with:
        'x0' : torch.Tensor (dim,) — flattened initial state, randomised

    The graph structure and parameters are fixed (shared from net_template).
    Only the decision variables D, G, q are re-randomised for each sample.

    During training, each sample provides a starting point x0. The network
    is reconstructed via net_template.clone() + unflatten(x0), and the loss
    is the fixed-point residual accumulated over T unrolled FB steps — no
    ground-truth GNE is needed.

    Parameters
    ----------
    num_samples  : number of problem instances to generate
    net_template : EnergyNetwork used as template — graph and parameters
                   are shared (read-only); state is re-randomised per sample
    device       : torch.device
    """

    def __init__(
        self,
        num_samples: int,
        net_template: EnergyNetwork,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.num_samples = num_samples
        self.device      = device
        self.dim         = net_template.dim

        print(f"Generating {num_samples} dataset samples...")

        x0_list = []
        for i in range(num_samples):
            # Clone template (shares parameters), then re-randomise state
            net = net_template.clone()
            net._init_state()

            # Detach: x0 is a fixed starting point, not part of any compute graph
            x0 = net.flatten().detach().clone()
            x0_list.append(x0)

            if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                print(f"  {i+1}/{num_samples} samples generated")

        # Stack into a single tensor for efficient batch indexing
        self.x0_data = torch.stack(x0_list).to(device)  # (num_samples, dim)

        print("Dataset generation complete.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns
        -------
        dict with:
            'x0' : torch.Tensor (dim,) — initial flattened state for this sample
        """
        return {"x0": self.x0_data[idx]}