import torch
from torch.utils.data import Dataset
import numpy as np
from energy_network import EnergyNetwork
from init_parameters import init_parameters_graph, init_parameters_fixed_topology



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

        # Pre-allocate tensor instead of building a list and stacking
        self.x0_data = torch.empty(num_samples, self.dim, device=device)

        for i in range(num_samples):
            # Clone template (shares parameters), then re-randomise state
            net = net_template.clone()
            net._init_state()

            # Detach: x0 is a fixed starting point, not part of any compute graph
            self.x0_data[i] = net.flatten().detach()

            if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                print(f"  {i+1}/{num_samples} samples generated")

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


class MultiTopologyDataset(Dataset):
    """
    Dataset with diverse graph topologies for robust GNN training.
 
    Each sample is a different problem instance with its own graph
    topology and parameters. This forces the GNN to learn perturbations
    that generalise across graph structures.
 
    Topologies are drawn uniformly from the provided list.
    Each sample stores its own net_template (graph + parameters) and x0.
 
    Parameters
    ----------
    num_samples    : total number of problem instances
    N              : number of prosumers per instance
    topologies     : list of topologies to sample from
    sparsity_range : (min, max) sparsity for "sparse" topology
    device         : torch.device
    seed           : random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int,
        N: int = 5,
        topologies: list = None,
        sparsity_range: tuple = (0.2, 0.8),
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        super().__init__()

        if topologies is None:
            topologies = ["complete", "tree", "sparse"]

        self.num_samples = num_samples
        self.device      = device
        self.N           = N

        rng = np.random.default_rng(seed)
        n_topos = len(topologies)

        self.samples = []  # list of (net_template, x0, rho_admm)

        print(f"Generating {num_samples} multi-topology samples (N={N})...")

        topo_counts = {t: 0 for t in topologies}

        for i in range(num_samples):
            topo = topologies[rng.integers(n_topos)]
            topo_counts[topo] += 1

            sparsity = float(rng.uniform(*sparsity_range)) if topo == "sparse" else 0.5

            sample_seed = int(rng.integers(1_000_000))
            net, _, _, _, rho_admm = init_parameters_graph(
                N=N,
                topology=topo,
                seed=sample_seed,
                sparsity=sparsity,
                device=device,
            )

            # Randomise initial state
            net._init_state()
            x0 = net.flatten().detach().clone()

            self.samples.append((net, x0, rho_admm))

            if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                print(f"  {i+1}/{num_samples} samples generated")

        print(f"Dataset generation complete.")
        print(f"  Topology distribution: {topo_counts}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns
        -------
        dict with:
            'net_template' : EnergyNetwork — graph + parameters for this sample
            'x0'           : torch.Tensor  — initial flattened state
            'rho_admm'     : float         — ADMM penalty for this topology
        """
        net, x0, rho_admm = self.samples[idx]
        return {"net_template": net, "x0": x0, "rho_admm": rho_admm}
    

class FixedTopologyDataset(Dataset):
    """
    Dataset with fixed graph topology but randomized parameters.
 
    Models the realistic deployment scenario: the physical network
    (which prosumers are connected to which) stays the same, but
    operating conditions change across samples:
      - Weather variations  → different G_res (solar/wind output)
      - Seasonal patterns   → different D_star (consumption habits)
      - Market fluctuations → different c (trading costs)
      - Equipment changes   → different G_max (flexibility capacity)
 
    Each sample has its own EnergyNetwork with the SAME neighbors
    dict but DIFFERENT physical parameters, all guaranteed feasible.
 
    Use this when:
      - You have a known, fixed network topology
      - You want the GNN to generalize across operating conditions
        rather than across graph structures
      - You want to simulate realistic day-to-day variability
 
    The GNN's weight-sharing architecture handles this naturally:
    same graph structure means same edge_index, so batching is
    straightforward if all samples have the same N.
 
    Parameters
    ----------
    num_samples : total number of problem instances
    neighbors   : dict {n: list[int]} — fixed graph adjacency
                  (can be built via build_graph or provided manually)
    device      : torch.device
    seed        : random seed for reproducibility
    """
 
    def __init__(
        self,
        num_samples: int,
        neighbors: dict,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        super().__init__()
 
        self.num_samples = num_samples
        self.device      = device
        self.N           = len(neighbors)
        self.neighbors   = neighbors
 
        rng = np.random.default_rng(seed)
 
        self.samples = []  # list of (net, x0, rho_admm)
 
        print(f"Generating {num_samples} fixed-topology samples "
              f"(N={self.N}, |edges|={sum(len(v) for v in neighbors.values()) // 2})...")
 
        for i in range(num_samples):
            sample_seed = int(rng.integers(1_000_000))
 
            net, _, _, _, rho_admm = init_parameters_fixed_topology(
                neighbors=neighbors,
                seed=sample_seed,
                device=device,
            )
 
            # Randomize initial state (D, G, q)
            net._init_state()
            x0 = net.flatten().detach().clone()
 
            self.samples.append((net, x0, rho_admm))
 
            if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                print(f"  {i+1}/{num_samples} samples generated")
 
        print("Dataset generation complete.")
 
    def __len__(self):
        return self.num_samples
 
    def __getitem__(self, idx):
        """
        Returns
        -------
        dict with:
            'net_template' : EnergyNetwork — network with this sample's parameters
            'x0'           : torch.Tensor  — initial flattened state
            'rho_admm'     : float         — ADMM penalty (same for all samples
                                             since topology is fixed)
        """
        net, x0, rho_admm = self.samples[idx]
        return {"net_template": net, "x0": x0, "rho_admm": rho_admm}