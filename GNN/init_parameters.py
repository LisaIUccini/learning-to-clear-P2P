import torch
import numpy as np
import networkx as nx

from prosumer import Prosumer
from energy_network import EnergyNetwork

tol = 1e-6
max_iter = 60
# ------------------------------------------------------------------
# Graph builder  (invariato)
# ------------------------------------------------------------------

def build_graph(N: int, topology: str, seed: int = None,
                sparsity: float = 0.5) -> dict:
    """
    Build a graph with N nodes and return neighbors dict.

    Parameters
    ----------
    N        : number of nodes
    topology : one of "complete", "star", "tree", "cycle", "path", "sparse"
    seed     : random seed
    sparsity : fraction of edges removed from complete graph [0,1).
               Only used when topology="sparse".

    Returns
    -------
    neighbors : dict {n: list of neighbor indices}
    """
    if topology == "complete":
        G = nx.complete_graph(N)

    elif topology == "star":
        G = nx.star_graph(N - 1)
        rng = np.random.default_rng(seed)
        non_root = list(range(1, N))
        n_extra  = max(0, N // 3)
        attempts = added = 0
        while added < n_extra and attempts < 1000:
            u, v = rng.choice(non_root, size=2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1
            attempts += 1

    elif topology == "tree":
        rng  = np.random.default_rng(seed)
        G    = nx.random_labeled_tree(N, seed=int(rng.integers(1e6)))
        degrees = dict(G.degree())
        root    = max(degrees, key=degrees.get)
        mapping = {root: 0}
        for new_idx, old_idx in enumerate(
            (n for n in G.nodes if n != root), start=1
        ):
            mapping[old_idx] = new_idx
        G = nx.relabel_nodes(G, mapping)

    elif topology == "cycle":
        G = nx.cycle_graph(N)

    elif topology == "path":
        G = nx.path_graph(N)

    elif topology == "sparse":
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
        max_edges = N * (N - 1) // 2
        n_edges   = max(N - 1, round((1.0 - sparsity) * max_edges))
        rng = np.random.default_rng(seed)
        while True:
            G = nx.gnm_random_graph(N, m=n_edges, seed=int(rng.integers(1e6)))
            if nx.is_connected(G):
                break

    else:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            f"Choose from: complete, star, tree, cycle, path, sparse."
        )

    return {n: sorted(list(G.neighbors(n))) for n in range(N)}


# ------------------------------------------------------------------
# Helper: build c and k dicts from per-edge functions
# ------------------------------------------------------------------

def _build_c_k(neighbors: dict, c_fn, k_fn) -> tuple:
    """
    Build c and k dicts given per-edge scalar functions.

    Returns
    -------
    c : dict {(m, n): scalar tensor}  — trading cost c_{nm}
    k : dict {(m, n): scalar tensor}  — capacity k_{mn}
    """
    c, k = {}, {}
    for n in neighbors:
        for m in neighbors[n]:
            c[(m, n)] = torch.tensor(c_fn(m, n), dtype=torch.float32)
            k[(m, n)] = torch.tensor(k_fn(m, n), dtype=torch.float32)
    return c, k


# ------------------------------------------------------------------
# Core builder: raw arrays -> list[Prosumer] -> EnergyNetwork
# ------------------------------------------------------------------

def build_network(
    N: int,
    neighbors: dict,
    D_min: torch.Tensor,
    D_max: torch.Tensor,
    G_min: torch.Tensor,
    G_max: torch.Tensor,
    G_res: torch.Tensor,
    a_tilde: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    D_star: torch.Tensor,
    c: dict,
    k: dict,
    device: torch.device,
) -> EnergyNetwork:
    """
    Construct an EnergyNetwork from raw parameter tensors.

    Each prosumer is built individually with its own slice of the
    parameter tensors and its own per-neighbor c/k dicts.

    Parameters
    ----------
    All tensors are shape (N,) for node-level parameters.
    c, k : dict {(m, n): scalar tensor}

    Returns
    -------
    EnergyNetwork
    """
    prosumers = []
    for n in range(N):
        # Per-neighbor c and k for node n: {m: tensor}
        c_n = {m: c[(m, n)] for m in neighbors[n]}
        k_n = {m: k[(m, n)] for m in neighbors[n]}

        p = Prosumer(
            node_id = n,
            neighbors = neighbors[n],
            D_min   = D_min[n],
            D_max   = D_max[n],
            G_min   = G_min[n],
            G_max   = G_max[n],
            G_res   = G_res[n],
            a_tilde = a_tilde[n],
            a       = a[n],
            b       = b[n],
            D_star  = D_star[n],
            c       = c_n,
            k       = k_n,
            device  = device,
        )
        prosumers.append(p)

    return EnergyNetwork(prosumers, neighbors, device)


# ------------------------------------------------------------------
# Solver hyperparameters  (shared across all init functions)
# ------------------------------------------------------------------

def _solver_params(max_iter: int = 100, tol: float = tol, gamma: float = 0.1):
    return {"gamma": gamma, "max_iter": max_iter, "tol": tol}


# ------------------------------------------------------------------
# Fixed instances
# ------------------------------------------------------------------

def initN3(device: torch.device = torch.device("cpu")) -> tuple:
    """
    Small test network — N=3, fully connected.

    Returns
    -------
    (net, gamma, max_iter, tol)  where net is an EnergyNetwork.
    """
    N         = 3
    neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    D_min   = torch.tensor([0.0,  0.0,  0.0],  dtype=torch.float32)
    D_max   = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor([10.0, 0.0, 0.0],   dtype=torch.float32)
    G_res   = torch.tensor([0.0,  3.0, 5.0],   dtype=torch.float32)
    a_tilde = torch.tensor([5.0,  15.0, 10.0], dtype=torch.float32)
    D_star  = torch.tensor([6.0,  3.0,  3.0],  dtype=torch.float32)
    a       = torch.tensor([4.0,  4.0,  4.0],  dtype=torch.float32)
    b       = torch.zeros(N, dtype=torch.float32)

    c, k = _build_c_k(
        neighbors,
        c_fn=lambda m, n: 0.1,
        k_fn=lambda m, n: 10.0 if m == 0 else 5.0,
    )

    net = build_network(N, neighbors, D_min, D_max, G_min, G_max, G_res,
                        a_tilde, a, b, D_star, c, k, device)
    return net, 0.1, 100, tol, 3.0  # rho_admm=3 for N=3 complete


def initN10(device: torch.device = torch.device("cpu")) -> tuple:
    """
    Medium network — N=10.
    Node 0 is the root (grid interface), connected to all others.

    Returns
    -------
    (net, gamma, max_iter, tol)
    """
    N = 10
    neighbors = {
        0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        1: [0, 2, 3], 2: [0, 1, 4], 3: [0, 1, 5],
        4: [0, 2, 6], 5: [0, 3, 7], 6: [0, 4, 8],
        7: [0, 5, 9], 8: [0, 6, 9], 9: [0, 7, 8],
    }

    D_min   = torch.zeros(N, dtype=torch.float32)
    D_max   = torch.full((N,), 10.0, dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor([10., 0., 6., 0., 0., 0., 8., 0., 0., 0.], dtype=torch.float32)
    G_res   = torch.tensor([0., 2., 3., 1., .5, 2.5, 1.5, 0., 3., 1.], dtype=torch.float32)
    D_star  = torch.tensor([6., 4., 5., 6., 5., 4., 5., 6., 4., 5.],   dtype=torch.float32)
    a_tilde = torch.tensor([10.,12., 8.,10., 9.,11.,10.,12., 9.,10.],   dtype=torch.float32)
    a       = torch.tensor([3., 2., 4., 3., 2.5,3.5, 4., 3., 2., 3.],  dtype=torch.float32)
    b       = torch.zeros(N, dtype=torch.float32)

    c, k = _build_c_k(
        neighbors,
        c_fn=lambda m, n: 0.1,
        k_fn=lambda m, n: 10.0 if (m == 0 or n == 0) else 5.0,
    )

    net = build_network(N, neighbors, D_min, D_max, G_min, G_max, G_res,
                        a_tilde, a, b, D_star, c, k, device)
    return net, 0.1, 100, tol, 5.0  # rho_admm=5 for N=10 star-like


def initN30(device: torch.device = torch.device("cpu")) -> tuple:
    """
    Large radial tree — N=30. Node 0 is the root.

    Returns
    -------
    (net, gamma, max_iter, tol)
    """
    N = 30
    neighbors = {
        0:  [1, 2],       1:  [0, 3, 4],    2:  [0, 5, 6],
        3:  [1, 7, 8],    4:  [1, 9],        5:  [2, 10],
        6:  [2, 11, 12],  7:  [3, 13, 14],   8:  [3, 15],
        9:  [4, 16],      10: [5, 16],        11: [6, 17],
        12: [6],           13: [7, 18],        14: [7, 19],
        15: [8],           16: [9, 10, 20],    17: [11, 21],
        18: [13, 22],      19: [14],           20: [16, 23, 24],
        21: [17, 25],      22: [18],           23: [20, 26],
        24: [20],          25: [21, 27],        26: [23],
        27: [25, 28],      28: [27, 29],        29: [28],
    }

    D_min = torch.zeros(N, dtype=torch.float32)
    D_max = torch.tensor([
         0.,20.,20.,15.,10.,10.,15., 8., 8., 6.,
         6., 8., 5., 6., 6., 5., 7., 6., 5., 5.,
         7., 5., 4., 5., 4., 4., 4., 3., 3., 3.,
    ], dtype=torch.float32)
    G_min = torch.zeros(N, dtype=torch.float32)
    G_max = torch.tensor([
        50., 8., 8., 5., 0., 0., 5., 3., 0., 3.,
         0., 3., 0., 2., 0., 0., 2., 0., 2., 0.,
         2., 0., 0., 2., 0., 0., 0., 0., 0., 0.,
    ], dtype=torch.float32)
    G_res = torch.tensor([
        0., 2., 1.5, 3., 4., 3.5, 2., 2.5, 4., 3.,
        3.5, 2., 4.5, 3., 3.5, 4., 2.5, 3., 2., 3.5,
        2.5, 3., 4., 2., 4., 3., 4., 3., 3.5, 4.,
    ], dtype=torch.float32)
    a_tilde = torch.tensor([
         0.,10.,10., 8.,12.,12., 8.,10.,10.,12.,
        12.,10.,15.,12.,12.,15.,10.,12.,15.,15.,
        10.,12.,15.,12.,15.,12.,15.,12.,12.,15.,
    ], dtype=torch.float32)
    D_star = torch.tensor([
         0.,14.,14.,10., 7., 7.,10., 5.5, 5.5, 4.,
         4., 5.5, 3.5, 4., 4., 3.5, 5., 4., 3.5, 3.5,
         5., 3.5, 3., 3.5, 3., 3., 3., 2., 2., 2.,
    ], dtype=torch.float32)
    a = torch.tensor([
        1., 4., 4., 5., 4., 4., 5., 6., 4., 5.,
        4., 6., 4., 6., 4., 4., 5., 4., 6., 4.,
        5., 4., 4., 5., 4., 4., 4., 4., 4., 4.,
    ], dtype=torch.float32)
    b = torch.zeros(N, dtype=torch.float32)

    def c_fn(m, n):
        if m == 0 or n == 0:   return 0.5
        elif m <= 2 or n <= 2: return 0.2
        else:                  return 0.1

    def k_fn(m, n):
        if m == 0 or n == 0:   return 20.0
        elif m <= 2 or n <= 2: return 10.0
        else:                  return 5.0

    c, k = _build_c_k(neighbors, c_fn, k_fn)

    net = build_network(N, neighbors, D_min, D_max, G_min, G_max, G_res,
                        a_tilde, a, b, D_star, c, k, device)
    return net, 0.1, 300, tol, 12.0  # rho_admm=12 for N=30 tree


# ------------------------------------------------------------------
# ADMM penalty heuristic
# ------------------------------------------------------------------

def compute_rho_admm(neighbors: dict, base: float = 2.0, rho_min: float = 2.0) -> float:
    """
    Compute a reasonable ADMM penalty parameter based on graph structure.

    Heuristic: rho = base * sqrt(avg_degree), clamped to [rho_min, inf).

    Typical values (base=2.0):
        - tree     (avg_degree ~ 2)  : rho ~ 2.8
        - sparse   (avg_degree ~ 3-4): rho ~ 3.5 - 4.0
        - complete N=3  (avg_degree=2): rho ~ 2.8
        - complete N=5  (avg_degree=4): rho ~ 4.0
        - complete N=10 (avg_degree=9): rho ~ 6.0

    Parameters
    ----------
    neighbors : dict {n: list[int]} — graph adjacency
    base      : scaling coefficient (increase if ADMM converges too slowly)
    rho_min   : minimum rho value

    Returns
    -------
    rho_admm : float
    """
    N = len(neighbors)
    total_degree = sum(len(nbrs) for nbrs in neighbors.values())
    avg_degree   = total_degree / N
    rho = round(base * float(np.sqrt(avg_degree)), 1)
    return max(rho, rho_min)


# ------------------------------------------------------------------
# Parametric instance (random graph + random parameters)
# ------------------------------------------------------------------

def init_parameters_graph(
    N: int,
    topology: str = "tree",
    seed: int = 0,
    sparsity: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Initialize a random EnergyNetwork for N prosumers.

    Returns
    -------
    (net, gamma, max_iter, tol, rho_admm)
    """
    rng       = np.random.default_rng(seed)
    neighbors = build_graph(N, topology, seed, sparsity)

    D_min   = torch.zeros(N, dtype=torch.float32)
    D_max   = torch.tensor(rng.uniform(5., 20., N),  dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor(rng.uniform(0.,  8., N),  dtype=torch.float32)
    G_res   = torch.tensor(rng.uniform(0.,  4., N),  dtype=torch.float32)
    D_star  = torch.tensor(rng.uniform(2., 14., N),  dtype=torch.float32)
    a_tilde = torch.tensor(rng.uniform(8., 15., N),  dtype=torch.float32)
    a       = torch.tensor(rng.uniform(2.,  6., N),  dtype=torch.float32)
    b       = torch.zeros(N, dtype=torch.float32)

    # Root node (node 0): grid interface
    D_max[0]   = 0.0
    G_max[0]   = 50.0
    a_tilde[0] = 0.0
    D_star[0]  = 0.0

    c, k = {}, {}
    for n in range(N):
        for m in neighbors[n]:
            if m == 0 or n == 0:
                c_val = float(rng.uniform(0.3, 0.5))
                k_val = float(rng.uniform(15., 20.))
            else:
                c_val = float(rng.uniform(0.1, 0.3))
                k_val = float(rng.uniform(5., 10.))
            c[(n, m)] = torch.tensor(c_val, dtype=torch.float32)
            if (n, m) not in k:
                k[(n, m)] = torch.tensor(k_val, dtype=torch.float32)
                k[(m, n)] = torch.tensor(k_val, dtype=torch.float32)

    net = build_network(N, neighbors, D_min, D_max, G_min, G_max, G_res,
                        a_tilde, a, b, D_star, c, k, device)

    rho_admm = compute_rho_admm(neighbors)

    return net, 0.1, max_iter, tol, rho_admm


# ------------------------------------------------------------------
# Graph plotting  (invariato)
# ------------------------------------------------------------------

def plot_graph(neighbors: dict, title: str = "Energy Network Graph",
               save_path: str = None):
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    G = nx.Graph()
    N = len(neighbors)
    G.add_nodes_from(range(N))
    for n, nbrs in neighbors.items():
        for m in nbrs:
            G.add_edge(n, m)

    degrees  = dict(G.degree())
    max_deg  = max(degrees.values())
    if max_deg >= N // 2:
        root   = max(degrees, key=degrees.get)
        shells = [[root], [n for n in range(N) if n != root]]
        pos    = nx.shell_layout(G, nlist=shells)
    else:
        pos = nx.spring_layout(G, seed=42)

    node_colors = ["#e05c5c" if n == 0 else "#5c8de0" for n in range(N)]
    node_sizes  = [600 if n == 0 else 400 for n in range(N)]

    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color="#aaaaaa", width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9,
                            font_color="white", font_weight="bold")

    legend = [
        Patch(facecolor="#e05c5c", label="Root (grid interface)"),
        Patch(facecolor="#5c8de0", label="Prosumer"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)

    n_edges      = G.number_of_edges()
    max_possible = N * (N - 1) // 2
    sparsity     = 1.0 - n_edges / max_possible if max_possible > 0 else 0.0
    ax.set_title(
        f"{title}\nN={N} nodes, {n_edges} edges, sparsity={sparsity:.2f}",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graph saved to {save_path}")
    else:
        plt.show()
    plt.close()