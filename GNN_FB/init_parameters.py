import torch
import numpy as np
import networkx as nx


"""
params = init_parameters(N=10, topology="tree", seed=42)
params = init_parameters(N=30, topology="sparse", sparsity=0.7, seed=0)
params = init_parameters(N=3,  topology="complete")
"""

# ------------------------------------------------------------------
# Graph builder
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
        # Node 0 connected to all, plus random peer edges among non-root nodes
        G = nx.star_graph(N - 1)
        rng = np.random.default_rng(seed)
        non_root = list(range(1, N))
        # Add ~N/3 extra peer edges
        n_extra = max(0, N // 3)
        attempts = 0
        added = 0
        while added < n_extra and attempts < 1000:
            u, v = rng.choice(non_root, size=2, replace=False)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1
            attempts += 1

    elif topology == "tree":
        # Random tree with node 0 as root (Prüfer sequence)
        rng = np.random.default_rng(seed)
        G = nx.random_labeled_tree(N, seed=int(rng.integers(1e6)))
        # Relabel so node with highest degree becomes node 0
        degrees = dict(G.degree())
        root = max(degrees, key=degrees.get)
        mapping = {root: 0}
        other = [n for n in G.nodes if n != root]
        for new_idx, old_idx in enumerate(other, start=1):
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

    neighbors = {n: sorted(list(G.neighbors(n))) for n in range(N)}
    return neighbors


# ------------------------------------------------------------------
# Main init function
# ------------------------------------------------------------------

def init_parameters_graph(
    N: int,
    topology: str = "tree",
    seed: int = 0,
    sparsity: float = 0.5,
    n_test = 0
) -> tuple:
    """
    Initialize EnergyNetwork parameters for N prosumers.

    Parameter ranges are calibrated on the original initN3/N10/N30:
        D_max   ~ U[5, 20]      (demand capacity)
        G_max   ~ U[0, 8]       (flexible generation)
        G_res   ~ U[0, 4]       (renewable generation)
        D_star  ~ U[2, 14]      (desired demand)
        a_tilde ~ U[8, 15]      (demand cost curvature)
        a       ~ U[2,  6]      (generation cost curvature)
        b       = 0             (linear generation cost, as in originals)
        c       ~ U[0.1, 0.5]   (trading cost)
        k       ~ U[5, 20]      (trading capacity)

    Node 0 is treated as the grid interface (root):
        D_max[0]   = 0   (root does not consume)
        G_max[0]   = 50  (root can inject from main grid)
        a_tilde[0] = 0
        D_star[0]  = 0

    Parameters
    ----------
    N        : number of prosumers
    topology : graph topology (see build_graph)
    seed     : random seed
    sparsity : only used when topology="sparse"

    Returns
    -------
    tuple of 16 values matching EnergyNetwork constructor
    """
    rng = np.random.default_rng(seed)

    # --- Graph ---
    neighbors = build_graph(N, topology, seed, sparsity)

    # --- Solver parameters ---
    gamma    = 0.1
    max_iter = 50
    tol      = 1e-6

    # --- Node parameters ---
    D_min   = torch.zeros(N, dtype=torch.float32)
    D_max   = torch.tensor(rng.uniform(5.0, 20.0, N),  dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor(rng.uniform(0.0,  8.0, N),  dtype=torch.float32)
    G_res   = torch.tensor(rng.uniform(0.0,  4.0, N),  dtype=torch.float32)
    D_star  = torch.tensor(rng.uniform(2.0, 14.0, N),  dtype=torch.float32)
    a_tilde = torch.tensor(rng.uniform(8.0, 15.0, N),  dtype=torch.float32)
    a       = torch.tensor(rng.uniform(2.0,  6.0, N),  dtype=torch.float32)
    b       = torch.zeros(N, dtype=torch.float32)

    # Node 0 = grid interface (root): no demand, unlimited supply
    D_max[0]   = 0.0
    G_max[0]   = 50.0
    a_tilde[0] = 0.0
    D_star[0]  = 0.0

    # --- Edge parameters ---
    c = {}
    k = {}
    for n in range(N):
        for m in neighbors[n]:
            # Higher cost and capacity for links involving root node
            if m == 0 or n == 0:
                c_val = float(rng.uniform(0.3, 0.5))
                k_val = float(rng.uniform(15.0, 20.0))
            else:
                c_val = float(rng.uniform(0.1, 0.3))
                k_val = float(rng.uniform(5.0, 10.0))
            c[(n, m)] = torch.tensor(c_val, dtype=torch.float32)
            # Capacity is symmetric — set both directions on first encounter
            if (n, m) not in k:
                k[(n, m)] = torch.tensor(k_val, dtype=torch.float32)
                k[(m, n)] = torch.tensor(k_val, dtype=torch.float32)


    return (
        N, gamma, max_iter, tol,
        neighbors,
        D_min, D_max, G_min, G_max, G_res,
        D_star, a_tilde, a, b, c, k,
    )



def _build_c_k(neighbors, c_fn, k_fn):
    """
    Helper to build c and k dicts given per-edge functions.

    Parameters
    ----------
    neighbors : dict {n: [m1, m2, ...]}
    c_fn : callable (m, n) -> float   trading cost c_{nm}
    k_fn : callable (m, n) -> float   capacity k_{mn}

    Returns
    -------
    c : dict {(m, n): tensor}
    k : dict {(m, n): tensor}
    """
    c = {}
    k = {}
    for n in neighbors:
        for m in neighbors[n]:
            c[(m, n)] = torch.tensor(c_fn(m, n), dtype=torch.float32)
            k[(m, n)] = torch.tensor(k_fn(m, n), dtype=torch.float32)
    return c, k


def plot_graph(neighbors: dict, title: str = "Energy Network Graph",
               save_path: str = None):
    """
    Plot the energy network graph from a neighbors dict.

    Parameters
    ----------
    neighbors : dict {n: list of neighbor indices}
    title     : plot title
    save_path : if given, save figure here; otherwise show interactively
    """
    import matplotlib.pyplot as plt

    G = nx.Graph()
    N = len(neighbors)
    G.add_nodes_from(range(N))
    for n, nbrs in neighbors.items():
        for m in nbrs:
            G.add_edge(n, m)

    # Layout: spring for general graphs, shell if star-like
    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    if max_deg >= N // 2:
        # Star or hub-dominated: put highest-degree node at center
        root = max(degrees, key=degrees.get)
        shells = [[root], [n for n in range(N) if n != root]]
        pos = nx.shell_layout(G, nlist=shells)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node colors: root (node 0) in a different color
    node_colors = ["#e05c5c" if n == 0 else "#5c8de0" for n in range(N)]
    node_sizes  = [600 if n == 0 else 400 for n in range(N)]

    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, edge_color="#aaaaaa", width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9,
                            font_color="white", font_weight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#e05c5c", label="Root (grid interface)"),
        Patch(facecolor="#5c8de0", label="Prosumer"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)

    # Stats
    n_edges = G.number_of_edges()
    max_possible = N * (N - 1) // 2
    sparsity = 1.0 - n_edges / max_possible if max_possible > 0 else 0.0
    ax.set_title(
        f"{title}\n"
        f"N={N} nodes, {n_edges} edges, sparsity={sparsity:.2f}",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graph saved to {save_path}")
    else:
        plt.show()

    plt.close()


def initN3():
    """
    Small test network with N=3 prosumers, fully connected.

    Returns
    -------
    N, gamma, max_iter, tol,
    neighbors, D_min, D_max, G_min, G_max, G_res,
    D_star, a_tilde, a, b, c, k
    """
    N        = 3
    gamma    = 0.1
    max_iter = 100
    tol      = 1e-6

    neighbors = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }

    D_min   = torch.tensor([0.0,  0.0,  0.0],  dtype=torch.float32)
    D_max   = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor([10.0, 0.0, 0.0],   dtype=torch.float32)
    G_res   = torch.tensor([0.0,  3.0, 5.0],   dtype=torch.float32)

    a_tilde = torch.tensor([5.0,  15.0, 10.0], dtype=torch.float32)
    D_star  = torch.tensor([6.0,  3.0,  3.0],  dtype=torch.float32)
    a       = torch.tensor([4.0,  4.0,  4.0],  dtype=torch.float32)
    b       = torch.zeros(N, dtype=torch.float32)

    def c_fn(m, n):
        return 0.1

    def k_fn(m, n):
        return 10.0 if m == 0 else 5.0

    c, k = _build_c_k(neighbors, c_fn, k_fn)

    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, \
           G_res, D_star, a_tilde, a, b, c, k


def initN10():
    """
    Medium network with N=10 prosumers.
    Node 0 is the root (grid interface), connected to all others.

    Returns
    -------
    N, gamma, max_iter, tol,
    neighbors, D_min, D_max, G_min, G_max, G_res,
    D_star, a_tilde, a, b, c, k
    """
    N        = 10
    gamma    = 0.1
    max_iter = 100
    tol      = 1e-6

    neighbors = {
        0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        1: [0, 2, 3],
        2: [0, 1, 4],
        3: [0, 1, 5],
        4: [0, 2, 6],
        5: [0, 3, 7],
        6: [0, 4, 8],
        7: [0, 5, 9],
        8: [0, 6, 9],
        9: [0, 7, 8],
    }

    D_min   = torch.zeros(N, dtype=torch.float32)
    D_max   = torch.full((N,), 10.0, dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor(
        [10.0, 0.0, 6.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        dtype=torch.float32
    )
    G_res   = torch.tensor(
        [0.0, 2.0, 3.0, 1.0, 0.5, 2.5, 1.5, 0.0, 3.0, 1.0],
        dtype=torch.float32
    )
    D_star  = torch.tensor(
        [6.0, 4.0, 5.0, 6.0, 5.0, 4.0, 5.0, 6.0, 4.0, 5.0],
        dtype=torch.float32
    )
    a_tilde = torch.tensor(
        [10.0, 12.0, 8.0, 10.0, 9.0, 11.0, 10.0, 12.0, 9.0, 10.0],
        dtype=torch.float32
    )
    a       = torch.tensor(
        [3.0, 2.0, 4.0, 3.0, 2.5, 3.5, 4.0, 3.0, 2.0, 3.0],
        dtype=torch.float32
    )
    b       = torch.zeros(N, dtype=torch.float32)

    def c_fn(m, n):
        return 0.1

    def k_fn(m, n):
        # Root node links have higher capacity
        return 10.0 if (m == 0 or n == 0) else 5.0

    c, k = _build_c_k(neighbors, c_fn, k_fn)

    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, \
           G_res, D_star, a_tilde, a, b, c, k


def initN30():
    """
    Large radial tree network with N=30 prosumers.
    Node 0 is the root (main grid interface).

    Returns
    -------
    N, gamma, max_iter, tol,
    neighbors, D_min, D_max, G_min, G_max, G_res,
    D_star, a_tilde, a, b, c, k
    """
    N        = 30
    gamma    = 0.1
    max_iter = 300
    tol      = 1e-6

    neighbors = {
        0:  [1, 2],
        1:  [0, 3, 4],
        2:  [0, 5, 6],
        3:  [1, 7, 8],
        4:  [1, 9],
        5:  [2, 10],
        6:  [2, 11, 12],
        7:  [3, 13, 14],
        8:  [3, 15],
        9:  [4, 16],
        10: [5, 16],
        11: [6, 17],
        12: [6],
        13: [7, 18],
        14: [7, 19],
        15: [8],
        16: [9, 10, 20],
        17: [11, 21],
        18: [13, 22],
        19: [14],
        20: [16, 23, 24],
        21: [17, 25],
        22: [18],
        23: [20, 26],
        24: [20],
        25: [21, 27],
        26: [23],
        27: [25, 28],
        28: [27, 29],
        29: [28],
    }

    D_min = torch.zeros(N, dtype=torch.float32)
    D_max = torch.tensor([
         0.0, 20.0, 20.0, 15.0, 10.0, 10.0, 15.0,  8.0,  8.0,  6.0,
         6.0,  8.0,  5.0,  6.0,  6.0,  5.0,  7.0,  6.0,  5.0,  5.0,
         7.0,  5.0,  4.0,  5.0,  4.0,  4.0,  4.0,  3.0,  3.0,  3.0,
    ], dtype=torch.float32)

    G_min = torch.zeros(N, dtype=torch.float32)
    G_max = torch.tensor([
        50.0, 8.0, 8.0, 5.0, 0.0, 0.0, 5.0, 3.0, 0.0, 3.0,
         0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0,
         2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=torch.float32)

    G_res = torch.tensor([
        0.0, 2.0, 1.5, 3.0, 4.0, 3.5, 2.0, 2.5, 4.0, 3.0,
        3.5, 2.0, 4.5, 3.0, 3.5, 4.0, 2.5, 3.0, 2.0, 3.5,
        2.5, 3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 3.0, 3.5, 4.0,
    ], dtype=torch.float32)

    a_tilde = torch.tensor([
         0.0, 10.0, 10.0,  8.0, 12.0, 12.0,  8.0, 10.0, 10.0, 12.0,
        12.0, 10.0, 15.0, 12.0, 12.0, 15.0, 10.0, 12.0, 15.0, 15.0,
        10.0, 12.0, 15.0, 12.0, 15.0, 12.0, 15.0, 12.0, 12.0, 15.0,
    ], dtype=torch.float32)

    D_star = torch.tensor([
         0.0, 14.0, 14.0, 10.0,  7.0,  7.0, 10.0,  5.5,  5.5,  4.0,
         4.0,  5.5,  3.5,  4.0,  4.0,  3.5,  5.0,  4.0,  3.5,  3.5,
         5.0,  3.5,  3.0,  3.5,  3.0,  3.0,  3.0,  2.0,  2.0,  2.0,
    ], dtype=torch.float32)

    a = torch.tensor([
        1.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 6.0, 4.0, 5.0,
        4.0, 6.0, 4.0, 6.0, 4.0, 4.0, 5.0, 4.0, 6.0, 4.0,
        5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    ], dtype=torch.float32)

    b = torch.zeros(N, dtype=torch.float32)

    def c_fn(m, n):
        if m == 0 or n == 0:
            return 0.5   # grid-connected link: higher tariff
        elif m <= 2 or n <= 2:
            return 0.2   # links adjacent to main hubs
        else:
            return 0.1   # peer-to-peer deeper links

    def k_fn(m, n):
        if m == 0 or n == 0:
            return 20.0
        elif m <= 2 or n <= 2:
            return 10.0
        else:
            return 5.0

    c, k = _build_c_k(neighbors, c_fn, k_fn)

    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, \
           G_res, D_star, a_tilde, a, b, c, k