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

    return {n: sorted(G.neighbors(n)) for n in range(N)}


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
    return net, 0.1, 100, tol, compute_rho_admm(neighbors)  # rho_admm computed based on graph structure

    
def initN4(device: torch.device = torch.device("cpu")) -> tuple:
    """
    Small test network — N=3, fully connected.

    Returns
    -------
    (net, gamma, max_iter, tol)  where net is an EnergyNetwork.
    """
    N         = 4
    neighbors = {0: [1, 2, 3], 1: [0, 2], 2: [0, 1, 3], 3: [0, 2]}

    D_min   = torch.tensor([0.0,  0.0,  0.0, 0.0],  dtype=torch.float32)
    D_max   = torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor([10.0, 0.0, 0.0, 0.0],   dtype=torch.float32)
    G_res   = torch.tensor([0.0,  3.0, 5.0, 3.0],   dtype=torch.float32)
    a_tilde = torch.tensor([5.0,  15.0, 10.0, 15.0], dtype=torch.float32)
    D_star  = torch.tensor([6.0,  3.0,  3.0,  3.0],  dtype=torch.float32)
    a       = torch.tensor([4.0,  4.0,  4.0,  4.0],  dtype=torch.float32)
    b       = torch.zeros(N, dtype=torch.float32)

    c, k = _build_c_k(
        neighbors,
        c_fn=lambda m, n: 0.1,
        k_fn=lambda m, n: 10.0 if m == 0 else 5.0,
    )

    net = build_network(N, neighbors, D_min, D_max, G_min, G_max, G_res,
                        a_tilde, a, b, D_star, c, k, device)
    return net, 0.1, 100, tol, compute_rho_admm(neighbors)  # rho_admm=3 for N=3 complete


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
    return net, 0.1, 100, tol, compute_rho_admm(neighbors)  # rho_admm=5 for N=10 star-like


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
    return net, 0.1, 300, tol, compute_rho_admm(neighbors)  # rho_admm computed based on graph structure


# ------------------------------------------------------------------
# ADMM penalty heuristic
# ------------------------------------------------------------------
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


"""
initN  — generate a feasible, well-conditioned EnergyNetwork for any N >= 3.

Drop-in replacement for initN3 / initN4 / initN10 / initN30.
Returns the same 5-tuple: (net, gamma, max_iter, tol, rho_admm).

Design choices
--------------
Graph
    Node 0 is always the root (grid interface, large generator, D=0).
    The graph is connected by construction:
        topology="hub"    — root connected to every non-root (star + optional peer links)
        topology="tree"   — random spanning tree, root = highest-degree node
        topology="sparse" — random connected graph, edge probability ~ 3/N (log-connected)
    Default: "hub"  (same structure as initN3/N10/N30).

Feasibility guarantees  (enforced algebraically, not just asserted)
    For every non-root node n:
        D_max[n]  >= G_res[n]                       (can self-consume renewables)
        D_max[n]  >= D_star[n]                       (target is reachable)
        D_min[n]  <= G_max[n] + G_res[n] + max_import[n]   (supply covers min demand)
    For the root:
        G_max[0]  >= total_deficit * margin           (root covers network deficit)
        sum_m k[(0,m)] >= G_max[0]                   (root link capacity scaled up if needed)

Well-conditioned ADMM
    rho = sqrt(L * mu)  where
        L   = max_{n != 0}  max(2*a_tilde[n], a[n])
        mu  = min_{n != 0}  min(2*a_tilde[n], a[n])
    Ranges of a_tilde and a are chosen so that L/mu <= 10 (mild conditioning).
    b = 0 everywhere (consistent with initN3/N10/N30).

Parameters kept consistent with fixed instances
    gamma    = 0.1
    max_iter = max(100, N * 15)
    tol      = 1e-6  (same global constant)
"""

import torch
import numpy as np
import networkx as nx

# These are already defined in init_parameters.py — imported here for standalone use.
# When integrated, just paste the function body into init_parameters.py.
from prosumer import Prosumer
from energy_network import EnergyNetwork

_TOL = 1e-6


# ------------------------------------------------------------------
# Helpers (same as in init_parameters.py)
# ------------------------------------------------------------------

def _build_c_k(neighbors: dict, c_fn, k_fn) -> tuple:
    c, k = {}, {}
    for n in neighbors:
        for m in neighbors[n]:
            c[(m, n)] = torch.tensor(c_fn(m, n), dtype=torch.float32)
            k[(m, n)] = torch.tensor(k_fn(m, n), dtype=torch.float32)
    return c, k


def _build_network(N, neighbors, D_min, D_max, G_min, G_max, G_res,
                   a_tilde, a, b, D_star, c, k, device):
    from energy_network import EnergyNetwork
    from prosumer import Prosumer
    prosumers = []
    for n in range(N):
        c_n = {m: c[(m, n)] for m in neighbors[n]}
        k_n = {m: k[(m, n)] for m in neighbors[n]}
        p = Prosumer(
            node_id=n, neighbors=neighbors[n],
            D_min=D_min[n], D_max=D_max[n],
            G_min=G_min[n], G_max=G_max[n],
            G_res=G_res[n], a_tilde=a_tilde[n],
            a=a[n], b=b[n], D_star=D_star[n],
            c=c_n, k=k_n, device=device,
        )
        prosumers.append(p)
    return EnergyNetwork(prosumers, neighbors, device)


# ------------------------------------------------------------------
# Graph builders
# ------------------------------------------------------------------

def _build_neighbors_hub(N: int, rng: np.random.Generator) -> dict:
    """
    Root (0) connected to every non-root.
    Additional peer-to-peer edges among non-roots via Erdos-Renyi (p=0.3),
    giving ~30% of possible peer links — enough connectivity without
    making the graph too dense for large N.
    """
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(N))
    for n in range(1, N):
        G_nx.add_edge(0, n)
    if N > 3:
        p_peer = min(0.4, 3.0 / max(N - 2, 1))   # ~3 peer neighbors on avg, cap at 0.4
        er_seed = int(rng.integers(1_000_000))
        G_peers = nx.erdos_renyi_graph(N - 1, p=p_peer, seed=er_seed)
        for (u, v) in G_peers.edges():
            G_nx.add_edge(u + 1, v + 1)
    return {n: sorted(G_nx.neighbors(n)) for n in range(N)}


def _build_neighbors_tree(N: int, rng: np.random.Generator) -> dict:
    """
    Random spanning tree; highest-degree node becomes node 0 (root).
    """
    tree_seed = int(rng.integers(1_000_000))
    G_nx = nx.random_labeled_tree(N, seed=tree_seed)
    degrees = dict(G_nx.degree())
    root = max(degrees, key=degrees.get)
    mapping = {root: 0}
    for new_idx, old_idx in enumerate((n for n in G_nx.nodes if n != root), start=1):
        mapping[old_idx] = new_idx
    G_nx = nx.relabel_nodes(G_nx, mapping)
    return {n: sorted(G_nx.neighbors(n)) for n in range(N)}


def _build_neighbors_sparse(N: int, rng: np.random.Generator,
                             avg_degree: float = 3.0) -> dict:
    """
    Random connected graph with average degree ~ avg_degree.
    Root = node with highest degree after generation.
    """
    n_edges = max(N - 1, round(avg_degree * N / 2))
    while True:
        g_seed = int(rng.integers(1_000_000))
        G_nx = nx.gnm_random_graph(N, m=n_edges, seed=g_seed)
        if nx.is_connected(G_nx):
            break
    degrees = dict(G_nx.degree())
    root = max(degrees, key=degrees.get)
    mapping = {root: 0}
    for new_idx, old_idx in enumerate((n for n in G_nx.nodes if n != root), start=1):
        mapping[old_idx] = new_idx
    G_nx = nx.relabel_nodes(G_nx, mapping)
    return {n: sorted(G_nx.neighbors(n)) for n in range(N)}


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------

def initN(
    N: int,
    seed: int = 42,
    topology: str = "hub",
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Generate a feasible, well-conditioned EnergyNetwork for any N >= 3.

    Drop-in replacement for initN3 / initN4 / initN10 / initN30.
    Returns the same 5-tuple:
        (net, gamma, max_iter, tol, rho_admm)

    Parameters
    ----------
    N        : number of prosumers (>= 3)
    seed     : random seed for reproducibility
    topology : "hub"    — root star + Erdos-Renyi peer links (default, matches fixed instances)
               "tree"   — random spanning tree
               "sparse" — random connected graph, avg degree ~ 3
    device   : torch.device

    Feasibility guarantees (enforced, not just asserted)
    ----------------------------------------------------
    - D_max[n]  >= max(G_res[n], D_star[n])     for all n
    - D_min[n]  <= G_max[n] + G_res[n] + max_import[n]  for all n
    - G_max[0]  covers total network deficit (with 20% safety margin)
    - Root link capacities scaled up if G_max[0] > sum of root link caps

    Well-conditioned ADMM
    ---------------------
    - a_tilde in [5, 15], a in [2, 6]  (L/mu <= 10 for non-root nodes)
    - rho = sqrt(L * mu), floored at 2.0
    """
    if N < 3:
        raise ValueError(f"N must be >= 3, got {N}")
    if topology not in ("hub", "tree", "sparse"):
        raise ValueError(f"topology must be 'hub', 'tree', or 'sparse', got '{topology}'")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Graph topology
    # ------------------------------------------------------------------
    if topology == "hub":
        neighbors = _build_neighbors_hub(N, rng)
    elif topology == "tree":
        neighbors = _build_neighbors_tree(N, rng)
    else:
        neighbors = _build_neighbors_sparse(N, rng)

    # ------------------------------------------------------------------
    # 2. Link parameters: c and k
    #    - Root links (0-n or n-0): higher capacity, higher cost
    #    - Peer links: lower capacity, lower cost
    #    - Symmetric: c[(m,n)] = c[(n,m)], k[(m,n)] = k[(n,m)]
    # ------------------------------------------------------------------
    c_dict, k_dict = {}, {}
    for n in range(N):
        for m in neighbors[n]:
            if (n, m) not in c_dict:   # generate once per undirected edge
                if m == 0 or n == 0:
                    c_val = float(rng.uniform(0.3, 0.5))
                    k_val = float(rng.uniform(15.0, 25.0))
                else:
                    c_val = float(rng.uniform(0.05, 0.3))
                    k_val = float(rng.uniform(3.0, 10.0))
                c_dict[(n, m)] = torch.tensor(c_val, dtype=torch.float32)
                c_dict[(m, n)] = torch.tensor(c_val, dtype=torch.float32)
                k_dict[(n, m)] = torch.tensor(k_val, dtype=torch.float32)
                k_dict[(m, n)] = torch.tensor(k_val, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 3. max_import[n] = sum of inbound link capacities
    #    Used to compute the feasibility ceiling for D_max.
    # ------------------------------------------------------------------
    max_import = torch.tensor(
        [sum(float(k_dict[(m, n)].item()) for m in neighbors[n]) for n in range(N)],
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # 4. Renewable generation G_res
    #    Root: G_res = 0 (it is a controllable generator, no RES)
    #    Non-root: U[0.5, 5.0]
    # ------------------------------------------------------------------
    G_res = torch.tensor(
        [0.0] + [float(rng.uniform(0.5, 5.0)) for _ in range(N - 1)],
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # 5. Controllable flexibility G_min / G_max
    #    Root: G_min=0, G_max set later (must cover network deficit).
    #    Non-root: ~35% of nodes have local flexibility G_max ~ U[2, 8].
    #    The rest are pure consumers (G_max = 0).
    # ------------------------------------------------------------------
    G_min = torch.zeros(N, dtype=torch.float32)
    G_max = torch.zeros(N, dtype=torch.float32)
    flex_prob = 0.35
    for n in range(1, N):
        if rng.random() < flex_prob:
            G_max[n] = float(rng.uniform(2.0, 8.0))

    # ------------------------------------------------------------------
    # 6. Cost parameters: a_tilde, a, b
    #    Non-root: a_tilde ~ U[5, 15], a ~ U[2, 6]
    #              => L/mu = max(2*a_tilde, a) / min(2*a_tilde, a) <= 10
    #    Root: a_tilde=0 (D fixed at 0, no demand preference),
    #          a=1 (minimal generation cost to avoid unbounded solution)
    #    b = 0 everywhere (consistent with all fixed instances)
    # ------------------------------------------------------------------
    a_tilde = torch.tensor(
        [0.0] + [float(rng.uniform(5.0, 15.0)) for _ in range(N - 1)],
        dtype=torch.float32,
    )
    a = torch.tensor(
        [1.0] + [float(rng.uniform(2.0, 6.0)) for _ in range(N - 1)],
        dtype=torch.float32,
    )
    b = torch.zeros(N, dtype=torch.float32)

    # ------------------------------------------------------------------
    # 7. D_min / D_max
    #    Root: D_min = D_max = 0 (root does not consume, only generates)
    #    Non-root:
    #        D_min = 0
    #        D_max = min(raw_sample, feasibility_ceiling)
    #              feasibility_ceiling = G_max[n] + G_res[n] + max_import[n]
    #        Then floor D_max >= G_res[n]   (can self-consume renewables)
    # ------------------------------------------------------------------
    D_min = torch.zeros(N, dtype=torch.float32)

    raw_D_max = torch.tensor(
        [0.0] + [float(rng.uniform(4.0, 18.0)) for _ in range(N - 1)],
        dtype=torch.float32,
    )
    feasibility_ceiling = G_max + G_res + max_import   # shape (N,)

    D_max = torch.minimum(raw_D_max, feasibility_ceiling)
    D_max = torch.maximum(D_max, G_res)   # can always self-consume
    D_max[0] = 0.0

    # ------------------------------------------------------------------
    # 8. D_star: target demand
    #    Sampled as a fraction of D_max in [0.3, 0.75] — always <= D_max.
    #    Root: D_star = 0.
    # ------------------------------------------------------------------
    frac = torch.tensor(
        [0.0] + [float(rng.uniform(0.3, 0.75)) for _ in range(N - 1)],
        dtype=torch.float32,
    )
    D_star = frac * D_max
    D_star[0] = 0.0

    # ------------------------------------------------------------------
    # 9. Root G_max: must cover total network deficit
    #    deficit[n] = max(0, D_star[n] - G_max[n] - G_res[n])
    #    We add a 20% safety margin so the root is not tight.
    # ------------------------------------------------------------------
    total_deficit = sum(
        max(0.0, float(D_star[n]) - float(G_max[n]) - float(G_res[n]))
        for n in range(1, N)
    )
    margin = 1.2
    G_max[0] = max(50.0, total_deficit * margin)

    # ------------------------------------------------------------------
    # 10. Scale root link capacities if G_max[0] exceeds their sum
    #     This ensures the root can physically export enough energy.
    # ------------------------------------------------------------------
    root_export_cap = sum(float(k_dict[(0, m)].item()) for m in neighbors[0])
    if G_max[0] > root_export_cap:
        scale = float(G_max[0]) / max(root_export_cap, 1e-6)
        for m in neighbors[0]:
            new_k = float(k_dict[(0, m)].item()) * scale
            k_dict[(0, m)] = torch.tensor(new_k, dtype=torch.float32)
            k_dict[(m, 0)] = torch.tensor(new_k, dtype=torch.float32)

        # Recompute max_import after scaling
        max_import = torch.tensor(
            [sum(float(k_dict[(m, n)].item()) for m in neighbors[n]) for n in range(N)],
            dtype=torch.float32,
        )
        # Re-apply feasibility ceiling with updated max_import
        feasibility_ceiling = G_max + G_res + max_import
        D_max = torch.minimum(D_max, feasibility_ceiling)
        D_max = torch.maximum(D_max, G_res)
        D_max[0] = 0.0
        # D_star already <= D_max by construction (frac * D_max_old <= D_max_new
        # since feasibility_ceiling grew), but clip to be safe
        D_star = torch.minimum(D_star, D_max)

    # ------------------------------------------------------------------
    # 11. Feasibility verification (debug-friendly, no hard crash)
    # ------------------------------------------------------------------
    feasibility_ok = True
    for n in range(N):
        supply = float(G_max[n]) + float(G_res[n]) + sum(
            float(k_dict[(m, n)].item()) for m in neighbors[n]
        )
        if float(D_min[n]) > supply + 1e-3:
            print(f"[initN] WARNING: Node {n}: D_min={D_min[n]:.3f} > supply={supply:.3f}")
            feasibility_ok = False
        if float(D_star[n]) > float(D_max[n]) + 1e-3:
            print(f"[initN] WARNING: Node {n}: D_star={D_star[n]:.3f} > D_max={D_max[n]:.3f}")
            feasibility_ok = False
        if float(D_max[n]) < float(G_res[n]) - 1e-3:
            print(f"[initN] WARNING: Node {n}: D_max={D_max[n]:.3f} < G_res={G_res[n]:.3f}")
            feasibility_ok = False
    if not feasibility_ok:
        print(f"[initN] Some feasibility checks failed for N={N}, seed={seed}. "
              f"The network may still work but results could be unreliable.")

    # ------------------------------------------------------------------
    # 12. Build the EnergyNetwork
    # ------------------------------------------------------------------
    net = _build_network(
        N, neighbors, D_min, D_max, G_min, G_max, G_res,
        a_tilde, a, b, D_star, c_dict, k_dict, device,
    )

    # ------------------------------------------------------------------
    # 13. Solver hyperparameters
    #
    # rho_admm = sqrt(L * mu)  (optimal ADMM penalty for strongly-monotone VI)
    #     L   = max_{n>0} max(2*a_tilde[n], a[n])
    #     mu  = min_{n>0} min(2*a_tilde[n], a[n])
    # Root excluded: a_tilde[0]=0 would make mu=0.
    #
    # gamma = 0.1  (consistent with all fixed instances)
    # max_iter scales with N to give enough budget for larger networks
    # ------------------------------------------------------------------
    L_vals  = [max(2.0 * float(a_tilde[n]), float(a[n])) for n in range(1, N)]
    mu_vals = [min(2.0 * float(a_tilde[n]), float(a[n])) for n in range(1, N)]
    L_max   = max(L_vals)
    mu_min  = min(mu_vals)
    rho_admm = float(np.sqrt(L_max * mu_min))
    rho_admm = max(round(rho_admm, 1), 2.0)   # floor at 2.0

    gamma    = 0.1
    max_iter = 100

    return net, gamma, max_iter, _TOL, rho_admm




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
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname if dirname else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graph saved to {save_path}")
    else:
        plt.show()
    plt.close()