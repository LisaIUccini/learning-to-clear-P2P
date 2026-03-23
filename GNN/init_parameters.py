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
    Initialize a random EnergyNetwork for N prosumers with
    physically realistic parameters and guaranteed feasibility.

    Parameter design rationale (from analysis of initN3/N10/N30
    and P2P energy market literature):

    - G_max: 55-80% of non-root nodes are pure consumers (G_max=0).
      Nodes with flexibility have G_max in [2, 8].
    - G_res: every node has some renewable generation, U[0.5, 5].
    - D_max: scales with position in network. Near-root nodes have
      higher demand capacity. Clipped for feasibility.
    - D_star: set as a fraction of D_max, typically 30-75%.
    - a_tilde: demand sensitivity, U[5, 15] (from fixed instances).
    - a: flexibility cost curvature, U[2, 6].
    - c: trading cost — root links more expensive (grid fees).
    - k: link capacity — root links 2-4x larger than peer links.

    Returns
    -------
    (net, gamma, max_iter, tol, rho_admm)
    """
    rng       = np.random.default_rng(seed)
    neighbors = build_graph(N, topology, seed, sparsity)

    # ------------------------------------------------------------------
    # Step 1: Flexibility (G_max) — sparse, like real microgrids
    #
    # In real P2P markets, most prosumers are pure consumers or have
    # only solar panels (captured by G_res). Controllable flexibility
    # (batteries, micro-CHP) is expensive and owned by ~20-40% of nodes.
    # ------------------------------------------------------------------
    G_min = torch.zeros(N, dtype=torch.float32)
    G_max = torch.zeros(N, dtype=torch.float32)

    # Fraction of non-root nodes with controllable flexibility: 20-45%
    flex_prob = float(rng.uniform(0.2, 0.45))
    for n in range(1, N):
        if rng.random() < flex_prob:
            G_max[n] = float(rng.uniform(2., 8.))
        # else G_max[n] = 0 (pure consumer)

    # ------------------------------------------------------------------
    # Step 2: Renewable generation (G_res) — every node has some
    #
    # In modern distribution networks, nearly all prosumers have
    # rooftop solar or small wind. G_res=0 is unrealistic.
    # Root node has G_res=0 (it's the grid interface, not a prosumer).
    # ------------------------------------------------------------------
    G_res = torch.tensor(
        [0.0] + [float(rng.uniform(0.5, 5.0)) for _ in range(N - 1)],
        dtype=torch.float32
    )

    # ------------------------------------------------------------------
    # Step 3: Link capacities — root links larger (grid connection)
    #
    # Root-to-prosumer links represent the grid connection point,
    # with higher capacity (15-25 kW). Peer-to-peer links are
    # smaller (3-10 kW), representing local interconnections.
    # Symmetry: k_mn = k_nm (same physical cable).
    # ------------------------------------------------------------------
    c, k = {}, {}
    for n in range(N):
        for m in neighbors[n]:
            # Trading cost: root links have grid fees
            if m == 0 or n == 0:
                c_val = float(rng.uniform(0.3, 0.5))
                k_val = float(rng.uniform(15., 25.))
            else:
                c_val = float(rng.uniform(0.05, 0.3))
                k_val = float(rng.uniform(3., 10.))
            c[(n, m)] = torch.tensor(c_val, dtype=torch.float32)
            if (n, m) not in k:
                k[(n, m)] = torch.tensor(k_val, dtype=torch.float32)
                k[(m, n)] = torch.tensor(k_val, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Step 4: Import capacity per node (needed for D_max feasibility)
    # ------------------------------------------------------------------
    max_import = torch.zeros(N, dtype=torch.float32)
    for n in range(N):
        max_import[n] = sum(float(k[(m, n)].item()) for m in neighbors[n])

    # ------------------------------------------------------------------
    # Step 5: Demand bounds (D_min, D_max)
    #
    # D_min = 0 for all nodes (can always choose not to consume).
    # D_max reflects the maximum load the node can draw:
    #   - Must satisfy: D_max >= G_res (can always self-consume RES)
    #   - Must satisfy: D_max <= G_max + G_res + max_import (feasibility)
    #   - In real networks, D_max correlates with connectivity
    #     (well-connected nodes can import more, so can consume more).
    # ------------------------------------------------------------------
    D_min = torch.zeros(N, dtype=torch.float32)

    # Base D_max: random, then clipped for feasibility
    D_max_raw = torch.tensor(
        [float(rng.uniform(4., 18.)) for _ in range(N)],
        dtype=torch.float32
    )

    # Feasibility ceiling: can't consume more than total available supply
    D_max_feasible = G_max + G_res + max_import
    D_max = torch.minimum(D_max_raw, D_max_feasible)

    # Feasibility floor: at least G_res (can self-consume renewables)
    D_max = torch.maximum(D_max, G_res)

    # Root: no demand
    D_max[0] = 0.0
    D_min[0] = 0.0

    # ------------------------------------------------------------------
    # Step 6: Target demand (D_star) — fraction of D_max
    #
    # D_star represents the prosumer's preferred consumption level.
    # From fixed instances: D_star/D_max is typically 0.3-0.75.
    # D_star MUST be in [D_min, D_max] for the cost function to
    # have its minimum inside the feasible set.
    # ------------------------------------------------------------------
    D_star_ratio = torch.tensor(
        [float(rng.uniform(0.3, 0.75)) for _ in range(N)],
        dtype=torch.float32
    )
    D_star = D_star_ratio * D_max
    D_star[0] = 0.0

    # ------------------------------------------------------------------
    # Step 7: Cost function parameters
    #
    # a_tilde: demand utility curvature. Higher = more sensitive to
    #   deviating from D_star. Range [5, 15] from fixed instances.
    #
    # a: flexibility activation cost curvature. Higher = more expensive
    #   to use controllable generation. Range [2, 6].
    #   For nodes with G_max=0, 'a' doesn't matter (G is always 0),
    #   but we set it anyway for consistency.
    #
    # b: linear flexibility cost. Set to 0 (as in all fixed instances).
    #   The constraint -b/a >= G_min (from paper eq. 6) is automatically
    #   satisfied since b=0 and G_min=0.
    # ------------------------------------------------------------------
    a_tilde = torch.tensor(
        [0.0] + [float(rng.uniform(5., 15.)) for _ in range(N - 1)],
        dtype=torch.float32
    )
    a = torch.tensor(
        [1.0] + [float(rng.uniform(2., 6.)) for _ in range(N - 1)],
        dtype=torch.float32
    )
    b = torch.zeros(N, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Step 8: Root node — grid interface
    #
    # The root must be able to export enough to cover the network's
    # net energy deficit. Its G_max must cover the total deficit
    # of all non-root nodes, and its link capacities must allow
    # transporting this energy.
    # ------------------------------------------------------------------
    # Total deficit: energy that non-root nodes cannot self-supply
    total_deficit = 0.0
    for n in range(1, N):
        # Worst case: node needs D_star but can only self-generate G_res+G_max
        local_gen = float(G_max[n]) + float(G_res[n])
        deficit_n = max(0.0, float(D_star[n]) - local_gen)
        total_deficit += deficit_n

    # Root G_max: cover deficit with 50% margin
    root_export_cap = sum(float(k[(0, m)].item()) for m in neighbors[0])
    G_max[0] = max(50.0, total_deficit * 1.5)

    # Ensure root can actually export its generation
    if G_max[0] > root_export_cap:
        scale = float(G_max[0]) / max(root_export_cap, 1.0)
        for m in neighbors[0]:
            new_val = float(k[(0, m)].item()) * scale
            k[(0, m)] = torch.tensor(new_val, dtype=torch.float32)
            k[(m, 0)] = torch.tensor(new_val, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Step 9: Feasibility verification
    # ------------------------------------------------------------------
    for n in range(N):
        local_supply_max = float(G_max[n]) + float(G_res[n]) + float(max_import[n])
        assert float(D_min[n]) <= local_supply_max, (
            f"Node {n}: D_min={D_min[n]:.2f} > max supply {local_supply_max:.2f}")
        assert float(D_max[n]) >= float(G_res[n]), (
            f"Node {n}: D_max={D_max[n]:.2f} < G_res={G_res[n]:.2f}")
        assert float(D_star[n]) <= float(D_max[n]), (
            f"Node {n}: D_star={D_star[n]:.2f} > D_max={D_max[n]:.2f}")

    net = build_network(N, neighbors, D_min, D_max, G_min, G_max, G_res,
                        a_tilde, a, b, D_star, c, k, device)

    rho_admm = compute_rho_admm(neighbors)

    return net, 0.1, max_iter, tol, rho_admm

# ------------------------------------------------------------------
# Fixed-topology parametric instance (random parameters, fixed graph)
# ------------------------------------------------------------------
 
def init_parameters_fixed_topology(
    neighbors: dict,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """
    Generate a random feasible EnergyNetwork on a FIXED topology.
 
    The graph structure (neighbors) is provided and never changes.
    All physical parameters (G_res, G_max, D_max, D_star, a_tilde,
    a, c, k) are randomized for each call, modelling different
    operating conditions on the same physical network:
      - Different weather → different G_res (solar/wind output)
      - Different seasons → different D_star (consumption habits)
      - Market changes   → different c (trading costs)
      - Equipment aging  → different G_max (flexibility capacity)
 
    Feasibility is guaranteed by the same constraints as
    init_parameters_graph (see that function for details).
 
    Parameters
    ----------
    neighbors : dict {n: list[int]} — fixed graph adjacency
    seed      : random seed for parameter generation
    device    : torch.device
 
    Returns
    -------
    (net, gamma, max_iter, tol, rho_admm)
    """
    rng = np.random.default_rng(seed)
    N   = len(neighbors)
 
    # ------------------------------------------------------------------
    # Flexibility (G_max) — sparse: 20-45% of non-root nodes
    # ------------------------------------------------------------------
    G_min = torch.zeros(N, dtype=torch.float32)
    G_max = torch.zeros(N, dtype=torch.float32)
 
    flex_prob = float(rng.uniform(0.2, 0.45))
    for n in range(1, N):
        if rng.random() < flex_prob:
            G_max[n] = float(rng.uniform(2., 8.))
 
    # ------------------------------------------------------------------
    # Renewable generation — every non-root node has some
    # ------------------------------------------------------------------
    G_res = torch.tensor(
        [0.0] + [float(rng.uniform(0.5, 5.0)) for _ in range(N - 1)],
        dtype=torch.float32
    )
 
    # ------------------------------------------------------------------
    # Link capacities and trading costs
    # ------------------------------------------------------------------
    c, k = {}, {}
    for n in range(N):
        for m in neighbors[n]:
            if m == 0 or n == 0:
                c_val = float(rng.uniform(0.3, 0.5))
                k_val = float(rng.uniform(15., 25.))
            else:
                c_val = float(rng.uniform(0.05, 0.3))
                k_val = float(rng.uniform(3., 10.))
            c[(n, m)] = torch.tensor(c_val, dtype=torch.float32)
            if (n, m) not in k:
                k[(n, m)] = torch.tensor(k_val, dtype=torch.float32)
                k[(m, n)] = torch.tensor(k_val, dtype=torch.float32)
 
    # ------------------------------------------------------------------
    # Import capacity per node
    # ------------------------------------------------------------------
    max_import = torch.zeros(N, dtype=torch.float32)
    for n in range(N):
        max_import[n] = sum(float(k[(m, n)].item()) for m in neighbors[n])
 
    # ------------------------------------------------------------------
    # Demand bounds — clipped for feasibility
    # ------------------------------------------------------------------
    D_min = torch.zeros(N, dtype=torch.float32)
    D_max_raw = torch.tensor(
        [float(rng.uniform(4., 18.)) for _ in range(N)],
        dtype=torch.float32
    )
    D_max_feasible = G_max + G_res + max_import
    D_max = torch.minimum(D_max_raw, D_max_feasible)
    D_max = torch.maximum(D_max, G_res)
    D_max[0] = 0.0
 
    # ------------------------------------------------------------------
    # Target demand — fraction of D_max
    # ------------------------------------------------------------------
    D_star_ratio = torch.tensor(
        [float(rng.uniform(0.3, 0.75)) for _ in range(N)],
        dtype=torch.float32
    )
    D_star = D_star_ratio * D_max
    D_star[0] = 0.0
 
    # ------------------------------------------------------------------
    # Cost parameters
    # ------------------------------------------------------------------
    a_tilde = torch.tensor(
        [0.0] + [float(rng.uniform(5., 15.)) for _ in range(N - 1)],
        dtype=torch.float32
    )
    a = torch.tensor(
        [1.0] + [float(rng.uniform(2., 6.)) for _ in range(N - 1)],
        dtype=torch.float32
    )
    b = torch.zeros(N, dtype=torch.float32)
 
    # ------------------------------------------------------------------
    # Root: cover network deficit
    # ------------------------------------------------------------------
    total_deficit = 0.0
    for n in range(1, N):
        local_gen = float(G_max[n]) + float(G_res[n])
        deficit_n = max(0.0, float(D_star[n]) - local_gen)
        total_deficit += deficit_n
 
    root_export_cap = sum(float(k[(0, m)].item()) for m in neighbors[0])
    G_max[0] = max(50.0, total_deficit * 1.5)
 
    if G_max[0] > root_export_cap:
        scale = float(G_max[0]) / max(root_export_cap, 1.0)
        for m in neighbors[0]:
            new_val = float(k[(0, m)].item()) * scale
            k[(0, m)] = torch.tensor(new_val, dtype=torch.float32)
            k[(m, 0)] = torch.tensor(new_val, dtype=torch.float32)
 
    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------
    for n in range(N):
        max_import_n = sum(float(k[(m, n)].item()) for m in neighbors[n])
        local_supply_max = float(G_max[n]) + float(G_res[n]) + max_import_n
        assert float(D_min[n]) <= local_supply_max, (
            f"Node {n}: D_min={D_min[n]:.2f} > max supply {local_supply_max:.2f}")
        assert float(D_star[n]) <= float(D_max[n]), (
            f"Node {n}: D_star={D_star[n]:.2f} > D_max={D_max[n]:.2f}")
 
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
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname if dirname else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graph saved to {save_path}")
    else:
        plt.show()
    plt.close()