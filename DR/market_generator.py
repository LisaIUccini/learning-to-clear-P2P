"""
market_generator.py
====================

Random generation of P2P energy-market instances.

We split the process in two clean stages:

    1. sample_spec(N)  ->  MarketSpec      (pure numpy, no torch)
           - random connected, symmetric trading graph
           - per-node / per-edge parameters in sensible ranges
           - built to be feasible by construction (Assumption 1)

    2. build(spec)     ->  EnergyNetwork   (torch objects)

Keeping the spec torch-free makes instances easy to validate, save,
reload and reproduce (important for the training dataset later on).

Design choices worth remembering
--------------------------------
* Graph is undirected (m in Omega_n  <=>  n in Omega_m) and connected.
* Link capacity is symmetric on an edge: kappa_{mn} = kappa_{nm} (paper).
* Trading cost c_{nm} may be asymmetric (per directed edge).
* Feasibility: demand bounds are placed around an operating point
  (G0, D0 = G0 + Delta G) so that q ~ 0 is an interior feasible point.
* Generation coefficients satisfy  -b_n / a_n >= G_min_n  by construction
  (paper's regularity condition), which here implies G_min_n <= 0
  (G interpreted as *net* flexibility / storage).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class MarketConfig:
    """Ranges (inclusive) from which instance parameters are drawn."""

    # --- graph ---
    graph_model: str = "sparse"          # sparse | random | ring | watts_strogatz | erdos_renyi | grid
    density: float = 0.3                 # (sparse) edge density in [0,1]:
                                         #   0 -> connected spanning tree (sparsest)
                                         #   1 -> complete graph (densest)
    avg_degree: float = 3.0              # target mean degree (random / ER)
    ws_k: int = 4                        # ring-lattice degree (watts_strogatz)
    ws_beta: float = 0.2                 # rewiring prob (watts_strogatz)

    # --- demand / usage benefit ---
    a_tilde_range: Tuple[float, float] = (0.5, 1.5)   # usage curvature ~a_n
    D_spread_lo_range: Tuple[float, float] = (0.5, 2.0)  # D0 - D_min
    D_spread_hi_range: Tuple[float, float] = (0.5, 2.0)  # D_max - D0

    # --- generation cost  1/2 a G^2 + b G ---
    a_range: Tuple[float, float] = (0.5, 2.0)         # a_n > 0
    b_range: Tuple[float, float] = (0.2, 1.0)         # b_n > 0
    Gmin_margin_range: Tuple[float, float] = (0.1, 1.0)  # G_min = -b/a - margin
    Gmax_range: Tuple[float, float] = (2.0, 5.0)      # upper generation bound

    # --- operating point (feasibility anchor) ---
    G0_frac_range: Tuple[float, float] = (0.2, 0.7)   # G0 = frac * Gmax  (>= 0)
    G_res_range: Tuple[float, float] = (0.0, 2.0)     # renewable injection Delta G

    # --- trading ---
    kappa_range: Tuple[float, float] = (1.0, 4.0)     # link capacity (symmetric)
    c_range: Tuple[float, float] = (0.05, 0.30)       # per-unit trading cost


# ======================================================================
# Instance specification (pure numpy / python)
# ======================================================================
@dataclass
class MarketSpec:
    """A fully-specified market instance, torch-free."""
    N: int
    neighbors: Dict[int, List[int]]
    # per-node arrays, indexed by node id (shape (N,))
    D_min: np.ndarray
    D_max: np.ndarray
    D_star: np.ndarray
    G_min: np.ndarray
    G_max: np.ndarray
    G_res: np.ndarray
    a_tilde: np.ndarray
    a: np.ndarray
    b: np.ndarray
    # per-edge dicts keyed by (m, n) meaning q_{mn} (n's decision toward m)
    c: Dict[Tuple[int, int], float]
    kappa: Dict[Tuple[int, int], float]

    @property
    def edges_undirected(self) -> List[Tuple[int, int]]:
        return _undirected_pairs(self.neighbors)


@dataclass
class MarketPrimitives:
    """The independent random draws `MarketGenerator` derives a MarketSpec
    from -- the inputs to the feasibility-preserving formulas in
    `MarketGenerator._build_spec` (regularity condition -b/a >= G_min,
    D_star inside its box, the G+G_res/D balance anchor).

    Perturbing THESE (see `MarketGenerator.perturb_spec`) and re-running
    the SAME derivation formulas keeps every invariant intact by
    construction; perturbing the derived MarketSpec fields directly
    (D_min, G_min, D_star, ...) independently would not -- e.g. jittering
    `a`, `b` and `G_min` separately could break -b/a >= G_min, which the
    original draw satisfies only because G_min is DERIVED from a, b and a
    positive margin, not sampled independently.
    """
    a: np.ndarray
    b: np.ndarray
    Gmax: np.ndarray
    margin: np.ndarray            # >= 0; Gmin = -b/a - margin
    G0_frac: np.ndarray           # in (0,1); G0 = G0_frac * Gmax
    G_res: np.ndarray
    spread_lo: np.ndarray
    spread_hi: np.ndarray
    Dstar_jitter: np.ndarray      # D_star = clip(D0 + jitter, D_min, D_max)
    a_tilde: np.ndarray
    c: Dict[Tuple[int, int], float]
    kappa: Dict[Tuple[int, int], float]   # symmetric per undirected pair


def _undirected_pairs(neighbors: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """Each undirected pair (m, n) once, m < n order not enforced."""
    seen, out = set(), []
    for n in neighbors:
        for m in neighbors[n]:
            e = (min(m, n), max(m, n))
            if e not in seen:
                seen.add(e)
                out.append(e)
    return out


# ======================================================================
# Graph generation (numpy only, guaranteed connected + symmetric)
# ======================================================================
def _sample_sparse(N: int, density: float,
                   rng: np.random.Generator) -> Dict[int, set]:
    """Connected graph with a tunable edge density in [0, 1].

    The graph always starts from a random spanning tree (N-1 edges, the
    sparsest connected structure). `density` linearly interpolates the
    number of edges between that tree and the complete graph:

        n_edges = round( (N-1) + density * (N(N-1)/2 - (N-1)) )

    so density=0 gives the tree and density=1 gives the complete graph.
    """
    density = float(np.clip(density, 0.0, 1.0))
    adj = _spanning_tree(N, rng)                 # connected, N-1 edges
    min_edges = N - 1
    max_edges = N * (N - 1) // 2
    target = int(round(min_edges + density * (max_edges - min_edges)))
    target = min(max(target, min_edges), max_edges)

    current = sum(len(v) for v in adj.values()) // 2
    guard = 0
    while current < target and guard < 100 * max_edges:
        u, v = int(rng.integers(0, N)), int(rng.integers(0, N))
        guard += 1
        if u == v or v in adj[u]:
            continue
        adj[u].add(v); adj[v].add(u)
        current += 1
    return adj


def _spanning_tree(N: int, rng: np.random.Generator) -> Dict[int, set]:
    """Random spanning tree over N nodes -> guarantees connectivity."""
    adj: Dict[int, set] = {i: set() for i in range(N)}
    order = rng.permutation(N)
    for i in range(1, N):
        u = order[i]
        v = order[rng.integers(0, i)]      # attach to an earlier node
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj


def _add_edges_to_degree(adj: Dict[int, set], N: int, avg_degree: float,
                         rng: np.random.Generator) -> None:
    """Add random edges until the mean degree reaches (about) avg_degree."""
    target_edges = int(round(avg_degree * N / 2.0))
    current = sum(len(v) for v in adj.values()) // 2
    max_edges = N * (N - 1) // 2
    target_edges = min(target_edges, max_edges)
    guard = 0
    while current < target_edges and guard < 50 * max_edges:
        u, v = rng.integers(0, N), rng.integers(0, N)
        guard += 1
        if u == v or v in adj[u]:
            continue
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
        current += 1


def _grid_graph(N: int) -> Dict[int, set]:
    """4-neighbour grid; rows x cols as close to square as possible."""
    rows = int(np.floor(np.sqrt(N)))
    while rows > 1 and N % rows != 0:
        rows -= 1
    cols = N // rows
    adj: Dict[int, set] = {i: set() for i in range(N)}
    idx = lambda r, c: r * cols + c
    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            if c + 1 < cols:
                j = idx(r, c + 1); adj[i].add(j); adj[j].add(i)
            if r + 1 < rows:
                j = idx(r + 1, c); adj[i].add(j); adj[j].add(i)
    return adj


def _ring_graph(N: int, k: int = 2) -> Dict[int, set]:
    """k-nearest ring lattice (k even)."""
    adj: Dict[int, set] = {i: set() for i in range(N)}
    half = max(1, k // 2)
    for i in range(N):
        for d in range(1, half + 1):
            j = (i + d) % N
            adj[i].add(j); adj[j].add(i)
    return adj


def _watts_strogatz(N: int, k: int, beta: float,
                    rng: np.random.Generator) -> Dict[int, set]:
    """Watts-Strogatz small-world; reconnect to keep it connected."""
    adj = _ring_graph(N, k)
    for i in range(N):
        for j in list(adj[i]):
            if j <= i:
                continue
            if rng.random() < beta:
                adj[i].discard(j); adj[j].discard(i)
                choices = [w for w in range(N) if w != i and w not in adj[i]]
                if choices:
                    w = int(rng.choice(choices))
                    adj[i].add(w); adj[w].add(i)
                else:
                    adj[i].add(j); adj[j].add(i)   # revert
    _ensure_connected(adj, N, rng)
    return adj


def _is_connected(adj: Dict[int, set], N: int) -> bool:
    if N == 0:
        return True
    seen, dq = {0}, deque([0])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); dq.append(v)
    return len(seen) == N


def _ensure_connected(adj: Dict[int, set], N: int,
                      rng: np.random.Generator) -> None:
    """Link separate components until the graph is connected."""
    while not _is_connected(adj, N):
        seen, dq = {0}, deque([0])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); dq.append(v)
        outside = [w for w in range(N) if w not in seen]
        u = int(rng.choice(list(seen)))
        v = int(rng.choice(outside))
        adj[u].add(v); adj[v].add(u)


def _sample_graph(N: int, cfg: MarketConfig,
                  rng: np.random.Generator) -> Dict[int, List[int]]:
    if N < 2:
        raise ValueError("Need at least 2 nodes.")
    model = cfg.graph_model
    if model == "sparse":
        adj = _sample_sparse(N, cfg.density, rng)
    elif model == "random":
        adj = _spanning_tree(N, rng)
        _add_edges_to_degree(adj, N, cfg.avg_degree, rng)
    elif model == "erdos_renyi":
        adj = {i: set() for i in range(N)}
        p = min(1.0, cfg.avg_degree / (N - 1))
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < p:
                    adj[i].add(j); adj[j].add(i)
        _ensure_connected(adj, N, rng)
    elif model == "watts_strogatz":
        adj = _watts_strogatz(N, cfg.ws_k, cfg.ws_beta, rng)
    elif model == "ring":
        adj = _ring_graph(N, cfg.ws_k)
    elif model == "grid":
        adj = _grid_graph(N)
        _ensure_connected(adj, N, rng)
    else:
        raise ValueError(f"Unknown graph_model '{model}'.")

    return {i: sorted(adj[i]) for i in range(N)}


# ======================================================================
# Generator
# ======================================================================
class MarketGenerator:
    """Samples MarketSpec instances and builds EnergyNetwork objects."""

    def __init__(self, config: Optional[MarketConfig] = None):
        self.cfg = config if config is not None else MarketConfig()

    # --- stage 1: pure spec -------------------------------------------
    def _sample_primitives(self, N: int, neighbors: Dict[int, List[int]],
                           rng: np.random.Generator) -> MarketPrimitives:
        """Draw the independent random inputs a fresh MarketSpec is built
        from (see `MarketPrimitives`), each uniform over its `MarketConfig`
        range."""
        cfg = self.cfg

        def U(lohi):
            return rng.uniform(lohi[0], lohi[1], size=N)

        c: Dict[Tuple[int, int], float] = {}
        for n in range(N):
            for m in neighbors[n]:
                c[(m, n)] = float(rng.uniform(*cfg.c_range))
        kappa: Dict[Tuple[int, int], float] = {}
        for (i, j) in _undirected_pairs(neighbors):
            cap = float(rng.uniform(*cfg.kappa_range))   # symmetric per edge
            kappa[(i, j)] = cap
            kappa[(j, i)] = cap

        return MarketPrimitives(
            a=U(cfg.a_range), b=U(cfg.b_range), Gmax=U(cfg.Gmax_range),
            margin=U(cfg.Gmin_margin_range), G0_frac=U(cfg.G0_frac_range),
            G_res=U(cfg.G_res_range), spread_lo=U(cfg.D_spread_lo_range),
            spread_hi=U(cfg.D_spread_hi_range),
            Dstar_jitter=rng.uniform(-0.5, 0.5, size=N),
            a_tilde=U(cfg.a_tilde_range), c=c, kappa=kappa,
        )

    @staticmethod
    def _build_spec(N: int, neighbors: Dict[int, List[int]],
                    prim: MarketPrimitives) -> MarketSpec:
        """Derive a MarketSpec from `prim` -- the SAME formulas regardless
        of whether `prim` came from a fresh draw or from `perturb_spec`, so
        every feasibility invariant holds either way (see
        `MarketPrimitives`'s docstring)."""
        Gmin = -prim.b / prim.a - prim.margin           # ensures -b/a >= Gmin
        G0 = prim.G0_frac * prim.Gmax                   # in (0, Gmax)
        D0 = G0 + prim.G_res                             # feasible anchor
        D_min = np.maximum(0.0, D0 - prim.spread_lo)
        D_max = D0 + prim.spread_hi
        D_star = np.clip(D0 + prim.Dstar_jitter, D_min, D_max)
        return MarketSpec(
            N=N, neighbors=neighbors,
            D_min=D_min, D_max=D_max, D_star=D_star,
            G_min=Gmin, G_max=prim.Gmax, G_res=prim.G_res,
            a_tilde=prim.a_tilde, a=prim.a, b=prim.b, c=prim.c, kappa=prim.kappa,
        )

    def sample_spec(self, N: int, seed: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None,
                    neighbors: Optional[Dict[int, List[int]]] = None
                    ) -> MarketSpec:
        if rng is None:
            rng = np.random.default_rng(seed)
        cfg = self.cfg

        # `neighbors`, if given, fixes the graph and skips drawing a new one
        # -- used to build datasets where only parameters/initial state vary
        # across instances (controlled experiments, see DatasetConfig.fixed_graph)
        if neighbors is None:
            neighbors = _sample_graph(N, cfg, rng)

        prim = self._sample_primitives(N, neighbors, rng)
        return self._build_spec(N, neighbors, prim)

    def perturb_spec(self, ref_primitives: MarketPrimitives,
                     neighbors: Dict[int, List[int]], pct: float,
                     rng: np.random.Generator) -> MarketSpec:
        """A new MarketSpec whose PRIMITIVES are `ref_primitives`, each
        independently scaled by a factor drawn uniformly in
        [1-pct, 1+pct] -- e.g. pct=0.1 for +/-10%. Reuses `_build_spec`
        (the exact derivation `sample_spec` uses), so the regularity
        condition, the D_star/box containment, and the balance anchor all
        still hold, by construction -- not just approximately.

        `kappa` is perturbed once PER UNDIRECTED PAIR (not once per
        directed entry) so the result stays symmetric, matching how
        `_sample_primitives` draws it in the first place.
        """
        def jitter(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return x * rng.uniform(1.0 - pct, 1.0 + pct, size=x.shape)

        pairs = _undirected_pairs(neighbors)
        kappa: Dict[Tuple[int, int], float] = {}
        for (i, j) in pairs:
            cap = float(ref_primitives.kappa[(i, j)]
                       * rng.uniform(1.0 - pct, 1.0 + pct))
            kappa[(i, j)] = cap
            kappa[(j, i)] = cap
        c = {k: float(v * rng.uniform(1.0 - pct, 1.0 + pct))
            for k, v in ref_primitives.c.items()}

        prim = MarketPrimitives(
            a=jitter(ref_primitives.a), b=jitter(ref_primitives.b),
            Gmax=jitter(ref_primitives.Gmax), margin=jitter(ref_primitives.margin),
            G0_frac=np.clip(jitter(ref_primitives.G0_frac), 0.01, 0.99),
            G_res=jitter(ref_primitives.G_res),
            spread_lo=jitter(ref_primitives.spread_lo),
            spread_hi=jitter(ref_primitives.spread_hi),
            Dstar_jitter=jitter(ref_primitives.Dstar_jitter),
            a_tilde=jitter(ref_primitives.a_tilde), c=c, kappa=kappa,
        )
        N = len(ref_primitives.a)
        return self._build_spec(N, neighbors, prim)

    # --- stage 2: build torch objects ---------------------------------
    def build(self, spec: MarketSpec, device=None, init: bool = True,
              seed: Optional[int] = None):
        """Turn a MarketSpec into an EnergyNetwork (imports torch lazily)."""
        import torch
        from prosumer import Prosumer
        from energy_network import EnergyNetwork

        device = device if device is not None else torch.device("cpu")
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device); gen.manual_seed(seed)

        prosumers = []
        for n in range(spec.N):
            nb = spec.neighbors[n]
            c_n = {m: float(spec.c[(m, n)]) for m in nb}
            k_n = {m: float(spec.kappa[(m, n)]) for m in nb}
            p = Prosumer(
                node_id=n, neighbors=nb,
                D_min=float(spec.D_min[n]), D_max=float(spec.D_max[n]),
                G_min=float(spec.G_min[n]), G_max=float(spec.G_max[n]),
                G_res=float(spec.G_res[n]), a_tilde=float(spec.a_tilde[n]),
                a=float(spec.a[n]), b=float(spec.b[n]),
                D_star=float(spec.D_star[n]), c=c_n, k=k_n,
                device=device, init=False,
            )
            prosumers.append(p)

        net = EnergyNetwork(prosumers, spec.neighbors, device=device)
        if init:
            net.init_state(generator=gen)
        return net

    # --- convenience ---------------------------------------------------
    def sample(self, N: int, device=None, seed: Optional[int] = None):
        """Sample a spec and build the network in one call."""
        spec = self.sample_spec(N, seed=seed)
        return self.build(spec, device=device, seed=seed)


# ======================================================================
# Standalone feasibility check on a spec (numpy, no torch)
# ======================================================================
def spec_is_feasible(spec: MarketSpec) -> Dict[str, bool]:
    """Verify the spec admits an interior feasible point (q ~ 0 anchor).

    Checks, per node, that the box [G_min+G_res, G_max+G_res] overlaps the
    demand box (D_min, D_max), i.e. some (D, G) with q=0 satisfies balance;
    and the regularity condition -b/a >= G_min.
    """
    lo = spec.G_min + spec.G_res
    hi = spec.G_max + spec.G_res
    overlap = (hi > spec.D_min) & (lo < spec.D_max)         # interiors intersect
    reg = (-spec.b / spec.a) >= spec.G_min - 1e-9
    box_ok = (spec.D_min <= spec.D_max).all() and (spec.G_min <= spec.G_max).all()
    return {
        "balance_anchor": bool(overlap.all()),
        "regularity": bool(reg.all()),
        "boxes_ok": bool(box_ok),
        "D_star_in_box": bool(((spec.D_star >= spec.D_min - 1e-9) &
                               (spec.D_star <= spec.D_max + 1e-9)).all()),
    }


# ======================================================================
# Dataset of market instances (for training the learned perturbation)
# ======================================================================
@dataclass
class DatasetConfig:
    """How to draw a collection of market instances.

    `N` and `density` accept either a fixed value or a (min, max) range:

        N=10,        density=0.3          -> same size and density, but a
                                            different graph and different
                                            parameters in every instance
        N=(8, 40),   density=(0.1, 0.6)   -> size and density vary too
                                            (zero-shot generalisation)

    For a first training pass keep `N` and `density` fixed and rely on the
    seed to vary graph/parameters across instances; widen the ranges later
    once the pipeline is validated on a single size.

    `fixed_graph=True` draws ONE graph (using `N`/`density`/`graph_models`,
    resolved once from `seed`) and reuses it for every instance -- only the
    market parameters and the initial state differ. Useful as a controlled
    setting: it removes graph-topology variation as a confound, so any
    difference in convergence behaviour across instances can be attributed
    to parameters/initialisation alone. `N` and `density` must be fixed
    values (not ranges) in this mode.
    """
    n_instances: int = 64
    N: Union[int, Tuple[int, int]] = 10
    density: Union[float, Tuple[float, float]] = 0.3
    graph_models: Sequence[str] = ("sparse",)
    seed: int = 0
    fixed_graph: bool = False


class MarketDataset:
    """A reproducible collection of market instances.

    Only the torch-free `MarketSpec`s are stored; the torch objects are
    built on demand by `build`, so a large dataset stays cheap in memory.
    """

    def __init__(self, config: Optional[DatasetConfig] = None,
                 market_config: Optional[MarketConfig] = None):
        self.cfg = config if config is not None else DatasetConfig()
        self.market_config = (market_config if market_config is not None
                              else MarketConfig())
        self.specs: List[MarketSpec] = []
        self._draw()

    # ------------------------------------------------------------------
    def _draw(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)

        fixed_neighbors = None
        if cfg.fixed_graph:
            if isinstance(cfg.N, (tuple, list)) or isinstance(cfg.density, (tuple, list)):
                raise ValueError("fixed_graph=True requires fixed N and density "
                                "(not ranges), since one graph is shared by "
                                "every instance.")
            N0 = int(cfg.N)
            model0 = str(rng.choice(list(cfg.graph_models)))
            mc0 = replace(self.market_config, graph_model=model0, density=float(cfg.density))
            # a graph drawn once, independently of the per-instance streams below
            graph_rng = np.random.default_rng(cfg.seed)
            fixed_neighbors = _sample_graph(N0, mc0, graph_rng)

        for i in range(cfg.n_instances):
            if cfg.fixed_graph:
                N = int(cfg.N)
                dens = float(cfg.density)
                model = model0
            else:
                N = (int(rng.integers(cfg.N[0], cfg.N[1] + 1))
                    if isinstance(cfg.N, (tuple, list)) else int(cfg.N))
                dens = (float(rng.uniform(*cfg.density))
                        if isinstance(cfg.density, (tuple, list))
                        else float(cfg.density))
                model = str(rng.choice(list(cfg.graph_models)))

            mc = replace(self.market_config, graph_model=model, density=dens)
            gen = MarketGenerator(mc)
            # a distinct stream per instance keeps the dataset reproducible
            self.specs.append(gen.sample_spec(N, seed=int(cfg.seed) * 100003 + i,
                                              neighbors=fixed_neighbors))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, i: int) -> MarketSpec:
        return self.specs[i]

    def build(self, i: int, device=None, seed: Optional[int] = None):
        """Instantiate instance `i` as an EnergyNetwork with a random state."""
        gen = MarketGenerator(self.market_config)
        return gen.build(self.specs[i], device=device,
                         seed=(self.cfg.seed + i if seed is None else seed))

    def split(self, train_frac: float = 0.8):
        """Split into (train, test) datasets without redrawing anything."""
        n_tr = int(round(train_frac * len(self.specs)))
        a, b = MarketDataset.__new__(MarketDataset), MarketDataset.__new__(MarketDataset)
        for part, specs in ((a, self.specs[:n_tr]), (b, self.specs[n_tr:])):
            part.cfg = self.cfg
            part.market_config = self.market_config
            part.specs = list(specs)
        return a, b

    def summary(self) -> Dict[str, object]:
        sizes = [s.N for s in self.specs]
        edges = [len(s.edges_undirected) for s in self.specs]
        return {
            "instances": len(self.specs),
            "N": (min(sizes), max(sizes)),
            "undirected_edges": (min(edges), max(edges)),
            "graph_models": sorted({self.market_config.graph_model}
                                   | set(self.cfg.graph_models)),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"MarketDataset(instances={s['instances']}, "
                f"N={s['N']}, edges={s['undirected_edges']})")


# ======================================================================
# Clustered dataset: a few REFERENCE markets, each expanded by small
# (+/- pct) perturbations -- a more realistic alternative to the fully
# iid sampling `DatasetConfig` uses, where every instance's parameters are
# independent draws over the WHOLE MarketConfig range regardless of any
# other instance. Here every instance instead belongs to one of a handful
# of "clusters" (one physical grid, a handful of typical operating
# regimes, small day-to-day variation within each) -- closer to how a
# deployed market would actually vary from one clearing round to the next.
# ======================================================================
@dataclass
class ClusteredDatasetConfig:
    """See module-level note above. Train and test share the SAME
    `n_references` reference markets (graph fixed, drawn once) but use
    DISJOINT perturbed instances -- distinct rng streams, so no perturbed
    instance is ever shared between train and test, only the reference
    "cluster centre" is. This tests generalisation to new day-to-day
    variation on a market whose character the network has already seen,
    not generalisation to an unseen market character (see
    `MarketGenerator.perturb_spec` for how a cluster member is drawn).
    """
    n_references: int = 5
    n_train_per_ref: int = 10
    n_test_per_ref: int = 2
    pct: float = 0.10           # +/- fraction perturbation around each reference
    N: int = 10
    density: float = 0.7
    graph_model: str = "sparse"
    seed: int = 42


def build_clustered_datasets(
    cfg: ClusteredDatasetConfig, market_config: Optional[MarketConfig] = None
) -> Tuple[MarketDataset, MarketDataset, List[MarketSpec]]:
    """Build (train_ds, test_ds, reference_specs) per `ClusteredDatasetConfig`.

    ONE graph, drawn once and shared by every reference and every
    perturbed instance (train and test alike) -- same convention as
    `DatasetConfig(fixed_graph=True)`, so results are directly comparable
    to the existing fixed-graph experiments. `reference_specs` is returned
    too (not used downstream, but handy for diagnostics/plots that want to
    show "how far is this test instance from its own reference").
    """
    mc = replace(market_config if market_config is not None else MarketConfig(),
                graph_model=cfg.graph_model, density=cfg.density)
    gen = MarketGenerator(mc)

    graph_rng = np.random.default_rng(cfg.seed)
    neighbors = _sample_graph(cfg.N, mc, graph_rng)

    # distinct, non-overlapping rng streams: references, then train
    # perturbations, then test perturbations -- so "new perturbation,
    # same reference" for test is never a draw train could also have made.
    ref_rng = np.random.default_rng(cfg.seed + 1)
    train_rng = np.random.default_rng(cfg.seed + 1_000_001)
    test_rng = np.random.default_rng(cfg.seed + 2_000_001)

    references: List[MarketPrimitives] = []
    reference_specs: List[MarketSpec] = []
    for _ in range(cfg.n_references):
        prim = gen._sample_primitives(cfg.N, neighbors, ref_rng)
        references.append(prim)
        reference_specs.append(gen._build_spec(cfg.N, neighbors, prim))

    train_specs: List[MarketSpec] = []
    test_specs: List[MarketSpec] = []
    for prim in references:
        for _ in range(cfg.n_train_per_ref):
            train_specs.append(gen.perturb_spec(prim, neighbors, cfg.pct, train_rng))
        for _ in range(cfg.n_test_per_ref):
            test_specs.append(gen.perturb_spec(prim, neighbors, cfg.pct, test_rng))

    def _wrap(specs: List[MarketSpec], n_instances: int) -> MarketDataset:
        ds = MarketDataset.__new__(MarketDataset)
        ds.cfg = DatasetConfig(n_instances=n_instances, N=cfg.N, density=cfg.density,
                               graph_models=(cfg.graph_model,), seed=cfg.seed,
                               fixed_graph=True)
        ds.market_config = mc
        ds.specs = specs
        return ds

    train_ds = _wrap(train_specs, len(train_specs))
    test_ds = _wrap(test_specs, len(test_specs))
    return train_ds, test_ds, reference_specs