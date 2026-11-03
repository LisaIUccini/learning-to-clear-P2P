"""Random P2P energy-market instances (paper Sec. 2.1 and App. C.1).

    graph G = (N, E), connected, undirected
    prosumer n: D_n (flexible demand), G_n (controllable generation),
                q_mn for m in Omega_n (bilateral trade with neighbour m)
    fixed parameter: Delta G_n >= 0 (RES injection, `G_res`)

    X^loc_n:  D_min_n <= D_n <= D_max_n,  G_min_n <= G_n <= G_max_n,
              q_mn <= kappa_mn,  D_n = G_n + Delta G_n + sum_m q_mn

    J_n(x_n) = a~_n (D_n - D*_n)^2 + 1/2 a_n G_n^2 + b_n G_n + sum_m c_nm q_mn

Every instance is feasible by construction (Assumption 1): an anchor point
(G0, D0 = G0 + Delta G, q = 0) is drawn first and every box is placed around
it (`build_spec`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

Range = Tuple[float, float]


@dataclass
class MarketConfig:
    """Sampling ranges of a base market (paper Table 1). All cost
    coefficients are strictly positive, which makes every J_n convex."""

    density: float = 0.3            # 0 -> spanning tree, 1 -> complete graph

    a_tilde_range: Range = (0.5, 1.5)   # demand-comfort curvature
    a_range: Range = (0.5, 2.0)         # generation-cost curvature
    b_range: Range = (0.2, 1.0)         # generation-cost slope
    c_range: Range = (0.05, 0.18)       # per-unit trading cost; max(c) < min(b)
    kappa_range: Range = (1.0, 4.0)     # link capacity

    Gmax_range: Range = (2.0, 8.0)      # upper generation bound
    Gmin_range: Range = (0.0, 1.0)      # lower generation bound
    G_res_range: Range = (0.5, 2.0)     # Delta G_n (RES)
    D_spread_range: Range = (0.4, 1.0)  # half-width of the demand box around the anchor D0


@dataclass
class MarketSpec:
    """A fully specified instance (numpy only)."""

    N: int
    neighbors: Dict[int, List[int]]
    D_min: np.ndarray; D_max: np.ndarray; D_star: np.ndarray
    G_min: np.ndarray; G_max: np.ndarray; G_res: np.ndarray
    a_tilde: np.ndarray; a: np.ndarray; b: np.ndarray
    c: Dict[Tuple[int, int], float]        # keyed (m, n) -> c_nm
    kappa: Dict[Tuple[int, int], float]    # keyed (m, n) -> kappa_mn = kappa_nm


@dataclass
class _Primitives:
    """The independent random draws `_build_spec` derives a `MarketSpec` from."""

    a_tilde: np.ndarray; a: np.ndarray; b: np.ndarray
    G_max: np.ndarray
    G_min_raw: np.ndarray      # clipped to G_max - eps by _build_spec
    G_res: np.ndarray
    G0_frac: np.ndarray        # in (0,1): G0 = G_min + G0_frac * (G_max - G_min)
    D_spread: np.ndarray
    D_star_jitter: np.ndarray  # D_star = clip(D0 + jitter, D_min, D_max)
    c: Dict[Tuple[int, int], float]
    kappa: Dict[Tuple[int, int], float]


# ----------------------------------------------------------------------
# Graph: a random spanning tree plus extra edges up to `density`
# ----------------------------------------------------------------------
def _spanning_tree(N: int, rng: np.random.Generator) -> Dict[int, set]:
    adj = {i: set() for i in range(N)}
    order = rng.permutation(N)
    for i in range(1, N):
        u, v = int(order[i]), int(order[rng.integers(0, i)])
        adj[u].add(v); adj[v].add(u)
    return adj


def sample_graph(N: int, density: float, rng: np.random.Generator) -> Dict[int, List[int]]:
    """Connected undirected graph; `density` in [0,1] interpolates the edge
    count between a spanning tree (N-1 edges) and the complete graph."""
    adj = _spanning_tree(N, rng)
    min_e, max_e = N - 1, N * (N - 1) // 2
    target = int(round(min_e + np.clip(density, 0.0, 1.0) * (max_e - min_e)))
    cur = min_e
    guard = 0
    while cur < target and guard < 100 * max_e:
        u, v = int(rng.integers(0, N)), int(rng.integers(0, N))
        guard += 1
        if u == v or v in adj[u]:
            continue
        adj[u].add(v); adj[v].add(u)
        cur += 1
    return {i: sorted(adj[i]) for i in range(N)}


def is_connected(adj: Dict[int, List[int]], N: int) -> bool:
    seen, dq = {0}, deque([0])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); dq.append(v)
    return len(seen) == N


# ----------------------------------------------------------------------
# Primitives -> feasible spec
# ----------------------------------------------------------------------
def _sample_primitives(N: int, neighbors: Dict[int, List[int]], cfg: MarketConfig,
                       rng: np.random.Generator) -> _Primitives:
    def U(lo, hi, size=N):
        return rng.uniform(lo, hi, size=size)

    c: Dict[Tuple[int, int], float] = {}
    for n in range(N):
        for m in neighbors[n]:
            c[(m, n)] = float(rng.uniform(*cfg.c_range))

    kappa: Dict[Tuple[int, int], float] = {}
    seen_pairs = set()
    for n in range(N):
        for m in neighbors[n]:
            pair = (min(m, n), max(m, n))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            cap = float(rng.uniform(*cfg.kappa_range))
            kappa[(pair[0], pair[1])] = cap
            kappa[(pair[1], pair[0])] = cap

    return _Primitives(
        a_tilde=U(*cfg.a_tilde_range), a=U(*cfg.a_range), b=U(*cfg.b_range),
        G_max=U(*cfg.Gmax_range), G_min_raw=U(*cfg.Gmin_range), G_res=U(*cfg.G_res_range),
        G0_frac=U(0.2, 0.8), D_spread=U(*cfg.D_spread_range), D_star_jitter=U(-0.3, 0.3),
        c=c, kappa=kappa,
    )


def _build_spec(N: int, neighbors: Dict[int, List[int]], prim: _Primitives) -> MarketSpec:
    """Feasibility-preserving derivation: the q = 0 anchor lies inside every box."""
    G_max = prim.G_max
    G_min = np.minimum(prim.G_min_raw, G_max - 1e-3)
    G_res = prim.G_res
    G0 = G_min + prim.G0_frac * (G_max - G_min)          # anchor, in (G_min,G_max)
    D0 = G0 + G_res                                       # feasible anchor, q=0
    D_min = np.maximum(0.0, D0 - prim.D_spread)
    D_max = D0 + prim.D_spread
    D_star = np.clip(D0 + prim.D_star_jitter, D_min, D_max)
    return MarketSpec(N=N, neighbors=neighbors, D_min=D_min, D_max=D_max, D_star=D_star,
                       G_min=G_min, G_max=G_max, G_res=G_res, a_tilde=prim.a_tilde,
                       a=prim.a, b=prim.b, c=prim.c, kappa=prim.kappa)


def sample_base(N: int, cfg: MarketConfig, seed: int) -> MarketSpec:
    """The base market (one community) every dataset of a density is built from."""
    rng = np.random.default_rng(seed)
    neighbors = sample_graph(N, cfg.density, rng)
    prim = _sample_primitives(N, neighbors, cfg, rng)
    return _build_spec(N, neighbors, prim)


def spec_is_feasible(spec: MarketSpec) -> bool:
    """Is the q = 0 anchor inside every box (Assumption 1)?"""
    ok = True
    for n in range(spec.N):
        lo = spec.G_min[n] + spec.G_res[n]
        hi = spec.G_max[n] + spec.G_res[n]
        ok &= (hi >= spec.D_min[n] - 1e-9) and (lo <= spec.D_max[n] + 1e-9)
        ok &= spec.D_min[n] <= spec.D_max[n] and spec.G_min[n] <= spec.G_max[n]
    return bool(ok)


def build_network(spec: MarketSpec, device=None, rng: Optional[torch.Generator] = None):
    """MarketSpec -> `Network` (float32), with a random initial state x^0
    drawn inside every local box (`Network.random_init`).

    With `rng=None` the draw consumes torch's GLOBAL random state: several
    training/evaluation paths rely on that (their x^0 is overwritten, or the
    draw order is part of what makes a run reproducible), so keep it."""
    from .network import Network, Prosumer

    device = device or torch.device("cpu")

    def t(v):
        return torch.tensor(float(v), dtype=torch.float32, device=device)

    prosumers = []
    for n in range(spec.N):
        nb = spec.neighbors[n]
        c_n = {m: t(spec.c[(m, n)]) for m in nb}
        k_n = {m: t(spec.kappa[(m, n)]) for m in nb}
        prosumers.append(Prosumer(
            node_id=n, neighbors=nb, device=device,
            D_min=t(spec.D_min[n]), D_max=t(spec.D_max[n]),
            G_min=t(spec.G_min[n]), G_max=t(spec.G_max[n]),
            G_res=t(spec.G_res[n]), a_tilde=t(spec.a_tilde[n]),
            a=t(spec.a[n]), b=t(spec.b[n]), D_star=t(spec.D_star[n]),
            c=c_n, k=k_n,
        ))
    net = Network(prosumers, spec.neighbors, device=device)
    net.random_init(rng=rng)
    return net
