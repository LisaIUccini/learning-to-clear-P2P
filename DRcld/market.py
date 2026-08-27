"""
market.py
=========

Random P2P energy-market instances, per the model of the paper (Sec. II-A):

    graph G = (N, E), connected, undirected
    prosumer n: D_n (flexible demand), G_n (controllable generation),
                q_mn for m in Omega_n (bilateral trade with neighbour m)
    fixed parameter: Delta G_n >= 0 (RES injection)

    X^loc_n(theta_n):  D_min_n <= D_n <= D_max_n
                        G_min_n <= G_n <= G_max_n
                        q_mn <= kappa_mn                 for all m
                        D_n = G_n + Delta G_n + sum_m q_mn

    J_n(x_n) = a~_n (D_n-D*_n)^2 + 1/2 a_n G_n^2 + b_n G_n + sum_m c_nm q_mn
               (Eq. 3, additive constants dropped)

Assumption 1 only requires X(theta) = X^loc(theta) cap X^rec nonempty. Every
instance is built feasible by construction: pick an "anchor" point
(G0, D0 = G0 + Delta G) with q = 0, and place every box so the anchor lies
strictly inside it.

Two ways to get a dataset
--------------------------------------------------------------------------
`sample_dataset`     -- fully i.i.d.: every instance is an independent draw
                         over the whole `MarketConfig` range (graph, size,
                         parameters all vary freely). Maximally
                         heterogeneous; good for a hard generalisation test.

`sample_base` +
`sample_cluster_dataset` -- ONE reference market (graph + parameters),
                         expanded into many instances that are all small
                         (+/- `pct`) perturbations of it: node count and
                         topology jitter by about `pct`, and every surviving
                         parameter is jittered by +/- `pct`; genuinely new
                         nodes/edges (from growth) get a fresh draw, since
                         they have no reference value to jitter. This is a
                         much easier, more homogeneous learning target than
                         the fully i.i.d. dataset -- closer to "the same
                         physical grid on slightly different days" than to
                         "a different market every time".

Perturbing the PRIMITIVES (the independent random draws) and re-running the
SAME derivation formula (`build_spec`) -- rather than jittering D_min,
G_min, ... directly -- is what keeps every instance feasible by
construction, no matter how the primitives were perturbed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class MarketConfig:
    """Sampling ranges for a random instance. All costs strictly positive,
    which is what makes each J_n convex (a~_n, a_n > 0, Eq. 3)."""

    density: float = 0.3            # 0 -> spanning tree, 1 -> complete graph

    a_tilde_range: Tuple[float, float] = (0.5, 1.5)   # demand-benefit curvature
    a_range: Tuple[float, float] = (0.5, 2.0)         # generation-cost curvature
    b_range: Tuple[float, float] = (0.2, 1.0)         # generation-cost slope
    c_range: Tuple[float, float] = (0.05, 0.30)       # per-unit trading cost
    kappa_range: Tuple[float, float] = (1.0, 4.0)     # link capacity

    Gmax_range: Tuple[float, float] = (2.0, 5.0)      # upper generation bound
    Gmin_range: Tuple[float, float] = (-2.0, 0.0)     # lower generation bound
    G_res_range: Tuple[float, float] = (0.0, 2.0)     # Delta G_n (RES)
    D_spread_range: Tuple[float, float] = (0.5, 2.0)  # half-width of D's box
                                                       # around the anchor D0


@dataclass
class MarketSpec:
    """A fully-specified instance, numpy only (cheap to sample/inspect)."""

    N: int
    neighbors: Dict[int, List[int]]
    D_min: np.ndarray; D_max: np.ndarray; D_star: np.ndarray
    G_min: np.ndarray; G_max: np.ndarray; G_res: np.ndarray
    a_tilde: np.ndarray; a: np.ndarray; b: np.ndarray
    c: Dict[Tuple[int, int], float]        # keyed (m, n) -> c_nm
    kappa: Dict[Tuple[int, int], float]    # keyed (m, n) -> kappa_mn = kappa_nm


@dataclass
class MarketPrimitives:
    """The independent random draws `build_spec` derives a `MarketSpec`
    from. Perturbing THESE and re-running `build_spec` keeps every
    feasibility invariant intact by construction; perturbing D_min, G_min,
    D_star, ... directly would not (e.g. jittering a, b and G_min_raw
    separately could place the anchor G0 outside [G_min, G_max])."""

    a_tilde: np.ndarray; a: np.ndarray; b: np.ndarray
    G_max: np.ndarray
    G_min_raw: np.ndarray      # pre-clip; build_spec clips to G_max - eps
    G_res: np.ndarray
    G0_frac: np.ndarray        # in (0,1): G0 = G_min + G0_frac * (G_max-G_min)
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
    """Connected undirected graph; density in [0,1] interpolates the edge
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


def _is_connected(adj: Dict[int, List[int]], N: int) -> bool:
    seen, dq = {0}, deque([0])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); dq.append(v)
    return len(seen) == N


# ----------------------------------------------------------------------
# Primitives -> feasible spec (the ONE derivation, shared by fresh draws
# and perturbed ones)
# ----------------------------------------------------------------------
def sample_primitives(N: int, neighbors: Dict[int, List[int]], cfg: MarketConfig,
                       rng: np.random.Generator) -> MarketPrimitives:
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

    return MarketPrimitives(
        a_tilde=U(*cfg.a_tilde_range), a=U(*cfg.a_range), b=U(*cfg.b_range),
        G_max=U(*cfg.Gmax_range), G_min_raw=U(*cfg.Gmin_range), G_res=U(*cfg.G_res_range),
        G0_frac=U(0.2, 0.8), D_spread=U(*cfg.D_spread_range), D_star_jitter=U(-0.3, 0.3),
        c=c, kappa=kappa,
    )


def build_spec(N: int, neighbors: Dict[int, List[int]], prim: MarketPrimitives) -> MarketSpec:
    """The feasibility-preserving derivation (Assumption 1: X(theta) != empty),
    shared by fresh draws and perturbed ones alike."""
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


def sample_market(N: int, cfg: MarketConfig, seed: int) -> MarketSpec:
    rng = np.random.default_rng(seed)
    neighbors = sample_graph(N, cfg.density, rng)
    assert _is_connected(neighbors, N)
    prim = sample_primitives(N, neighbors, cfg, rng)
    return build_spec(N, neighbors, prim)


def spec_is_feasible(spec: MarketSpec) -> bool:
    """Check the q=0 anchor used to build `spec` is genuinely inside every
    box (Assumption 1: X(theta) nonempty)."""
    ok = True
    for n in range(spec.N):
        lo = spec.G_min[n] + spec.G_res[n]
        hi = spec.G_max[n] + spec.G_res[n]
        ok &= (hi >= spec.D_min[n] - 1e-9) and (lo <= spec.D_max[n] + 1e-9)
        ok &= spec.D_min[n] <= spec.D_max[n] and spec.G_min[n] <= spec.G_max[n]
    return bool(ok)


# ----------------------------------------------------------------------
# Perturbing a base instance: topology (+/- pct node count, some rewiring)
# and primitives (+/- pct on every value; fresh draw for anything genuinely
# new)
# ----------------------------------------------------------------------
def _bfs_prefix(neighbors: Dict[int, List[int]], k: int, start: int,
                 rng: np.random.Generator) -> List[int]:
    """First k nodes in BFS order from `start` -- always a CONNECTED
    induced subgraph on the original ids, since every node after the first
    is discovered via an edge from an already-included node."""
    seen, order, dq = {start}, [start], deque([start])
    while dq and len(order) < k:
        u = dq.popleft()
        for v in neighbors[u]:
            if v not in seen:
                seen.add(v); order.append(v); dq.append(v)
                if len(order) >= k:
                    break
    if len(order) < k:                       # guard; unreachable for connected input
        rest = [n for n in neighbors if n not in seen]
        rng.shuffle(rest)
        order += rest[: k - len(order)]
    return order[:k]


def _perturb_topology(neighbors0: Dict[int, List[int]], N0: int, pct: float,
                       rng: np.random.Generator
                       ) -> Tuple[int, Dict[int, List[int]], Dict[int, Optional[int]]]:
    """A new graph within about `pct` of the base one, in both node count
    and edge count. Returns (N_new, neighbors_new, origin), `origin[i]` the
    base node id node `i` came from, or None if `i` is genuinely new.

    Shrinking takes a random connected induced subgraph (BFS prefix, always
    connected); growing attaches each new node to a uniformly random
    existing node (also always connected). Either way, a further ~pct extra
    random edges are added (never removed) for topology variation at fixed
    size -- this can only help connectivity, never hurt it.
    """
    delta_max = max(1, round(pct * N0))
    dN = int(rng.integers(-delta_max, delta_max + 1))
    N_new = max(2, N0 + dN)

    if N_new <= N0:
        kept = _bfs_prefix(neighbors0, N_new, int(rng.integers(0, N0)), rng)
        old_to_new = {old: i for i, old in enumerate(kept)}
        kept_set = set(kept)
        nbrs = {old_to_new[old]: [old_to_new[m] for m in neighbors0[old] if m in kept_set]
                for old in kept}
        origin = {i: old for i, old in enumerate(kept)}
    else:
        nbrs = {old: list(neighbors0[old]) for old in range(N0)}
        origin = {i: i for i in range(N0)}
        for new_id in range(N0, N_new):
            anchor = int(rng.integers(0, new_id))
            nbrs[new_id] = [anchor]
            nbrs[anchor].append(new_id)
            origin[new_id] = None

    n_extra = max(0, round(pct * sum(len(v) for v in nbrs.values()) / 2.0))
    added, guard = 0, 0
    while added < n_extra and guard < 20 * max(n_extra, 1):
        guard += 1
        u, v = int(rng.integers(0, N_new)), int(rng.integers(0, N_new))
        if u == v or v in nbrs[u]:
            continue
        nbrs[u].append(v); nbrs[v].append(u)
        added += 1

    neighbors_new = {i: sorted(set(nbrs[i])) for i in range(N_new)}
    return N_new, neighbors_new, origin


def _perturb_topology_shared_factor(base_neighbors: Dict[int, List[int]], N0: int,
                                     factor: float, rng: np.random.Generator
                                     ) -> Tuple[int, Dict[int, List[int]], Dict[int, Optional[int]]]:
    """Topology counterpart of `sample_common_scale_dataset`'s single
    shared parameter factor: ONE already-drawn `factor` (e.g. 1 +/- pct)
    determines BOTH the new node count and the new edge count together,
    instead of `_perturb_topology`'s independent draws (a random node-
    count change, plus a SEPARATE extra-edge count that is a fixed
    function of pct, not tied to that same draw). Node changes use the
    same connectivity-preserving construction as `_perturb_topology`
    (BFS-prefix to shrink, random attachment to grow); edges are then
    added OR REMOVED to hit the target count, edge removal only ever
    dropping an edge if the graph stays connected without it.
    """
    N_new = max(2, round(N0 * factor))
    E0 = sum(len(v) for v in base_neighbors.values()) // 2
    E_target = max(N_new - 1, round(E0 * factor))

    if N_new <= N0:
        kept = _bfs_prefix(base_neighbors, N_new, int(rng.integers(0, N0)), rng)
        old_to_new = {old: i for i, old in enumerate(kept)}
        kept_set = set(kept)
        nbrs = {old_to_new[old]: [old_to_new[m] for m in base_neighbors[old] if m in kept_set]
                for old in kept}
        origin = {i: old for i, old in enumerate(kept)}
    else:
        nbrs = {old: list(base_neighbors[old]) for old in range(N0)}
        origin = {i: i for i in range(N0)}
        for new_id in range(N0, N_new):
            anchor = int(rng.integers(0, new_id))
            nbrs[new_id] = [anchor]
            nbrs[anchor].append(new_id)
            origin[new_id] = None

    current_edges = sum(len(v) for v in nbrs.values()) // 2

    guard = 0
    while current_edges < E_target and guard < 50 * max(E_target, 1):
        guard += 1
        u, v = int(rng.integers(0, N_new)), int(rng.integers(0, N_new))
        if u == v or v in nbrs[u]:
            continue
        nbrs[u].append(v); nbrs[v].append(u)
        current_edges += 1

    if current_edges > E_target:
        candidates = [(u, v) for u in range(N_new) for v in nbrs[u] if v > u]
        rng.shuffle(candidates)
        for (u, v) in candidates:
            if current_edges <= E_target:
                break
            nbrs[u].remove(v); nbrs[v].remove(u)
            if _is_connected(nbrs, N_new):
                current_edges -= 1
            else:
                nbrs[u].append(v); nbrs[v].append(u)   # revert, would disconnect

    neighbors_new = {i: sorted(set(nbrs[i])) for i in range(N_new)}
    return N_new, neighbors_new, origin


def sample_common_scale_topology_dataset(base_neighbors: Dict[int, List[int]],
                                          base_prim: MarketPrimitives, cfg: MarketConfig,
                                          n_instances: int, pct: float, seed: int
                                          ) -> List[MarketSpec]:
    """The topology-only counterpart of `sample_common_scale_dataset`:
    market parameters stay EXACTLY fixed (for every node/edge inherited
    from the base -- no jitter at all, unlike `sample_cluster_dataset`)
    -- only the GRAPH varies, via one shared factor in [1-pct,1+pct] per
    instance applied to node count and edge count together
    (`_perturb_topology_shared_factor`). Genuinely new nodes/edges
    (possible when the graph grows) have no base value to keep fixed, so
    they get a fresh draw from `cfg`'s ranges -- reuses `_perturb_primitives`
    with pct=0.0, which makes every INHERITED value an exact copy
    (`x * uniform(1,1) == x`) while still drawing fresh values for
    anything genuinely new.
    """
    N0 = len(base_neighbors)
    specs = []
    for i in range(n_instances):
        rng = np.random.default_rng(seed * 100_003 + i)
        factor = 1.0 + float(rng.uniform(-pct, pct))
        N_new, neighbors_new, origin = _perturb_topology_shared_factor(base_neighbors, N0, factor, rng)
        prim_new = _perturb_primitives(base_neighbors, base_prim, N_new, neighbors_new,
                                        origin, cfg, 0.0, rng)
        specs.append(build_spec(N_new, neighbors_new, prim_new))
    return specs


def _perturb_primitives(base_neighbors: Dict[int, List[int]], prim0: MarketPrimitives,
                         N_new: int, neighbors_new: Dict[int, List[int]],
                         origin: Dict[int, Optional[int]], cfg: MarketConfig,
                         pct: float, rng: np.random.Generator,
                         factor: Optional[float] = None) -> MarketPrimitives:
    """If `factor` is given, every inherited value is scaled by that SAME
    pre-drawn factor (the shared-factor mode, `pct` is then ignored) --
    used by `sample_common_scale_combined_dataset` to apply one coherent
    parameter-wide shift, the primitives analogue of
    `_perturb_topology_shared_factor`. Otherwise (`factor=None`, the
    default) every value draws its OWN independent factor in
    [1-pct, 1+pct], as used by `sample_cluster_dataset`/`sample_param_only_dataset`."""
    def jitter(x: float) -> float:
        f = factor if factor is not None else rng.uniform(1.0 - pct, 1.0 + pct)
        return float(x * f)

    fields = ["a_tilde", "a", "b", "G_max", "G_min_raw", "G_res", "G0_frac", "D_spread",
              "D_star_jitter"]
    out = {f: np.empty(N_new) for f in fields}

    fresh_cache: Dict[int, MarketPrimitives] = {}
    for i in range(N_new):
        old = origin.get(i)
        if old is not None:
            for f in fields:
                out[f][i] = jitter(float(getattr(prim0, f)[old]))
            out["G0_frac"][i] = float(np.clip(out["G0_frac"][i], 0.01, 0.99))
        else:
            fresh = sample_primitives(1, {0: []}, cfg, rng)
            for f in fields:
                out[f][i] = float(getattr(fresh, f)[0])

    base_edge = {o: set(base_neighbors[o]) for o in range(len(base_neighbors))}

    def inherited(n: int, m: int) -> bool:
        on, om = origin.get(n), origin.get(m)
        return on is not None and om is not None and om in base_edge[on]

    c_new: Dict[Tuple[int, int], float] = {}
    for n in range(N_new):
        for m in neighbors_new[n]:
            if inherited(n, m) and (origin[m], origin[n]) in prim0.c:
                c_new[(m, n)] = jitter(prim0.c[(origin[m], origin[n])])
            else:
                c_new[(m, n)] = float(rng.uniform(*cfg.c_range))

    kappa_new: Dict[Tuple[int, int], float] = {}
    seen_pairs = set()
    for n in range(N_new):
        for m in neighbors_new[n]:
            pair = (min(m, n), max(m, n))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            base_pair = ((min(origin[n], origin[m]), max(origin[n], origin[m]))
                         if inherited(n, m) else None)
            if base_pair is not None and base_pair in prim0.kappa:
                cap = jitter(prim0.kappa[base_pair])
            else:
                cap = float(rng.uniform(*cfg.kappa_range))
            kappa_new[(pair[0], pair[1])] = cap
            kappa_new[(pair[1], pair[0])] = cap

    return MarketPrimitives(a_tilde=out["a_tilde"], a=out["a"], b=out["b"],
                             G_max=out["G_max"], G_min_raw=out["G_min_raw"],
                             G_res=out["G_res"], G0_frac=out["G0_frac"],
                             D_spread=out["D_spread"], D_star_jitter=out["D_star_jitter"],
                             c=c_new, kappa=kappa_new)


def sample_base(N0: int, cfg: MarketConfig, seed: int
                ) -> Tuple[Dict[int, List[int]], MarketPrimitives, MarketSpec]:
    """The single reference market a cluster is built around."""
    rng = np.random.default_rng(seed)
    neighbors = sample_graph(N0, cfg.density, rng)
    prim = sample_primitives(N0, neighbors, cfg, rng)
    return neighbors, prim, build_spec(N0, neighbors, prim)


def sample_cluster_dataset(base_neighbors: Dict[int, List[int]], base_prim: MarketPrimitives,
                            cfg: MarketConfig, n_instances: int, pct: float, seed: int
                            ) -> List[MarketSpec]:
    """`n_instances` instances, each independently perturbed from the SAME
    base by about `pct` in node count, topology, and every parameter (see
    module docstring). Two calls with different `seed`s (same base) draw
    disjoint perturbations -- used for train/test on the same market
    character without ever sharing an exact instance."""
    N0 = len(base_neighbors)
    specs = []
    for i in range(n_instances):
        rng = np.random.default_rng(seed * 100_003 + i)
        N_new, neighbors_new, origin = _perturb_topology(base_neighbors, N0, pct, rng)
        prim_new = _perturb_primitives(base_neighbors, base_prim, N_new, neighbors_new,
                                        origin, cfg, pct, rng)
        specs.append(build_spec(N_new, neighbors_new, prim_new))
    return specs


def sample_param_only_dataset(base_neighbors: Dict[int, List[int]], base_prim: MarketPrimitives,
                               cfg: MarketConfig, n_instances: int, pct: float, seed: int
                               ) -> List[MarketSpec]:
    """Like `sample_cluster_dataset`, but the graph is left EXACTLY fixed
    (no topology perturbation at all) -- only primitive jitter is applied,
    isolating parameter diversity from topology diversity. EVERY individual
    parameter of EVERY prosumer/edge gets its OWN independent +/-pct draw
    (reuses `_perturb_primitives` with an identity origin map, so nothing
    is topologically new). Measured result: this is a much LARGER effective
    perturbation of the market's aggregate character than "each parameter
    within pct" suggests, because ~100+ independent draws rarely cancel out
    -- see runs/param_variation_independent_per_param (42% win-rate, worse
    than the fully diverse cluster dataset). `sample_common_scale_dataset`
    below is the more restrained alternative: one shared factor for the
    WHOLE parameter set, not one per parameter."""
    N0 = len(base_neighbors)
    identity_origin = {i: i for i in range(N0)}
    specs = []
    for i in range(n_instances):
        rng = np.random.default_rng(seed * 100_003 + i)
        prim_new = _perturb_primitives(base_neighbors, base_prim, N0, base_neighbors,
                                        identity_origin, cfg, pct, rng)
        specs.append(build_spec(N0, base_neighbors, prim_new))
    return specs


def sample_common_scale_dataset(base_neighbors: Dict[int, List[int]], base_prim: MarketPrimitives,
                                 n_instances: int, pct: float, seed: int) -> List[MarketSpec]:
    """The graph is EXACTLY fixed, same as `sample_param_only_dataset` --
    but here ONE shared scale factor (1 +/- pct, a single draw) is applied
    to the WHOLE parameter set together, per instance, instead of drawing
    an independent factor for every individual parameter. The entire
    market shifts coherently -- "the same market, scaled up or down by up
    to pct" -- rather than each of its ~100+ numbers moving independently
    within pct (which is a fundamentally different, much larger-in-
    aggregate kind of variation; see `sample_param_only_dataset`'s note).
    """
    N0 = len(base_neighbors)
    specs = []
    for i in range(n_instances):
        rng = np.random.default_rng(seed * 100_003 + i)
        factor = 1.0 + float(rng.uniform(-pct, pct))
        prim_new = MarketPrimitives(
            a_tilde=base_prim.a_tilde * factor,
            a=base_prim.a * factor,
            b=base_prim.b * factor,
            G_max=base_prim.G_max * factor,
            G_min_raw=base_prim.G_min_raw * factor,
            G_res=base_prim.G_res * factor,
            G0_frac=np.clip(base_prim.G0_frac * factor, 0.01, 0.99),
            D_spread=base_prim.D_spread * factor,
            D_star_jitter=base_prim.D_star_jitter * factor,
            c={k: v * factor for k, v in base_prim.c.items()},
            kappa={k: v * factor for k, v in base_prim.kappa.items()},
        )
        specs.append(build_spec(N0, base_neighbors, prim_new))
    return specs


def sample_common_scale_combined_dataset(base_neighbors: Dict[int, List[int]],
                                          base_prim: MarketPrimitives, cfg: MarketConfig,
                                          n_instances: int, pct: float, seed: int
                                          ) -> List[MarketSpec]:
    """Combines `sample_common_scale_dataset` (shared parameter factor) and
    `sample_common_scale_topology_dataset` (shared topology factor): BOTH
    axes vary together, per instance -- but via TWO separately-drawn
    shared factors (one for topology, one for parameters), not one factor
    driving both. Topology and parameter scale are conceptually different
    kinds of shift (structural vs. numeric), so coupling them to the same
    draw would be an arbitrary correlation, not a real constraint of the
    problem. Genuinely new nodes/edges (from topology growth) have no base
    value to scale and draw fresh from `cfg`'s ranges, same as
    `sample_common_scale_topology_dataset`.
    """
    N0 = len(base_neighbors)
    specs = []
    for i in range(n_instances):
        rng = np.random.default_rng(seed * 100_003 + i)
        topo_factor = 1.0 + float(rng.uniform(-pct, pct))
        param_factor = 1.0 + float(rng.uniform(-pct, pct))
        N_new, neighbors_new, origin = _perturb_topology_shared_factor(base_neighbors, N0, topo_factor, rng)
        prim_new = _perturb_primitives(base_neighbors, base_prim, N_new, neighbors_new,
                                        origin, cfg, 0.0, rng, factor=param_factor)
        specs.append(build_spec(N_new, neighbors_new, prim_new))
    return specs


# ----------------------------------------------------------------------
# Build torch objects
# ----------------------------------------------------------------------
def build_network(spec: MarketSpec, device=None, rng: torch.Generator = None):
    """MarketSpec -> network.Network, with a random initial state x^0 drawn
    inside every local box (see network.Network.random_init)."""
    from network import Network, Prosumer

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


def sample_dataset(n_instances: int, N_range: Tuple[int, int], cfg: MarketConfig,
                    seed: int) -> List[MarketSpec]:
    """`n_instances` fully independent specs, sizes drawn uniformly in
    `N_range` (inclusive) -- maximal heterogeneity, used as a harder,
    out-of-cluster generalisation test alongside `sample_cluster_dataset`."""
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n_instances):
        N = int(rng.integers(N_range[0], N_range[1] + 1))
        specs.append(sample_market(N, cfg, seed=seed * 100_003 + i))
    return specs
