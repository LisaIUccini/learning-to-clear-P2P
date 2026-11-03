"""Market instances derived from a base market (paper Sec. 5.1, App. C.1).

- level+width shifts: the levels of demand, flexible generation and RES are
  rescaled by U(1 - pct_level, 1 + pct_level) and the widths of the operating
  intervals by U(1 - pct_width, 1 + pct_width), with factors common to all
  prosumers (in-distribution: 10% / 4%; OOD: 20/30/40% with 8%);
- one prosumer more or less (OOD).

Every draw is seeded by (seed, slot i, attempt), so a dataset is a pure
function of its arguments.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from . import market
from .market import MarketConfig, MarketSpec


def perturb_level_width(base: MarketSpec, f_level_D: float, f_width_D: float,
                        f_level_G: float, f_width_G: float, f_gres: float) -> MarketSpec:
    """D* and the demand box move together under `f_level_D` (D*'s relative
    position inside the box is preserved); the box width is rescaled by
    `f_width_D`. The generation box moves (its centre) under `f_level_G` and
    its width under `f_width_G`; G_res is rescaled by `f_gres`. Graph and
    cost coefficients are unchanged."""
    width_D = base.D_max - base.D_min
    rel_pos_D = np.where(width_D > 1e-9, (base.D_star - base.D_min) / np.maximum(width_D, 1e-9), 0.5)
    D_star_new = base.D_star * f_level_D
    width_D_new = width_D * f_width_D
    D_min_new = D_star_new - rel_pos_D * width_D_new
    D_max_new = D_min_new + width_D_new

    G_center = (base.G_min + base.G_max) / 2.0 * f_level_G
    width_G_new = (base.G_max - base.G_min) * f_width_G
    G_min_new = G_center - width_G_new / 2.0
    G_max_new = G_center + width_G_new / 2.0

    G_res_new = base.G_res * f_gres
    return dataclasses.replace(base, D_star=D_star_new, G_res=G_res_new,
                                D_min=D_min_new, D_max=D_max_new,
                                G_min=G_min_new, G_max=G_max_new)


def draw_level_width(base: MarketSpec, seed: int, slot: int, attempt: int,
                     pct_level: float, pct_width: float) -> Optional[MarketSpec]:
    """One seeded level+width draw; None if it is infeasible or makes any of
    D*, G_res, D_min, D_max, G_min, G_max negative (the caller redraws)."""
    rng = np.random.default_rng(seed * 100_003 + slot * 1009 + attempt)
    f_level_D = float(rng.uniform(1 - pct_level, 1 + pct_level))
    f_width_D = float(rng.uniform(1 - pct_width, 1 + pct_width))
    f_level_G = float(rng.uniform(1 - pct_level, 1 + pct_level))
    f_width_G = float(rng.uniform(1 - pct_width, 1 + pct_width))
    f_gres = float(rng.uniform(1 - pct_level, 1 + pct_level))
    spec = perturb_level_width(base, f_level_D, f_width_D, f_level_G, f_width_G, f_gres)
    all_nonnegative = bool((spec.D_star >= 0).all() and (spec.G_res >= 0).all()
                           and (spec.D_min >= 0).all() and (spec.D_max >= 0).all()
                           and (spec.G_min >= 0).all() and (spec.G_max >= 0).all())
    if not (all_nonnegative and market.spec_is_feasible(spec)):
        return None
    return spec


def sample_level_width_dataset(base: MarketSpec, n_instances: int, seed: int,
                               pct_level: float, pct_width: float,
                               max_retries: int = 50) -> Tuple[List[MarketSpec], int]:
    """`[base] + (n_instances - 1)` feasible level+width draws (slot 0 is the
    unperturbed base market, as in every official train/val/test set), and
    the number of draws discarded on the way."""
    specs = [base]
    n_discarded = 0
    for i in range(n_instances - 1):
        for attempt in range(max_retries):
            spec = draw_level_width(base, seed, i, attempt, pct_level, pct_width)
            if spec is not None:
                specs.append(spec)
                break
            n_discarded += 1
        else:
            raise RuntimeError(f"instance {i}: no feasible draw found in {max_retries} retries "
                               f"at pct_level={pct_level}, pct_width={pct_width}")
    return specs, n_discarded


# ----------------------------------------------------------------------
# One prosumer more or less (OOD)
# ----------------------------------------------------------------------
def remove_one_node(spec: MarketSpec, removed: int) -> Optional[MarketSpec]:
    """`spec` without prosumer `removed` (nodes relabelled 0..N-2); None if
    the graph disconnects or the instance becomes infeasible."""
    keep = sorted(set(range(spec.N)) - {removed})
    remap = {old: new for new, old in enumerate(keep)}
    N_new = len(keep)
    neighbors_new = {remap[n]: sorted(remap[m] for m in spec.neighbors[n] if m != removed) for n in keep}
    if not market.is_connected(neighbors_new, N_new):
        return None
    idx = np.array(keep)
    c_new = {(remap[m], remap[n]): v for (m, n), v in spec.c.items() if m != removed and n != removed}
    kappa_new = {(remap[m], remap[n]): v for (m, n), v in spec.kappa.items() if m != removed and n != removed}
    new_spec = MarketSpec(
        N=N_new, neighbors=neighbors_new,
        D_min=spec.D_min[idx], D_max=spec.D_max[idx], D_star=spec.D_star[idx],
        G_min=spec.G_min[idx], G_max=spec.G_max[idx], G_res=spec.G_res[idx],
        a_tilde=spec.a_tilde[idx], a=spec.a[idx], b=spec.b[idx],
        c=c_new, kappa=kappa_new,
    )
    if not market.spec_is_feasible(new_spec):
        return None
    return new_spec


def add_one_node(spec: MarketSpec, rng: np.random.Generator, new_degree: int,
                 cfg: MarketConfig) -> Optional[MarketSpec]:
    """`spec` plus one prosumer (index N) drawn from `cfg`'s ranges, attached
    to `new_degree` random existing prosumers with fresh c/kappa per new
    link; None if the instance is infeasible."""
    N_old = spec.N
    new_degree = max(1, min(new_degree, N_old))
    attach_to = rng.choice(N_old, size=new_degree, replace=False).tolist()

    a_tilde = float(rng.uniform(*cfg.a_tilde_range))
    a = float(rng.uniform(*cfg.a_range))
    b = float(rng.uniform(*cfg.b_range))
    G_max = float(rng.uniform(*cfg.Gmax_range))
    G_min_raw = float(rng.uniform(*cfg.Gmin_range))
    G_min = min(G_min_raw, G_max - 1e-3)
    G_res = float(rng.uniform(*cfg.G_res_range))
    G0_frac = float(rng.uniform(0.2, 0.8))
    D_spread = float(rng.uniform(*cfg.D_spread_range))
    D_star_jitter = float(rng.uniform(-0.3, 0.3))
    G0 = G_min + G0_frac * (G_max - G_min)
    D0 = G0 + G_res
    D_min = max(0.0, D0 - D_spread)
    D_max = D0 + D_spread
    D_star = float(np.clip(D0 + D_star_jitter, D_min, D_max))

    neighbors_new = {n: list(v) for n, v in spec.neighbors.items()}
    neighbors_new[N_old] = sorted(attach_to)
    for m in attach_to:
        neighbors_new[m] = sorted(neighbors_new[m] + [N_old])

    c_new = dict(spec.c)
    kappa_new = dict(spec.kappa)
    for m in attach_to:
        c_new[(m, N_old)] = float(rng.uniform(*cfg.c_range))
        c_new[(N_old, m)] = float(rng.uniform(*cfg.c_range))
        cap = float(rng.uniform(*cfg.kappa_range))
        kappa_new[(m, N_old)] = cap
        kappa_new[(N_old, m)] = cap

    new_spec = MarketSpec(
        N=N_old + 1, neighbors=neighbors_new,
        D_min=np.append(spec.D_min, D_min), D_max=np.append(spec.D_max, D_max),
        D_star=np.append(spec.D_star, D_star),
        G_min=np.append(spec.G_min, G_min), G_max=np.append(spec.G_max, G_max),
        G_res=np.append(spec.G_res, G_res),
        a_tilde=np.append(spec.a_tilde, a_tilde), a=np.append(spec.a, a), b=np.append(spec.b, b),
        c=c_new, kappa=kappa_new,
    )
    if not market.spec_is_feasible(new_spec):
        return None
    return new_spec


def sample_one_node_changes(base: MarketSpec, cfg: MarketConfig, n_instances: int,
                            seed: int) -> Tuple[List[MarketSpec], List[str]]:
    """Add or remove one prosumer with equal probability (the new one gets as
    many neighbours as the base market's rounded mean degree); invalid draws
    are skipped. Returns (specs, operations) with operations in {"add", "remove"}."""
    degrees = [len(v) for v in base.neighbors.values()]
    mean_degree = round(sum(degrees) / len(degrees))
    rng = np.random.default_rng(seed)
    specs, ops = [], []
    attempts = 0
    while len(specs) < n_instances and attempts < n_instances * 20:
        attempts += 1
        if bool(rng.integers(0, 2)):
            spec = add_one_node(base, rng, mean_degree, cfg)
            op = "add"
        else:
            spec = remove_one_node(base, int(rng.integers(0, base.N)))
            op = "remove"
        if spec is None:
            continue
        specs.append(spec)
        ops.append(op)
    return specs, ops
