"""
build_realistic_training_dataset.py
======================================

Training dataset for the "train directly on parameter diversity, fixed
x0 and topology" design (agreed with the user's supervisor): a NEW base
market at density=1.0 (SAME graph as every other experiment this
project -- `sample_graph`'s draw only depends on `density`/N/seed, not
on the primitive ranges below, so changing the ranges leaves the graph
identical), but with realistic, literature-grounded parameter ranges
(household-prosumer scale, kW) instead of the original abstract
defaults -- see the auto-memory project notes for the literature this
is grounded in (Belgioioso et al., arXiv:2107.13444, and residential
demand/PV-capacity figures). Crucially, `Gmin_range` is now (0.0, 1.0)
instead of the original (-2.0, 0.0): dispatchable generation is
non-negative by definition in the reference paper (p_di >= 0), and the
supervisor asked for ALL SIX of D*, G_res, D_min, D_max, G_min, G_max to
stay >= 0, which the old negative-G_min default violated.

Each training instance is a DIRECT multiplicative perturbation of the
base market's own SIX quantities (D_star, G_res, D_min, D_max, G_min,
G_max) -- NOT a re-derivation through the primitives/anchor pipeline --
by SIX independent factors, ONE PER QUANTITY, SHARED across all N nodes
of that instance (matching the established `build_dstar_gres_dataset`
convention): D*/G_res (target/forecast-like quantities) get the wider
+/-15% factor, the four box bounds (more stable, physical/contractual)
get the narrower +/-10% factor -- originally +/-25% for D*/G_res, but
that made D* (whose OWN box only moves by +/-10%) clip to its box edge
in 45.6% of node-instances (a 200-draw check), diluting the intended
diversity into a pile-up at the boundary; +/-15% cut that down a lot
while keeping D*/G_res meaningfully more variable than the bounds.
Graph, costs (a_tilde/a/b/c), and
capacities (kappa) stay EXACTLY fixed. x0 is NOT part of this dataset at
all -- the training script deploys a single SHARED fixed x0 across every
instance (see the caller), unlike every prior experiment in this
project where x0 was the (sole) source of variation.

Because perturbing D_min/D_max/G_min/G_max independently CAN break
Assumption-1 feasibility (unlike the primitives pipeline's built-in
guarantee), every instance is feasibility-checked; D_star is re-clipped
into its own [D_min,D_max] after perturbation (it must always lie inside
its box by construction), and infeasible draws are counted, not silently
kept.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import dataclasses

import numpy as np

import config
import market

# realistic household-prosumer-scale ranges (kW), grounded in residential
# demand/PV-capacity literature -- see auto-memory project notes
REALISTIC_MARKET_CONFIG = market.MarketConfig(
    density=1.0,
    Gmax_range=(2.0, 8.0),     # residential PV capacity, 1-10 kWp typical
    Gmin_range=(0.0, 1.0),     # >=0: dispatchable generation is non-negative
    G_res_range=(0.5, 2.0),    # reserved/non-dispatchable (e.g. baseline PV forecast)
    D_spread_range=(0.4, 1.0),  # demand flexibility: ~15-35% of typical D0
)


def build_realistic_base_market(base_seed: int = 0, density: float = 1.0):
    """The new base market: SAME graph as every other density=1.0 seed=0
    experiment in this project (when density=1.0, the default), but
    realistic-range primitives. `density` overrides REALISTIC_MARKET_CONFIG's
    own density field (e.g. 0.8/0.5/0.3, matching the project's original
    cross-density campaign) while keeping every other field (Gmax/Gmin/
    G_res/D_spread ranges) identical."""
    mc = REALISTIC_MARKET_CONFIG if density == 1.0 else dataclasses.replace(REALISTIC_MARKET_CONFIG, density=density)
    return market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)


def build_realistic_base_market_regular(base_seed: int = 0, degree: int = 8):
    """Like `build_realistic_base_market`, but the topology is k-REGULAR
    (every node has exactly `degree` neighbors, zero degree heterogeneity)
    instead of density-driven -- isolates degree heterogeneity as the
    experimental variable (see `market.sample_regular_graph`)."""
    return market.sample_base_regular(config.BASE_MARKET_N, REALISTIC_MARKET_CONFIG, degree=degree, seed=base_seed)


def perturb_six_direct(base_spec: market.MarketSpec, f_dstar: float, f_gres: float,
                        f_dmin: float, f_dmax: float, f_gmin: float, f_gmax: float
                        ) -> market.MarketSpec:
    """Multiply each of the six quantities by its OWN factor (shared across
    all nodes), independently of the others. D_star is re-clipped into its
    own [D_min,D_max] afterward (it must lie inside its box by definition);
    everything else (graph, a_tilde/a/b/c, kappa) is untouched."""
    D_min_new = base_spec.D_min * f_dmin
    D_max_new = base_spec.D_max * f_dmax
    G_min_new = base_spec.G_min * f_gmin
    G_max_new = base_spec.G_max * f_gmax
    D_star_new = np.clip(base_spec.D_star * f_dstar, D_min_new, D_max_new)
    G_res_new = base_spec.G_res * f_gres
    return dataclasses.replace(base_spec, D_star=D_star_new, G_res=G_res_new,
                                D_min=D_min_new, D_max=D_max_new,
                                G_min=G_min_new, G_max=G_max_new)


def sample_six_perturbed_dataset(base_spec: market.MarketSpec, n_instances: int,
                                  seed: int, pct: float = 0.20, max_retries: int = 50):
    """n_instances perturbed specs (ONE uniform +/-pct factor for all six
    quantities, independently drawn -- simpler than the earlier split
    target/bounds percentages) + per-instance factor log + count of
    infeasible draws DISCARDED along the way (not returned -- each slot is
    resampled, with a fresh factor draw, until feasible or `max_retries` is
    hit, so the returned list is always exactly `n_instances` long and
    always feasible)."""
    specs, log_rows = [], []
    n_infeasible = 0
    for i in range(n_instances):
        for attempt in range(max_retries):
            rng = np.random.default_rng(seed * 100_003 + i * 1009 + attempt)
            f_dstar = float(rng.uniform(1 - pct, 1 + pct))
            f_gres = float(rng.uniform(1 - pct, 1 + pct))
            f_dmin = float(rng.uniform(1 - pct, 1 + pct))
            f_dmax = float(rng.uniform(1 - pct, 1 + pct))
            f_gmin = float(rng.uniform(1 - pct, 1 + pct))
            f_gmax = float(rng.uniform(1 - pct, 1 + pct))
            spec = perturb_six_direct(base_spec, f_dstar, f_gres, f_dmin, f_dmax, f_gmin, f_gmax)
            ok = market.spec_is_feasible(spec)
            if not ok:
                n_infeasible += 1
                continue
            specs.append(spec)
            log_rows.append((i, f_dstar, f_gres, f_dmin, f_dmax, f_gmin, f_gmax, ok))
            break
        else:
            raise RuntimeError(f"instance {i}: no feasible draw found in {max_retries} retries at pct={pct}")
    return specs, log_rows, n_infeasible


# ======================================================================
# "level + width" perturbation -- an economically-motivated alternative to
# perturbing D_min/D_max (and G_min/G_max) fully independently.
# ======================================================================
#
# `perturb_six_direct` above multiplies D_min and D_max by two INDEPENDENT
# factors. Nothing stops them moving toward each other: on a 75-instance
# training set this let the D-box collapse from a base width of 0.932 down
# to 0.085 for one node (an 11x squeeze), while the single FIXED x0 (shared
# across every instance, per the supervisor's design) stayed put -- landing
# x0 up to ~9.7 box-half-widths outside its own box for that instance
# (measured: 72% of the 75 training instances had x0 outside at least one
# node's box at all). `h_D`/`h_G` (gnn.py) are documented as always lying in
# [-1,1] -- true in every other experiment (x0 was always FRESHLY drawn
# inside whatever box was active), silently violated here for the first
# time. This is a real, noisy, hard-to-learn-from source of "heterogeneity"
# masquerading as market-parameter diversity, not a clean signal.
#
# Fix: separate two economically DIFFERENT kinds of variation instead of
# perturbing D_min/D_max (or G_min/G_max) independently:
#   - LEVEL (pct_level, wide, default 0.20): how much demand/generation
#     shifts overall (weather, season) -- moves D_star (and the D-box with
#     it, preserving D*'s relative position inside the box) / the G-box's
#     CENTER together, and G_res independently (it has no box of its own).
#   - WIDTH (pct_width, narrow, default 0.08): flexibility/capacity RANGE,
#     a more stable, contractual/physical property -- jitters the D-box's
#     and G-box's WIDTH only, always by a strictly positive factor.
# The box can never collapse or invert by construction (width is perturbed
# directly, never as the difference of two independently-moving numbers),
# so this also removes almost all of the earlier box-collapse pathology at
# the source, rather than papering over its symptom in the GNN's features.
def perturb_six_level_width(base_spec: market.MarketSpec, f_level_D: float, f_width_D: float,
                             f_level_G: float, f_width_G: float, f_gres: float
                             ) -> market.MarketSpec:
    """D* and the D-box move TOGETHER under `f_level_D` (D*'s relative
    position inside [D_min,D_max] is preserved exactly, so no re-clipping
    is needed -- unlike `perturb_six_direct`); the D-box's WIDTH is jittered
    separately by `f_width_D`. G_min/G_max move together (their CENTRE) under
    `f_level_G`, with the G-box's WIDTH jittered separately by `f_width_G`.
    G_res (no box of its own) is perturbed independently by `f_gres`, as
    before. Everything else (graph, a_tilde/a/b/c, kappa) untouched."""
    width_D = base_spec.D_max - base_spec.D_min
    rel_pos_D = np.where(width_D > 1e-9, (base_spec.D_star - base_spec.D_min) / np.maximum(width_D, 1e-9), 0.5)
    D_star_new = base_spec.D_star * f_level_D
    width_D_new = width_D * f_width_D
    D_min_new = D_star_new - rel_pos_D * width_D_new
    D_max_new = D_min_new + width_D_new

    G_center = (base_spec.G_min + base_spec.G_max) / 2.0 * f_level_G
    width_G_new = (base_spec.G_max - base_spec.G_min) * f_width_G
    G_min_new = G_center - width_G_new / 2.0
    G_max_new = G_center + width_G_new / 2.0

    G_res_new = base_spec.G_res * f_gres
    return dataclasses.replace(base_spec, D_star=D_star_new, G_res=G_res_new,
                                D_min=D_min_new, D_max=D_max_new,
                                G_min=G_min_new, G_max=G_max_new)


def sample_level_width_perturbed_dataset(base_spec: market.MarketSpec, n_instances: int,
                                          seed: int, pct_level: float = 0.20,
                                          pct_width: float = 0.08, max_retries: int = 50):
    """Like `sample_six_perturbed_dataset`, but using `perturb_six_level_width`
    (5 independent factors instead of 6: D* and the D-box share one level
    factor, G_min/G_max share one level factor, each box gets its own
    narrower width factor, G_res is independent) -- see that function's
    docstring for why. Also enforces all six quantities >=0 (the
    supervisor's requirement; `perturb_six_direct`'s multiplicative-of-a-
    non-negative-base scheme made this automatic, but the level+width
    reconstruction of D_min/G_min does not, since a small centre minus half
    a jittered width CAN go negative). Same discard+resample contract as
    `sample_six_perturbed_dataset`: returns exactly `n_instances` feasible,
    all-non-negative specs, plus a factor log and the count of draws
    discarded along the way."""
    specs, log_rows = [], []
    n_infeasible = 0
    for i in range(n_instances):
        for attempt in range(max_retries):
            rng = np.random.default_rng(seed * 100_003 + i * 1009 + attempt)
            f_level_D = float(rng.uniform(1 - pct_level, 1 + pct_level))
            f_width_D = float(rng.uniform(1 - pct_width, 1 + pct_width))
            f_level_G = float(rng.uniform(1 - pct_level, 1 + pct_level))
            f_width_G = float(rng.uniform(1 - pct_width, 1 + pct_width))
            f_gres = float(rng.uniform(1 - pct_level, 1 + pct_level))
            spec = perturb_six_level_width(base_spec, f_level_D, f_width_D, f_level_G, f_width_G, f_gres)
            all_ge0 = bool((spec.D_star >= 0).all() and (spec.G_res >= 0).all()
                           and (spec.D_min >= 0).all() and (spec.D_max >= 0).all()
                           and (spec.G_min >= 0).all() and (spec.G_max >= 0).all())
            ok = all_ge0 and market.spec_is_feasible(spec)
            if not ok:
                n_infeasible += 1
                continue
            specs.append(spec)
            log_rows.append((i, f_level_D, f_width_D, f_level_G, f_width_G, f_gres, ok))
            break
        else:
            raise RuntimeError(f"instance {i}: no feasible draw found in {max_retries} retries "
                               f"at pct_level={pct_level}, pct_width={pct_width}")
    return specs, log_rows, n_infeasible


# ======================================================================
# "x0-anchored" perturbation -- guarantees x0 is ALWAYS inside the box, by
# CONSTRUCTION, instead of building the box from the market's own D*/level
# and patching x0 back in afterward (`adjust_x0_to_spec`) when it drifts
# out. The user's own framing: change the bounds, but the one fixed x0
# value must ALWAYS be included in the bound -- and then the bound itself
# can be made much bigger or much smaller.
#
# For each node, the base market gives a FIXED distance from x0 to each
# edge of its box (`dist_below = x0.D - D_min`, `dist_above = D_max -
# x0.D`, both >0 since x0 was drawn inside the base box). A perturbed
# instance rebuilds the box around x0 by scaling those two distances by
# TWO INDEPENDENT positive factors:
#
#     D_min_new = x0.D - dist_below * f_below
#     D_max_new = x0.D + dist_above * f_above
#
# Since f_below, f_above > 0 always, x0 is mathematically inside the new
# box for EVERY draw -- no post-hoc clip needed (unlike
# `adjust_x0_to_spec`, still useful as a zero-cost safety net for float
# edge cases). Because the two sides scale INDEPENDENTLY, x0's relative
# position inside the box still varies freely across instances (unlike
# scaling both sides by the same factor, which would keep x0's relative
# position exactly constant) -- this is what recreates the "how far is x0
# from a good spot" signal that made the old x0-diversity axis effective,
# without ever moving x0 itself or risking the v1-v4 box-collapse
# pathology (the box can shrink toward x0 but never invert past it).
# Same construction for G. D* and G_res keep their own independent
# "level" perturbation as before, D* re-clipped into the new box.
def perturb_six_x0_anchored(base_spec: market.MarketSpec, x0_D: "np.ndarray", x0_G: "np.ndarray",
                             f_D_below: float, f_D_above: float, f_G_below: float, f_G_above: float,
                             f_dstar_level: float, f_gres: float) -> market.MarketSpec:
    dist_D_below = x0_D - base_spec.D_min
    dist_D_above = base_spec.D_max - x0_D
    D_min_new = x0_D - dist_D_below * f_D_below
    D_max_new = x0_D + dist_D_above * f_D_above

    dist_G_below = x0_G - base_spec.G_min
    dist_G_above = base_spec.G_max - x0_G
    G_min_new = x0_G - dist_G_below * f_G_below
    G_max_new = x0_G + dist_G_above * f_G_above

    D_star_new = np.clip(base_spec.D_star * f_dstar_level, D_min_new, D_max_new)
    G_res_new = base_spec.G_res * f_gres
    return dataclasses.replace(base_spec, D_star=D_star_new, G_res=G_res_new,
                                D_min=D_min_new, D_max=D_max_new,
                                G_min=G_min_new, G_max=G_max_new)


def sample_x0_anchored_dataset(base_spec: market.MarketSpec, x0_D: "np.ndarray", x0_G: "np.ndarray",
                                n_instances: int, seed: int, f_range=(0.5, 2.0),
                                pct_level: float = 0.20, max_retries: int = 50):
    """Like `sample_level_width_perturbed_dataset`, but using
    `perturb_six_x0_anchored` (x0 guaranteed inside every box by
    construction -- 4 independent box-shape factors drawn from `f_range`,
    D*/G_res keep their own independent level factor from `pct_level`).
    Same discard+resample contract: enforces all six quantities >=0 and
    Assumption-1 feasibility, returns exactly `n_instances` valid specs."""
    f_lo, f_hi = f_range
    specs, log_rows = [], []
    n_infeasible = 0
    for i in range(n_instances):
        for attempt in range(max_retries):
            rng = np.random.default_rng(seed * 100_003 + i * 1009 + attempt)
            f_D_below = float(rng.uniform(f_lo, f_hi))
            f_D_above = float(rng.uniform(f_lo, f_hi))
            f_G_below = float(rng.uniform(f_lo, f_hi))
            f_G_above = float(rng.uniform(f_lo, f_hi))
            f_dstar_level = float(rng.uniform(1 - pct_level, 1 + pct_level))
            f_gres = float(rng.uniform(1 - pct_level, 1 + pct_level))
            spec = perturb_six_x0_anchored(base_spec, x0_D, x0_G, f_D_below, f_D_above,
                                            f_G_below, f_G_above, f_dstar_level, f_gres)
            all_ge0 = bool((spec.D_star >= 0).all() and (spec.G_res >= 0).all()
                           and (spec.D_min >= 0).all() and (spec.D_max >= 0).all()
                           and (spec.G_min >= 0).all() and (spec.G_max >= 0).all())
            ok = all_ge0 and market.spec_is_feasible(spec)
            if not ok:
                n_infeasible += 1
                continue
            specs.append(spec)
            log_rows.append((i, f_D_below, f_D_above, f_G_below, f_G_above, f_dstar_level, f_gres, ok))
            break
        else:
            raise RuntimeError(f"instance {i}: no feasible draw found in {max_retries} retries "
                               f"at f_range={f_range}, pct_level={pct_level}")
    return specs, log_rows, n_infeasible


def _perturb_node_level_width(base_spec: market.MarketSpec, n: int, f_level_D: float, f_width_D: float,
                               f_level_G: float, f_width_G: float, f_gres: float):
    """Same arithmetic as `perturb_six_level_width`, for ONE node only --
    used by `sample_level_width_per_node_dataset` to feasibility-check and
    retry each node independently (this feasibility check, `spec_is_feasible`,
    has no cross-node terms, so per-node retry is exact, not an approximation)."""
    width_D = base_spec.D_max[n] - base_spec.D_min[n]
    rel_pos_D = (base_spec.D_star[n] - base_spec.D_min[n]) / width_D if width_D > 1e-9 else 0.5
    D_star_new = base_spec.D_star[n] * f_level_D
    width_D_new = width_D * f_width_D
    D_min_new = D_star_new - rel_pos_D * width_D_new
    D_max_new = D_min_new + width_D_new

    G_center = (base_spec.G_min[n] + base_spec.G_max[n]) / 2.0 * f_level_G
    width_G_new = (base_spec.G_max[n] - base_spec.G_min[n]) * f_width_G
    G_min_new = G_center - width_G_new / 2.0
    G_max_new = G_center + width_G_new / 2.0

    G_res_new = base_spec.G_res[n] * f_gres
    return D_min_new, D_max_new, G_min_new, G_max_new, D_star_new, G_res_new


def _node_feasible(D_min, D_max, G_min, G_max, D_star, G_res) -> bool:
    ge0 = D_min >= 0 and D_max >= 0 and G_min >= 0 and G_max >= 0 and D_star >= 0 and G_res >= 0
    lo, hi = G_min + G_res, G_max + G_res
    feas = (hi >= D_min - 1e-9) and (lo <= D_max + 1e-9) and (D_min <= D_max) and (G_min <= G_max)
    return bool(ge0 and feas)


def sample_level_width_per_node_dataset(base_spec: market.MarketSpec, n_instances: int,
                                         seed: int, pct_level: float = 0.20,
                                         pct_width: float = 0.08, max_retries: int = 200):
    """Like `sample_level_width_perturbed_dataset`, but each of the 5 factors
    is drawn INDEPENDENTLY PER NODE (shape (N,), not a shared scalar) --
    `perturb_six_level_width` already broadcasts correctly either way, no
    change needed there. Rationale: every variant tried so far (v1-v12) used
    ONE shared factor per instance, applied identically to all N nodes --
    structurally only ~5 degrees of freedom per instance, far poorer than
    the x0-diversity axis's 10 INDEPENDENT per-node draws (each node's x0
    lands anywhere in its own box, unrelated to its neighbours) that
    produced that axis's much larger speedup (~2x) at a similar DR-iters cv.
    Per-node factors restore that same per-node independence on the
    parameter axis instead of x0 -- and are arguably MORE realistic too:
    there is no physical reason 10 different households' demand should
    shift by exactly the same percentage on a given day.

    Feasibility is resampled PER NODE, not per whole instance: `spec_is_
    feasible` has no cross-node terms, so requiring all N independently-
    drawn nodes to be simultaneously feasible in one combined draw (as a
    first attempt at this function did) makes the whole-instance acceptance
    probability collapse combinatorially (~10 independent coin flips all
    needing to land right) -- it hit `max_retries` before finding a single
    valid instance. Retrying each node on its own is both correct (given no
    cross-node coupling) and exponentially cheaper."""
    specs, log_rows = [], []
    n_infeasible = 0
    N = base_spec.N
    for i in range(n_instances):
        f_level_D = np.empty(N); f_width_D = np.empty(N); f_level_G = np.empty(N)
        f_width_G = np.empty(N); f_gres = np.empty(N)
        for n in range(N):
            rng = np.random.default_rng(seed * 100_003 + i * 1009 + n * 97)
            for attempt in range(max_retries):
                fD_l = float(rng.uniform(1 - pct_level, 1 + pct_level))
                fD_w = float(rng.uniform(1 - pct_width, 1 + pct_width))
                fG_l = float(rng.uniform(1 - pct_level, 1 + pct_level))
                fG_w = float(rng.uniform(1 - pct_width, 1 + pct_width))
                fgr = float(rng.uniform(1 - pct_level, 1 + pct_level))
                vals = _perturb_node_level_width(base_spec, n, fD_l, fD_w, fG_l, fG_w, fgr)
                if _node_feasible(*vals):
                    f_level_D[n], f_width_D[n], f_level_G[n], f_width_G[n], f_gres[n] = fD_l, fD_w, fG_l, fG_w, fgr
                    break
                n_infeasible += 1
            else:
                raise RuntimeError(f"instance {i} node {n}: no feasible draw found in "
                                   f"{max_retries} retries at pct_level={pct_level}, pct_width={pct_width}")
        spec = perturb_six_level_width(base_spec, f_level_D, f_width_D, f_level_G, f_width_G, f_gres)
        specs.append(spec)
        log_rows.append((i, f_level_D.copy(), f_width_D.copy(), f_level_G.copy(), f_width_G.copy(), f_gres.copy(), True))
    return specs, log_rows, n_infeasible


def sample_dstar_gres_only_dataset(base_spec: market.MarketSpec, n_instances: int,
                                    seed: int, pct: float = 0.20, max_retries: int = 50):
    """The SIMPLEST parameter-diversity dataset on this axis: perturbs ONLY
    D_star and G_res (one shared factor each per instance, matching
    `demo_fixed_market_on_dstar_gres_scale.py`'s original
    `build_dstar_gres_dataset` -- the design that degraded gracefully,
    zero-shot, on the OLD market: 51%->28% iteration reduction as pct grew
    0%->50%, no cliff). D_min/D_max/G_min/G_max are left EXACTLY fixed --
    unlike every other sampler in this module, there is no box to collapse
    or jitter, so the fixed x0 (drawn once from the base market) will
    remain inside every instance's box automatically (the box never
    changes), no clip/adjustment ever needed. D_star is re-clipped into
    the (fixed) box if the level shift pushes it outside; G_res has no box
    of its own. Same discard+resample contract as the other samplers here:
    returns exactly `n_instances` feasible, non-negative specs."""
    specs, log_rows = [], []
    n_infeasible = 0
    for i in range(n_instances):
        for attempt in range(max_retries):
            rng = np.random.default_rng(seed * 100_003 + i * 1009 + attempt)
            f_D = float(rng.uniform(1 - pct, 1 + pct))
            f_G = float(rng.uniform(1 - pct, 1 + pct))
            D_star_new = np.clip(base_spec.D_star * f_D, base_spec.D_min, base_spec.D_max)
            G_res_new = base_spec.G_res * f_G
            spec = dataclasses.replace(base_spec, D_star=D_star_new, G_res=G_res_new)
            all_ge0 = bool((spec.D_star >= 0).all() and (spec.G_res >= 0).all())
            ok = all_ge0 and market.spec_is_feasible(spec)
            if not ok:
                n_infeasible += 1
                continue
            specs.append(spec)
            log_rows.append((i, f_D, f_G, ok))
            break
        else:
            raise RuntimeError(f"instance {i}: no feasible draw found in {max_retries} retries at pct={pct}")
    return specs, log_rows, n_infeasible


def adjust_x0_to_spec(x0_flat, net):
    """Project the ONE shared fixed x0 onto THIS instance's own box: clip D
    and G into [D_min,D_max]/[G_min,G_max] (q is untouched -- kappa, the
    edge capacities, is never perturbed by this dataset, so q never needs
    adjusting). A no-op whenever x0 is already inside (the common case
    under the level+width scheme above); otherwise the SMALLEST possible
    correction (snap to the nearest edge) -- "leggermente aggiustarlo", not
    a fresh random redraw. Takes/returns a flattened [D,G,Q] torch tensor,
    `net` already built from the (perturbed) spec so its D_min_vec etc.
    reflect that instance."""
    import torch
    D, G, Q = net.get_DGQ(x0_flat)
    D_adj = torch.clamp(D, net.D_min_vec, net.D_max_vec)
    G_adj = torch.clamp(G, net.G_min_vec, net.G_max_vec)
    return torch.cat([D_adj, G_adj, Q])


if __name__ == "__main__":
    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=0)
    print(f"base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}")

    specs, log_rows, n_infeasible = sample_six_perturbed_dataset(base_spec, n_instances=200, seed=config.TRAIN_SEED)
    print(f"\n200 draws: {n_infeasible}/200 infeasible ({n_infeasible/200:.1%})")

    print(f"\nfirst 5 instances' factors:")
    print(f"{'i':>3} {'f_D*':>7} {'f_Gres':>7} {'f_Dmin':>7} {'f_Dmax':>7} {'f_Gmin':>7} {'f_Gmax':>7} {'feasible':>9}")
    for i, fD, fG, fDm, fDM, fGm, fGM, ok in log_rows[:5]:
        print(f"{i:3d} {fD:7.3f} {fG:7.3f} {fDm:7.3f} {fDM:7.3f} {fGm:7.3f} {fGM:7.3f} {str(ok):>9}")

    print(f"\ninstance 0, per-node values (perturbed):")
    s0 = specs[0]
    print(f"{'node':>5} {'D_star':>8} {'G_res':>8} {'D_min':>8} {'D_max':>8} {'G_min':>8} {'G_max':>8}")
    for n in range(s0.N):
        print(f"{n:5d} {s0.D_star[n]:8.3f} {s0.G_res[n]:8.3f} {s0.D_min[n]:8.3f} {s0.D_max[n]:8.3f} "
              f"{s0.G_min[n]:8.3f} {s0.G_max[n]:8.3f}")

    all_ge0 = all(
        bool((s.D_star >= 0).all() and (s.G_res >= 0).all() and (s.D_min >= 0).all()
             and (s.D_max >= 0).all() and (s.G_min >= 0).all() and (s.G_max >= 0).all())
        for s in specs
    )
    print(f"\nall 6 quantities >=0 across all 200 instances: {all_ge0}")
