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


def build_realistic_base_market(base_seed: int = 0):
    """The new base market: SAME graph as every other density=1.0 seed=0
    experiment in this project, but realistic-range primitives."""
    return market.sample_base(config.BASE_MARKET_N, REALISTIC_MARKET_CONFIG, seed=base_seed)


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
