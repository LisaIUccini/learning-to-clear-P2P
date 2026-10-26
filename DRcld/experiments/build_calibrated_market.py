"""
build_calibrated_market.py
==========================================

Second-generation market-instance generator, recalibrated 2026-10-14 per
literature grounding supplied by the user (Le Cadre thesis Sec. 2.3/6.2.1,
Sousa et al. 2019, Belgioioso et al. arXiv:2107.13444). Unlike
`REALISTIC_MARKET_CONFIG` (every primitive an independent i.i.d. draw),
several quantities here are DERIVED from others:

    a_tilde_n = b_tilde / max(D_star_n, D_star_floor)^2   (NOT sampled)
    a_n, b_n  : technology-class procedure below (p_n^max sampled PER
                CLASS, a_n DERIVED from p_n^max/b_n/G_max_n)
    kappa_mn  ~ U(kappa_frac_lo, kappa_frac_hi) * D_bar    (D_bar = this
                instance's own mean D_star_n, not a fixed constant)
    c_nm      = s_e - g_m  (s_e = s_mn = s_nm, ONE symmetric base cost per
                edge; g_m = a per-NODE seller discount, one draw per node,
                not per edge -- see "SELLER-DEPENDENT c_nm" below for why
                this replaced an earlier independent-per-direction scheme),
                clipped below min_n(b_n)

TECHNOLOGY CLASSES stay FIXED across every perturbed instance of a given
base market (like topology and c_nm) -- assigned once when the base
market is built, never redrawn by the level+width perturbation (which
only ever touches D*/G_res/D_min/D_max/G_min/G_max, unchanged from
`build_realistic_training_dataset.py`).

Procedure (user-confirmed 2026-10-14), per prosumer:
  1. assign a class from a fixed mix (40% low, 40% medium, 20% high)
  2. sample G_max_n (Gbar_n) from the class's own size range
  3. sample b_n ~ U(0.045, 0.07) -- SAME range for every class (Belgioioso's
     dispatchable linear cost; heterogeneity lives in a_n, per Le Cadre)
  4. sample p_n^max (marginal cost at full output) from the class's own
     range, derive a_n = (p_n^max - b_n) / G_max_n
  5. verify b_n > max_m(c_nm) and p_n^max <= 0.16 (grid price); re-draw
     the instance if violated

d_n (a constant cost-function offset, does not affect the equilibrium) is
NOT modeled -- equivalent to d_n=0, matching the current cost function
(no constant term) -- state this explicitly in the paper, not a silent
omission.

a_n RATIOS across classes (0.06 : 1 : 2.5, low:medium:high) are the one
number taken directly from literature (Sousa et al. 2019's RES/coal/gas
bracket midpoints) -- NOT a project choice. Absolute class SIZES (Gbar_n
ranges) are the project's own modeling choice (stated as such, not cited
as empirical fact) following Belgioioso's convention of assigning
heterogeneous-size units to an arbitrary prosumer subset.

Kept FULLY SEPARATE from `build_realistic_training_dataset.py` /
`REALISTIC_MARKET_CONFIG` -- no existing checkpoint/experiment is
affected by this file.

SELLER-DEPENDENT c_nm (fixed 2026-10-14, supervisor review): the ORIGINAL
scheme here drew eps_mn/eps_nm INDEPENDENTLY per direction,
c_nm = c_base*(1+eps_nm) -- this was exactly the "small independent
epsilon" design the supervisor's own literature review had already
ruled out. Going around any cycle, the signed sum of c's does NOT cancel
under an independent asymmetry (it is a sum of independent random terms,
generically nonzero but NOT bounded away from ~0), so a near-zero- or
negative-cost circular trade (Le Cadre Prop. 5's arbitrage) is still
available on some cycles -- confirmed empirically: a 20000-iteration DR
run at density=0.6 (far past the 4000 cap) shows the RAW residual decay
steadily to ~2.7e-5 by k~6000-7000, then sit on a hard floor there
forever, never crossing the 1e-5 tolerance -- a persistent drift along
the degenerate cyclic-arbitrage direction, not merely slow convergence
(which would eventually cross the tolerance given enough iterations).
This explained the gamma grid search finding every density except 0.2
non-convergent at EVERY tested gamma.

Fix: c_nm = s_nm - g_m, s_nm = s_mn (ONE symmetric base cost per edge,
replacing the old per-direction c_base/eps), g_m >= 0 a DISCOUNT drawn
ONCE PER NODE (not per edge) acting on m whenever m is the SELLER in that
directed pair. Going around a cycle n_1 -> n_2 -> ... -> n_k -> n_1, the
sum of c's is now (sum of the k EDGE base costs s_e, each visited once,
all independently positive) MINUS (sum of the k NODE discounts g_{n_i},
each visited once) -- a deterministic quantity, bounded strictly away
from 0 as long as every g_m stays well below typical s_e (true by
construction: g ~ U(0, 0.005) vs s ~ U(0.01, 0.04), so even a k=2 cycle
keeps a safety margin of ~0.01). This is exactly Le Cadre Prop. 5's
arbitrage-free structure, not an independent per-direction perturbation.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

import config as proj_config
import market


# ----------------------------------------------------------------------
# Technology classes (Le Cadre Sec 6.2.1 / Sousa et al. 2019)
# ----------------------------------------------------------------------
# a_n ratios 0.06 : 1 : 2.5 (low:medium:high) ARE the literature number
# (Sousa's RES/coal/gas bracket midpoints ~0.075:1.33:3.37, normalised).
# Absolute p_n^max ranges below are back-solved so that, combined with the
# G_max/b_n ranges' own midpoints, the RESULTING a_n lands near the
# indicative values the user gave (low~0.0006, medium~0.01, high~0.025
# EUR/kWh^2) while respecting p_n^max<=0.16 always -- stated as a project
# modeling choice (the ABSOLUTE p_max brackets), not cited literature.
#
# "high"'s Gmax_range fixed 2026-10-14 (supervisor review): with the
# ORIGINAL (4.0, 8.0) range, a = markup/G_max landed in [0.0075, 0.0225]
# (median ~0.0125) for "high" vs [0.005, 0.025] (median ~0.01) for
# "medium" -- an actual medium:high ratio of ~1:1.25, not the ~1:2.5 the
# module docstring claimed (the claimed ratio would have been FALSE in
# the paper). Narrowing to (2.0, 4.0) restores it: a now spans
# [0.015, 0.045] (median ~0.025-0.03), recovering medium:high ~ 1:2.5 as
# intended -- a one-line fix, not a retraction of the cited ratio.
TECH_CLASSES = [
    {"name": "low",    "weight": 0.40, "Gmax_range": (3.0, 5.0), "markup_range": (0.001, 0.004)},
    {"name": "medium", "weight": 0.40, "Gmax_range": (1.0, 3.0), "markup_range": (0.015, 0.025)},
    {"name": "high",   "weight": 0.20, "Gmax_range": (2.0, 4.0), "markup_range": (0.060, 0.090)},
]
# OLD-REGIME CURVATURE variant (user-specified 2026-10-14, ablation): since
# a_n = markup/G_max_n BY CONSTRUCTION, a_n*Gmax_n = markup EXACTLY -- so
# "a*Gbar of the same order as b" means markup of the same order as b_n
# (~0.045-0.07), not TECH_CLASSES' literature-accurate but tiny "low"
# markup (0.001-0.004, a_n~0.0002-0.0013, ~200x smaller than "high"). This
# variant keeps every other new-generator mechanism (tech-class mix/
# Gmax_range, derived a_tilde, seller-discount c_nm, kappa~D_bar) but
# raises EVERY class's markup to be comparable to b_n, restoring roughly
# the OLD generator's near-uniform curvature (its a_range=(0.5,2.0) was
# only ~4x wide) instead of the new one's ~225x heterogeneity -- tests
# whether THAT heterogeneity (not c_nm, not gnn_scale, not unroll) is what
# makes this generator harder for a single shared learned correction than
# the old one ever was (see runs/official's unroll/slow-channel/scale/
# c_scheme ablations, all negative or no-op results on N=25).
TECH_CLASSES_OLD_CURVATURE = [
    {"name": "low",    "weight": 0.40, "Gmax_range": (3.0, 5.0), "markup_range": (0.030, 0.050)},
    {"name": "medium", "weight": 0.40, "Gmax_range": (1.0, 3.0), "markup_range": (0.040, 0.060)},
    {"name": "high",   "weight": 0.20, "Gmax_range": (2.0, 4.0), "markup_range": (0.060, 0.090)},
]
# markup = p_n^max - b_n, sampled DIRECTLY (not p_max and b independently)
# so a_n = markup/G_max_n >= 0 BY CONSTRUCTION -- independent p_max/b draws
# let b_n exceed p_max_n whenever their ranges overlap (they do for "low":
# p_max~0.06 vs b~0.045-0.07), silently producing a_n<0, which violates
# a_n>=0 (Sousa's own RES bracket is [0,0.15), never negative) -- caught
# empirically on the first calibration smoke test, not a hypothetical.
B_RANGE = (0.045, 0.070)          # b_n, SAME range for every class (Belgioioso)
P_GRID = 0.1624                    # grid price ceiling, EUR/kWh (Belgioioso: d_mg=0.1624/b_h)
C_RANGE = (0.01, 0.04)             # s_e (symmetric per-edge base cost), EUR/kWh (between
                                    # Belgioioso's grid tariff 0.01 and its c_di=0.045, kept
                                    # < min_n(b_n) by construction)
G_DISCOUNT_RANGE = (0.0, 0.005)    # g_m, per-node seller discount, c_nm = s_nm - g_m
                                    # (fixed 2026-10-14, replacing independent per-direction
                                    # asymmetry -- see module docstring "SELLER-DEPENDENT c_nm")
D_STAR_FLOOR = 0.3                 # kW, floor on D_n for a_tilde_n's denominator, user-confirmed


@dataclass
class CalibratedConfig:
    density: float = 1.0
    # ~unchanged from REALISTIC_MARKET_CONFIG (Le Cadre Tab.2/Fig.7: G_res
    # means 0.7-1.4 kW already coherent; D_spread controls the box
    # half-width D* ends up in -- verified post-hoc to land ~[0.5,3] kW)
    Gmin_range: Tuple[float, float] = (0.0, 1.0)
    G_res_range: Tuple[float, float] = (0.5, 2.0)
    D_spread_range: Tuple[float, float] = (0.4, 1.0)
    b_tilde: float = 1.0           # PROVISIONAL -- calibrated by the sanity-check loop
                                    # (target: median demand curtailment < ~20%)
    kappa_frac_range: Tuple[float, float] = (0.5, 1.0)   # kappa_mn ~ U(.) * D_bar
    curvature_regime: str = "literature"   # "literature" (default, TECH_CLASSES, the
                                             # 0.06:1:2.5 Sousa ratio) | "old_regime"
                                             # (TECH_CLASSES_OLD_CURVATURE, user-specified
                                             # 2026-10-14 ablation, see that constant's comment)
    c_scheme: str = "seller_discount"   # "seller_discount" (default, arbitrage-free,
                                         # c_nm=s_nm-g_m) | "independent_wide" (Le Cadre
                                         # case (b), user-specified 2026-10-14: c_nm, c_mn
                                         # independently ~ U(*C_RANGE), full range each --
                                         # NOT the earlier buggy +/-5% narrow perturbation
                                         # around a shared base (that narrow spread is what
                                         # made cycle sums land near zero; `market.py`'s own
                                         # OLD/pre-recalibration generator -- the one every
                                         # well-performing "combined" experiment used -- drew
                                         # c[(m,n)] fully independently from a WIDE range,
                                         # (0.05,0.30), one draw per directed pair, which this
                                         # mode reproduces at the new, narrower C_RANGE scale)


def sample_calibrated_base_market(base_seed: int, cfg: CalibratedConfig, max_retries: int = 20,
                                   N: int = None):
    """Returns (neighbors, spec, tier_names) for one base market under the
    calibrated scheme. Mirrors `market.sample_base`'s contract (graph via
    `market.sample_graph`, same N=config.BASE_MARKET_N by default) but
    with the new staged derivation instead of `sample_primitives`/
    `build_spec`. Retries the WHOLE market (fresh rng) if step 5's
    verification fails for any node -- classes/G_max/b/p_max/c are cheap
    to redraw, no reason to patch individual nodes.

    `N`: override the market size (default None -> config.BASE_MARKET_N,
    unchanged behaviour for every existing caller). Added 2026-10-14 for
    a one-off N=25 scaling test -- every other primitive's SAMPLING range
    (tech-class mix/ranges, B_RANGE, C_RANGE, etc.) stays exactly as
    calibrated for N=10; only the number of prosumers/graph size changes."""
    N = proj_config.BASE_MARKET_N if N is None else N
    if cfg.curvature_regime == "literature":
        classes = TECH_CLASSES
    elif cfg.curvature_regime == "old_regime":
        classes = TECH_CLASSES_OLD_CURVATURE
    else:
        raise ValueError(f"unknown curvature_regime {cfg.curvature_regime!r}")

    for attempt in range(max_retries):
        rng = np.random.default_rng(base_seed * 1_000_003 + attempt)
        neighbors = market.sample_graph(N, cfg.density, rng)

        # --- G/D boxes via the SAME anchor-point construction as
        # market.build_spec (unchanged mechanism) ---
        tier_idx = rng.choice(len(classes), size=N, p=[t["weight"] for t in classes])
        G_max = np.array([rng.uniform(*classes[tier_idx[n]]["Gmax_range"]) for n in range(N)])
        G_min_raw = rng.uniform(*cfg.Gmin_range, size=N)
        G_min = np.minimum(G_min_raw, G_max - 1e-3)
        G_res = rng.uniform(*cfg.G_res_range, size=N)
        G0_frac = rng.uniform(0.2, 0.8, size=N)
        G0 = G_min + G0_frac * (G_max - G_min)
        D0 = G0 + G_res
        D_spread = rng.uniform(*cfg.D_spread_range, size=N)
        D_min = np.maximum(0.0, D0 - D_spread)
        D_max = D0 + D_spread
        D_star_jitter = rng.uniform(-0.3, 0.3, size=N)
        D_star = np.clip(D0 + D_star_jitter, D_min, D_max)

        # --- a_tilde_n = b_tilde / max(D*_n, floor)^2 (DERIVED) ---
        a_tilde = cfg.b_tilde / np.maximum(D_star, D_STAR_FLOOR) ** 2

        # --- technology classes -> b_n, markup -> p_n^max, a_n (DERIVED) ---
        b = rng.uniform(*B_RANGE, size=N)
        markup = np.array([rng.uniform(*classes[tier_idx[n]]["markup_range"]) for n in range(N)])
        p_max = b + markup          # by construction: p_max > b, so a_n >= 0 always
        a = markup / G_max          # = (p_max - b) / G_max

        # --- c_nm = s_nm - g_m: s symmetric per edge (clipped below
        # min_n(b_n)), g_m a per-NODE seller discount drawn ONCE (not per
        # edge) -- see module docstring "SELLER-DEPENDENT c_nm" for why
        # this (not an independent per-direction draw) is what keeps every
        # cycle's cost sum bounded away from the Prop. 5 arbitrage floor ---
        b_min = float(b.min())
        c = {}
        if cfg.c_scheme == "seller_discount":
            g_discount = rng.uniform(*G_DISCOUNT_RANGE, size=N)
            for n in range(N):
                for m in neighbors[n]:
                    if (m, n) in c:
                        continue
                    s_e = min(rng.uniform(*C_RANGE), 0.95 * b_min)
                    c[(m, n)] = s_e - g_discount[m]   # n imports from m: m is the seller
                    c[(n, m)] = s_e - g_discount[n]   # m imports from n: n is the seller
        elif cfg.c_scheme == "independent_wide":
            # Le Cadre case (b): c_nm, c_mn drawn FULLY INDEPENDENTLY, each
            # over the WHOLE C_RANGE (not a narrow perturbation around a
            # shared base) -- matches `market.py`'s old pre-recalibration
            # generator's own per-directed-pair independent draw exactly,
            # just clipped to stay below 0.95*min(b) (the new scheme's own
            # safety margin, kept here too).
            c_cap = 0.95 * b_min
            for n in range(N):
                for m in neighbors[n]:
                    c[(m, n)] = min(float(rng.uniform(*C_RANGE)), c_cap)
        else:
            raise ValueError(f"unknown c_scheme {cfg.c_scheme!r}")

        # --- verification (step 5): b_n > max_m(c_nm), p_n^max <= grid price ---
        max_c_per_node = np.zeros(N)
        for (m, n), val in c.items():
            max_c_per_node[n] = max(max_c_per_node[n], val)
        ok = bool((b > max_c_per_node).all() and (p_max <= P_GRID + 1e-9).all())
        if not ok:
            continue

        # --- kappa_mn ~ U(frac) * D_bar (THIS instance's own mean D*) ---
        D_bar = float(D_star.mean())
        kappa = {}
        seen = set()
        for n in range(N):
            for m in neighbors[n]:
                pair = (min(m, n), max(m, n))
                if pair in seen:
                    continue
                seen.add(pair)
                cap = float(rng.uniform(*cfg.kappa_frac_range)) * D_bar
                kappa[(pair[0], pair[1])] = cap
                kappa[(pair[1], pair[0])] = cap

        spec = market.MarketSpec(N=N, neighbors=neighbors, D_min=D_min, D_max=D_max, D_star=D_star,
                                  G_min=G_min, G_max=G_max, G_res=G_res,
                                  a_tilde=a_tilde, a=a, b=b, c=c, kappa=kappa)
        tier_names = [classes[i]["name"] for i in tier_idx]
        return neighbors, spec, tier_names

    raise RuntimeError(f"base_seed={base_seed}: no valid market found in {max_retries} retries "
                        f"(b_n > max_m(c_nm) or p_max<=grid price kept failing)")
