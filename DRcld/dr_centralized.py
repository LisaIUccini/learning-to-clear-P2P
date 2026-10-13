"""
dr_centralized.py
==================

The SAME Douglas-Rachford iteration as `algorithm.run_dr`
(`Prosumer.dr_step` + `Network.communicate`, no GNN), but executed by a
SINGLE coordinator holding the network's ENTIRE state and every
prosumer's private parameters -- using the vectorised functions
`Network.prox_local` / `Network.project_reciprocity` that `network.py`
already exposes for training (its own docstring calls them "the
centralised... twin" of the per-node distributed path).

Why this exists: `algorithm.run_dr` needs exactly one round of one-hop
message exchange (`Network.communicate()`) PER ITERATION, because q_mn and
q_nm live on two different prosumers. A single coordinator that already
holds the whole state needs NO such exchange -- the reflection/projection
step (line 4 below) is a pure function of the vector it already has. The
only communication a centralised solver needs AT ALL is a one-time
gather (every prosumer's private cost/box parameters -> the coordinator,
to build `Network`) and a one-time broadcast (the converged (D,G,q) ->
every prosumer) -- a FIXED cost, independent of how many DR iterations run
internally, unlike every other method compared in
`experiments/compare_a2dr_vs_drslope.py` (distributed DR, A2DR, DR+
learned), whose communication scales with iteration count. The price is
that every prosumer must reveal its private data to that one coordinator
-- the privacy/single-point-of-failure trade-off a distributed algorithm
exists to avoid in the first place.

Iteration-by-iteration, this reproduces this project's own DR ordering
(projection first, then prox -- see algorithm.py's docstring/Prosumer.
dr_step, which swaps the roles of f/h relative to the textbook DRS
ordering, but is still a valid DRS instance):

    qbar = Pi(v^k)                  # Network.project_reciprocity (identity on D,G)
    r    = 2*qbar - v^k             # reflection (matches dr_step's rD,rG=D,G exactly,
                                     # since qbar=D,G there -- same formula, no special case)
    z    = prox_tf(r)               # Network.prox_local
    v^{k+1} = v^k + lam*(z - qbar)  # matches Prosumer.dr_step's D_new/G_new/q_new exactly

Numerically this is expected to track `algorithm.run_dr` almost exactly,
but not necessarily bit-for-bit: `Network.prox_local` adds one extra
differentiable Newton correction step after the bisection (needed for
training, absent from `Prosumer.local_prox`'s pure-bisection output), so
the two can differ by a handful of iterations near the tolerance boundary.
"""

from __future__ import annotations

import torch

from algorithm import DRConfig, DRResult


def run_dr_centralized(net, cfg: DRConfig, record: bool = True) -> DRResult:
    """Same calling convention as `algorithm.run_dr`: seed `net`'s state
    (net.flatten()) before calling, `cfg` is the identical `DRConfig`."""
    net.reset()
    v = net.flatten()
    residual = []
    shadow_residual = []
    cost = []
    converged = False
    k_run = 0
    prev_shadow = net.project_reciprocity(v) if record else None

    with torch.no_grad():
        for k in range(cfg.n_iter):
            k_run = k + 1
            qbar = net.project_reciprocity(v)
            r = 2.0 * qbar - v
            z = net.prox_local(cfg.gamma, r, n_bisect=cfg.bisect)
            v_new = v + cfg.lam * (z - qbar)

            step_residual = float(torch.linalg.vector_norm(v_new - v))
            v = v_new
            if record:
                residual.append(step_residual)
                shadow = net.project_reciprocity(v)
                shadow_residual.append(float(torch.linalg.vector_norm(shadow - prev_shadow)))
                cost.append(float(net.cost_at(shadow)))
                prev_shadow = shadow
            if step_residual < cfg.tol:
                converged = True
                break

    net.unflatten(v)
    return DRResult(n_iter=k_run, converged=converged, residual=residual,
                     shadow_residual=shadow_residual, cost=cost)
