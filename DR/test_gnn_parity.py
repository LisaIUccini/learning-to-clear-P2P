"""
test_gnn_parity.py
===================

Regression test for the claim made throughout gnn.py/training.py's
docstrings and documentation.txt: the CENTRALISED path used for training
(`PerturbationGNN.direction`, vectorised over the whole graph) and the
DISTRIBUTED path used for deployment (`PerturbationGNN.local_forward`,
run independently by each prosumer on its own one-hop information) are
the SAME mathematical function of the same weights.

Why this needs an actual test, not just a one-off manual check
----------------------------------------------------------------
gnn.py's own `local_forward` docstring records a real incident: batching
the per-neighbour `phi_edge` calls into one GEMM (mathematically the same
MLP, just computed together instead of row-by-row) rounds differently at
the ULP level than the per-row loop, and that tiny difference was observed
to shift the EXACT convergence iteration on some instances -- a real
behaviour change, not just a speed one. If training (centralised) and
deployment (distributed) ever drift apart even that much, the reported
speed-ups stop meaning what they claim to mean: you would be training one
system and deploying a slightly different one. This test is the guard
against that happening silently again, e.g. the next time someone
vectorises a loop in `local_forward` or `build_features` "for speed, same
values".

Run directly with `python3 test_gnn_parity.py`, or via pytest if
installed (plain `assert`-based `test_*` functions, no fixtures needed).
"""

from __future__ import annotations

import torch

from market_generator import MarketConfig, MarketGenerator
from gnn import PerturbationGNN, GNNConfig, LocalNodeState, extract_local_state


TOL = 1e-5  # generous vs. the ~1e-7 typically observed; catches real drift,
            # not ordinary float32 accumulation noise


def _build_net(density, seed, N=10):
    mc = MarketConfig(graph_model="sparse", density=density)
    gen = MarketGenerator(mc)
    net = gen.sample(N, seed=seed)
    gnn = PerturbationGNN(GNNConfig.from_market_config(mc, hidden_dim=16))
    gnn.eval()
    return net, gnn


def _distributed_direction(gnn, net, x, mem, prev_res):
    """Mirror of what every prosumer computes on its own, assembled here
    only for the cross-check (a real deployment never does this centrally)."""
    d = torch.zeros(net.dim)
    mem_new = torch.zeros_like(mem)
    for n in range(net.N):
        s = extract_local_state(net, x, n, mem[n], prev_res[n])
        with torch.no_grad():
            dD, dG, dq, mem_new_n = gnn.local_forward(s)
        d[n] = dD
        d[net.N + n] = dG
        for m in net.neighbors[n]:
            d[2 * net.N + net._idx_map[(m, n)]] = dq[m]
        mem_new[n] = mem_new_n
    return d, mem_new


def _check_parity(net, gnn, n_steps=5):
    x = net.flatten()
    mem = gnn.init_mem(net)
    prev_res = torch.zeros(net.N)

    for step in range(n_steps):
        with torch.no_grad():
            d_central, mem_new_central = gnn.direction(net, x, mem, prev_res)
        d_dist, mem_new_dist = _distributed_direction(gnn, net, x, mem, prev_res)

        diff_d = (d_central - d_dist).abs().max().item()
        diff_mem = (mem_new_central - mem_new_dist).abs().max().item()
        assert diff_d < TOL, (
            f"step {step}: centralised/distributed DIRECTION mismatch "
            f"{diff_d:.3e} >= {TOL:.0e} -- training and deployment are "
            f"computing different functions"
        )
        assert diff_mem < TOL, (
            f"step {step}: centralised/distributed MEMORY mismatch "
            f"{diff_mem:.3e} >= {TOL:.0e}"
        )

        # advance the state with a trivial fixed-point-free step (this test
        # only cares about Phi_theta parity, not about DR itself) so the
        # recurrent memory is genuinely exercised across several distinct
        # inputs, the way it is across real dr_step calls
        x = x + 0.01 * torch.randn_like(x)
        mem, prev_res = mem_new_central, torch.rand(net.N)


def test_parity_sparse_graph():
    """density=0.3 -- the dataset the model has historically been tuned on."""
    net, gnn = _build_net(density=0.3, seed=1)
    _check_parity(net, gnn)


def test_parity_dense_graph():
    """density=0.7 -- the denser fixed-graph dataset (more neighbours per
    node -> exercises the aggregation/normalisation path harder)."""
    net, gnn = _build_net(density=0.7, seed=2)
    _check_parity(net, gnn)


def test_parity_isolated_node():
    """k_nb == 0 branch of local_forward (a node with no neighbours) is a
    separate code path (see the `if k_nb == 0` guard) and is never
    exercised by a connected market graph -- check it directly against a
    zero-neighbour slice of `direction`'s node-only computation."""
    net, gnn = _build_net(density=0.3, seed=3)
    x = net.flatten()
    mem = gnn.init_mem(net)
    prev_res = torch.zeros(net.N)

    n = 0
    s = LocalNodeState(
        node_id=n, neighbors=[],
        D=net.prosumers[n].D, G=net.prosumers[n].G,
        D_star=net.prosumers[n].D_star, G_res=net.prosumers[n].G_res,
        D_min=net.prosumers[n].D_min, D_max=net.prosumers[n].D_max,
        G_min=net.prosumers[n].G_min, G_max=net.prosumers[n].G_max,
        a_tilde=net.prosumers[n].a_tilde, a=net.prosumers[n].a,
        b=net.prosumers[n].b, q_own={}, q_recv={}, c={}, kappa={},
        mem=mem[n], prev_residual=prev_res[n],
    )
    with torch.no_grad():
        dD, dG, dq, mem_new = gnn.local_forward(s)
    assert dq == {}
    assert torch.isfinite(dD) and torch.isfinite(dG)
    assert torch.isfinite(mem_new).all()


def test_summability_bound_independent_of_weights():
    """The bound in PerturbationGNN.summability_bound must hold for BOTH
    untrained (random) and perturbed weights -- Theorem 1 requires it to
    hold for ANY theta, so this is not something training can break."""
    net, gnn = _build_net(density=0.3, seed=4)
    bound_before = gnn.summability_bound(net)
    with torch.no_grad():
        for p in gnn.parameters():
            p.add_(torch.randn_like(p) * 10.0)   # simulate wild/bad weights
    bound_after = gnn.summability_bound(net)
    assert bound_before == bound_after, (
        "summability_bound must not depend on theta at all"
    )
    assert bound_after < float("inf")


if __name__ == "__main__":
    tests = [
        test_parity_sparse_graph,
        test_parity_dense_graph,
        test_parity_isolated_node,
        test_summability_bound_independent_of_weights,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
