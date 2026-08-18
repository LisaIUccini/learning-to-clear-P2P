"""
test_gnn_parity.py
===================

Correctness check tying the paper's central claim -- training is
centralised, inference is fully distributed, and both compute the SAME
function of the SAME weights -- to the code: `PerturbationGNN.forward_local`
(run independently by every prosumer, using only its own state and the q
it received at line 3) must agree numerically with
`PerturbationGNN.forward_centralised` (the vectorised mirror used only for
training). If these ever disagree, the model being trained is not the
model being deployed.
"""

import torch

import market
from gnn import PerturbationGNN, GNNConfig


def check(N: int, density: float, seed: int, atol: float = 1e-5) -> float:
    mc = market.MarketConfig(density=density)
    spec = market.sample_market(N, mc, seed=seed)
    net = market.build_network(spec, device=torch.device("cpu"))

    torch.manual_seed(seed)
    gnn = PerturbationGNN(GNNConfig.from_market_config(mc))
    gnn.eval()

    net.communicate()      # line 3: populate q_received before either path reads it
    x = net.flatten()
    mem = gnn.init_mem(N, net.device)
    # nonzero, distinct-per-node prev_residual/stall so both new features are
    # actually exercised by the parity check (both paths default to zero
    # at k=0, which would silently skip testing this feature otherwise)
    g = torch.Generator().manual_seed(seed)
    prev_residual = torch.rand(N, generator=g)
    stall = torch.rand(N, generator=g)
    for i, p in enumerate(net.prosumers):
        p.prev_residual = prev_residual[i]
        p.stall_ema = stall[i]
    with torch.no_grad():
        dir_fast_c, dir_slow_c, mem_new_c = gnn.forward_centralised(net, x, mem, prev_residual, stall)

    max_err = 0.0
    for n, p in enumerate(net.prosumers):
        state = gnn.local_state(p)
        with torch.no_grad():
            (dD_f, dG_f, dq_f), (dD_s, dG_s, dq_s), mem_new_l = gnn.forward_local(state)
        err = max(
            float((dD_f - dir_fast_c[n]).abs()), float((dD_s - dir_slow_c[n]).abs()),
            float((dG_f - dir_fast_c[net.N + n]).abs()), float((dG_s - dir_slow_c[net.N + n]).abs()),
            float((mem_new_l - mem_new_c[n]).abs().max()),
        )
        for m in p.neighbors:
            i = net.idx_map[(m, n)]
            err = max(err, float((dq_f[m] - dir_fast_c[2 * net.N + i]).abs()),
                      float((dq_s[m] - dir_slow_c[2 * net.N + i]).abs()))
        max_err = max(max_err, err)
    assert max_err < atol, f"parity mismatch: {max_err}"
    return max_err


if __name__ == "__main__":
    worst = 0.0
    for N in (1, 2, 5, 9, 14):
        for density in (0.0, 0.3, 0.8):
            for seed in range(3):
                e = check(N, density, seed=seed * 97 + N)
                worst = max(worst, e)
    print(f"OK -- centralised/local parity holds, worst-case error {worst:.2e}")
