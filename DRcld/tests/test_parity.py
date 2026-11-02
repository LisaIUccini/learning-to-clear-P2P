"""Training is centralised, inference is distributed: both paths must compute
the same function of the same weights, otherwise the trained model is not
the deployed one."""

import pytest
import torch

from l2p2p import gnn as gnn_mod
from l2p2p.fixed_point import DRConfig, run_dr
from l2p2p.gnn import GNNConfig, PerturbationGNN
from l2p2p.market import MarketConfig, build_network, sample_base
from l2p2p.training import TrainConfig, rollout

CASES = [(N, density, seed) for N in (1, 2, 5, 9, 14) for density in (0.0, 0.3, 0.8) for seed in range(3)]


def _model(mc, seed, **kw):
    torch.manual_seed(seed)
    model = PerturbationGNN(GNNConfig.from_market_config(mc, **kw))
    model.eval()
    return model


@pytest.mark.parametrize("N,density,seed", CASES)
def test_gnn_forward_local_matches_centralised(N, density, seed):
    seed = seed * 97 + N
    mc = MarketConfig(density=density)
    net = build_network(sample_base(N, mc, seed=seed), device=torch.device("cpu"))
    gnn = _model(mc, seed)

    net.communicate()
    x = net.flatten()
    mem = gnn.init_mem(N, net.device)
    # nonzero per-node prev_residual / stall, so those features are exercised too
    g = torch.Generator().manual_seed(seed)
    prev_residual = torch.rand(N, generator=g)
    stall = torch.rand(N, generator=g)
    for i, p in enumerate(net.prosumers):
        p.prev_residual = prev_residual[i]
        p.stall_ema = stall[i]
    with torch.no_grad():
        dir_fast_c, dir_slow_c, mem_new_c = gnn.forward_centralised(net, x, mem, prev_residual, stall)

    for n, p in enumerate(net.prosumers):
        with torch.no_grad():
            (dD_f, dG_f, dq_f), (dD_s, dG_s, dq_s), mem_new_l = gnn.forward_local(gnn.local_state(p))
        errors = [(dD_f - dir_fast_c[n]).abs(), (dD_s - dir_slow_c[n]).abs(),
                  (dG_f - dir_fast_c[net.N + n]).abs(), (dG_s - dir_slow_c[net.N + n]).abs(),
                  (mem_new_l - mem_new_c[n]).abs().max()]
        for m in p.neighbors:
            i = net.idx_map[(m, n)]
            errors += [(dq_f[m] - dir_fast_c[2 * net.N + i]).abs(), (dq_s[m] - dir_slow_c[2 * net.N + i]).abs()]
        assert max(float(e) for e in errors) < 1e-5


@pytest.mark.parametrize("density,with_gnn", [(0.3, False), (0.3, True), (0.8, True)])
def test_distributed_iteration_matches_training_rollout(density, with_gnn):
    """30 iterations of Algorithm 1 run prosumer by prosumer (`run_dr`) and
    vectorised (`training.rollout`) from the same x^0: same residuals."""
    mc = MarketConfig(density=density)
    spec = sample_base(10, mc, seed=5)
    gen = torch.Generator().manual_seed(11)
    net = build_network(spec, rng=gen)
    x0 = net.flatten().clone()
    model = _model(mc, 3, rho=0.97, scale=1.8, scale_slow=0.4) if with_gnn else None

    if with_gnn:
        gnn_mod.deploy(model, net)
    res = run_dr(net, DRConfig(gamma=4.0, n_iter=30, tol=0.0, bisect=30), record=True)

    net2 = build_network(spec)
    _, residuals = rollout(net2, model, TrainConfig(gamma=4.0, bisect=30, unroll=30), x0=x0, differentiable=False)
    for r_dist, r_vec in zip(res.residual, residuals):
        assert abs(r_dist - float(r_vec)) <= 1e-4 * max(r_dist, 1e-6)
