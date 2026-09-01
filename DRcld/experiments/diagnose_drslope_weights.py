"""
diagnose_drslope_weights.py
==============================

Print the actual omega weights `dr_slope` would compute for a few
instances of a given market, to see WHERE weight concentrates and whether
the reference DR trajectory's local log-decay rate is smooth (one clean
plateau) or noisy/irregular (multiple false plateaus, weight scattered).

Usage: python3 diagnose_drslope_weights.py <density> <seed> <gamma> <unroll>
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import torch

import config
import market
import algorithm
import train as train_mod

DENSITY = float(sys.argv[1])
SEED = int(sys.argv[2])
GAMMA = float(sys.argv[3])
UNROLL = int(sys.argv[4])

device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)

train_cfg = train_mod.TrainConfig(gamma=GAMMA, lam=config.DR_LAMBDA, bisect=config.DR_BISECT_STEPS,
                                   unroll=UNROLL, gamma_mono=2.0, loss_mode="asymmetric")

for inst in [1000, 1003, 1006]:
    gen = torch.Generator(device=device); gen.manual_seed(config.TRAIN_SEED * 7919 + inst)
    net = market.build_network(base_spec, device=device, rng=gen)
    x0 = net.flatten().detach()
    _, residuals_dr = train_mod.rollout(net, None, train_cfg, x0=x0, differentiable=False)

    import gnn as gnn_mod
    scale = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(mc)).physical_scale(net)
    denom = torch.linalg.vector_norm(scale) + 1e-9
    rho_dr = torch.stack(residuals_dr) / denom
    log_rho_dr = torch.log(rho_dr + 1e-9)
    log_decrease = torch.clamp(log_rho_dr[:-1] - log_rho_dr[1:], min=0.0)
    typical = log_decrease.mean().clamp(min=1e-9)
    omega_local = typical / (log_decrease + 0.1 * typical)
    omega_local = torch.cat([omega_local, omega_local[-1:]])
    omega = omega_local / omega_local.mean()

    print(f"\n=== instance seed={inst} (unroll={UNROLL}) ===")
    print(f"{'k':>5}  {'rho_dr':>12}  {'log_decrease':>13}  {'omega':>8}")
    for k in range(0, UNROLL, max(UNROLL // 30, 1)):
        print(f"{k:5d}  {rho_dr[k].item():12.4e}  "
              f"{(log_decrease[k].item() if k < len(log_decrease) else 0):13.5f}  "
              f"{omega[k].item():8.3f}")
    print(f"  max omega: {omega.max().item():.2f} at k={omega.argmax().item()}   "
          f"top-5 omega positions: {omega.topk(5).indices.tolist()}")
