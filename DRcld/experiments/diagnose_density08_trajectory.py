"""
diagnose_density08_trajectory.py
===================================

Every hyperparameter combo tried on density=0.8/seed=0 converges to the
SAME qualitative failure: a rigid "learned" pace stuck ABOVE most of plain
DR's own range. Instead of guessing more hyperparameters, print the actual
residual trajectory (not just the final iteration count) for a few
instances, comparing plain DR vs the learned model step by step, to see
WHERE the learned trajectory falls behind (early divergence? a plateau
DR doesn't have? active oscillation?).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import torch

import config
import market
import algorithm
import gnn as gnn_mod

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/cross_density/seed0_density0.8_gamma8_unroll406_scale10_asymmetric/phi_w.pt"
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
RHO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.97536
SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

device = torch.device("cpu")
mc = market.MarketConfig(density=0.8)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=0)

model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=RHO, scale=SCALE, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(blob["state_dict"])
model.eval()

dr_cfg = algorithm.DRConfig(gamma=GAMMA, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                             tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

for i in [0, 6]:  # instance 0 (a "fast" DR one) and 6 (relatively "slow" DR one)
    seed = config.TEST_SEED * 7919 + i
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net_dr = market.build_network(base_spec, device=device, rng=gen)
    res_dr = algorithm.run_dr(net_dr, dr_cfg, record=True)

    gen2 = torch.Generator(device=device); gen2.manual_seed(seed)
    net_l = market.build_network(base_spec, device=device, rng=gen2)
    gnn_mod.deploy(model, net_l)
    res_l = algorithm.run_dr(net_l, dr_cfg, record=True)

    print(f"\n=== instance {i} ===  DR n_iter={res_dr.n_iter}  learned n_iter={res_l.n_iter}")
    max_len = max(len(res_dr.residual), len(res_l.residual))
    print(f"{'k':>5}  {'DR residual':>14}  {'learned residual':>17}")
    for k in range(0, max_len, 20):
        dr_r = res_dr.residual[k] if k < len(res_dr.residual) else res_dr.residual[-1]
        l_r = res_l.residual[k] if k < len(res_l.residual) else res_l.residual[-1]
        marker = ""
        if k < len(res_dr.residual) and k == len(res_dr.residual) - 1:
            marker += "  <- DR stop"
        if k < len(res_l.residual) and k == len(res_l.residual) - 1:
            marker += "  <- learned stop"
        print(f"{k:5d}  {dr_r:14.6e}  {l_r:17.6e}{marker}")
