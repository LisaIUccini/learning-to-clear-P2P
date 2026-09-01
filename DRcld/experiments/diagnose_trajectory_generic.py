"""
diagnose_trajectory_generic.py
=================================

Generic version of diagnose_density08_trajectory.py: print the residual
trajectory (DR vs learned) for a few instances of an arbitrary
density/checkpoint, to see WHERE the learned trajectory falls behind.

Usage: python3 diagnose_trajectory_generic.py <density> <seed> <gamma>
       <rho> <scale> <ckpt_path> <instance_idx1> [instance_idx2] ...
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import torch

import config
import market
import algorithm
import gnn as gnn_mod

DENSITY = float(sys.argv[1])
SEED = int(sys.argv[2])
GAMMA = float(sys.argv[3])
RHO = float(sys.argv[4])
SCALE = float(sys.argv[5])
CKPT = sys.argv[6]
INSTANCES = [int(a) for a in sys.argv[7:]]

device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)

model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=RHO, scale=SCALE, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(blob["state_dict"])
model.eval()

dr_cfg = algorithm.DRConfig(gamma=GAMMA, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                             tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

for i in INSTANCES:
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
    step = max(max_len // 40, 1)
    print(f"{'k':>5}  {'DR residual':>14}  {'learned residual':>17}")
    for k in range(0, max_len, step):
        dr_r = res_dr.residual[k] if k < len(res_dr.residual) else res_dr.residual[-1]
        l_r = res_l.residual[k] if k < len(res_l.residual) else res_l.residual[-1]
        marker = ""
        if k < len(res_dr.residual) and k == len(res_dr.residual) - 1:
            marker += "  <- DR stop"
        if k < len(res_l.residual) and k == len(res_l.residual) - 1:
            marker += "  <- learned stop"
        print(f"{k:5d}  {dr_r:14.6e}  {l_r:17.6e}{marker}")
