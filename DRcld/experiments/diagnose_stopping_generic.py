"""
diagnose_stopping_generic.py
===============================

Generic version of diagnose_asymmetric_stopping.py: force a few instances
to run well past their official stop point (tol forced near 0) and print
the cost gap over iterations, confirming a borderline QP-check miss is a
benign stopping-rule artifact (gap keeps shrinking smoothly) and not a
wrong equilibrium (gap plateaus at a real offset).

Usage: python3 diagnose_stopping_generic.py <density> <seed> <gamma> <rho>
       <scale> <ckpt_path> <instance_idx1> [instance_idx2] ...
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import torch

import config
import market
import algorithm
import gnn as gnn_mod
from equilibrium import solve_equilibrium_qp

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

dr_cfg_force = algorithm.DRConfig(gamma=GAMMA, lam=config.DR_LAMBDA, n_iter=2000, tol=1e-12,
                                   bisect=config.DR_BISECT_STEPS)

for i in INSTANCES:
    gen = torch.Generator(device=device); gen.manual_seed(config.TEST_SEED * 7919 + i)
    net = market.build_network(base_spec, device=device, rng=gen)
    j_star = solve_equilibrium_qp(net)
    gnn_mod.deploy(model, net)
    net.reset()
    print(f"\n=== instance {i+1} (j_star={j_star:.6f}) ===")
    official_stop = None
    with torch.no_grad():
        for k in range(dr_cfg_force.n_iter):
            net.communicate()
            for p in net.prosumers:
                p.dr_step(dr_cfg_force.gamma, dr_cfg_force.lam, n_bisect=dr_cfg_force.bisect)
            r = net.aux_residual()
            if official_stop is None and r < 1e-5:
                official_stop = k + 1
            if (k + 1) % 40 == 0 or (official_stop is not None and k + 1 == official_stop):
                shadow = net.project_reciprocity(net.flatten())
                cost = float(net.cost_at(shadow))
                gap = abs(cost - j_star)
                marker = "  <-- official stop (r<1e-5)" if k + 1 == official_stop else ""
                print(f"  k={k+1:4d}  residual={r:.2e}  cost_gap={gap:.3e}{marker}")
            if official_stop is not None and k + 1 > official_stop + 300:
                break
    gnn_mod.undeploy(net)
