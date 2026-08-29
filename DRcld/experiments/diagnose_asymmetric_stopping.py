"""
diagnose_asymmetric_stopping.py
=================================

The asymmetric_loss result (density=1.0, seed=0) fails the standard QP
genuineness check at tol=1e-4 on ALL 30 test instances (worst gap 1.45e-4),
unlike prior runs where only a handful of instances land in that borderline
band. Investigate: is this a miscalibrated stopping rule (DR's own
iterate-residual tol=1e-5 no longer implying a cost gap < 1e-4 for this
trajectory shape), or has the learned trajectory actually gotten stuck at
the wrong point? Runs a few instances well PAST their reported stopping
iteration (tol forced to ~0, so the loop only stops at n_iter) and prints
the cost gap over iterations to see whether it keeps shrinking toward zero
(genuine, just-past-boundary convergence) or plateaus at a fixed offset
(a real correctness problem).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch

import config
import market
import algorithm
import gnn as gnn_mod
from equilibrium import solve_equilibrium_qp

DENSITY = 1.0
SEED = 0
GAMMA = 4.0
device = torch.device("cpu")
mc = market.MarketConfig(density=DENSITY)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=SEED)

model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=0.9667, scale=1.0, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
CKPT = "runs/density_1.0/seed0_gamma4_unroll300_scale10_epochs60_asymmetric/phi_w.pt"
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(blob["state_dict"])
model.eval()

# force the loop to run far past its normal stopping point
dr_cfg_force = algorithm.DRConfig(gamma=GAMMA, lam=config.DR_LAMBDA,
                                   n_iter=800, tol=1e-12, bisect=config.DR_BISECT_STEPS)

CHECK_INSTANCES = [0, 8, 13, 27]  # 1-indexed 1, 9, 14, 28 in the earlier report -- 28 was the worst QP gap
for i in CHECK_INSTANCES:
    net = market.build_network(base_spec, device=device)
    gen = torch.Generator(device=device); gen.manual_seed(config.TEST_SEED * 7919 + i)
    net = market.build_network(base_spec, device=device, rng=gen)
    j_star = solve_equilibrium_qp(net)

    gnn_mod.deploy(model, net)
    net.reset()
    prev_shadow = net.project_reciprocity(net.flatten())
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
    gnn_mod.undeploy(net)
