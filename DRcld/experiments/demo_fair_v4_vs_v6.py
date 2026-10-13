"""
demo_fair_v4_vs_v6.py
==========================================

TRUE apples-to-apples zero-shot test: evaluates the `realistic_market_x0_diversity_v4_wbehind6`
checkpoint (x0-diversity training) on v6's OWN EXACT test protocol -- the SAME 30 test instances
(same seed=TRAIN_SEED+1) AND the SAME shared, fixed, raw/unclipped x0 (same seed=TRAIN_SEED) that
v6 was trained and evaluated on -- instead of v4's own native fresh-random-x0 convention. Isolates
whether v4 really "generalizes better than dedicated training" (as the raw zero-shot number
suggested, 1.42x/93%) or whether that gap was actually explained by an easier evaluation protocol
(fresh random x0 per instance, which is trivially easier than one shared x0 that isn't guaranteed
to sit anywhere sensible relative to each perturbed instance's own box).

Usage: python3 demo_fair_v4_vs_v6.py [weights_path] [n_test] [pct_level] [pct_width]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

import torch

import config
import market
import algorithm
import gnn as gnn_mod
from build_realistic_training_dataset import build_realistic_base_market, sample_level_width_perturbed_dataset, REALISTIC_MARKET_CONFIG
from demo_train_realistic_params_fixed_x0 import evaluate_fixed_x0


def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "runs/cross_density/realistic_market_x0_diversity_v4_wbehind6/phi_w.pt"
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    pct_level = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20
    pct_width = float(sys.argv[4]) if len(sys.argv) > 4 else 0.08

    device = torch.device("cpu")
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=4.0, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=0.973, scale=1.0, rho_slow=0.99, scale_slow=0.4))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"loaded v4 checkpoint: {weights_path}", flush=True)

    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=0)

    # EXACTLY v6's own protocol: same shared fixed x0, same test-set seed
    fixed_x0_gen = torch.Generator(device=device); fixed_x0_gen.manual_seed(config.TRAIN_SEED)
    fixed_x0_net = market.build_network(base_spec, device=device, rng=fixed_x0_gen)
    fixed_x0_base = fixed_x0_net.flatten().detach().clone()

    test_specs, _, n_infeasible = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=pct_level, pct_width=pct_width)
    test_specs = [base_spec] + test_specs
    test_x0 = [fixed_x0_base] * len(test_specs)
    print(f"test set: v6's EXACT {n_test} instances (seed={config.TRAIN_SEED + 1}), v6's EXACT shared "
          f"fixed/unclipped x0 (seed={config.TRAIN_SEED}) -- {n_infeasible} infeasible discarded", flush=True)
    print("-" * 70)

    evaluate_fixed_x0(test_specs, test_x0, model, dr_cfg, device, label="v4-on-v6-protocol")


if __name__ == "__main__":
    main()
