#!/usr/bin/env python3
"""
table_ek_comparison.py
========================
Al punto di arresto (residual < tol), per DR e DR+GNN, per ogni densita'
(mediana sulle 30 istanze ufficiali di test): gap di ottimalita' relativo,
violazione dei vincoli locali (letti dai dati salvati), e ||z^k - x_tilde^k||
(calcolato ora direttamente, catturando z^k e x_tilde^k separatamente --
non e' uguale al residual salvato per DR+GNN, la cui x^{k+1} include anche
la perturbazione v^k sommata sopra il passo DR, vedi verify_ek.py).

30 istanze x 2 metodi x 6 densita'. DR puro puo' richiedere molte piu'
iterazioni di DR+GNN (fino a migliaia), quindi questo gira in background.
"""
from __future__ import annotations

import os
import sys
import json

_ROOT = "/home/lisa/energy_market_fixed_point/DRcld"
_EXPERIMENTS = os.path.join(_ROOT, "experiments")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EXPERIMENTS)

import numpy as np
import torch
torch.set_num_threads(1)

import config
import market
import gnn as gnn_mod
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_ek_data")
os.makedirs(OUT_DIR, exist_ok=True)

DCFG = {
    0.2: dict(base_seed=3, gamma=4.0, gnn_rho=0.9765, rho_slow=0.99, scale_slow=0.0,
              ckpt="official_N10_d0.2_seed3_replicate/phi_w.pt"),
    0.4: dict(base_seed=3, gamma=4.0, gnn_rho=0.9686, rho_slow=0.99, scale_slow=0.4,
              ckpt="official_N10_d0.4_seed3_replicate/phi_w.pt"),
    0.5: dict(base_seed=1, gamma=6.0, gnn_rho=0.9825, rho_slow=0.99, scale_slow=0.0,
              ckpt="official_N10_d0.5_seed1_replicate/phi_w.pt"),
    0.6: dict(base_seed=3, gamma=4.0, gnn_rho=0.9688, rho_slow=0.99, scale_slow=0.0,
              ckpt="official_N10_d0.6_seed3_replicate/phi_w.pt"),
    0.8: dict(base_seed=0, gamma=4.0, gnn_rho=0.9914, rho_slow=0.99, scale_slow=0.4,
              ckpt="official_N10_d0.8_seed0_replicate_fixed/phi_w.pt"),
    1.0: dict(base_seed=2, gamma=6.0, gnn_rho=0.9926, rho_slow=0.99, scale_slow=0.4,
              ckpt="official_N10_d1.0_seed2_replicate/phi_w.pt"),
}
N_INSTANCES = 30
MAX_ITER = config.DR_MAX_ITER
TOL = config.DR_TOLERANCE


def rollout_capture(net, gamma, lam, n_bisect, use_gnn):
    net.reset()
    e_k = None
    for k in range(MAX_ITER):
        net.communicate()
        z_parts, xt_parts = [], []
        for p in net.prosumers:
            qbar = p.shadow_q()
            rD, rG = p.D, p.G
            rq = {m: 2.0 * qbar[m] - p.q[m] for m in p.neighbors}
            zD, zG, zq = p.local_prox(gamma, rD, rG, rq, n_bisect, tau=0.0)

            z_parts.append(zD); z_parts.append(zG); z_parts += [zq[m] for m in p.neighbors]
            xt_parts.append(rD); xt_parts.append(rG); xt_parts += [qbar[m] for m in p.neighbors]

            D_new = p.D + lam * (zD - p.D)
            G_new = p.G + lam * (zG - p.G)
            q_new = {m: p.q[m] + lam * (zq[m] - qbar[m]) for m in p.neighbors}
            if use_gnn and p.gnn_agent is not None:
                state = p.gnn_agent.gnn.local_state(p)
                v_D, v_G, v_q, mem_new = p.gnn_agent.perturbation(state, p._round)
                D_new = D_new + v_D; G_new = G_new + v_G
                q_new = {m: q_new[m] + v_q[m] for m in p.neighbors}
                p._gnn_mem = mem_new
            parts = [(D_new - p.D) ** 2, (G_new - p.G) ** 2]
            parts += [(q_new[m] - p.q[m]) ** 2 for m in p.neighbors]
            p.residual_sq = torch.stack(parts).sum()
            p.D, p.G, p.q = D_new, G_new, q_new
            p._round += 1

        z_vec = torch.cat([t.reshape(-1) for t in z_parts])
        xt_vec = torch.cat([t.reshape(-1) for t in xt_parts])
        e_k = float(torch.linalg.vector_norm(z_vec - xt_vec))

        r = net.aux_residual()
        if r < TOL:
            return k + 1, e_k
    return MAX_ITER, e_k


def main():
    device = torch.device("cpu")
    results = {}
    for density, d in DCFG.items():
        mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                                  G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=(0.05, 0.18))
        _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=d["base_seed"])
        test_specs, _, _ = sample_level_width_perturbed_dataset(
            base_spec, n_instances=N_INSTANCES - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
        test_specs = [base_spec] + test_specs

        model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
            mc, rho=d["gnn_rho"], scale=1.0, rho_slow=d["rho_slow"], scale_slow=d["scale_slow"],
            hidden_dim=config.GNN_HIDDEN_DIM))
        ckpt_path = os.path.join(_EXPERIMENTS, "runs", "official", d["ckpt"])
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.eval()

        ek_dr, ek_learned = [], []
        for i, spec in enumerate(test_specs):
            seed = config.TEST_SEED * 7919 + i
            gen = torch.Generator(device=device); gen.manual_seed(seed)
            net_dr = market.build_network(spec, device=device, rng=gen)
            with torch.no_grad():
                k_dr, e_dr = rollout_capture(net_dr, d["gamma"], config.DR_LAMBDA, config.DR_BISECT_STEPS, use_gnn=False)

            gen2 = torch.Generator(device=device); gen2.manual_seed(seed)
            net_l = market.build_network(spec, device=device, rng=gen2)
            gnn_mod.deploy(model, net_l)
            with torch.no_grad():
                k_l, e_l = rollout_capture(net_l, d["gamma"], config.DR_LAMBDA, config.DR_BISECT_STEPS, use_gnn=True)

            ek_dr.append(e_dr); ek_learned.append(e_l)
            print(f"  density={density} {i+1}/{N_INSTANCES}: DR k={k_dr} e^k={e_dr:.3e}  |  "
                  f"DR+GNN k={k_l} e^k={e_l:.3e}", flush=True)

        results[density] = {
            "ek_dr_median": float(np.median(ek_dr)), "ek_learned_median": float(np.median(ek_learned)),
            "ek_dr_all": ek_dr, "ek_learned_all": ek_learned,
        }
        print(f"density={density}: median e^k DR={np.median(ek_dr):.3e}  DR+GNN={np.median(ek_learned):.3e}", flush=True)

    out_path = os.path.join(OUT_DIR, "table_ek_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
