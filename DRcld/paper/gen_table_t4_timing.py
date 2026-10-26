#!/usr/bin/env python3
"""
gen_table_t4_timing.py
========================
Valutazione extra (NON training) per la Tabella T4: tempo medio per
agente per iterazione della prox locale (bisezione, Network.local_prox)
e di una valutazione di Phi_w (LocalAgent.perturbation: messaggi,
aggregazione, GRU, teste di uscita), sulle stesse ~5 istanze di test
ufficiali per densita', con il checkpoint gia' allenato (nessun training).

Replica ESATTAMENTE la sequenza di Network.dr_step (stesse chiamate,
stesso ordine), solo per isolare i due tempi con time.perf_counter()
attorno a local_prox e a gnn_agent.perturbation separatamente -- nessuna
modifica ai file sorgente, solo chiamate dirette dallo script esterno.

CPU singolo thread (torch.set_num_threads(1)), float64 (dtype di default
del progetto), eseguito in isolamento (non in parallelo con altri job) per
una misura pulita.

Salva un JSON in paper/table_t4_data/.
"""
from __future__ import annotations

import os
import sys
import json
import time
import platform

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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_t4_data")
os.makedirs(OUT_DIR, exist_ok=True)

# density -> (base_seed, gamma, gnn_rho, rho_slow, scale_slow, checkpoint relpath)
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
N_INSTANCES = 5
BISECT_STEPS = config.DR_BISECT_STEPS
MAX_ITER = config.DR_MAX_ITER
TOL = config.DR_TOLERANCE


def timed_rollout(net, gamma, lam, n_bisect):
    """Replica Network.run_dr's loop + Prosumer.dr_step, cronometrando
    separatamente local_prox e gnn_agent.perturbation per ogni agente a
    ogni iterazione. Ritorna (prox_times, gnn_times) -- liste piatte, un
    elemento per (agente, iterazione)."""
    prox_times, gnn_times = [], []
    net.reset()
    for k in range(MAX_ITER):
        net.communicate()
        for p in net.prosumers:
            qbar = p.shadow_q()
            rD, rG = p.D, p.G
            rq = {m: 2.0 * qbar[m] - p.q[m] for m in p.neighbors}

            t0 = time.perf_counter()
            zD, zG, zq = p.local_prox(gamma, rD, rG, rq, n_bisect, tau=0.0)
            prox_times.append(time.perf_counter() - t0)

            D_new = p.D + lam * (zD - p.D)
            G_new = p.G + lam * (zG - p.G)
            q_new = {m: p.q[m] + lam * (zq[m] - qbar[m]) for m in p.neighbors}

            if p.gnn_agent is not None:
                t0 = time.perf_counter()
                state = p.gnn_agent.gnn.local_state(p)
                v_D, v_G, v_q, mem_new = p.gnn_agent.perturbation(state, p._round)
                gnn_times.append(time.perf_counter() - t0)
                D_new = D_new + v_D
                G_new = G_new + v_G
                q_new = {m: q_new[m] + v_q[m] for m in p.neighbors}
                p._gnn_mem = mem_new

            parts = [(D_new - p.D) ** 2, (G_new - p.G) ** 2]
            parts += [(q_new[m] - p.q[m]) ** 2 for m in p.neighbors]
            p.residual_sq = torch.stack(parts).sum()
            p.D, p.G, p.q = D_new, G_new, q_new
            p._round += 1

        r = net.aux_residual()
        if r < TOL:
            break
    return prox_times, gnn_times


def main():
    device = torch.device("cpu")
    results = {}
    for density, dcfg in DCFG.items():
        mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                                  G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=(0.05, 0.18))
        _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=dcfg["base_seed"])
        test_specs, _, _ = sample_level_width_perturbed_dataset(
            base_spec, n_instances=N_INSTANCES - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
        test_specs = [base_spec] + test_specs

        model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
            mc, rho=dcfg["gnn_rho"], scale=1.0, rho_slow=dcfg["rho_slow"], scale_slow=dcfg["scale_slow"],
            hidden_dim=config.GNN_HIDDEN_DIM))
        ckpt_path = os.path.join(_EXPERIMENTS, "runs", "official", dcfg["ckpt"])
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.eval()

        all_prox, all_gnn = [], []
        for i, spec in enumerate(test_specs):
            seed = config.TEST_SEED * 7919 + i
            gen = torch.Generator(device=device); gen.manual_seed(seed)
            net = market.build_network(spec, device=device, rng=gen)
            gnn_mod.deploy(model, net)
            with torch.no_grad():
                prox_t, gnn_t = timed_rollout(net, dcfg["gamma"], config.DR_LAMBDA, BISECT_STEPS)
            all_prox.extend(prox_t); all_gnn.extend(gnn_t)
            print(f"  density={density} instance {i+1}/{N_INSTANCES}: "
                  f"{len(prox_t)} prox calls, {len(gnn_t)} gnn calls", flush=True)

        prox_ms = float(np.mean(all_prox)) * 1000
        gnn_ms = float(np.mean(all_gnn)) * 1000
        results[density] = {
            "prox_ms_per_agent_iter": prox_ms, "gnn_ms_per_agent_iter": gnn_ms,
            "overhead_pct": gnn_ms / prox_ms * 100, "n_prox_samples": len(all_prox), "n_gnn_samples": len(all_gnn),
        }
        print(f"density={density}: prox={prox_ms:.4f}ms  gnn={gnn_ms:.4f}ms  "
              f"overhead={gnn_ms/prox_ms*100:.1f}%", flush=True)

    out = {
        "hardware": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "n_threads": torch.get_num_threads(),
        "dtype": "float64 (project default)",
        "results": results,
    }
    out_path = os.path.join(OUT_DIR, "table_t4_timing.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
