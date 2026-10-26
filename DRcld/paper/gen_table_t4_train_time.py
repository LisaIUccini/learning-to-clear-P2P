#!/usr/bin/env python3
"""
gen_table_t4_train_time.py
=============================
Misura ISOLATA (nessun altro job in esecuzione in parallelo) del tempo di
training di Phi_w: stessa identica costruzione di train_official_pilot.py
(generator="old", stessi gamma/unroll/gnn_rho/omega_mode per densita',
letti dai log reali -- i moduli GAMMA_STAR/UNROLL_STAR del file originale
sono STALE/sovrascritti via CLI override per le run ufficiali, quindi
NON vanno riusati), ma con epochs=3 invece di 25, cronometrando SOLO
trainer.fit(...) ed estrapolando linearmente a 25 epoche.

Thread singolo, CPU, float64 (default del progetto).

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

import torch
torch.set_num_threads(1)

import config
import market
import algorithm
import gnn as gnn_mod
import train as train_mod
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_t4_data")
os.makedirs(OUT_DIR, exist_ok=True)

TIMED_EPOCHS = 3
FULL_EPOCHS = 25
N_TRAIN = 50

# valori REALI delle run ufficiali (letti dai train.log, non dai default stale del modulo)
DCFG = {
    0.2: dict(base_seed=3, gamma=4.0, unroll=426, gnn_rho=0.9765, rho_slow=0.99, scale_slow=0.0,
              omega_mode="fixed", omega_power=1.5),
    0.4: dict(base_seed=3, gamma=4.0, unroll=318, gnn_rho=0.9686, rho_slow=0.99, scale_slow=0.4,
              omega_mode="fixed", omega_power=1.5),
    0.5: dict(base_seed=1, gamma=6.0, unroll=570, gnn_rho=0.9825, rho_slow=0.99, scale_slow=0.0,
              omega_mode="fixed", omega_power=1.5),
    0.6: dict(base_seed=3, gamma=4.0, unroll=321, gnn_rho=0.9688, rho_slow=0.99, scale_slow=0.0,
              omega_mode="fixed", omega_power=1.5),
    0.8: dict(base_seed=0, gamma=4.0, unroll=1159, gnn_rho=0.9914, rho_slow=0.99, scale_slow=0.4,
              omega_mode="dr_slope", omega_power=0.0),
    1.0: dict(base_seed=2, gamma=6.0, unroll=1354, gnn_rho=0.9926, rho_slow=0.99, scale_slow=0.4,
              omega_mode="dr_slope", omega_power=0.0),
}


def main():
    device = torch.device("cpu")
    results = {}
    for density, d in DCFG.items():
        torch.manual_seed(config.TRAIN_SEED)
        mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                                  G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=(0.05, 0.18))
        _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=d["base_seed"])
        train_specs, _, _ = sample_level_width_perturbed_dataset(
            base_spec, n_instances=N_TRAIN - 1, seed=config.TRAIN_SEED, pct_level=0.10, pct_width=0.04)
        train_specs = [base_spec] + train_specs
        val_specs, _, _ = sample_level_width_perturbed_dataset(
            base_spec, n_instances=9, seed=config.TRAIN_SEED + 2, pct_level=0.10, pct_width=0.04)
        val_specs = [base_spec] + val_specs

        dr_cfg = algorithm.DRConfig(gamma=d["gamma"], lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                     tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
        train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                           unroll=d["unroll"], gamma_mono=config.TRAIN_GAMMA_MONO,
                                           epochs=TIMED_EPOCHS, w_behind=2.0, w_ahead=1.0,
                                           lr=config.TRAIN_LR, lr_final_frac=0.01,
                                           seed=config.TRAIN_SEED, loss_mode="asymmetric",
                                           omega_mode=d["omega_mode"], omega_power=d["omega_power"])
        model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
            mc, rho=d["gnn_rho"], scale=1.0, rho_slow=d["rho_slow"], scale_slow=d["scale_slow"],
            hidden_dim=config.GNN_HIDDEN_DIM))
        trainer = train_mod.Trainer(model, train_cfg)

        t0 = time.perf_counter()
        trainer.fit(train_specs, val_specs=val_specs, device=device)
        elapsed = time.perf_counter() - t0

        full_min = elapsed / TIMED_EPOCHS * FULL_EPOCHS / 60.0
        results[density] = {"timed_epochs": TIMED_EPOCHS, "timed_seconds": elapsed,
                             "extrapolated_25ep_minutes": full_min, "unroll_K": d["unroll"]}
        print(f"density={density}: {TIMED_EPOCHS} epochs in {elapsed:.1f}s  ->  "
              f"extrapolated 25 epochs = {full_min:.1f} min", flush=True)

    out = {
        "hardware": platform.processor() or platform.machine(), "platform": platform.platform(),
        "n_threads": torch.get_num_threads(), "timed_epochs": TIMED_EPOCHS, "full_epochs": FULL_EPOCHS,
        "results": results,
    }
    out_path = os.path.join(OUT_DIR, "table_t4_train_time.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
