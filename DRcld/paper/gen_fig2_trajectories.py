#!/usr/bin/env python3
"""
gen_fig2_trajectories.py
=========================
Valutazione extra (NON training, NON ri-sweep di iperparametri) per
ottenere le traiettorie (cost, violation, residual, per iterazione) di
A2DR e di L2WS distribuito (ENTRAMBE le loss, reg e asym) sulle stesse 30
istanze di test ufficiali, per tutte e 6 le densita' -- richiesto dalla
Figura 2 (iterazioni-vs-accuratezza/tolleranza, stile Moret), che altrimenti
avrebbe solo le curve DR/DR+GNN/L2WS-cent (gia' disponibili senza calcolo
extra dai dati ufficiali).

- A2DR: usa gamma/eta/mmax gia' calibrati (stessi di a2dr_vs_official_d*),
  stesso x0 fresh-random per istanza (stesso seed di compare_a2dr_vs_official.py).
  A2DRResult traccia anche `violation` (aggiunta minima a a2dr.py, stessa
  convenzione di DRResult).
- L2WS distribuito: carica il checkpoint h_w_dist.pt gia' allenato
  (nessun training), calcola il warm start via forward_centralised, poi
  DR con record=True a partire da quel warm start -- stessa convenzione
  di train_l2ws.py's run_pair. Entrambe le loss (regression/asymmetric),
  ciascuna col proprio checkpoint.
- j_star ricalcolato con equilibrium.solve_equilibrium_qp per ogni
  istanza (stessa funzione di diagnostics.py), non riusato da altri file.

Se un JSON esiste gia' per una densita' e contiene gia' la loss L2WS
richiesta CON il campo residual, quella parte viene riusata (non
ricalcolata) -- solo i pezzi mancanti (A2DR, l'altra loss L2WS) vengono
calcolati. Salva un JSON per densita' in paper/fig2_data/.
"""
from __future__ import annotations

import os
import sys
import json

_ROOT = "/home/lisa/energy_market_fixed_point/DRcld"
_EXPERIMENTS = os.path.join(_ROOT, "experiments")
_L2WS_DIR = os.path.join(_EXPERIMENTS, "l2ws_benchmark")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EXPERIMENTS)
sys.path.insert(0, _L2WS_DIR)

import torch
torch.set_num_threads(1)

import config
import market
import algorithm
import a2dr
from equilibrium import solve_equilibrium_qp
from build_realistic_training_dataset import sample_level_width_perturbed_dataset
from distributed_model import DistributedWarmStartGNN, DistributedWarmStartConfig, DistributedPredictor, NodeEdgeNormalizer

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_data")
os.makedirs(OUT_DIR, exist_ok=True)

# density -> (base_seed, dr_gamma, a2dr_eta, a2dr_mmax)
DENSITY_CFG = {
    0.2: dict(base_seed=3, gamma=4.0, eta=1e-4, mmax=3),
    0.4: dict(base_seed=3, gamma=4.0, eta=1e-4, mmax=5),
    0.5: dict(base_seed=1, gamma=6.0, eta=1e-4, mmax=5),
    0.6: dict(base_seed=3, gamma=4.0, eta=3e-5, mmax=5),
    0.8: dict(base_seed=0, gamma=4.0, eta=1e-4, mmax=3),
    1.0: dict(base_seed=2, gamma=6.0, eta=3e-5, mmax=5),
}
LOSSES = ["regression", "asymmetric"]
N_TEST = 30
COMPUTE_A2DR = True


def load_distributed_predictor(density, base_seed, loss, device):
    ckpt_path = os.path.join(_L2WS_DIR, "runs", "l2ws", "distributed",
                              f"l2ws_N10_d{density}_seed{base_seed}_distributed_{loss}", "h_w_dist.pt")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = DistributedWarmStartConfig(**blob["cfg"])
    gnn = DistributedWarmStartGNN(cfg)
    gnn.load_state_dict(blob["state_dict"])
    gnn.eval()
    normalizer = NodeEdgeNormalizer.__new__(NodeEdgeNormalizer)
    normalizer.node_mean = blob["node_mean"]; normalizer.node_std = blob["node_std"]
    normalizer.edge_mean = blob["edge_mean"]; normalizer.edge_std = blob["edge_std"]
    predictor = DistributedPredictor(gnn, normalizer, device=device)
    return predictor, cfg.n_rounds


def already_have_l2ws(existing, loss):
    d = existing.get("l2ws_dist", {}).get(loss) if existing else None
    return bool(d and d.get("instances") and "residual" in d["instances"][0])


def already_have_a2dr(existing):
    d = existing.get("a2dr") if existing else None
    return bool(d and d.get("instances") and "residual" in d["instances"][0])


def main():
    device = torch.device("cpu")
    for density, dcfg in DENSITY_CFG.items():
        print(f"=== density={density} ===", flush=True)
        base_seed = dcfg["base_seed"]
        out_path = os.path.join(OUT_DIR, f"fig2_extra_trajectories_d{density}.json")
        existing = json.load(open(out_path)) if os.path.exists(out_path) else {}
        # migra lo schema vecchio: l2ws_dist piatto (una sola loss) -> keyed per loss
        if "l2ws_dist" in existing and "loss" in existing["l2ws_dist"]:
            old = existing["l2ws_dist"]
            existing["l2ws_dist"] = {old["loss"]: old}

        need_a2dr = COMPUTE_A2DR and not already_have_a2dr(existing)
        need_losses = [l for l in LOSSES if not already_have_l2ws(existing, l)]
        if not need_a2dr and not need_losses:
            print(f"  already complete, skipping", flush=True)
            continue

        mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                                  G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=(0.05, 0.18))
        _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
        test_specs, _, n_disc = sample_level_width_perturbed_dataset(
            base_spec, n_instances=N_TEST - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
        test_specs = [base_spec] + test_specs
        print(f"  test set: {len(test_specs)} instances ({n_disc} infeasible discarded)  "
              f"need_a2dr={need_a2dr}  need_losses={need_losses}", flush=True)

        dr_cfg = algorithm.DRConfig(gamma=dcfg["gamma"], lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                     tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
        a2_cfg = a2dr.A2DRConfig(gamma=dcfg["gamma"], lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                  tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS,
                                  mmax=dcfg["mmax"], eta=dcfg["eta"])

        predictors = {}
        for loss in need_losses:
            predictors[loss], L = load_distributed_predictor(density, base_seed, loss, device)
            print(f"  loaded distributed L2WS checkpoint (loss={loss}, L={L})", flush=True)

        a2dr_insts = []
        l2ws_insts = {loss: [] for loss in need_losses}
        for i, spec in enumerate(test_specs):
            seed = config.TEST_SEED * 7919 + i
            net_j = market.build_network(spec, device=device)
            j_star = solve_equilibrium_qp(net_j)
            j_denom = max(abs(j_star), 1e-9)

            res_a2 = None
            if need_a2dr:
                gen = torch.Generator(device=device); gen.manual_seed(seed)
                net_rand = market.build_network(spec, device=device, rng=gen)
                x0 = net_rand.flatten().clone()
                net_a2 = market.build_network(spec, device=device)
                net_a2.unflatten(x0)
                res_a2 = a2dr.run_a2dr(net_a2, a2_cfg, record=True)
                a2dr_insts.append({
                    "j_star": j_star, "n_iter": res_a2.n_iter, "converged": res_a2.converged,
                    "cost": res_a2.cost, "violation": res_a2.violation, "residual": res_a2.residual,
                    "relative_gap": [abs(c - j_star) / j_denom for c in res_a2.cost],
                })

            msg = f"  {i + 1:2d}/{len(test_specs)}  "
            if res_a2:
                msg += f"A2DR {res_a2.n_iter:5d} it ({'conv' if res_a2.converged else 'CAP'})  |  "
            for loss in need_losses:
                net_l = market.build_network(spec, device=device)
                warm_start = predictors[loss].predict(spec, differentiable=False)
                net_l.unflatten(warm_start)
                res_l = algorithm.run_dr(net_l, dr_cfg, record=True)
                L = predictors[loss].gnn.cfg.n_rounds
                l2ws_insts[loss].append({
                    "j_star": j_star, "n_iter": res_l.n_iter, "converged": res_l.converged, "L": L,
                    "cost": res_l.cost, "violation": res_l.violation, "residual": res_l.residual,
                    "relative_gap": [abs(c - j_star) / j_denom for c in res_l.cost],
                })
                msg += f"L2WS-{loss[:3]}-dist(+L) {res_l.n_iter + L:5d} it ({'conv' if res_l.converged else 'CAP'})  "
            print(msg, flush=True)

        out = existing if existing else {"density": density, "base_seed": base_seed, "n_test": len(test_specs)}
        out.setdefault("l2ws_dist", {})
        if need_a2dr:
            out["a2dr"] = {"eta": dcfg["eta"], "mmax": dcfg["mmax"], "gamma": dcfg["gamma"], "instances": a2dr_insts}
        for loss in need_losses:
            L = predictors[loss].gnn.cfg.n_rounds
            out["l2ws_dist"][loss] = {"loss": loss, "L": L, "instances": l2ws_insts[loss]}

        with open(out_path, "w") as f:
            json.dump(out, f)
        print(f"  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
