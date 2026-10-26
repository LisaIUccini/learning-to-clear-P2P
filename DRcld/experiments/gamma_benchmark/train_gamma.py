"""
train_gamma.py
==========================================

Trains gamma_gnn.GammaGNN: predicts per-node, instance-adaptive DR prox
step sizes gamma_i (NOT an additive correction like Phi_w, NOT a warm
start like L2WS) -- the classical, unmodified DR update runs throughout,
only the gamma VALUE each node plugs into its own local prox differs.
See gamma_gnn.py's module docstring for the full motivation (ported from
Doerks et al. 2025's "Local alpha_i" ADMM task) and the convergence
caveat (per-node-varying gamma has no proven theorem for this project's
own DR map yet -- validated here only empirically).

Evaluation convention: SAME shared x0 between plain DR and gamma-learned
DR (matches demo.run_pair / Phi_w's own isolation design) -- this is NOT
a warm-start method, so the fair comparison is "does changing gamma per
node help from the SAME start", not L2WS's cold-vs-predicted-start design.

Usage: python3 train_gamma.py [density=0.2] [run_name] [n_train=50]
       [n_test=30] [epochs=25] [base_seed=3] [n_nodes=10] [gamma_base=4.0]
       [unroll=46] [hidden_dim=32] [n_rounds=3] [lr=3e-3]
       [c_old_range=0.05,0.18]
"""
from __future__ import annotations

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_EXPERIMENTS = _os.path.dirname(_HERE)
_ROOT = _os.path.dirname(_EXPERIMENTS)
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _EXPERIMENTS)
_sys.path.insert(0, _os.path.join(_EXPERIMENTS, "l2ws_benchmark"))

import json
import os
import statistics as stats
import sys

import torch
torch.set_num_threads(1)

import config
import market
import algorithm
from gamma_gnn import GammaGNN, GammaGNNConfig, NodeEdgeNormalizer, GammaPredictor, rollout_gamma
from build_realistic_training_dataset import sample_level_width_perturbed_dataset


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


class _RolloutCfg:
    def __init__(self, lam, bisect, unroll):
        self.lam = lam
        self.bisect = bisect
        self.unroll = unroll


def physical_scale(net) -> torch.Tensor:
    half_D = (net.D_max_vec - net.D_min_vec) / 2.0
    half_G = (net.G_max_vec - net.G_min_vec) / 2.0
    return torch.cat([half_D, half_G, net.k_edge])


def regression_loss(z_final, z_star, scale, eps=1e-9):
    denom = torch.linalg.vector_norm(scale) + eps
    return torch.linalg.vector_norm(z_final - z_star) / denom


def compute_fixed_points(specs, dr_cfg, device, label=""):
    z_stars = []
    n_nonconverged = 0
    for spec in specs:
        net = market.build_network(spec, device=device)
        res = algorithm.run_dr(net, dr_cfg, record=False)
        if not res.converged:
            n_nonconverged += 1
        z_stars.append(net.flatten().detach())
    if n_nonconverged:
        print(f"  [{label}] WARNING: {n_nonconverged}/{len(specs)} fixed-point targets did NOT "
              f"converge within {dr_cfg.n_iter} iterations (used anyway)", flush=True)
    return z_stars


def fit_gamma(gnn, normalizer, train_specs, val_specs, x0_seed_base, rollout_cfg,
              epochs, lr, grad_clip, seed, device, dr_cfg):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)
    history = {"train": [], "val": []}

    print(f"  precomputing {len(train_specs)} train + {len(val_specs or [])} val fixed-point "
          f"targets (plain DR to convergence, no grad)...", flush=True)
    train_z_star = compute_fixed_points(train_specs, dr_cfg, device, label="train targets")
    val_z_star = compute_fixed_points(val_specs, dr_cfg, device, label="val targets") if val_specs else None

    def _instance_loss(spec, seed_i, z_star, differentiable):
        gen = torch.Generator(device=device); gen.manual_seed(seed_i)
        net = market.build_network(spec, device=device, rng=gen)
        x0 = net.flatten().detach()
        gamma_vec = gnn.forward_centralised(net, normalizer)
        if not differentiable:
            gamma_vec = gamma_vec.detach()
        x_final, _ = rollout_gamma(net, gamma_vec, rollout_cfg, x0=x0, differentiable=differentiable)
        scale = physical_scale(net)
        return regression_loss(x_final, z_star, scale)

    for epoch in range(epochs):
        gnn.train()
        total = 0.0
        for i, spec in enumerate(train_specs):
            opt.zero_grad()
            loss = _instance_loss(spec, x0_seed_base * 7919 + i, train_z_star[i], differentiable=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), grad_clip)
            opt.step()
            total += loss.item()
        train_loss = total / max(len(train_specs), 1)
        history["train"].append(train_loss)

        val_loss = None
        if val_specs:
            gnn.eval()
            with torch.no_grad():
                val_total = sum(float(_instance_loss(s, x0_seed_base * 7919 + 100000 + i, val_z_star[i],
                                                       differentiable=False))
                                 for i, s in enumerate(val_specs))
            val_loss = val_total / max(len(val_specs), 1)
            history["val"].append(val_loss)

        msg = f"  epoch {epoch + 1:3d}/{epochs} | train {train_loss:+.4f}"
        if val_loss is not None:
            msg += f" | val {val_loss:+.4f}"
        print(msg, flush=True)
    return history


def main():
    density = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"gamma_N10_d{density}_seed3"
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 25
    base_seed = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    n_nodes = int(sys.argv[7]) if len(sys.argv) > 7 else config.BASE_MARKET_N
    gamma_base = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0
    unroll = int(sys.argv[9]) if len(sys.argv) > 9 else 46
    hidden_dim = int(sys.argv[10]) if len(sys.argv) > 10 else 32
    n_rounds = int(sys.argv[11]) if len(sys.argv) > 11 else 3
    lr = float(sys.argv[12]) if len(sys.argv) > 12 else 3e-3
    c_old_range = (tuple(float(x) for x in sys.argv[13].split(",")) if len(sys.argv) > 13
                   else (0.05, 0.18))

    out_dir = os.path.join(_HERE, "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "train.log"), "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"writing all artifacts to {out_dir}/  (density={density}  N={n_nodes}  "
          f"gamma_base={gamma_base}  unroll={unroll}  hidden_dim={hidden_dim}  n_rounds={n_rounds}  "
          f"n_train={n_train}  n_test={n_test}  epochs={epochs}  base_seed={base_seed}  lr={lr})",
          flush=True)
    print("=" * 70)
    print("GAMMA-GNN -- learned per-node DR step size (ADMM-paper-inspired prototype)")
    print(f"density={density}  base_seed={base_seed}  gamma_base={gamma_base}  unroll(train k)={unroll}")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)

    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    _, _, base_spec = market.sample_base(n_nodes, mc, seed=base_seed)
    n_edges = sum(len(v) for v in base_spec.neighbors.values()) // 2
    print(f"  N={base_spec.N}  edges={n_edges}", flush=True)

    train_specs, _, n_disc_train = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct_level=0.10, pct_width=0.04)
    train_specs = [base_spec] + train_specs
    test_specs, _, n_disc_test = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
    test_specs = [base_spec] + test_specs
    n_val = 10
    val_specs, _, n_disc_val = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_val - 1, seed=config.TRAIN_SEED + 2, pct_level=0.10, pct_width=0.04)
    val_specs = [base_spec] + val_specs
    print(f"  train: {n_train} ({n_disc_train} discarded)  |  test: {n_test} ({n_disc_test} discarded)  "
          f"|  val: {n_val} ({n_disc_val} discarded)", flush=True)

    normalizer = NodeEdgeNormalizer(train_specs, device=device)
    gnn = GammaGNN(GammaGNNConfig(hidden_dim=hidden_dim, n_rounds=n_rounds, gamma_base=gamma_base))
    predictor = GammaPredictor(gnn, normalizer, device=device)
    print(f"  model: {gnn.n_parameters()} parameters (independent of N)", flush=True)

    dr_cfg = algorithm.DRConfig(gamma=gamma_base, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    rollout_cfg = _RolloutCfg(lam=config.DR_LAMBDA, bisect=config.DR_BISECT_STEPS, unroll=unroll)

    print(f"\ntraining gamma-GNN (k={unroll} unrolled steps)...", flush=True)
    history = fit_gamma(gnn, normalizer, train_specs, val_specs, config.TRAIN_SEED, rollout_cfg,
                         epochs, lr, grad_clip=config.TRAIN_GRAD_CLIP, seed=config.TRAIN_SEED,
                         device=device, dr_cfg=dr_cfg)
    torch.save({"state_dict": gnn.state_dict(), "cfg": gnn.cfg.__dict__,
                "node_mean": normalizer.node_mean, "node_std": normalizer.node_std,
                "edge_mean": normalizer.edge_mean, "edge_std": normalizer.edge_std},
               os.path.join(out_dir, "gamma_gnn.pt"))

    print(f"\nevaluating: plain DR vs DR with learned per-node gamma, SAME shared x0, "
          f"in-distribution test set", flush=True)
    print("-" * 70)
    dr_iters, gamma_iters, speedups, pcts = [], [], [], []
    for i, spec in enumerate(test_specs):
        seed = config.TEST_SEED * 7919 + i
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        net_dr = market.build_network(spec, device=device, rng=gen)
        x0 = net_dr.flatten().clone()
        res_dr = algorithm.run_dr(net_dr, dr_cfg, record=False)

        net_g = market.build_network(spec, device=device)
        net_g.unflatten(x0)
        gamma_vec = predictor.predict(spec, differentiable=False)
        x_final, residuals = rollout_gamma(net_g, gamma_vec, _RolloutCfg(
            lam=config.DR_LAMBDA, bisect=config.DR_BISECT_STEPS, unroll=config.DR_MAX_ITER),
            x0=x0, differentiable=False)
        n_g = len(residuals)
        converged_g = bool(residuals and float(residuals[-1]) < config.DR_TOLERANCE)
        # rollout_gamma doesn't early-stop on tol (fixed unroll budget) --
        # find the first iteration where residual < tol for a fair iteration count
        n_g_conv = next((k + 1 for k, r in enumerate(residuals) if float(r) < config.DR_TOLERANCE), n_g)

        sp = res_dr.n_iter / max(n_g_conv, 1)
        pct = (res_dr.n_iter - n_g_conv) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); gamma_iters.append(n_g_conv); speedups.append(sp); pcts.append(pct)
        print(f"  [gamma-gnn/d{density}] {i + 1:2d}/{n_test}  DR {res_dr.n_iter:5d} it  |  "
              f"gamma-DR {n_g_conv:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)  "
              f"gamma range=[{gamma_vec.min().item():.2f},{gamma_vec.max().item():.2f}]", flush=True)

    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"\n  [gamma-gnn/d{density}] mean speed-up {stats.mean(speedups):.2f}x   "
          f"median {stats.median(speedups):.2f}x   min {min(speedups):.2f}x   max {max(speedups):.2f}x   "
          f"win-rate {win:.0%} ({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)
    print(f"  [gamma-gnn/d{density}] mean % reduction: {stats.mean(pcts):.1f}%  median: {stats.median(pcts):.1f}%",
          flush=True)

    results = {"n_test": n_test, "density": density, "base_seed": base_seed, "gamma_base": gamma_base,
               "dr_iters": dr_iters, "gamma_iters": gamma_iters, "speedups": speedups, "pcts": pcts,
               "summary": {"mean": stats.mean(speedups), "median": stats.median(speedups), "win_rate": win}}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE -- {out_dir}/", flush=True)
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
