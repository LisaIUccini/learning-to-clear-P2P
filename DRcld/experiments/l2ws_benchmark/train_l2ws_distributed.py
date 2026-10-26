"""
train_l2ws_distributed.py
==========================================

DISTRIBUTED L2WS: same benchmark as train_l2ws.py (same market/dataset
pipeline, same loss functions, same evaluation convention), but using
`distributed_model.DistributedWarmStartGNN` (message-passing, L one-time
communication rounds) instead of `model.WarmStartMLP` (one monolithic MLP
reading every node's private data at once). First prototype -- see
distributed_model.py's module docstring for the architecture and the
communication-cost reasoning.

Usage: python3 train_l2ws_distributed.py [density=0.2] [run_name]
       [n_train=50] [n_test=30] [epochs=50] [base_seed=3] [n_nodes=10]
       [gamma=4.0] [unroll=46] [hidden_dim=32] [n_rounds=2] [lr=3e-3]
       [c_old_range=0.05,0.18] [loss_mode=regression]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os
import statistics as stats
import sys

import torch
torch.set_num_threads(1)

import config
import market
import algorithm
import train as train_mod
from diagnostics import save_instance_plots
from demo import plot_results, plot_training_loss, write_run_notes
from plot_style import stats_from
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

from distributed_model import (DistributedWarmStartGNN, DistributedWarmStartConfig,
                                NodeEdgeNormalizer, DistributedPredictor)
# reuse the SAME loss functions / fixed-point & reference-trajectory
# precomputation already validated for the centralised benchmark -- none of
# them are tied to WarmStartMLP, they just operate on (z_final, residuals)
from train_l2ws import (fp_residual_loss, regression_loss, compute_fixed_points,
                         compute_reference_trajectories, evaluate_with_curves, run_pair,
                         physical_scale)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def fit_distributed(gnn: DistributedWarmStartGNN, normalizer: NodeEdgeNormalizer, train_specs, val_specs,
                     dr_train_cfg: train_mod.TrainConfig, epochs: int, lr: float, grad_clip: float,
                     seed: int, device, loss_mode: str, dr_cfg: algorithm.DRConfig,
                     gamma_mono: float, w_behind: float, w_ahead: float, omega_mode: str, omega_power: float):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)
    history = {"train": [], "val": []}

    train_z_star, val_z_star = None, None
    train_ref, val_ref = None, None
    if loss_mode == "regression":
        print(f"  precomputing {len(train_specs)} train + {len(val_specs or [])} val fixed-point "
              f"targets (plain DR to convergence, no grad)...", flush=True)
        train_z_star = compute_fixed_points(train_specs, dr_cfg, device, label="train targets")
        if val_specs:
            val_z_star = compute_fixed_points(val_specs, dr_cfg, device, label="val targets")
    elif loss_mode == "asymmetric":
        print(f"  precomputing {len(train_specs)} train + {len(val_specs or [])} val cold-start "
              f"reference trajectories (plain DR, {dr_train_cfg.unroll} steps, no grad)...", flush=True)
        train_ref = compute_reference_trajectories(train_specs, dr_train_cfg, device, seed=seed, label="train ref")
        if val_specs:
            val_ref = compute_reference_trajectories(val_specs, dr_train_cfg, device, seed=seed + 1, label="val ref")

    def _instance_loss(spec, differentiable, z_star=None, residuals_ref=None):
        net = market.build_network(spec, device=device)
        warm_start = gnn.forward_centralised(net, normalizer)
        if not differentiable:
            warm_start = warm_start.detach()
        x_final, residuals = train_mod.rollout(net, None, dr_train_cfg, x0=warm_start, differentiable=differentiable)
        scale = physical_scale(net)
        if loss_mode == "regression":
            return regression_loss(x_final, z_star, scale)
        if loss_mode == "asymmetric":
            return train_mod.asymmetric_loss(residuals, residuals_ref, scale, gamma_mono, 1e-9,
                                              w_behind, w_ahead, omega_power, omega_mode)
        return fp_residual_loss(residuals, scale)

    for epoch in range(epochs):
        gnn.train()
        total = 0.0
        for i, spec in enumerate(train_specs):
            opt.zero_grad()
            z_star = train_z_star[i] if train_z_star is not None else None
            ref = train_ref[i] if train_ref is not None else None
            loss = _instance_loss(spec, differentiable=True, z_star=z_star, residuals_ref=ref)
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
                val_total = sum(float(_instance_loss(s, differentiable=False,
                                                       z_star=(val_z_star[i] if val_z_star is not None else None),
                                                       residuals_ref=(val_ref[i] if val_ref is not None else None)))
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
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"l2ws_dist_N10_d{density}_seed3"
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    base_seed = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    n_nodes = int(sys.argv[7]) if len(sys.argv) > 7 else config.BASE_MARKET_N
    dr_gamma = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0
    unroll = int(sys.argv[9]) if len(sys.argv) > 9 else 46
    hidden_dim = int(sys.argv[10]) if len(sys.argv) > 10 else 32
    n_rounds = int(sys.argv[11]) if len(sys.argv) > 11 else 2
    lr = float(sys.argv[12]) if len(sys.argv) > 12 else 3e-3
    c_old_range = (tuple(float(x) for x in sys.argv[13].split(",")) if len(sys.argv) > 13
                   else (0.05, 0.18))
    loss_mode = sys.argv[14] if len(sys.argv) > 14 else "regression"
    assert loss_mode in ("regression", "fp_residual", "asymmetric"), f"unknown loss_mode {loss_mode!r}"
    gamma_mono = float(sys.argv[15]) if len(sys.argv) > 15 else config.TRAIN_GAMMA_MONO
    w_behind = float(sys.argv[16]) if len(sys.argv) > 16 else config.TRAIN_W_BEHIND
    w_ahead = float(sys.argv[17]) if len(sys.argv) > 17 else config.TRAIN_W_AHEAD
    omega_mode = sys.argv[18] if len(sys.argv) > 18 else "dr_slope"
    omega_power = float(sys.argv[19]) if len(sys.argv) > 19 else 0.0
    shared_weights = (sys.argv[20].lower() != "false") if len(sys.argv) > 20 else True
    update_type = sys.argv[21] if len(sys.argv) > 21 else "gru"

    out_dir = os.path.join("runs", "l2ws", "distributed", run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "train.log"), "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"writing all artifacts to {out_dir}/  (density={density}  N={n_nodes}  gamma={dr_gamma}  "
          f"unroll={unroll}  hidden_dim={hidden_dim}  n_rounds={n_rounds}  shared_weights={shared_weights}  "
          f"update_type={update_type}  n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"base_seed={base_seed}  lr={lr}  loss_mode={loss_mode})",
          flush=True)
    print("=" * 70)
    print("DISTRIBUTED L2WS -- message-passing GNN warm-start (prototype)")
    print(f"h_w = L={n_rounds}-round message-passing GNN ({'shared' if shared_weights else 'separate'} "
          f"weights, {update_type} update), PLAIN DR thereafter")
    print(f"density={density}  base_seed={base_seed}  gamma={dr_gamma}  unroll(train k)={unroll}")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)

    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    _, _, base_spec = market.sample_base(n_nodes, mc, seed=base_seed)
    n_edges = sum(len(v) for v in base_spec.neighbors.values()) // 2
    print(f"  N={base_spec.N}  edges={n_edges}", flush=True)

    train_specs, _, n_discarded_train = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct_level=0.10, pct_width=0.04)
    train_specs = [base_spec] + train_specs
    test_specs, _, n_discarded_test = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=0.10, pct_width=0.04)
    test_specs = [base_spec] + test_specs
    n_val = 10
    val_specs, _, n_discarded_val = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_val - 1, seed=config.TRAIN_SEED + 2, pct_level=0.10, pct_width=0.04)
    val_specs = [base_spec] + val_specs
    print(f"  train: {n_train} ({n_discarded_train} discarded)  |  test: {n_test} "
          f"({n_discarded_test} discarded)  |  val: {n_val} ({n_discarded_val} discarded)", flush=True)

    normalizer = NodeEdgeNormalizer(train_specs, device=device)
    gnn = DistributedWarmStartGNN(DistributedWarmStartConfig(
        hidden_dim=hidden_dim, n_rounds=n_rounds, shared_weights=shared_weights, update_type=update_type))
    predictor = DistributedPredictor(gnn, normalizer, device=device)
    print(f"  model: {gnn.n_parameters()} parameters (independent of N)", flush=True)

    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    dr_train_cfg = train_mod.TrainConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, bisect=config.DR_BISECT_STEPS,
                                          unroll=unroll)

    print(f"\ntraining distributed h_w (loss_mode={loss_mode}, k={unroll} steps)...", flush=True)
    history = fit_distributed(gnn, normalizer, train_specs, val_specs, dr_train_cfg, epochs, lr,
                               grad_clip=config.TRAIN_GRAD_CLIP, seed=config.TRAIN_SEED, device=device,
                               loss_mode=loss_mode, dr_cfg=dr_cfg, gamma_mono=gamma_mono,
                               w_behind=w_behind, w_ahead=w_ahead, omega_mode=omega_mode, omega_power=omega_power)
    torch.save({"state_dict": gnn.state_dict(), "cfg": gnn.cfg.__dict__,
                "node_mean": normalizer.node_mean, "node_std": normalizer.node_std,
                "edge_mean": normalizer.edge_mean, "edge_std": normalizer.edge_std},
               os.path.join(out_dir, "h_w_dist.pt"))
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print(f"\nevaluating: cold-start plain DR vs plain DR warm-started by distributed h_w(theta), "
          f"in-distribution test set", flush=True)
    print("-" * 70)
    test_res = evaluate_with_curves(test_specs, predictor, dr_cfg, device, seed=config.TEST_SEED,
                                     label=f"l2ws-dist/d{density}")
    st = stats_from(test_res)
    print(f"  mean % reduction (per-instance average): mean={st['pct_mean']:.1f}%  "
          f"median={st['pct_median']:.1f}%", flush=True)

    subtitle = (f"Distributed L2WS benchmark, density={density}, gamma={dr_gamma}  |  "
                f"trained on {n_train} instances, {epochs} epochs, unroll(k)={unroll}, L={n_rounds} rounds  |  "
                f"tested on {n_test}, cold-start baseline")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    write_run_notes(out_dir, f"""\
# {run_name}

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L={n_rounds}
one-time communication rounds instead of reading every node's private data into one global MLP.

density={density}, base_seed={base_seed}, gamma={dr_gamma}, training unroll k={unroll},
hidden_dim={hidden_dim}, n_rounds={n_rounds}, lr={lr}, loss_mode={loss_mode}.

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}.
Mean % reduction: {st['pct_mean']:.1f}%, median {st['pct_median']:.1f}%.
""")
    print(f"\nDONE -- {out_dir}/", flush=True)
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
