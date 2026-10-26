"""
train_l2ws.py
==========================================

L2WS (Learning to Warm-Start) benchmark -- Sambharya, Hall, Amos & Stellato,
"Learning to Warm-Start Fixed-Point Optimization Algorithms" (JMLR 2024),
applied to this project's Douglas-Rachford p2p market-clearing fixed point.

Unlike this project's own Phi_w (gnn.py / train.py), which perturbs EVERY DR
iteration, this benchmark's feedforward MLP h_w(theta) predicts ONLY the
starting point z^0 for an otherwise completely unmodified DR loop:

    theta (market primitives) --MLP--> z^0 = h_w(theta) --[unroll plain DR]--> z^k

Trained end-to-end by differentiating through `cfg.unroll` un-perturbed DR
steps (train.rollout(net, gnn=None, ...)) against one of two losses (CLI
`loss_mode`, paper Sec. 3.3):
  - "regression" (Eq. 6, DEFAULT since 2026-10-15): ||z - z*(theta)||,
    z*(theta) = plain DR run to convergence, precomputed once per instance
    (no grad). Switched to from "fp_residual" after diagnosing that the
    latter can be satisfied by a DR PLATEAU point (tiny step size without
    true proximity to the equilibrium) -- see regression_loss's docstring.
  - "fp_residual" (Eq. 5): ||T_theta(z) - z|| at the final unrolled step,
    no ground truth needed, but vulnerable to the plateau above for this
    operator.
See model.py's module docstring for the full architecture rationale.

Dataset / DR hyperparameters deliberately mirror
`experiments/train_official_pilot.py`'s `generator="old"` branch EXACTLY
(same MarketConfig overrides, same level+width +/-10%/+/-4% perturbation,
same c_old_range=(0.05,0.18) c<b fix, same train/test/val seed convention)
so this benchmark's numbers are directly comparable to the already-measured
DR+Phi_w results on the same (density, base_seed) combination -- e.g.
density=0.2/base_seed=3/gamma=4.0/unroll=46 reached 6.54x/100%/+83.0% with
Phi_w (runs/official/official_N10_d0.2_seed3_replicate).

Usage: python3 train_l2ws.py [density=0.2] [run_name] [n_train=50]
       [n_test=30] [epochs=50] [base_seed=3] [n_nodes=10] [gamma=4.0]
       [unroll=46] [hidden_dim=64] [n_hidden_layers=1] [lr=3e-3]
       [c_old_range=0.05,0.18] [loss_mode=regression|fp_residual|asymmetric]
       [gamma_mono=2.0] [w_behind=2.0] [w_ahead=1.0] [omega_mode=dr_slope]
       [omega_power=0.0]   (last 5 args only matter for loss_mode=asymmetric)
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os
import statistics as stats
import sys

import numpy as np
import torch

import config
import market
import algorithm
import train as train_mod
from diagnostics import save_instance_plots
from demo import plot_results, plot_training_loss, write_run_notes
from plot_style import stats_from
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

from model import (WarmStartMLP, WarmStartConfig, ThetaNormalizer, L2WSPredictor,
                    build_theta, theta_dim, physical_scale)


class _Tee:
    """Mirrors every `print` to both the terminal and a log file inside the
    run's own `out_dir` -- same convention as train_official_pilot.py."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def run_pair(spec, predictor: L2WSPredictor, dr_cfg: algorithm.DRConfig, device, seed, record=False):
    """Cold-start plain DR (fresh random x0, this project's usual baseline
    AND the paper's own "cold start" baseline) vs. plain DR warm-started
    from h_w(theta) -- deliberately DIFFERENT starting points (not the
    shared-x0 design of demo.run_pair, which isolates a PER-STEP
    correction's effect): the entire point of a warm start is that its
    chosen starting point should already be better than a naive one."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net_dr = market.build_network(spec, device=device, rng=gen)
    res_dr = algorithm.run_dr(net_dr, dr_cfg, record=record)

    net_l = market.build_network(spec, device=device)   # state irrelevant, overwritten next
    warm_start = predictor.predict(spec, differentiable=False)
    net_l.unflatten(warm_start)
    res_learned = algorithm.run_dr(net_l, dr_cfg, record=record)

    return res_dr, res_learned


def evaluate_with_curves(specs, predictor, dr_cfg, device, seed, label):
    dr_iters, learned_iters, speedups, curves_dr, curves_learned = [], [], [], [], []
    for i, spec in enumerate(specs):
        res_dr, res_learned = run_pair(spec, predictor, dr_cfg, device, seed=seed * 7919 + i, record=True)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter); speedups.append(sp)
        curves_dr.append(np.asarray(res_dr.residual)); curves_learned.append(np.asarray(res_learned.residual))
        print(f"  [{label}] {i + 1:2d}/{len(specs)}  DR {res_dr.n_iter:5d} it  |  "
              f"L2WS {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)
    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"  [{label}] mean speed-up {stats.mean(speedups):.2f}x   median {stats.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)
    return {"dr": dr_iters, "learned": learned_iters, "speedup": speedups,
            "curves_dr": curves_dr, "curves_learned": curves_learned}


def fp_residual_loss(residuals, scale, eps=1e-9):
    """Paper Eq. 5: ell^fp_theta(z) = ||T_theta(z) - z||_2, evaluated at
    the FINAL unrolled step only (not averaged over the trajectory --
    unlike this project's own stability_loss/asymmetric_loss, which DO
    average, because those train a PER-STEP correction active throughout;
    here only the starting point is learned, so only where the trajectory
    ENDS UP after k steps is the thing being optimised). Normalised by the
    instance's own physical scale for comparability across instances
    whose raw units differ, matching train.py's own convention."""
    denom = torch.linalg.vector_norm(scale) + eps
    return residuals[-1] / denom


def regression_loss(z_final, z_star, scale, eps=1e-9):
    """Paper Eq. 6: ell^reg_theta(z) = ||z - z*(theta)||_2, z*(theta) a
    KNOWN fixed point of T_theta (here: plain DR run to this project's
    normal convergence tolerance, no grad -- see
    `compute_fixed_points` below). Diagnosed 2026-10-15: the fp-residual
    loss (above) can be satisfied by a point where DR's STEP SIZE has
    collapsed without the trajectory actually being near the true
    equilibrium -- DR is only an AVERAGED (not contractive) operator for
    this market, and this project has documented long near-stationary
    plateaus elsewhere (train.py's hinge_loss docstring) where the
    residual stays tiny for hundreds/thousands of iterations before truly
    converging. Regressing directly against a genuine fixed point sidesteps
    that proxy entirely -- it cannot be satisfied by a plateau point that
    merely LOOKS converged by step size alone."""
    denom = torch.linalg.vector_norm(scale) + eps
    return torch.linalg.vector_norm(z_final - z_star) / denom


def compute_fixed_points(specs, dr_cfg: algorithm.DRConfig, device, label=""):
    """Runs PLAIN DR (no grad, no GNN/MLP) to this project's normal
    convergence tolerance for each spec, once, and returns the FINAL
    auxiliary iterate x^k (net.flatten() after convergence) -- a genuine
    fixed point of T_theta in the SAME space the warm start lives in.
    NOT `equilibrium.solve_equilibrium_qp`'s primal/shadow optimum: that
    lives in the feasible [D,G,Q] space (xbar's limit), which only
    coincides with the raw auxiliary iterate's own limit once DR has
    actually converged -- exactly what we're computing here directly, so
    there is no reason to go through the QP solver for this."""
    z_stars = []
    n_nonconverged = 0
    for i, spec in enumerate(specs):
        net = market.build_network(spec, device=device)
        res = algorithm.run_dr(net, dr_cfg, record=False)
        if not res.converged:
            n_nonconverged += 1
        z_stars.append(net.flatten().detach())
    if n_nonconverged:
        print(f"  [{label}] WARNING: {n_nonconverged}/{len(specs)} fixed-point targets did NOT converge "
              f"within {dr_cfg.n_iter} iterations (used anyway, as the best available target)", flush=True)
    return z_stars


def compute_reference_trajectories(specs, dr_train_cfg: train_mod.TrainConfig, device, seed: int, label=""):
    """For `loss_mode="asymmetric"`: a COLD-START plain-DR residual
    trajectory per instance (no grad), `dr_train_cfg.unroll` steps, same
    length as the warm-started trajectory being trained. Precomputed once
    (fixed across epochs) rather than resampled, so the training target is
    stable -- analogous to `compute_fixed_points` for the regression loss.
    Diagnosed 2026-10-15 (user's own idea): Phi_w's own asymmetric_loss
    (train.py) compares a PER-STEP-CORRECTED trajectory against the SAME
    instance's UNCORRECTED trajectory from the SAME x0 -- that pairing
    doesn't exist here, since h_w only ever touches x0 (there is no
    "same x0, with vs without" pair once x0 itself is the thing chosen).
    The natural analogue: compare the warm-started trajectory's per-step
    residual decay against a COLD-START reference trajectory (different
    x0) -- this penalises falling behind the reference's decay rate at
    EVERY step along the unroll window, not just the final one, which is
    exactly the property fp_residual_loss lacks (it only ever looks at the
    LAST step, so it cannot detect or penalise an early plateau that the
    trajectory never escapes)."""
    refs = []
    for i, spec in enumerate(specs):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed * 1_000_003 + i)
        net = market.build_network(spec, device=device, rng=gen)
        _, residuals_ref = train_mod.rollout(net, None, dr_train_cfg, x0=None, differentiable=False)
        refs.append(residuals_ref)
    return refs


def fit_warm_start(mlp: WarmStartMLP, normalizer: ThetaNormalizer, train_specs, val_specs,
                    dr_train_cfg: train_mod.TrainConfig, epochs: int, lr: float,
                    lr_final_frac, grad_clip: float, seed: int, device,
                    loss_mode: str = "regression", dr_cfg: "algorithm.DRConfig" = None,
                    gamma_mono: float = 2.0, w_behind: float = 2.0, w_ahead: float = 1.0,
                    omega_mode: str = "dr_slope", omega_power: float = 0.0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    sched = None
    if lr_final_frac is not None:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * lr_final_frac)
    history = {"train": [], "val": []}

    train_z_star, val_z_star = None, None
    train_ref, val_ref = None, None
    if loss_mode == "regression":
        assert dr_cfg is not None, "regression loss needs dr_cfg to compute fixed-point targets"
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
        net = market.build_network(spec, device=device)   # state irrelevant; rollout uses explicit x0
        theta = build_theta(spec).to(device)
        theta_n = normalizer(theta)
        warm_start = mlp(theta_n) if differentiable else mlp(theta_n).detach()
        x_final, residuals = train_mod.rollout(net, None, dr_train_cfg, x0=warm_start, differentiable=differentiable)
        scale = physical_scale(net)
        if loss_mode == "regression":
            return regression_loss(x_final, z_star, scale)
        if loss_mode == "asymmetric":
            return train_mod.asymmetric_loss(residuals, residuals_ref, scale, gamma_mono, 1e-9,
                                              w_behind, w_ahead, omega_power, omega_mode)
        return fp_residual_loss(residuals, scale)

    for epoch in range(epochs):
        mlp.train()
        total = 0.0
        for i, spec in enumerate(train_specs):
            opt.zero_grad()
            z_star = train_z_star[i] if train_z_star is not None else None
            ref = train_ref[i] if train_ref is not None else None
            loss = _instance_loss(spec, differentiable=True, z_star=z_star, residuals_ref=ref)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), grad_clip)
            opt.step()
            total += loss.item()
        if sched is not None:
            sched.step()
        train_loss = total / max(len(train_specs), 1)
        history["train"].append(train_loss)

        val_loss = None
        if val_specs:
            mlp.eval()
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
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"l2ws_N10_d{density}_seed3"
    n_train = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    n_test = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    base_seed = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    n_nodes = int(sys.argv[7]) if len(sys.argv) > 7 else config.BASE_MARKET_N
    dr_gamma = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0
    unroll = int(sys.argv[9]) if len(sys.argv) > 9 else 46
    hidden_dim = int(sys.argv[10]) if len(sys.argv) > 10 else 64
    n_hidden_layers = int(sys.argv[11]) if len(sys.argv) > 11 else 1
    lr = float(sys.argv[12]) if len(sys.argv) > 12 else 3e-3
    c_old_range = (tuple(float(x) for x in sys.argv[13].split(",")) if len(sys.argv) > 13
                   else (0.05, 0.18))
    # loss_mode="regression" (Eq. 6, paper) -- default, since 2026-10-15: the
    # original "fp_residual" (Eq. 5) exploited a DR plateau (small step size
    # without true proximity to the equilibrium) on this market's operator;
    # see regression_loss's docstring. "asymmetric" (2026-10-15, user's own
    # idea): reuses Phi_w's own asymmetric_loss (train.py) -- self-supervised
    # like fp_residual (no ground-truth z* needed), but compares the
    # warm-started trajectory against a COLD-START reference trajectory
    # STEP BY STEP across the whole unroll window, not just the final step
    # -- see compute_reference_trajectories's docstring for why this should
    # resist the plateau exploit that broke fp_residual.
    loss_mode = sys.argv[14] if len(sys.argv) > 14 else "regression"
    assert loss_mode in ("regression", "fp_residual", "asymmetric"), f"unknown loss_mode {loss_mode!r}"
    # asymmetric-loss hyperparameters (Phi_w's own TrainConfig/config.py
    # defaults, since the point is testing THE SAME loss mechanism, not a
    # density-specific hand-tuned recipe)
    gamma_mono = float(sys.argv[15]) if len(sys.argv) > 15 else config.TRAIN_GAMMA_MONO
    w_behind = float(sys.argv[16]) if len(sys.argv) > 16 else config.TRAIN_W_BEHIND
    w_ahead = float(sys.argv[17]) if len(sys.argv) > 17 else config.TRAIN_W_AHEAD
    omega_mode = sys.argv[18] if len(sys.argv) > 18 else "dr_slope"
    omega_power = float(sys.argv[19]) if len(sys.argv) > 19 else 0.0

    out_dir = os.path.join("runs", "l2ws", "centralized", run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "train.log"), "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"writing all artifacts to {out_dir}/  (density={density}  N={n_nodes}  "
          f"gamma={dr_gamma}  unroll={unroll}  hidden_dim={hidden_dim}  n_hidden_layers={n_hidden_layers}  "
          f"n_train={n_train}  n_test={n_test}  epochs={epochs}  base_seed={base_seed}  lr={lr}  "
          f"loss_mode={loss_mode})", flush=True)
    print("=" * 70)
    print("L2WS BENCHMARK -- Sambharya, Hall, Amos & Stellato (JMLR 2024)")
    print(f"warm-start MLP h_w(theta) only, PLAIN DR thereafter (no per-step correction)")
    print(f"density={density}  base_seed={base_seed}  gamma={dr_gamma}  unroll(train k)={unroll}")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)

    # Same generator/overrides as train_official_pilot.py's generator="old"
    # branch -- the c<b fix, applied project-wide 2026-10-14.
    mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                              G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0), c_range=c_old_range)
    _, _, base_spec = market.sample_base(n_nodes, mc, seed=base_seed)
    print(f"  [generator=old, c_old_range={c_old_range}, c<b guaranteed: "
          f"max(c)={c_old_range[1]} < min(b)={mc.b_range[0]}]")
    n_edges = sum(len(v) for v in base_spec.neighbors.values()) // 2   # undirected, for display only
    ref_net = market.build_network(base_spec, device=device)
    out_dim, n_directed = ref_net.dim, ref_net.E   # Network.flatten()'s Q block is DIRECTED edges (2x undirected)
    print(f"  N={base_spec.N}  edges={n_edges} (directed={n_directed})  theta_dim={theta_dim(base_spec)}  "
          f"out_dim(=2N+E_directed)={out_dim}", flush=True)

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

    # Fit the theta normalizer on the TRAINING set only.
    train_thetas = [build_theta(s) for s in train_specs]
    normalizer = ThetaNormalizer(train_thetas)

    # Box-midpoint init point for h_w's output bias (D mid, G mid, Q=0) --
    # see model.py's WarmStartMLP docstring.
    D_mid = (base_spec.D_min + base_spec.D_max) / 2.0
    G_mid = (base_spec.G_min + base_spec.G_max) / 2.0
    Q_zero = np.zeros(n_directed, dtype=np.float32)
    init_point = torch.from_numpy(np.concatenate([D_mid, G_mid, Q_zero]).astype(np.float32))

    mlp = WarmStartMLP(theta_dim(base_spec), out_dim,
                        WarmStartConfig(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers),
                        init_point=init_point)
    predictor = L2WSPredictor(mlp, normalizer, device=device)

    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    dr_train_cfg = train_mod.TrainConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, bisect=config.DR_BISECT_STEPS,
                                          unroll=unroll)

    print(f"\ntraining h_w (feedforward MLP, loss_mode={loss_mode}, k={unroll} steps)...", flush=True)
    if loss_mode == "asymmetric":
        print(f"  asymmetric_loss params: gamma_mono={gamma_mono}  w_behind={w_behind}  w_ahead={w_ahead}  "
              f"omega_mode={omega_mode}  omega_power={omega_power}", flush=True)
    history = fit_warm_start(mlp, normalizer, train_specs, val_specs, dr_train_cfg, epochs, lr,
                              lr_final_frac=None, grad_clip=config.TRAIN_GRAD_CLIP,
                              seed=config.TRAIN_SEED, device=device,
                              loss_mode=loss_mode, dr_cfg=dr_cfg,
                              gamma_mono=gamma_mono, w_behind=w_behind, w_ahead=w_ahead,
                              omega_mode=omega_mode, omega_power=omega_power)
    torch.save({"state_dict": mlp.state_dict(), "cfg": mlp.cfg.__dict__,
                "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std},
               os.path.join(out_dir, "h_w.pt"))
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print(f"\nevaluating: cold-start plain DR vs plain DR warm-started by h_w(theta), "
          f"in-distribution test set", flush=True)
    print("-" * 70)
    test_res = evaluate_with_curves(test_specs, predictor, dr_cfg, device, seed=config.TEST_SEED,
                                     label=f"l2ws/d{density}")
    st = stats_from(test_res)
    print(f"  mean % reduction (per-instance average): mean={st['pct_mean']:.1f}%  "
          f"median={st['pct_median']:.1f}%", flush=True)

    subtitle = (f"L2WS benchmark, density={density}, gamma={dr_gamma}  |  "
                f"trained on {n_train} instances, {epochs} epochs, unroll(k)={unroll}  |  "
                f"tested on {n_test}, cold-start baseline")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    save_instance_plots(test_specs, predictor, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**L2WS benchmark** (Sambharya, Hall, Amos & Stellato, JMLR 2024 -- "Learning to Warm-Start
Fixed-Point Optimization Algorithms"), applied to this project's DR p2p market-clearing fixed
point. A feedforward MLP h_w(theta) predicts ONLY the starting point z^0 for plain,
UN-perturbed DR -- the per-iteration DR operator itself is completely unmodified, unlike this
project's own Phi_w (which perturbs every step). Compared against Phi_w results on the SAME
(density={density}, base_seed={base_seed}) combination as the most directly comparable benchmark.

density={density}, base_seed={base_seed}, gamma={dr_gamma} (same as the Phi_w run on this
combination), training unroll k={unroll} (same calibrated value used for Phi_w), hidden_dim=
{hidden_dim}, n_hidden_layers={n_hidden_layers}, lr={lr}, loss_mode={loss_mode}.

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4%
perturbation, same "old" generator + c<b fix (c_old_range={c_old_range}) as
train_official_pilot.py.

**Baseline**: cold-start plain DR from a fresh random x0 (this project's and the paper's own
baseline) -- NOT the same-x0 isolation design used for Phi_w, since a warm start's entire point
is choosing a BETTER starting point, not correcting the trajectory from a shared one.

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}.
Mean % reduction: {st['pct_mean']:.1f}%, median {st['pct_median']:.1f}%.
""")
    print(f"\nDONE -- {out_dir}/", flush=True)
    sys.stdout = sys.__stdout__   # restore BEFORE closing, else interpreter
    log_file.close()              # shutdown's own stdout-flush hits a closed file


if __name__ == "__main__":
    main()
