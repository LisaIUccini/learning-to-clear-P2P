"""Learned warm start (L2WS) benchmark: training and evaluation shared by the
centralised MLP and the distributed GNN predictor (paper Sec. 5.2).

A predictor h_w maps an instance to a starting point z^0; plain,
unmodified DR runs from there. Training differentiates through `unroll`
DR steps from z^0 with one of two losses: "regression" (distance of z^K to a
fixed point computed beforehand, Sambharya et al., 2024) and "asymmetric"
(Eq. 16 against a cold-start DR trajectory).
"""

from __future__ import annotations

import torch

from .. import market
from ..fixed_point import run_dr
from ..training import asymmetric_loss, rollout

LOSSES = ("regression", "asymmetric")


def run_pair(spec, predictor, dr_cfg, device, seed, record=False):
    """Cold-start plain DR from a random x^0 (drawn with `seed`) vs plain DR
    warm-started at h_w(theta): different starting points by design."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net_dr = market.build_network(spec, device=device, rng=gen)
    res_dr = run_dr(net_dr, dr_cfg, record=record)

    net_l = market.build_network(spec, device=device)   # state overwritten next
    net_l.unflatten(predictor.predict(spec, differentiable=False))
    res_learned = run_dr(net_l, dr_cfg, record=record)
    return res_dr, res_learned


def regression_loss(z_final, z_star, scale, eps=1e-9):
    """||z^K - z*||, normalised; z* is a fixed point of DR (`compute_fixed_points`)."""
    denom = torch.linalg.vector_norm(scale) + eps
    return torch.linalg.vector_norm(z_final - z_star) / denom


def compute_fixed_points(specs, dr_cfg, device, label=""):
    """Plain DR run to convergence (no grad) from torch's global random
    state; returns the final auxiliary iterate of each instance."""
    z_stars = []
    n_nonconverged = 0
    for spec in specs:
        net = market.build_network(spec, device=device)
        res = run_dr(net, dr_cfg, record=False)
        if not res.converged:
            n_nonconverged += 1
        z_stars.append(net.flatten().detach())
    if n_nonconverged:
        print(f"  [{label}] WARNING: {n_nonconverged}/{len(specs)} fixed-point targets did NOT converge "
              f"within {dr_cfg.n_iter} iterations (used anyway, as the best available target)", flush=True)
    return z_stars


def compute_reference_trajectories(specs, train_cfg, device, seed):
    """Cold-start plain-DR residuals (`train_cfg.unroll` steps) used by the
    asymmetric loss; instance i starts from a seeded random x^0."""
    refs = []
    for i, spec in enumerate(specs):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed * 1_000_003 + i)
        net = market.build_network(spec, device=device, rng=gen)
        _, residuals_ref = rollout(net, None, train_cfg, x0=None, differentiable=False)
        refs.append(residuals_ref)
    return refs


def fit_warm_start(predictor, module, train_specs, val_specs, train_cfg, epochs, lr, grad_clip, seed,
                   device, loss_mode, dr_cfg, gamma_mono, w_behind, w_ahead, omega_mode, omega_power):
    """Adam on `module`'s parameters, one step per training instance and epoch.
    `predictor.warm_start(net, spec, differentiable)` gives z^0."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    history = {"train": [], "val": []}

    if loss_mode == "regression":
        print(f"  precomputing {len(train_specs)} train + {len(val_specs)} val fixed-point "
              f"targets (plain DR to convergence, no grad)...", flush=True)
        train_target = compute_fixed_points(train_specs, dr_cfg, device, label="train targets")
        val_target = compute_fixed_points(val_specs, dr_cfg, device, label="val targets")
    else:
        print(f"  precomputing {len(train_specs)} train + {len(val_specs)} val cold-start "
              f"reference trajectories (plain DR, {train_cfg.unroll} steps, no grad)...", flush=True)
        train_target = compute_reference_trajectories(train_specs, train_cfg, device, seed=seed)
        val_target = compute_reference_trajectories(val_specs, train_cfg, device, seed=seed + 1)

    def instance_loss(spec, differentiable, target):
        net = market.build_network(spec, device=device)   # state irrelevant: rollout starts from z^0
        warm_start = predictor.warm_start(net, spec, differentiable)
        x_final, residuals = rollout(net, None, train_cfg, x0=warm_start, differentiable=differentiable)
        scale = net.physical_scale()
        if loss_mode == "regression":
            return regression_loss(x_final, target, scale)
        return asymmetric_loss(residuals, target, scale, gamma_mono, 1e-9,
                               w_behind, w_ahead, omega_power, omega_mode)

    for epoch in range(epochs):
        module.train()
        total = 0.0
        for spec, target in zip(train_specs, train_target):
            opt.zero_grad()
            loss = instance_loss(spec, True, target)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
            opt.step()
            total += loss.item()
        train_loss = total / max(len(train_specs), 1)
        history["train"].append(train_loss)

        module.eval()
        with torch.no_grad():
            val_total = sum(float(instance_loss(s, False, t)) for s, t in zip(val_specs, val_target))
        val_loss = val_total / max(len(val_specs), 1)
        history["val"].append(val_loss)
        print(f"  epoch {epoch + 1:3d}/{epochs} | train {train_loss:+.4f} | val {val_loss:+.4f}", flush=True)
    return history
