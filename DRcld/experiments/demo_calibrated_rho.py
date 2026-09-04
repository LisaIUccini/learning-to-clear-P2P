"""
demo_calibrated_rho.py
=========================

PROTOTYPE (option "A" from the topology-generalization discussion): can a
per-INSTANCE calibrated decay rate (`gnn.CalibratedLocalAgent`) recover
some of the topology zero-shot weakness, WITHOUT retraining any weights?

Hypothesis: the fast channel's rho/scale are fixed constants tuned once
to the TRAINING topology's own mean DR convergence rate (the project's
standing rule: unroll=mean_DR/2, rho=1-10/unroll). When the graph changes
(zero-shot topology test), that timescale is silently stale -- the
correction decays at the training market's pace, not the test instance's.

Fix tested here: run a SHORT uncorrected DR calibration window (k_calib
iterations) on each test instance, fit an exponential decay rate to ITS
OWN residual trajectory, derive a fresh rho via the SAME rule used
throughout this project, then deploy the trained model with that
estimated rho (scale kept at the checkpoint's own trained value) and the
decay exponent reset to 0 at the deployment point -- continuing the SAME
trajectory, not restarting (the calibration iterations are not wasted,
they're the first k_calib steps of the real run).

Three methods compared per instance, same random x0:
    1. plain DR (baseline)
    2. DR + learned, FIXED checkpoint rho (current approach -- already
       measured as weak on topology zero-shot)
    3. DR + learned, CALIBRATED per-instance rho (this prototype)

Usage: python3 demo_calibrated_rho.py <run_name> <weights_path>
       <base_seed> <density> <dr_gamma> <gnn_rho> <gnn_scale>
       <topology_pct> [n_test] [k_calib]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import statistics as stats
import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod


def estimate_rho(calib_residuals, tol: float, skip: int = 5,
                  rho_min: float = 0.5, rho_max: float = 0.999):
    """Fit an exponential decay to the calibration-window residuals (log-
    linear regression, skipping the first `skip` -- often noisier/less
    representative of the trajectory's settled contraction rate), estimate
    how many MORE iterations this instance would need to reach `tol` at
    that rate, then apply the project's own standing rule
    (rho = 1 - 10/(mean_iters_est/2)) to the resulting total estimate.
    Returns None if the window doesn't show clean decay (too short, not
    monotone enough to fit) -- caller should fall back to the checkpoint's
    fixed rho in that case.
    """
    r = np.asarray(calib_residuals[skip:], dtype=float)
    r = r[r > 0]
    if len(r) < 5:
        return None
    k = np.arange(len(r))
    slope, _ = np.polyfit(k, np.log(r), 1)
    alpha = float(np.exp(slope))
    if not (0.0 < alpha < 1.0):
        return None
    r_last = float(r[-1])
    if r_last <= tol:
        n_remaining = 1.0
    else:
        n_remaining = float(np.log(tol / r_last) / np.log(alpha))
    mean_iters_est = len(calib_residuals) + max(n_remaining, 1.0)
    rho_new = 1.0 - 10.0 / max(mean_iters_est / 2.0, 10.0)
    return float(np.clip(rho_new, rho_min, rho_max)), mean_iters_est


def run_dr_calibrated(net, model, dr_cfg, k_calib: int):
    """Plain DR for k_calib steps, then deploy the trained model with a
    rho estimated from THIS instance's own calibration window (falls back
    to the checkpoint's fixed rho if the fit fails), continuing the same
    trajectory (no reset)."""
    net.reset()
    gnn_mod.undeploy(net)
    calib_residuals = []
    converged = False
    k_run = 0
    rho_used = None

    with torch.no_grad():
        for k in range(dr_cfg.n_iter):
            k_run = k + 1
            net.communicate()
            for p in net.prosumers:
                p.dr_step(dr_cfg.gamma, dr_cfg.lam, n_bisect=dr_cfg.bisect)
            r = net.aux_residual()
            if k < k_calib:
                calib_residuals.append(r)
            if k + 1 == k_calib:
                fit = estimate_rho(calib_residuals, dr_cfg.tol)
                if fit is not None:
                    rho_used, _ = fit
                else:
                    rho_used = model.cfg.rho
                gnn_mod.deploy_calibrated(model, net, rho_override=rho_used, k_offset=k_calib)
            if r < dr_cfg.tol:
                converged = True
                break

    return algorithm.DRResult(n_iter=k_run, converged=converged), rho_used


def main():
    run_name = sys.argv[1]
    weights_path = sys.argv[2]
    base_seed = int(sys.argv[3])
    density = float(sys.argv[4])
    dr_gamma = float(sys.argv[5])
    gnn_rho = float(sys.argv[6])
    gnn_scale = float(sys.argv[7])
    topo_pct = float(sys.argv[8])
    n_test = int(sys.argv[9]) if len(sys.argv) > 9 else 30
    k_calib = int(sys.argv[10]) if len(sys.argv) > 10 else 30

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (weights={weights_path}  base_seed={base_seed}  "
          f"density={density}  dr_gamma={dr_gamma}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"topo_pct={topo_pct}  n_test={n_test}  k_calib={k_calib})", flush=True)

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    base_nb, base_prim, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}",
          flush=True)
    test_specs = market.sample_common_scale_topology_dataset(
        base_nb, base_prim, mc, n_instances=n_test, pct=topo_pct, seed=config.TRAIN_SEED + 4)
    print(f"  topology test set built (+/-{topo_pct*100:.0f}%, same seed/mechanism as the earlier "
          f"topology_scale campaign, so directly comparable to that baseline)", flush=True)

    fixed_pct, calib_pct = [], []
    fixed_sp, calib_sp = [], []
    rhos_used = []
    for i, spec in enumerate(test_specs):
        seed = config.TEST_SEED * 7919 + i

        gen = torch.Generator(device=device); gen.manual_seed(seed)
        net_dr = market.build_network(spec, device=device, rng=gen)
        x0 = net_dr.flatten().clone()
        gnn_mod.undeploy(net_dr)
        res_dr = algorithm.run_dr(net_dr, dr_cfg, record=False)

        net_fixed = market.build_network(spec, device=device)
        net_fixed.unflatten(x0)
        gnn_mod.deploy(model, net_fixed)
        res_fixed = algorithm.run_dr(net_fixed, dr_cfg, record=False)

        net_calib = market.build_network(spec, device=device)
        net_calib.unflatten(x0)
        res_calib, rho_used = run_dr_calibrated(net_calib, model, dr_cfg, k_calib)
        rhos_used.append(rho_used)

        dr_it = res_dr.n_iter
        pf = (dr_it - res_fixed.n_iter) / max(dr_it, 1) * 100.0
        pc = (dr_it - res_calib.n_iter) / max(dr_it, 1) * 100.0
        sf = dr_it / max(res_fixed.n_iter, 1)
        sc = dr_it / max(res_calib.n_iter, 1)
        fixed_pct.append(pf); calib_pct.append(pc)
        fixed_sp.append(sf); calib_sp.append(sc)
        print(f"  {i + 1:2d}/{n_test}  DR {dr_it:5d} it  |  fixed-rho {res_fixed.n_iter:5d} it "
              f"({pf:+5.1f}%)  |  calibrated-rho(={rho_used:.4f}) {res_calib.n_iter:5d} it ({pc:+5.1f}%)",
              flush=True)

    def summarize(label, pct_vals, sp_vals):
        win = sum(s >= 1.0 for s in sp_vals) / len(sp_vals)
        print(f"\n[{label}]  mean_pct_reduction={stats.mean(pct_vals):6.2f}%  "
              f"median={stats.median(pct_vals):6.2f}%  min={min(pct_vals):6.2f}%  max={max(pct_vals):6.2f}%  "
              f"mean_speedup={stats.mean(sp_vals):5.2f}x  win-rate={win:.0%}", flush=True)

    summarize("fixed-rho (baseline, current approach)", fixed_pct, fixed_sp)
    summarize("calibrated-rho (prototype)", calib_pct, calib_sp)
    print(f"\nrho used per instance: mean={stats.mean(rhos_used):.4f}  "
          f"min={min(rhos_used):.4f}  max={max(rhos_used):.4f}  "
          f"(checkpoint's fixed rho was {gnn_rho})", flush=True)


if __name__ == "__main__":
    main()
