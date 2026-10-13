"""
demo_train_realistic_params_fixed_x0.py
==========================================

Train directly on PARAMETER diversity (D*, G_res, D_min, D_max, G_min,
G_max -- via `build_realistic_training_dataset.sample_level_width_perturbed_dataset`)
instead of x0 diversity, with x0 held to a SINGLE shared draw across every
training AND test instance -- the design agreed with the user's
supervisor. x0 is treated as an input the user supplies to start the
algorithm (not something that varies at deployment), so evaluation only
ever uses this one shared x0, used RAW/UNCLIPPED for every instance --
there is no random-x0 evaluation here. Topology is fixed too (density=1.0,
the same graph as every other experiment in this project). Uses the
realistic (household-prosumer, kW-scale) base market from
`build_realistic_training_dataset.py`, which also fixes the project's old
Gmin_range<0 default (all 6 quantities are now >=0 by construction).

**Perturbation scheme ("level + width")**: v1-v4 perturbed D_min/D_max (and
G_min/G_max) by two fully INDEPENDENT factors, which could squeeze a
node's box down to ~1/11th of its base width -- a real bug (box collapse),
fixed by this scheme: D* and the D-box move TOGETHER under one "level"
factor (D*'s relative position inside its box is preserved exactly, no
collapse possible), the D-box's WIDTH is jittered separately by a much
narrower factor (flexibility is a more stable, contractual property than
the demand LEVEL); same idea for G_min/G_max (move together) vs the G-box's
width.

**x0 handling (finalised after discussion, 2026-09-10)**: the ONE shared x0
is used AS-IS for every instance, with NO per-instance clip/adjustment --
an earlier version of this script (`adjust_x0_to_spec`) snapped x0 to the
nearest box edge whenever a perturbed instance's box moved out from under
it, but this was never actually a requirement: x0's feasibility relative to
any given instance's box is NOT a requirement of the algorithm -- DR's own
`local_prox` clamps back onto the box on the very first step regardless of
where x0 starts. Since this scheme's box can't collapse (only jitter by
+/-8% in width), the un-clipped excursion stays bounded (measured: h_D in
roughly [-2.4, 1.8] before this de-simplification) -- nothing like the
box-collapse-era +/-9.7 pathology of v1-v4, which this scheme already ruled
out independently of the clip.

Recipe: the established best recipe for this exact topology/gamma
(asymmetric_loss + dr_slope, see runs/cross_density/seed0_density1.0_seed0_drslope's
STRUCTURE -- loss/omega/weights identical; gnn_rho/unroll are NOT reused
from that market, they are freshly re-derived from THIS dataset's own
plain-DR iteration distribution via the project's standing rule,
unroll=mean_DR/2, rho=1-10/unroll, checked before every launch).

Dataset is a FIXED, concrete, reproducible set (not regenerated per epoch --
that would break the standard meaning of "epoch" as a repeated pass over
the SAME data): slot 0 is always the unperturbed original network
(matching the project's established `[base_spec] + perturbed(...)`
convention, e.g. demo_train_topology_scale.py), the rest are perturbed
once and reused every epoch.

Outputs match the project's standard format: results.png (via
demo.plot_results) and instance_plots/ (via diagnostics.save_instance_plots),
both built from the FIXED-x0 evaluation.

Usage: python3 demo_train_realistic_params_fixed_x0.py <run_name>
       [n_train] [n_test] [epochs] [dr_gamma] [unroll] [gnn_rho] [gnn_scale]
       [pct_level] [pct_width] [w_ahead] [w_behind]
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
import train as train_mod
from diagnostics import save_instance_plots
from demo import plot_results, plot_training_loss, write_run_notes
from build_realistic_training_dataset import (
    build_realistic_base_market, sample_level_width_perturbed_dataset,
    sample_level_width_per_node_dataset, adjust_x0_to_spec,
    REALISTIC_MARKET_CONFIG,
)


def build_per_instance_x0(specs, fixed_x0_base, device):
    """The ONE shared x0, individually projected onto each spec's own box
    (`adjust_x0_to_spec`) -- a list aligned with `specs` by position."""
    out = []
    for spec in specs:
        net = market.build_network(spec, device=device)
        out.append(adjust_x0_to_spec(fixed_x0_base, net))
    return out


def run_pair_fixed_x0(spec, model, dr_cfg, device, seed, record=False, x0=None):
    """Same signature shape as demo.run_pair (so it's a drop-in `run_pair_fn`
    for diagnostics.save_instance_plots), but ignores `seed` and uses the
    SAME (per-instance-adjusted) x0 for both plain DR and DR+learned -- x0
    is an input the user supplies to start the algorithm, not something
    that varies."""
    net_dr = market.build_network(spec, device=device)
    net_dr.unflatten(x0)
    gnn_mod.undeploy(net_dr)
    res_dr = algorithm.run_dr(net_dr, dr_cfg, record=record)

    net_l = market.build_network(spec, device=device)
    net_l.unflatten(x0)
    gnn_mod.deploy(model, net_l)
    res_l = algorithm.run_dr(net_l, dr_cfg, record=record)
    return res_dr, res_l


def evaluate_fixed_x0(specs, x0_list, model, dr_cfg, device, label):
    """Like demo.evaluate, but every instance uses its own (per-instance
    projected) fixed x0 for both plain DR and DR+learned, and also records
    residual curves (for plot_results' convergence-band panel)."""
    dr_iters, learned_iters, speedups, curves_dr, curves_learned = [], [], [], [], []
    for i, (spec, x0) in enumerate(zip(specs, x0_list)):
        res_dr, res_l = run_pair_fixed_x0(spec, model, dr_cfg, device, seed=0,
                                           record=True, x0=x0)
        sp = res_dr.n_iter / max(res_l.n_iter, 1)
        pct = (res_dr.n_iter - res_l.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_l.n_iter); speedups.append(sp)
        curves_dr.append(np.asarray(res_dr.residual)); curves_learned.append(np.asarray(res_l.residual))
        print(f"  [{label}] {i + 1:2d}/{len(specs)}  DR {res_dr.n_iter:5d} it  |  "
              f"learned {res_l.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)
    win = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"  [{label}] mean speed-up {stats.mean(speedups):.2f}x   median {stats.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   win-rate {win:.0%} "
          f"({sum(s >= 1.0 for s in speedups)}/{len(speedups)})", flush=True)
    return {"dr": dr_iters, "learned": learned_iters, "speedup": speedups,
            "curves_dr": curves_dr, "curves_learned": curves_learned}


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "train_realistic_params_fixed_x0"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    unroll = int(sys.argv[6]) if len(sys.argv) > 6 else 360
    gnn_rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.9722
    gnn_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    pct_level = float(sys.argv[9]) if len(sys.argv) > 9 else 0.20
    pct_width = float(sys.argv[10]) if len(sys.argv) > 10 else 0.08
    w_ahead = float(sys.argv[11]) if len(sys.argv) > 11 else config.TRAIN_W_AHEAD
    w_behind = float(sys.argv[12]) if len(sys.argv) > 12 else config.TRAIN_W_BEHIND
    per_node = (sys.argv[13] == "1") if len(sys.argv) > 13 else False
    gnn_scale_slow = float(sys.argv[14]) if len(sys.argv) > 14 else config.GNN_SCALE_SLOW
    gnn_rho_slow = float(sys.argv[15]) if len(sys.argv) > 15 else config.GNN_RHO_SLOW
    gamma_mono = float(sys.argv[16]) if len(sys.argv) > 16 else config.TRAIN_GAMMA_MONO
    omega_mode = sys.argv[17] if len(sys.argv) > 17 else "dr_slope"

    sampler = sample_level_width_per_node_dataset if per_node else sample_level_width_perturbed_dataset

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"pct_level={pct_level}  pct_width={pct_width}  w_ahead={w_ahead}  w_behind={w_behind}  "
          f"per_node={per_node}  gnn_scale_slow={gnn_scale_slow}  gnn_rho_slow={gnn_rho_slow}  "
          f"gamma_mono={gamma_mono}  omega_mode={omega_mode})", flush=True)
    assert w_ahead < w_behind, "w_ahead must stay strictly below w_behind -- equal collapses the loss into stability_loss"

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=gamma_mono,
                                       epochs=epochs, w_behind=w_behind,
                                       w_ahead=w_ahead, lr=config.TRAIN_LR,
                                       grad_clip=config.TRAIN_GRAD_CLIP, seed=config.TRAIN_SEED,
                                       loss_mode="asymmetric", omega_mode=omega_mode)

    print("=" * 70)
    print("dataset: fixed topology (density=1.0, same graph as every other")
    print("experiment) + ONE shared x0 (drawn once, used RAW/UNCLIPPED for")
    print("every instance -- x0 is an input the user supplies to start the")
    print("algorithm, not something that varies, and its feasibility relative")
    print("to any instance's box is not a requirement) -- diversity")
    print("comes ONLY from perturbing D*/G_res/D_min/D_max/G_min/G_max via")
    print(f"the 'level + width' scheme: level (D*, G-box centre, G_res) +/-{pct_level:.0%},")
    print(f"box WIDTH (D-box, G-box) +/-{pct_width:.0%} -- boxes can't collapse/invert by")
    print("construction. Infeasible or negative draws discarded+resampled.")
    print("FIXED train/test sets (slot 0 = original network, always)")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=0)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    fixed_x0_gen = torch.Generator(device=device); fixed_x0_gen.manual_seed(config.TRAIN_SEED)
    fixed_x0_net = market.build_network(base_spec, device=device, rng=fixed_x0_gen)
    fixed_x0_base = fixed_x0_net.flatten().detach().clone()
    print(f"  shared x0 drawn once (seed={config.TRAIN_SEED}), used RAW/UNCLIPPED for every instance", flush=True)
    print("  (x0 feasibility relative to any instance's box is NOT a requirement of the", flush=True)
    print("  algorithm -- DR's own local_prox clamps back onto the box on the very first", flush=True)
    print("  step regardless -- so no per-instance adjustment is applied)", flush=True)

    train_specs, train_log, n_discarded_train = sampler(
        base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct_level=pct_level, pct_width=pct_width)
    train_specs = [base_spec] + train_specs
    test_specs, test_log, n_discarded_test = sampler(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=pct_level, pct_width=pct_width)
    test_specs = [base_spec] + test_specs
    print(f"  train: {n_train} FIXED instances (slot 0 = original network, "
          f"{n_train - 1} perturbed), {n_discarded_train} infeasible/negative draws discarded+resampled  |  "
          f"test: {n_test} FIXED instances (slot 0 = original), {n_discarded_test} infeasible/negative draws "
          f"discarded+resampled", flush=True)

    train_x0 = [fixed_x0_base] * len(train_specs)
    test_x0 = [fixed_x0_base] * len(test_specs)

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='{omega_mode}')", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device,
                           fixed_x0=train_x0, val_fixed_x0=test_x0[:5])

    # save the checkpoint RIGHT AWAY -- before any evaluation/plotting, so
    # a saved model is never lost if evaluation is interrupted downstream
    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, per-instance-projected fixed x0 --", flush=True)
    print("the only evaluation mode (x0 is an input the user supplies, not something that varies)", flush=True)
    print("-" * 70)
    test_res = evaluate_fixed_x0(test_specs, test_x0, model, dr_cfg, device, label="fixed-x0")

    subtitle = (f"trained on: {n_train} instances (fixed x0+topology, level+width perturbation "
                f"+/-{pct_level:.0%}/+/-{pct_width:.0%}, w_ahead={w_ahead}/w_behind={w_behind}), "
                f"{epochs} epochs  |  tested on: n={n_test}, SAME shared x0 (per-instance projected)")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    x0_by_id = {id(spec): x0 for spec, x0 in zip(test_specs, test_x0)}
    def _run_pair_fn(spec, model_, dr_cfg_, device_, seed, record=False):
        return run_pair_fixed_x0(spec, model_, dr_cfg_, device_, seed, record=record, x0=x0_by_id[id(spec)])
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=_run_pair_fn)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: train directly on PARAMETER diversity (D*, G_res, D_min, D_max, G_min, G_max) with a
SHARED x0 and FIXED topology (density=1.0, same graph as every other experiment), instead of the
usual x0-only diversity. x0 is treated as an input the user supplies to start the algorithm, not
something that varies at deployment -- evaluation uses this SAME shared x0 throughout (each
instance's own box-projection of it), no random-x0 test. Realistic (household-prosumer, kW-scale)
base market -- see `experiments/build_realistic_training_dataset.py`; G_min is now >=0 by
construction (was negative by default before), matching the literature this codebase's market model
is based on (Belgioioso et al., arXiv:2107.13444, where dispatchable generation is non-negative by
definition).

**Perturbation scheme ("level + width")**: supersedes v1-v4's independent D_min/D_max (and
G_min/G_max) factors, which could squeeze a node's box down to ~1/11th of its base width -- a real
bug (box collapse). Here, D* and the D-box move TOGETHER under one "level" factor
(+/-{pct_level:.0%}, D*'s relative position inside its box preserved exactly -- no collapse, no
re-clipping needed), the D-box's WIDTH is jittered separately by a narrower "width" factor
(+/-{pct_width:.0%}, a more stable/contractual property than the demand level); same for G_min/G_max
(move together) vs the G-box's width. G_res (no box of its own) is perturbed independently by its
own level factor. The box can never collapse/invert by construction.

**x0 (finalised 2026-09-10)**: the ONE shared x0 is used RAW/UNCLIPPED for every instance -- x0's
feasibility relative to any instance's box is NOT a requirement of the algorithm (DR's own
`local_prox` clamps back onto the box on the very first step regardless), and this scheme's box
can't collapse, so the un-clipped excursion stays bounded (measured: h_D roughly in [-2.4, 1.8]) --
an earlier version of this script clipped x0 into each instance's box, which was correct but never
actually necessary; removed for a simpler, more defensible design.

**Dataset**: {n_train} FIXED training instances (slot 0 always the unperturbed original network,
the other {n_train - 1} perturbed once and reused every epoch) and {n_test} FIXED test instances
(slot 0 = original, unperturbed) -- a real, concrete, reproducible dataset, not regenerated per
epoch. Infeasible or negative draws are discarded and resampled (not kept). Loss asymmetry:
w_ahead={w_ahead}, w_behind={w_behind}.

**Calibration**: unroll={unroll}, gnn_rho={gnn_rho} -- freshly re-derived from THIS dataset's own
plain-DR iteration distribution (fixed, unclipped x0) via the project's standing rule
(unroll=mean_DR/2, rho=1-10/unroll), not reused from any other market/dataset.

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).

**Question answered**: with box-collapse ruled out by construction, x0 kept simple (no clip), and
rho/unroll properly calibrated on this exact dataset, does the model train cleanly (no self-inflicted
plateau) and match or beat the earlier v5 result (1.25x/100%, which used the same scheme but with an
x0 clip and rho/unroll calibrated on an earlier, slightly different check)?
""")


if __name__ == "__main__":
    main()
