"""
demo_train_realistic_params_fixed_x0.py
==========================================

Train directly on PARAMETER diversity (D*, G_res, D_min, D_max, G_min,
G_max -- via `build_realistic_training_dataset.sample_six_perturbed_dataset`)
instead of x0 diversity, with x0 held to a SINGLE FIXED value across every
training AND test instance -- the design agreed with the user's
supervisor. x0 is treated as an input the user supplies to start the
algorithm (not something that varies at deployment), so evaluation only
ever uses this one fixed x0 -- there is no random-x0 evaluation here.
Topology is fixed too (density=1.0, the same graph as every other
experiment in this project). Uses the realistic (household-prosumer,
kW-scale) base market from `build_realistic_training_dataset.py`, which
also fixes the project's old Gmin_range<0 default (all 6 quantities are
now >=0 by construction).

Recipe: the established best recipe for this exact topology/gamma
(asymmetric_loss + dr_slope, see runs/cross_density/seed0_density1.0_seed0_drslope),
just pointed at this new dataset/x0 design. unroll/rho are re-derived from
THIS market's own plain-DR mean (719 iterations over 8 draws -- checked
before launching, see the auto-memory project notes), not reused from the
old abstract-unit market, since the realistic-range magnitudes are not the
same market.

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
    build_realistic_base_market, sample_six_perturbed_dataset, REALISTIC_MARKET_CONFIG,
)


def run_pair_fixed_x0(spec, model, dr_cfg, device, seed, record=False, fixed_x0=None):
    """Same signature shape as demo.run_pair (so it's a drop-in `run_pair_fn`
    for diagnostics.save_instance_plots), but ignores `seed` and uses the
    SAME fixed_x0 for both plain DR and DR+learned -- x0 is an input the
    user supplies to start the algorithm, not something that varies."""
    net_dr = market.build_network(spec, device=device)
    net_dr.unflatten(fixed_x0)
    gnn_mod.undeploy(net_dr)
    res_dr = algorithm.run_dr(net_dr, dr_cfg, record=record)

    net_l = market.build_network(spec, device=device)
    net_l.unflatten(fixed_x0)
    gnn_mod.deploy(model, net_l)
    res_l = algorithm.run_dr(net_l, dr_cfg, record=record)
    return res_dr, res_l


def evaluate_fixed_x0(specs, model, dr_cfg, device, fixed_x0, label):
    """Like demo.evaluate, but every instance uses the SAME fixed x0 for
    both plain DR and DR+learned, and also records residual curves (for
    plot_results' convergence-band panel)."""
    dr_iters, learned_iters, speedups, curves_dr, curves_learned = [], [], [], [], []
    for i, spec in enumerate(specs):
        res_dr, res_l = run_pair_fixed_x0(spec, model, dr_cfg, device, seed=0,
                                           record=True, fixed_x0=fixed_x0)
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
    pct = float(sys.argv[9]) if len(sys.argv) > 9 else 0.20
    w_ahead = float(sys.argv[10]) if len(sys.argv) > 10 else config.TRAIN_W_AHEAD
    w_behind = float(sys.argv[11]) if len(sys.argv) > 11 else config.TRAIN_W_BEHIND

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"pct={pct}  w_ahead={w_ahead}  w_behind={w_behind})", flush=True)
    assert w_ahead < w_behind, "w_ahead must stay strictly below w_behind -- equal collapses the loss into stability_loss"

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=config.TRAIN_GAMMA_MONO,
                                       epochs=epochs, w_behind=w_behind,
                                       w_ahead=w_ahead, lr=config.TRAIN_LR,
                                       grad_clip=config.TRAIN_GRAD_CLIP, seed=config.TRAIN_SEED,
                                       loss_mode="asymmetric", omega_mode="dr_slope")

    print("=" * 70)
    print("dataset: fixed topology (density=1.0, same graph as every other")
    print("experiment) + fixed x0 (ONE draw, reused everywhere -- x0 is an")
    print("input the user supplies to start the algorithm, not something")
    print("that varies) -- diversity comes ONLY from perturbing ALL SIX of")
    print(f"D*/G_res/D_min/D_max/G_min/G_max by ONE UNIFORM +/-{pct:.0%} (independent")
    print("per quantity), all >=0. Infeasible draws discarded+resampled.")
    print("FIXED train/test sets (slot 0 = original network, always)")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=0)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    fixed_x0_gen = torch.Generator(device=device); fixed_x0_gen.manual_seed(config.TRAIN_SEED)
    fixed_x0_net = market.build_network(base_spec, device=device, rng=fixed_x0_gen)
    fixed_x0 = fixed_x0_net.flatten().detach().clone()
    print(f"  fixed x0 drawn once (seed={config.TRAIN_SEED}), reused for every instance", flush=True)

    train_specs, train_log, n_discarded_train = sample_six_perturbed_dataset(
        base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct=pct)
    train_specs = [base_spec] + train_specs
    test_specs, test_log, n_discarded_test = sample_six_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct=pct)
    test_specs = [base_spec] + test_specs
    print(f"  train: {n_train} FIXED instances (slot 0 = original network, "
          f"{n_train - 1} perturbed, +/-{pct:.0%}), {n_discarded_train} infeasible draws discarded+resampled  |  "
          f"test: {n_test} FIXED instances (slot 0 = original), {n_discarded_test} infeasible draws "
          f"discarded+resampled", flush=True)

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='dr_slope')", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device, fixed_x0=fixed_x0)

    # save the checkpoint RIGHT AWAY -- before any evaluation/plotting, so
    # a saved model is never lost if evaluation is interrupted downstream
    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, FIXED x0 (the only evaluation mode --", flush=True)
    print("x0 is an input the user supplies, not something that varies)", flush=True)
    print("-" * 70)
    test_res = evaluate_fixed_x0(test_specs, model, dr_cfg, device, fixed_x0, label="fixed-x0")

    subtitle = (f"trained on: {n_train} instances (fixed x0+topology, all 6 params +/-{pct:.0%}, "
                f"w_ahead={w_ahead}/w_behind={w_behind}), "
                f"{epochs} epochs  |  tested on: n={n_test}, SAME fixed x0")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    def _run_pair_fn(spec, model_, dr_cfg_, device_, seed, record=False):
        return run_pair_fixed_x0(spec, model_, dr_cfg_, device_, seed, record=record, fixed_x0=fixed_x0)
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=_run_pair_fn)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: train directly on PARAMETER diversity (D*, G_res, D_min, D_max, G_min, G_max) with
FIXED x0 and FIXED topology (density=1.0, same graph as every other experiment), instead of the
usual x0-only diversity. x0 is treated as an input the user supplies to start the algorithm, not
something that varies at deployment -- evaluation uses this SAME fixed x0 throughout, no random-x0
test. Realistic (household-prosumer, kW-scale) base market -- see
`experiments/build_realistic_training_dataset.py`; G_min is now >=0 by construction (was negative
by default before), matching the literature this codebase's market model is based on (Belgioioso
et al., arXiv:2107.13444, where dispatchable generation is non-negative by definition).

**Dataset**: {n_train} FIXED training instances (slot 0 always the unperturbed original network,
the other {n_train - 1} perturbed once and reused every epoch) and {n_test} FIXED test instances
(slot 0 = original, unperturbed) -- a real, concrete, reproducible dataset, not regenerated per
epoch. Each perturbed instance is a direct multiplicative perturbation of the SAME base market's
D*, G_res, D_min, D_max, G_min, G_max, all six by ONE UNIFORM +/-{pct:.0%} factor (independent
draw per quantity, shared across nodes) -- graph, costs, and capacities held exactly fixed.
Infeasible draws are discarded and resampled (not kept). Loss asymmetry: w_ahead={w_ahead},
w_behind={w_behind} (up from the project's usual w_ahead=1.0, since this checkpoint already
reaches 100% win-rate with room to trade some of that safety margin for larger magnitude). x0 is
the SAME single draw (seed={config.TRAIN_SEED}) for every training instance, every epoch, and every
test instance.

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).

**Question answered**: does training directly on parameter diversity (instead of x0 diversity),
with x0 fixed throughout (as the user supplies it), produce a model that generalises well across
the 6-quantity perturbation space?
""")


if __name__ == "__main__":
    main()
