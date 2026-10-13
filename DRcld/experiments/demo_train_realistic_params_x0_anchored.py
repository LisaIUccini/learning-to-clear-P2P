"""
demo_train_realistic_params_x0_anchored.py
=============================================

Variant of `demo_train_realistic_params_fixed_x0.py` (v5/v6, "level+width")
using `build_realistic_training_dataset.sample_x0_anchored_dataset` instead
(v7, "x0-anchored") -- the user's own framing: change the bounds, but the
ONE shared fixed x0 value must ALWAYS be included in the bound, and then
the bound itself can be made much bigger or much smaller.

**Why**: v5/v6 built the box from D*'s own level+width and then PATCHED x0
back in afterward (`adjust_x0_to_spec`, clip to the nearest edge) whenever
it drifted outside -- which happened often (~93% of instances) because a
+/-20% LEVEL shift of the box is large relative to a node's box width,
regardless of the (now-safe, non-collapsing) width jitter. v7 instead
builds each node's box AROUND x0 directly: the base market gives a FIXED
distance from x0 to each edge of its box (`dist_below`, `dist_above`,
both >0 by construction, since x0 was drawn inside the base box); a
perturbed instance rescales those two distances by INDEPENDENT positive
factors (`f_below`, `f_above`, drawn from `f_range`):

    D_min_new = x0.D - dist_below * f_below
    D_max_new = x0.D + dist_above * f_above

Since f_below, f_above > 0 always, x0 is mathematically inside the new box
for EVERY draw -- no post-hoc clip ever needed (v5/v6's most common source
of "adjustment" is eliminated by construction, not patched). Because the
two sides scale INDEPENDENTLY, x0's relative position inside the box still
varies freely across instances (this is what made v5/v6's "how far is x0
from a good spot" signal correlate with achieved speedup, r=0.428) --
without ever moving x0 itself, and without the v1-v4 box-collapse
pathology (the box can shrink toward x0 but never past it, never invert).
Same construction for G. D* and G_res keep their own independent "level"
perturbation, D* re-clipped into the new box (still genuinely needed: D*
is a separate quantity from x0, not anchored to it).

Recipe, dataset conventions, outputs: identical to
demo_train_realistic_params_fixed_x0.py -- see that file's docstring.
Shares `run_pair_fixed_x0`/`evaluate_fixed_x0`/`build_per_instance_x0`
from it rather than duplicating them (x0-per-instance adjustment is still
called here as a zero-cost safety net for float edge cases at the box
boundary -- expected to report ~0 adjustments needed, confirming the
by-construction guarantee holds).

Usage: python3 demo_train_realistic_params_x0_anchored.py <run_name>
       [n_train] [n_test] [epochs] [dr_gamma] [unroll] [gnn_rho] [gnn_scale]
       [f_lo] [f_hi] [pct_level] [w_ahead] [w_behind] [gnn_scale_slow] [gnn_rho_slow]

`gnn_scale_slow`/`gnn_rho_slow` (new, default to config.py's GNN_SCALE_SLOW=0.0/
GNN_RHO_SLOW=0.99, i.e. OFF unless overridden): the GNN's second, slow-decay
correction channel (gnn.py, k99~458 at rho_slow=0.99), built specifically to
keep helping past where the fast channel (rho=0.9722, k99~163) has already
decayed away -- never enabled in any run in this project so far (v1-v7 all
used scale_slow=0.0), despite DR needing 500-800+ iterations here, well past
where the fast channel's reach ends.
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
    build_realistic_base_market, sample_x0_anchored_dataset, adjust_x0_to_spec,
    REALISTIC_MARKET_CONFIG,
)
from demo_train_realistic_params_fixed_x0 import (
    build_per_instance_x0, run_pair_fixed_x0, evaluate_fixed_x0,
)


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "train_realistic_params_x0_anchored"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    unroll = int(sys.argv[6]) if len(sys.argv) > 6 else 360
    gnn_rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.9722
    gnn_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    f_lo = float(sys.argv[9]) if len(sys.argv) > 9 else 0.5
    f_hi = float(sys.argv[10]) if len(sys.argv) > 10 else 2.0
    pct_level = float(sys.argv[11]) if len(sys.argv) > 11 else 0.20
    w_ahead = float(sys.argv[12]) if len(sys.argv) > 12 else config.TRAIN_W_AHEAD
    w_behind = float(sys.argv[13]) if len(sys.argv) > 13 else config.TRAIN_W_BEHIND
    gnn_scale_slow = float(sys.argv[14]) if len(sys.argv) > 14 else config.GNN_SCALE_SLOW
    gnn_rho_slow = float(sys.argv[15]) if len(sys.argv) > 15 else config.GNN_RHO_SLOW

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"f_range=({f_lo},{f_hi})  pct_level={pct_level}  w_ahead={w_ahead}  w_behind={w_behind}  "
          f"gnn_scale_slow={gnn_scale_slow}  gnn_rho_slow={gnn_rho_slow})", flush=True)
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
    print("experiment) + ONE shared x0 (drawn once) -- x0-ANCHORED perturbation:")
    print("each node's box is rebuilt AROUND x0 by rescaling the base market's")
    print("own distance from x0 to each edge (D_min/D_max, G_min/G_max) by two")
    print(f"INDEPENDENT positive factors in [{f_lo},{f_hi}] -- x0 is mathematically")
    print("inside the box for every draw, by construction, no clip ever needed.")
    print(f"D*/G_res keep their own independent 'level' factor +/-{pct_level:.0%}.")
    print("Infeasible or negative draws discarded+resampled.")
    print("FIXED train/test sets (slot 0 = original network, always)")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=0)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    fixed_x0_gen = torch.Generator(device=device); fixed_x0_gen.manual_seed(config.TRAIN_SEED)
    fixed_x0_net = market.build_network(base_spec, device=device, rng=fixed_x0_gen)
    fixed_x0_base = fixed_x0_net.flatten().detach().clone()
    N = base_spec.N
    x0_D = fixed_x0_base[:N].numpy()
    x0_G = fixed_x0_base[N:2 * N].numpy()
    print(f"  shared x0 drawn once (seed={config.TRAIN_SEED}), every box rebuilt around it", flush=True)

    train_specs, train_log, n_discarded_train = sample_x0_anchored_dataset(
        base_spec, x0_D, x0_G, n_instances=n_train - 1, seed=config.TRAIN_SEED,
        f_range=(f_lo, f_hi), pct_level=pct_level)
    train_specs = [base_spec] + train_specs
    test_specs, test_log, n_discarded_test = sample_x0_anchored_dataset(
        base_spec, x0_D, x0_G, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1,
        f_range=(f_lo, f_hi), pct_level=pct_level)
    test_specs = [base_spec] + test_specs
    print(f"  train: {n_train} FIXED instances (slot 0 = original network, "
          f"{n_train - 1} perturbed), {n_discarded_train} infeasible/negative draws discarded+resampled  |  "
          f"test: {n_test} FIXED instances (slot 0 = original), {n_discarded_test} infeasible/negative draws "
          f"discarded+resampled", flush=True)

    train_x0 = build_per_instance_x0(train_specs, fixed_x0_base, device)
    test_x0 = build_per_instance_x0(test_specs, fixed_x0_base, device)
    n_adjusted_train = sum(1 for spec, x0 in zip(train_specs, train_x0)
                            if not torch.allclose(x0, fixed_x0_base))
    n_adjusted_test = sum(1 for spec, x0 in zip(test_specs, test_x0)
                           if not torch.allclose(x0, fixed_x0_base))
    print(f"  x0 needed adjusting (sanity check -- should be ~0, x0 is inside by construction): "
          f"{n_adjusted_train}/{n_train} train, {n_adjusted_test}/{n_test} test", flush=True)

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='dr_slope')", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device,
                           fixed_x0=train_x0, val_fixed_x0=test_x0[:5])

    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, per-instance-projected fixed x0 --", flush=True)
    print("the only evaluation mode (x0 is an input the user supplies, not something that varies)", flush=True)
    print("-" * 70)
    test_res = evaluate_fixed_x0(test_specs, test_x0, model, dr_cfg, device, label="fixed-x0")

    subtitle = (f"trained on: {n_train} instances (fixed x0+topology, x0-anchored perturbation "
                f"f=[{f_lo},{f_hi}]/level+/-{pct_level:.0%}, w_ahead={w_ahead}/w_behind={w_behind}), "
                f"{epochs} epochs  |  tested on: n={n_test}, SAME shared x0 (x0-anchored)")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    x0_by_id = {id(spec): x0 for spec, x0 in zip(test_specs, test_x0)}
    def _run_pair_fn(spec, model_, dr_cfg_, device_, seed, record=False):
        return run_pair_fixed_x0(spec, model_, dr_cfg_, device_, seed, record=record, x0=x0_by_id[id(spec)])
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=_run_pair_fn)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: v7 of the fixed-x0/parameter-diversity training axis -- "x0-anchored" perturbation
instead of v5/v6's "level+width". Each node's D-box and G-box are rebuilt AROUND the one shared
fixed x0 by rescaling the base market's own distance from x0 to each edge by two INDEPENDENT
positive factors in [{f_lo},{f_hi}] -- x0 is mathematically guaranteed inside every box, by
construction, for every draw. D*/G_res keep their own independent level factor (+/-{pct_level:.0%}),
D* re-clipped into the new box. Same topology (density=1.0), same realistic base market as v1-v6.

**Why**: v5/v6 built the box from D*'s level+width and patched x0 back in afterward when it drifted
out (~93% of instances needed a clip). v7 removes the need for that patch entirely -- and because the
two box-edge distances scale independently, x0's relative position inside the box still varies
freely (the "how far is x0 from a good spot" signal that correlated with speedup, r=0.428 in v5),
without the v1-v4 box-collapse risk (the box can shrink toward x0 but never invert past it).

**Dataset**: {n_train} FIXED training instances (slot 0 = unperturbed original), {n_test} FIXED test
instances (slot 0 = original). x0 needed adjusting in {n_adjusted_train}/{n_train} train and
{n_adjusted_test}/{n_test} test instances (sanity check on the by-construction guarantee).

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).

**Question answered**: does anchoring the box construction to x0 (guaranteeing it's always inside,
while still letting its relative position vary via independent below/above scale factors) improve
on v5's 1.25x/100% (previous best of this training axis)?
""")


if __name__ == "__main__":
    main()
