"""
demo_train_combined_levelwidth_x0diversity.py
==========================================

COMBINED diversity design: trains on BOTH axes at once -- market instances
perturbed via the "level+width" scheme (D*, G_res, D_min, D_max, G_min,
G_max; default +/-10% level, +/-4% width, milder than the +/-20%/+/-8%
used for the dedicated parameter-diversity axis) AND x0 left FREE (fresh
random draw per instance/epoch during training, `Trainer.fit`'s default
behaviour when `fixed_x0` is not passed -- the SAME x0-diversity mechanism
`realistic_market_x0_diversity_v4_wbehind6` uses on a single fixed market).

Motivation (2026-09-12): a zero-shot test of the v4 checkpoint (x0-diversity
only, trained on ONE fixed market) on a level+width +/-10%/+/-4% perturbed
dataset, evaluated with fresh random x0 (its native convention), scored an
apparently strong 1.55x/97% -- but a controlled re-test on v6's own EXACT
protocol (same test instances, v6's single SHARED FIXED x0) showed this was
an artifact of the easier fresh-random-x0 evaluation: v4 zero-shot only
scored 1.08x/70% there, clearly below v6's own dedicated training
(1.25x/100%). This run asks the real question directly, instead of via
zero-shot proxy: what happens if a model is TRAINED (not just evaluated)
on this exact combined distribution -- market diversity AND x0 diversity
together?

**Calibration** (measured 2026-09-12, standing rule unroll=mean/2,
rho=1-10/unroll, n=40 draws each = one level+width-perturbed instance +
one fresh random x0): mean=771.2, cv=0.2426, min=485, max=1229 ->
unroll=386, rho=0.9741. Notably higher cv than either axis alone (both
sources of heterogeneity stack).

**Architecture**: rho_slow=0.99/scale_slow=0.4 kept ON by default (matching
`realistic_market_x0_diversity_v4_wbehind6`'s own config -- this design is
an extension of v4's x0-diversity mechanism, not v6's fixed-x0 one, which
always had the slow channel off).

**Evaluation**: fresh random x0 per test instance (`evaluate_with_curves`),
matching the training distribution -- x0 is a genuine free variable here,
not a fixed input to project onto (unlike the fixed-x0/parameter-diversity
axis).

Usage: python3 demo_train_combined_levelwidth_x0diversity.py <run_name>
       [n_train] [n_test] [epochs] [dr_gamma] [unroll] [gnn_rho] [gnn_scale]
       [pct_level] [pct_width] [w_ahead] [w_behind] [gnn_rho_slow]
       [gnn_scale_slow] [gamma_mono] [omega_mode]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import statistics as stats
import sys

import torch

import config
import algorithm
import gnn as gnn_mod
import train as train_mod
from diagnostics import save_instance_plots
from demo import plot_results, plot_training_loss, write_run_notes, run_pair
from plot_style import stats_from
from demo_train_realistic_market_x0_diversity import evaluate_with_curves
from build_realistic_training_dataset import build_realistic_base_market, sample_level_width_perturbed_dataset, REALISTIC_MARKET_CONFIG


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "train_combined_levelwidth_x0diversity"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    dr_gamma = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    unroll = int(sys.argv[6]) if len(sys.argv) > 6 else 386
    gnn_rho = float(sys.argv[7]) if len(sys.argv) > 7 else 0.9741
    gnn_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    pct_level = float(sys.argv[9]) if len(sys.argv) > 9 else 0.10
    pct_width = float(sys.argv[10]) if len(sys.argv) > 10 else 0.04
    w_ahead = float(sys.argv[11]) if len(sys.argv) > 11 else 1.4
    w_behind = float(sys.argv[12]) if len(sys.argv) > 12 else 6.0
    gnn_rho_slow = float(sys.argv[13]) if len(sys.argv) > 13 else 0.99
    gnn_scale_slow = float(sys.argv[14]) if len(sys.argv) > 14 else 0.4
    gamma_mono = float(sys.argv[15]) if len(sys.argv) > 15 else config.TRAIN_GAMMA_MONO
    omega_mode = sys.argv[16] if len(sys.argv) > 16 else "dr_slope"
    hidden_dim = int(sys.argv[17]) if len(sys.argv) > 17 else config.GNN_HIDDEN_DIM
    lr_final_frac = float(sys.argv[18]) if len(sys.argv) > 18 and sys.argv[18] != "none" else None
    density = float(sys.argv[19]) if len(sys.argv) > 19 else 1.0
    omega_power = float(sys.argv[20]) if len(sys.argv) > 20 else 0.0
    base_seed = int(sys.argv[21]) if len(sys.argv) > 21 else 0

    out_dir = os.path.join("runs", "cross_density", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (n_train={n_train}  n_test={n_test}  epochs={epochs}  "
          f"dr_gamma={dr_gamma}  unroll={unroll}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"pct_level={pct_level}  pct_width={pct_width}  w_ahead={w_ahead}  w_behind={w_behind}  "
          f"gnn_rho_slow={gnn_rho_slow}  gnn_scale_slow={gnn_scale_slow}  gamma_mono={gamma_mono}  "
          f"omega_mode={omega_mode}  hidden_dim={hidden_dim}  lr_final_frac={lr_final_frac}  "
          f"density={density}  omega_power={omega_power}  base_seed={base_seed})", flush=True)
    assert w_ahead < w_behind, "w_ahead must stay strictly below w_behind"

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    mc = REALISTIC_MARKET_CONFIG
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=gamma_mono,
                                       epochs=epochs, w_behind=w_behind, w_ahead=w_ahead,
                                       lr=config.TRAIN_LR, lr_final_frac=lr_final_frac,
                                       grad_clip=config.TRAIN_GRAD_CLIP,
                                       seed=config.TRAIN_SEED, loss_mode="asymmetric", omega_mode=omega_mode,
                                       omega_power=omega_power)

    print("=" * 70)
    print("dataset: COMBINED diversity -- market instances perturbed via level+width")
    print(f"(+/-{pct_level:.0%} level, +/-{pct_width:.0%} width) AND x0 left FREE (fresh random")
    print("draw per instance/epoch during training -- NOT fixed). Realistic market,")
    print(f"density={density}, base_seed={base_seed}.")
    print("=" * 70)
    base_nb, base_prim, base_spec = build_realistic_base_market(base_seed=base_seed, density=density)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)

    train_specs, train_log, n_discarded_train = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_train - 1, seed=config.TRAIN_SEED, pct_level=pct_level, pct_width=pct_width)
    train_specs = [base_spec] + train_specs
    test_specs, test_log, n_discarded_test = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_test - 1, seed=config.TRAIN_SEED + 1, pct_level=pct_level, pct_width=pct_width)
    test_specs = [base_spec] + test_specs
    # SEPARATE validation set (own seed, no overlap with train or test) --
    # used only for the per-epoch "val" loss printed during training/for
    # hyperparameter choices; final reported numbers always come from
    # test_specs. n_val=10, seed=TRAIN_SEED+2 (distinct from train's
    # TRAIN_SEED and test's TRAIN_SEED+1). Previously this reused
    # test_specs[:5] (a genuine train/val/test leak, fixed 2026-10-14).
    n_val = 10
    val_specs, val_log, n_discarded_val = sample_level_width_perturbed_dataset(
        base_spec, n_instances=n_val - 1, seed=config.TRAIN_SEED + 2, pct_level=pct_level, pct_width=pct_width)
    val_specs = [base_spec] + val_specs
    print(f"  train: {n_train} instances (slot 0 = original), {n_discarded_train} infeasible discarded  |  "
          f"test: {n_test} instances (slot 0 = original), {n_discarded_test} infeasible discarded  |  "
          f"val: {n_val} instances (slot 0 = original), {n_discarded_val} infeasible discarded", flush=True)
    print("  x0: FRESH RANDOM per (instance, epoch) during training, fresh per test instance at eval", flush=True)

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='{omega_mode}')", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale, rho_slow=gnn_rho_slow, scale_slow=gnn_scale_slow,
        hidden_dim=hidden_dim))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=val_specs, device=device)

    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    print(f"saved {os.path.join(out_dir, 'phi_w.pt')}", flush=True)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print("\nevaluating: plain DR vs DR + Phi_w, FRESH RANDOM x0 per test instance", flush=True)
    print("-" * 70)
    test_res = evaluate_with_curves(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="combined")
    st = stats_from(test_res)
    print(f"  [combined] CORRECT mean % reduction (per-instance average, not 1-1/mean_speedup): "
          f"mean={st['pct_mean']:.1f}%  median={st['pct_median']:.1f}%", flush=True)

    subtitle = (f"trained on: {n_train} instances (level+width +/-{pct_level:.0%}/+/-{pct_width:.0%} "
                f"market diversity + free x0, w_ahead={w_ahead}/w_behind={w_behind}), {epochs} epochs  |  "
                f"tested on: n={n_test}, fresh random x0")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...", flush=True)
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: COMBINED diversity design -- market instances perturbed via level+width
(+/-{pct_level:.0%}/+/-{pct_width:.0%}) AND x0 left FREE (fresh random per instance/epoch), instead of
fixing one axis while varying the other (v6: fixed x0 + market diversity; v4: fixed market + x0
diversity). Realistic market, density=1.0, base_seed=0.

**Motivation**: a zero-shot test of v4's checkpoint (x0-diversity-only) on this exact perturbed
distribution looked deceptively strong (1.55x/97%) evaluated with fresh random x0, but a controlled
re-test on v6's own protocol (same test set, v6's single fixed x0) showed only 1.08x/70% -- below
v6's own dedicated training (1.25x/100%). This run trains directly on the combined distribution
instead of relying on zero-shot transfer from either single-axis model.

**Calibration**: unroll={unroll}, gnn_rho={gnn_rho}, freshly measured on this exact combined
distribution (level+width perturbed market + fresh random x0, n=40): mean=771.2, cv=0.2426.

**Recipe**: w_ahead={w_ahead}, w_behind={w_behind}, gamma_mono={gamma_mono}, omega_mode={omega_mode},
gnn_scale={gnn_scale}, gnn_rho_slow={gnn_rho_slow}, gnn_scale_slow={gnn_scale_slow} (slow channel kept
ON, matching v4's own config, since this design extends v4's x0-diversity mechanism).

**Results**: mean {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
min {min(test_res['speedup']):.2f}x, max {max(test_res['speedup']):.2f}x, win-rate
{sum(s >= 1.0 for s in test_res['speedup']) / len(test_res['speedup']):.0%}
({sum(s >= 1.0 for s in test_res['speedup'])}/{len(test_res['speedup'])}).
Mean % reduction (per-instance average, the correct definition): {st['pct_mean']:.1f}%, median {st['pct_median']:.1f}%.

**Density**: {density} (base_seed={base_seed}).
""")


if __name__ == "__main__":
    main()
