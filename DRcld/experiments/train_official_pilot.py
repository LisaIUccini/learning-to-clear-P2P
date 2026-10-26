"""
train_official_pilot.py
==========================================

First official training + in-distribution test on the RECALIBRATED market
generator (`build_calibrated_market.py`), one density at a time, using the
hyperparameters resolved earlier today (2026-10-14):

    gamma*(density), lambda=1.0 (config.DR_LAMBDA)   -- calibrate_gamma_
                                                         calibrated_market.py
    unroll, rho_10 (= 1-10/unroll, standing rule),
    rho_5  (= 1-5/unroll, pilot alternative)          -- calibrate_unroll_rho.py

Pilot comparison (user-specified): trains TWO models per density, one at
each rho, everything else identical, to see which decay schedule the
NEWLY-calibrated gamma actually prefers (the standing rule rho=1-10/K was
never checked against these new, very different gamma/k_DR numbers --
e.g. density=0.6 needs gamma=300, an order of magnitude above every other
density).

Per-test-instance log/plots now include everything the user asked for:
    - k_DR and k_learned (`evaluate_with_curves`'s per-instance print)
    - residual trajectory, both x^k and xbar^k (diagnostics.plot_instance,
      panels 1-2)
    - RELATIVE optimality gap |J(xbar^k)-J*|/|J*| (panel 3, normalised
      2026-10-14 -- was absolute before)
    - constraint violation dist(xbar^k, X_loc) (panel 4, NEW 2026-10-14,
      network.Network.dist_to_Xloc + algorithm.DRResult.violation)
  all on the SHADOW iterate xbar^k, saved as one 4-panel PNG per test
  instance via `diagnostics.save_instance_plots`.

Train/val/test are properly separated (own seeds: TRAIN_SEED/+1/+2, no
leak -- the 2026-10-14 validation-set fix), market instances perturbed via
level+width (+/-10%/+/-4%) with x0 left FREE (fresh random draw per
instance/epoch during training, fresh per test instance at eval) --
same "combined diversity" design as `demo_train_combined_levelwidth_
x0diversity.py`, just on the recalibrated generator + the new gamma/rho.
Slow channel kept OFF (config.GNN_SCALE_SLOW=0.0, the config default) --
this first pilot starts from the simpler fast-channel-only model, not
`demo_train_combined_levelwidth_x0diversity.py`'s own scale_slow=0.4
override; add it back later only if the fast channel alone proves
insufficient.

Usage: python3 train_official_pilot.py <density> [rho_variant=10|5]
       [run_name] [n_train=25] [n_test=30] [epochs=60] [base_seed=0]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import statistics as stats
import sys
from dataclasses import dataclass
from typing import Tuple

import torch

import config
import market
import algorithm
import gnn as gnn_mod
import train as train_mod
from diagnostics import save_instance_plots
from demo import plot_results, plot_training_loss, write_run_notes, run_pair
from plot_style import stats_from
from demo_train_realistic_market_x0_diversity import evaluate_with_curves
from build_calibrated_market import sample_calibrated_base_market, CalibratedConfig
from build_realistic_training_dataset import sample_level_width_perturbed_dataset

# resolved 2026-10-14 (seed=0 only, per user's confirmed decision to use the
# base market actually trained/tested on, not a cross-seed aggregate)
GAMMA_STAR = {0.2: 84.0, 0.4: 92.0, 0.5: 36.0, 0.6: 300.0, 0.8: 22.0, 1.0: 16.0}
UNROLL_STAR = {0.2: 46, 0.4: 111, 0.5: 87, 0.6: 183, 0.8: 29, 1.0: 129}
RHO10_STAR = {0.2: 0.7833, 0.4: 0.9101, 0.5: 0.8857, 0.6: 0.9453, 0.8: 0.6602, 1.0: 0.9222}
RHO5_STAR = {0.2: 0.8916, 0.4: 0.9550, 0.5: 0.9428, 0.6: 0.9726, 0.8: 0.8301, 1.0: 0.9611}


class _Tee:
    """Mirrors every `print` to both the terminal and a log file inside
    the run's own `out_dir` -- so the run's progress is always readable
    straight from its own folder (alongside phi_w.pt/results.png/NOTES.md),
    not only from whatever /tmp path happened to be used to launch it."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@dataclass
class _FeatureRanges:
    """Normalisation ranges for the GNN's own input features
    (`gnn.GNNConfig.from_market_config` only reads these 4 fields) --
    hand-computed from `build_calibrated_market.py`'s own constants, NOT
    the old `market.MarketConfig` ranges (which don't apply here: a_tilde
    is DERIVED not sampled, a varies by technology class, c = s - g).
    a_tilde: b_tilde/max(D*,0.3)^2, D* empirically ~0.5-7 in this scheme
    -> a_tilde typically ~0.02-4, wide margin kept for the D*<1 tail.
    a: markup/G_max across all 3 classes, exactly (0.0002, 0.045).
    b: B_RANGE, exactly (0.045, 0.070).
    c: s - g, s in C_RANGE clipped to 0.95*min(b), g in G_DISCOUNT_RANGE
    -> roughly (0.005, 0.043)."""
    a_tilde_range: Tuple[float, float] = (0.01, 4.0)
    a_range: Tuple[float, float] = (0.0, 0.05)
    b_range: Tuple[float, float] = (0.045, 0.070)
    c_range: Tuple[float, float] = (0.0, 0.045)


def main():
    density = float(sys.argv[1])
    rho_variant = sys.argv[2] if len(sys.argv) > 2 else "10"
    assert rho_variant in ("10", "5"), "rho_variant must be '10' or '5'"
    run_name = sys.argv[3] if len(sys.argv) > 3 else f"official_d{density}_rho{rho_variant}"
    n_train = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    n_test = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    epochs = int(sys.argv[6]) if len(sys.argv) > 6 else 60
    base_seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    # one-off market-size override (2026-10-14, N=25 scaling test): the
    # GAMMA_STAR/UNROLL_STAR/RHO*_STAR dicts above are N=10-specific
    # (config.BASE_MARKET_N's own calibration) -- passing n_nodes bypasses
    # them with explicit values from that size's OWN calibration
    # (calibrate_gamma_calibrated_market.py / calibrate_unroll_rho.py, both
    # run with their own n_nodes override).
    n_nodes = int(sys.argv[8]) if len(sys.argv) > 8 else None
    gamma_override = float(sys.argv[9]) if len(sys.argv) > 9 else None
    unroll_override = int(sys.argv[10]) if len(sys.argv) > 10 else None
    rho10_override = float(sys.argv[11]) if len(sys.argv) > 11 else None
    rho5_override = float(sys.argv[12]) if len(sys.argv) > 12 else None
    # slow-channel override (2026-10-14, N=25 ablation: a longer TRAINING
    # unroll diluted the gradient and made things worse -- see module
    # docstring's "slow channel kept OFF" note -- trying the slow channel
    # instead, which extends the correction's effective horizon WITHOUT
    # touching unroll at all).
    scale_slow_override = float(sys.argv[13]) if len(sys.argv) > 13 else None
    rho_slow_override = float(sys.argv[14]) if len(sys.argv) > 14 else None
    # fast-channel amplitude override (2026-10-14, N=25 ablation: slow
    # channel abandoned after a mis-scaled rho_slow made it catastrophic,
    # see runs/official/official_N25_d0.2_slowchannel -- testing the fast
    # channel's own amplitude next, never retuned away from 1.0 for N=25).
    gnn_scale_override = float(sys.argv[15]) if len(sys.argv) > 15 else None
    # c_nm generation scheme override (2026-10-14, Le Cadre case (b) ablation:
    # independent_wide reproduces market.py's OLD generator's own per-direction
    # independent draw, suspected to be why "combined" experiments on that
    # generator worked so much better than this one's default seller_discount).
    c_scheme = sys.argv[16] if len(sys.argv) > 16 else "seller_discount"
    curvature_regime = sys.argv[17] if len(sys.argv) > 17 else "literature"   # 2026-10-14 ablation
    # generator="old" (2026-10-14, supervisor-specified ablation): use
    # market.py's ORIGINAL pre-recalibration generator (the one every
    # "combined" experiment ran on) instead of build_calibrated_market.py,
    # with ONE fix applied -- c_range narrowed so c_nm < b_n is guaranteed
    # (old default c_range=(0.05,0.30) vs b_range=(0.2,1.0) let c_nm > b_n
    # on some edges: since reciprocity is an INEQUALITY, a prosumer then
    # profits from "selling" energy nobody absorbs -- a real economic flaw,
    # not present in Le Cadre's own numbers (trading costs 1-3 vs b=30,
    # always c<<b there). Independent per-direction asymmetry (market.py's
    # own c[(m,n)] = U(*c_range) draw, one per directed pair) is KEPT
    # exactly as-is, per the supervisor's "il resto non cambia, compresa
    # l'asimmetria indipendente".
    generator = sys.argv[18] if len(sys.argv) > 18 else "calibrated"
    c_old_range = (tuple(float(x) for x in sys.argv[19].split(",")) if len(sys.argv) > 19
                   else (0.05, 0.18))
    # exposed 2026-10-14 to replicate runs/cross_density/combined_density02_v2_mild
    # (density=0.2, base_seed=3, OLD generator -- the one genuine ~70%-reduction/
    # ~3.9x-mean-speedup result found in this project's history) exactly: that
    # run used omega_mode=fixed/omega_power=1.5 (NOT this script's own dr_slope
    # default) and a cosine LR anneal down to 1% of TRAIN_LR over its epochs.
    omega_mode = sys.argv[20] if len(sys.argv) > 20 else "dr_slope"
    omega_power = float(sys.argv[21]) if len(sys.argv) > 21 else 0.0
    lr_final_frac = (float(sys.argv[22]) if len(sys.argv) > 22 and sys.argv[22] != "none" else None)
    # w_ahead/w_behind exposed 2026-10-14 (several best historical "combined"
    # runs use 1.6/6.0 or 1.6/10.0, not this script's hardcoded 1.0/2.0).
    w_ahead = float(sys.argv[23]) if len(sys.argv) > 23 else 1.0
    w_behind = float(sys.argv[24]) if len(sys.argv) > 24 else 2.0
    if n_nodes is not None:
        assert None not in (gamma_override, unroll_override, rho10_override, rho5_override), \
            "n_nodes override requires gamma/unroll/rho10/rho5 overrides too (GAMMA_STAR etc. are N=10-only)"

    dr_gamma = gamma_override if gamma_override is not None else GAMMA_STAR[density]
    unroll = unroll_override if unroll_override is not None else UNROLL_STAR[density]
    if rho_variant == "10":
        gnn_rho = rho10_override if rho10_override is not None else RHO10_STAR[density]
    else:
        gnn_rho = rho5_override if rho5_override is not None else RHO5_STAR[density]

    out_dir = os.path.join("runs", "official", run_name)
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "train.log"), "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print(f"writing all artifacts to {out_dir}/  (density={density}  N={n_nodes or config.BASE_MARKET_N}  "
          f"c_scheme={c_scheme}  rho_variant={rho_variant}  gamma*={dr_gamma}  unroll={unroll}  "
          f"gnn_rho={gnn_rho}  n_train={n_train}  n_test={n_test}  epochs={epochs}  base_seed={base_seed}  "
          f"omega_mode={omega_mode}  omega_power={omega_power}  lr_final_frac={lr_final_frac})",
          flush=True)

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA, n_iter=config.DR_MAX_ITER,
                                 tol=config.DR_TOLERANCE, bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=config.TRAIN_GAMMA_MONO,
                                       epochs=epochs, w_behind=w_behind, w_ahead=w_ahead,
                                       lr=config.TRAIN_LR, lr_final_frac=lr_final_frac,
                                       seed=config.TRAIN_SEED, loss_mode="asymmetric",
                                       omega_mode=omega_mode, omega_power=omega_power)

    print("=" * 70)
    print(f"OFFICIAL PILOT -- recalibrated market generator, density={density}, base_seed={base_seed}")
    print(f"gamma*={dr_gamma} (resolved via plateau-center grid search), lambda={dr_cfg.lam}, "
          f"rho_variant={rho_variant} (gnn_rho={gnn_rho})")
    print("=" * 70)

    if generator == "calibrated":
        cfg = CalibratedConfig(density=density, c_scheme=c_scheme, curvature_regime=curvature_regime)
        neighbors, base_spec, tier_names = sample_calibrated_base_market(base_seed, cfg, N=n_nodes)
        feature_ranges = _FeatureRanges()
        print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}  "
              f"tiers: low={tier_names.count('low')} medium={tier_names.count('medium')} "
              f"high={tier_names.count('high')}", flush=True)
    elif generator == "old":
        # matches REALISTIC_MARKET_CONFIG (build_realistic_training_dataset.py)
        # exactly, the config every "combined" experiment actually ran on --
        # bare MarketConfig() defaults have Gmin_range=(-2,0) (negative!),
        # which sample_level_width_perturbed_dataset's own all-nonnegative
        # filter rejects every single draw of (not a c_old_range artifact).
        mc = market.MarketConfig(density=density, Gmax_range=(2.0, 8.0), Gmin_range=(0.0, 1.0),
                                  G_res_range=(0.5, 2.0), D_spread_range=(0.4, 1.0),
                                  c_range=c_old_range)
        _, _, base_spec = market.sample_base(n_nodes or config.BASE_MARKET_N, mc, seed=base_seed)
        feature_ranges = mc   # MarketConfig already HAS a_tilde_range/a_range/b_range/c_range
        print(f"  [generator=old, c_old_range={c_old_range}, c<b guaranteed: "
              f"max(c)={c_old_range[1]} < min(b)={mc.b_range[0]}]")
        print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}", flush=True)
    else:
        raise ValueError(f"unknown generator {generator!r}")

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
    print("  x0: FRESH RANDOM per (instance, epoch) during training, fresh per test instance at eval", flush=True)

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode='asymmetric', omega_mode='{omega_mode}')",
          flush=True)
    scale_slow = scale_slow_override if scale_slow_override is not None else config.GNN_SCALE_SLOW
    rho_slow = rho_slow_override if rho_slow_override is not None else config.GNN_RHO_SLOW
    print(f"slow channel: scale_slow={scale_slow}  rho_slow={rho_slow}", flush=True)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        feature_ranges, rho=gnn_rho, scale=(gnn_scale_override if gnn_scale_override is not None else 1.0),
        rho_slow=rho_slow,
        scale_slow=scale_slow, hidden_dim=config.GNN_HIDDEN_DIM))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=val_specs, device=device)

    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))

    print(f"\nevaluating: plain DR vs DR + Phi_w, in-distribution test set, FRESH RANDOM x0", flush=True)
    print("-" * 70)
    test_res = evaluate_with_curves(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                                     label=f"d{density}/rho{rho_variant}")
    st = stats_from(test_res)
    print(f"  mean % reduction (per-instance average): mean={st['pct_mean']:.1f}%  "
          f"median={st['pct_median']:.1f}%", flush=True)

    subtitle = (f"density={density}, gamma*={dr_gamma}, rho_variant={rho_variant} (rho={gnn_rho})  |  "
                f"trained on {n_train} instances, {epochs} epochs  |  tested on {n_test}, fresh x0")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots (4 panels: residual x^k, residual xbar^k, "
          "relative optimality gap, local constraint violation)...", flush=True)
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Official pilot** -- recalibrated market generator (build_calibrated_market.py), density={density},
base_seed={base_seed}. gamma*={dr_gamma} (plateau-center grid search, seed=0), lambda={dr_cfg.lam}
(kept at project default -- a 2026-10-14 lambda sweep found 15-40% gains at lambda>1 for most
densities, deferred to future work per user decision). rho_variant={rho_variant} (gnn_rho={gnn_rho},
unroll={unroll}).

**Train/val/test**: properly separated (seeds TRAIN_SEED/+1/+2), level+width +/-10%/+/-4% perturbation,
x0 free (fresh random per instance/epoch).

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
