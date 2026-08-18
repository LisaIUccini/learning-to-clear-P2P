"""
demo.py
=======

End-to-end demonstration: train the perturbation GNN centrally, deploy it
fully distributedly, and compare plain Douglas-Rachford against DR +
learned perturbation on held-out market instances.

Held-out data is IN-CLUSTER only for now (see market.py): small (+/- 10%)
perturbations of the SAME reference market used for training (disjoint
draws, never the exact training instances). `market.sample_dataset` still
gives a fully independent, out-of-cluster set for a harder generalisation
test later; this script just doesn't evaluate it at the moment.
"""

from __future__ import annotations

import os
import statistics as stats
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import market
import algorithm
import gnn as gnn_mod
import train as train_mod
from diagnostics import save_instance_plots
import plot_style
from plot_style import C_DR, C_LEARNED, C_WIN, C_LOSS, C_TRAIN, C_VAL, INK, INK_MUTED, clean_axes, median_band, stats_from


# ----------------------------------------------------------------------
def run_pair(spec, model, dr_cfg, device, seed, record=False):
    """Plain DR and DR+GNN from the SAME random initial state x0."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    net = market.build_network(spec, device=device, rng=gen)
    x0 = net.flatten().clone()

    gnn_mod.undeploy(net)
    res_dr = algorithm.run_dr(net, dr_cfg, record=record)

    net2 = market.build_network(spec, device=device)
    net2.unflatten(x0)
    gnn_mod.deploy(model, net2)
    res_learned = algorithm.run_dr(net2, dr_cfg, record=record)

    return res_dr, res_learned


def build_curriculum_dataset(base_nb, base_prim, mc, dr_cfg, device, n_final, pct, seed,
                              pool_multiplier=3, hard_frac=0.5, screen_cap=2000):
    """A training set biased toward SLOW/HARD instances (curriculum "B1"):
    the fast channel's fast-decay schedule is already validated; the new
    slow channel needs enough gradient signal from genuinely hard instances
    to learn anything, and those are a small minority under plain uniform
    cluster sampling (most instances converge in a few hundred iterations).

    Draws a POOL of `n_final * pool_multiplier` candidate cluster instances,
    measures each one's plain-DR iteration count (capped at `screen_cap`,
    cheap -- no GNN, no gradients), and keeps: the base instance, the
    `hard_frac` fraction of slots filled by the SLOWEST candidates, and the
    rest filled by uniformly random candidates from what remains -- so hard
    instances dominate the gradient without the easy/gentle regime (needed
    to keep the fast channel's win-rate) disappearing from training.
    """
    n_pool = max(n_final * pool_multiplier, n_final)
    pool_specs = market.sample_cluster_dataset(base_nb, base_prim, mc, n_instances=n_pool,
                                                 pct=pct, seed=seed)
    screen_cfg = algorithm.DRConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, n_iter=screen_cap,
                                     tol=dr_cfg.tol, bisect=dr_cfg.bisect)
    scored = []
    for spec in pool_specs:
        net = market.build_network(spec, device=device)
        res = algorithm.run_dr(net, screen_cfg, record=False)
        scored.append((res.n_iter, spec))
    scored.sort(key=lambda t: t[0], reverse=True)

    n_hard = max(0, round(hard_frac * (n_final - 1)))
    hard_specs = [spec for _, spec in scored[:n_hard]]
    remaining = [spec for _, spec in scored[n_hard:]]
    rng = np.random.default_rng(seed + 1)
    n_fill = max(0, (n_final - 1) - n_hard)
    fill_idx = rng.choice(len(remaining), size=min(n_fill, len(remaining)), replace=False)
    fill_specs = [remaining[i] for i in fill_idx]

    iters_hard = [n for n, _ in scored[:n_hard]]
    iters_fill = [scored[n_hard + i][0] for i in fill_idx]
    print(f"  curriculum: pool={n_pool}  hard picks: {n_hard} (iters {min(iters_hard, default=0)}"
          f"-{max(iters_hard, default=0)})  random fill: {len(fill_specs)} "
          f"(iters {min(iters_fill, default=0)}-{max(iters_fill, default=0)})")
    return hard_specs + fill_specs


def evaluate(specs, model, dr_cfg, device, seed, label):
    dr_iters, learned_iters, speedups = [], [], []
    for i, spec in enumerate(specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=seed * 7919 + i)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter)
        speedups.append(sp)
        flag = "" if (res_dr.converged and res_learned.converged) else "  [DID NOT CONVERGE]"
        print(f"  [{label}] {i + 1:2d}/{len(specs)}  N={spec.N:3d}  "
              f"DR {res_dr.n_iter:5d} it  |  learned {res_learned.n_iter:5d} it  "
              f"|  x{sp:5.2f} ({pct:+.1f}%){flag}")
    win_rate = sum(s >= 1.0 for s in speedups) / len(speedups)
    print(f"  [{label}] mean speed-up {stats.mean(speedups):.2f}x   "
          f"median {stats.median(speedups):.2f}x   "
          f"min {min(speedups):.2f}x   max {max(speedups):.2f}x   "
          f"win-rate {win_rate:.0%} ({sum(s >= 1.0 for s in speedups)}/{len(speedups)})")
    return {"dr": dr_iters, "learned": learned_iters, "speedup": speedups}


def write_run_notes(out_dir: str, text: str) -> None:
    """Every run folder gets a NOTES.md explaining what that specific test
    is, so runs/ stays self-documenting without needing runs/INDEX.md open
    side by side."""
    path = os.path.join(out_dir, "NOTES.md")
    with open(path, "w") as f:
        f.write(text.strip() + "\n")
    print(f"saved {path}")


# ----------------------------------------------------------------------
def plot_results(test_res, tol, out_path="results.png", subtitle=""):
    """2x2 layout: stats scoreboard (mean/median/min-max/win-rate), the
    below-the-line scatter, per-instance speed-up (green = win, red =
    loss against the 1.0x breakeven), and the median/IQR convergence
    band. No training-loss panel here -- see `plot_training_loss` for
    that, saved as its own file so it doesn't compete with these four for
    space. `subtitle` is a one-line "trained on: ... | tested on: ..."
    caption; no other title is drawn."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    st = stats_from(test_res)

    # -- top-left: stats scoreboard --
    ax = axes[0, 0]
    ax.axis("off")
    rows = [
        ("mean speed-up", f"{st['mean']:.2f}x", C_LEARNED),
        ("median speed-up", f"{st['median']:.2f}x", C_LEARNED),
        ("min / max", f"{st['min']:.2f}x / {st['max']:.2f}x", INK_MUTED),
        ("win-rate", f"{st['win_rate']:.0%}  ({st['n_win']}/{st['n']})", C_WIN),
    ]
    y0 = 0.86
    for row_label, value, accent in rows:
        ax.add_patch(plt.Rectangle((0.03, y0 - 0.02), 0.05, 0.16, transform=ax.transAxes,
                                    facecolor=accent, edgecolor="none"))
        ax.text(0.14, y0 + 0.06, value, transform=ax.transAxes, fontsize=20,
                fontweight="bold", color=INK, va="center")
        ax.text(0.14, y0 - 0.08, row_label, transform=ax.transAxes, fontsize=10.5,
                color=INK_MUTED, va="center")
        y0 -= 0.26

    # -- top-right: scatter, below the line = learned is faster --
    ax = axes[0, 1]
    lo = min(min(test_res["dr"]), min(test_res["learned"]))
    hi = max(max(test_res["dr"]), max(test_res["learned"]))
    ax.plot([lo, hi], [lo, hi], color="#b5b5b5", linestyle="--", linewidth=1.2, zorder=1)
    ax.scatter(test_res["dr"], test_res["learned"], s=46, color=C_LEARNED,
               edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("plain DR iterations"); ax.set_ylabel("DR + learned iterations")
    ax.set_title("below the line = learned model is faster")
    clean_axes(ax)

    # -- bottom-left: per-instance speed-up, in test-set order (matches
    # demo_run.log's numbering and instance_plots/instance_NN_*.png, so a
    # bar at position N is instance N+1 everywhere -- NOT sorted by size,
    # sorting would silently break that correspondence) --
    ax = axes[1, 0]
    speedups = np.asarray(test_res["speedup"])
    colors = [C_WIN if s >= 1.0 else C_LOSS for s in speedups]
    ax.bar(range(1, len(speedups) + 1), speedups, color=colors, width=0.72, zorder=3)
    ax.axhline(1.0, color=INK_MUTED, linestyle="--", linewidth=1.1, zorder=2)
    ax.set_xlabel("test instance"); ax.set_ylabel("speed-up (DR it / learned it)")
    ax.set_title("per-instance speed-up")
    clean_axes(ax)

    # -- bottom-right: residual convergence, median + IQR band --
    ax = axes[1, 1]
    med_dr, lo_dr, hi_dr = median_band(test_res["curves_dr"])
    med_l, lo_l, hi_l = median_band(test_res["curves_learned"])
    k_dr = np.arange(1, len(med_dr) + 1); k_l = np.arange(1, len(med_l) + 1)
    ax.plot(k_dr, med_dr, label="plain DR (median)", color=C_DR, linewidth=2)
    ax.fill_between(k_dr, lo_dr, hi_dr, color=C_DR, alpha=0.35, linewidth=0)
    ax.plot(k_l, med_l, label="DR + learned (median)", color=C_LEARNED, linewidth=2)
    ax.fill_between(k_l, lo_l, hi_l, color=C_LEARNED, alpha=0.35, linewidth=0)
    ax.axhline(tol, color=INK_MUTED, linestyle="--", linewidth=1.1, label="tolerance")
    ax.set_yscale("log")
    ax.set_xlabel("iteration k"); ax.set_ylabel(r"$\|x^{k+1}-x^k\|$  (median, IQR)")
    ax.set_title("convergence")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    clean_axes(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha="center", fontsize=10, color=INK_MUTED, style="italic")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_training_loss(history, out_path="training_loss.png", subtitle=""):
    """Separate figure, train/val hinge loss vs. epoch -- kept out of
    `plot_results` so the two don't compete for space. `subtitle` is a
    one-line "trained on: ..." caption; no other title is drawn."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history["train"]) + 1), history["train"],
            label="train", color=C_TRAIN, linewidth=2.2)
    if history.get("val"):
        ax.plot(range(1, len(history["val"]) + 1), history["val"],
                label="val", color=C_VAL, linewidth=2.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(frameon=False, fontsize=9.5)
    clean_axes(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    if subtitle:
        fig.text(0.5, 0.955, subtitle, ha="center", fontsize=10, color=INK_MUTED, style="italic")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


# ----------------------------------------------------------------------
def main():
    # every artifact from this run goes under runs/<run_name>/ -- nothing
    # is ever written to the DRcld/ top level, so no run can overwrite
    # another's results.png / phi_w.pt / instance_plots/. Pass a name:
    #     python3 demo.py my_experiment_name
    run_name = sys.argv[1] if len(sys.argv) > 1 else "latest"
    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/")

    device = torch.device("cpu")
    seed = 0
    torch.manual_seed(seed)

    mc = market.MarketConfig(density=0.3)
    dr_cfg = algorithm.DRConfig(gamma=4.0, lam=1.0, n_iter=4000, tol=1e-5, bisect=30)
    # loss_mode="asymmetric": continuous, UNCAPPED reward for beating plain
    # DR (more reward the further ahead) and continuous penalty for
    # lagging it (more penalty the further behind, weighted 2x to bias
    # toward win-rate) -- unlike "hinge" (flat zero gradient past a small
    # margin, which caps achievable average magnitude by construction).
    # unroll extended 200->350 while rho stays 0.95 (short, UNCHANGED --
    # the correction still fully decays by k~150-200): lets training see
    # more of the consequence of an early intervention without ever
    # giving the correction itself more time to interfere with a late
    # plateau -- the combination that broke A1 and A2+B1 changed the
    # decay too. See train.py's module docstring for the full comparison.
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=350, gamma_mono=2.0, epochs=20, lr=3e-3,
                                       seed=seed, loss_mode="asymmetric", w_behind=4.0, w_ahead=1.0)

    print("=" * 70)
    print("dataset: ONE reference market (N0=10), all other instances are")
    print("+/- 10% perturbations of it (topology, size, and parameters)")
    print("=" * 70)
    base_nb, base_prim, base_spec = market.sample_base(10, mc, seed=seed)
    train_specs = [base_spec] + market.sample_cluster_dataset(
        base_nb, base_prim, mc, n_instances=24, pct=0.10, seed=seed + 1)
    test_specs = market.sample_cluster_dataset(
        base_nb, base_prim, mc, n_instances=12, pct=0.10, seed=seed + 2)

    print("\nsanity check: plain DR converges on a few training instances")
    for spec in train_specs[:3]:
        net = market.build_network(spec, device=device)
        res = algorithm.run_dr(net, dr_cfg)
        print(f"  N={spec.N:3d}  iters={res.n_iter:5d}  converged={res.converged}  "
              f"feasible={net.is_feasible(1e-3)}")

    print(f"\ntraining Phi_w (centralised, self-supervised, loss_mode={train_cfg.loss_mode!r})")
    # DEFAULT/VALIDATED config (matches runs/hinge_v5_final, the best result
    # found so far: 72% win-rate). The architecture is dual-channel (gnn.py),
    # but scale_slow=0.0 here neutralises the slow channel completely (its
    # contribution is scale_slow * decay_slow(k) * direction -- exactly zero
    # regardless of what it learns), reproducing the validated single-channel
    # behaviour. Two attempts to actually USE a longer reach -- A1 (single
    # channel, slower decay: 72%->25% win-rate) and A2+B1 (dual channel +
    # curriculum: 72%->8%) -- both made things substantially WORSE; see
    # runs/a1_moderate_horizon/ and runs/a2b1_dual_channel_curriculum/.
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=0.95, scale=0.4, scale_slow=0.0))
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    print("\nevaluating: plain DR vs DR + Phi_w, fully distributed inference")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="in-cluster")

    print("\nrecording residual curves on the in-cluster test set...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = "trained on: reference market +/- 10% (topology, size, params)  |  tested on: same cluster, n=" + str(len(test_specs))
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"),
                        subtitle=f"trained on: reference market +/- 10% cluster, {len(train_specs)} instances, {train_cfg.epochs} epochs")

    print("\ngenerating per-instance diagnostic plots (residual vs. iteration,")
    print("shadow-residual vs. iteration, cost gap vs. iteration)...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    weights_path = os.path.join(out_dir, "phi_w.pt")
    trainer.save(weights_path)
    print(f"saved trained weights to {weights_path}")


if __name__ == "__main__":
    main()
