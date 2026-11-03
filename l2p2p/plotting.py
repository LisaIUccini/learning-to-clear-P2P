"""Diagnostic figures saved next to every run (results.png,
training_loss.png, instance_plots/*.png). Paper figures have their own
style, in paper/style.py.

A series keeps its colour in every figure: C_DR plain DR, C_LEARNED the
accelerated method, C_WIN/C_LOSS per-instance gain/loss, C_TRAIN/C_VAL
loss curves.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PASTEL = sns.color_palette("pastel")
C_DR = PASTEL[0]
C_LEARNED = PASTEL[1]
C_WIN = PASTEL[2]
C_LOSS = PASTEL[3]
C_TRAIN = PASTEL[4]
C_VAL = PASTEL[5]

INK = "#2b2b2b"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e3"

plt.rcParams.update({
    "font.size": 10.5,
    "axes.edgecolor": "#b5b5b5",
    "axes.labelcolor": INK_MUTED,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b5b5b5")
    ax.spines["bottom"].set_color("#b5b5b5")
    ax.tick_params(which="both", top=False, right=False)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def median_band(curves):
    """Per-iteration median and IQR of residual curves; shorter (converged)
    curves are padded with their last value."""
    max_len = max(len(c) for c in curves)
    padded = np.stack([np.concatenate([c, np.full(max_len - len(c), c[-1])]) for c in curves])
    return np.median(padded, axis=0), np.percentile(padded, 25, axis=0), np.percentile(padded, 75, axis=0)


def plot_results(test_res, summary, tol, out_path, subtitle=""):
    """2x2 overview of a test set: scoreboard, DR-vs-method scatter,
    per-instance reduction (in test-set order, matching instance_plots/),
    and the median/IQR residual curves."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    st = summary

    ax = axes[0, 0]
    ax.axis("off")
    rows = [
        ("mean reduction", f"{st['pct_mean']:+.0f}%", C_LEARNED),
        ("median reduction", f"{st['pct_median']:+.0f}%", C_LEARNED),
        ("min / max", f"{st['pct_min']:+.0f}% / {st['pct_max']:+.0f}%", INK_MUTED),
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

    ax = axes[1, 0]
    dr = np.asarray(test_res["dr"]); learned = np.asarray(test_res["learned"])
    pct = (dr - learned) / dr * 100.0
    colors = [C_WIN if p >= 0.0 else C_LOSS for p in pct]
    ax.bar(range(1, len(pct) + 1), pct, color=colors, width=0.72, zorder=3)
    ax.axhline(0.0, color=INK_MUTED, linestyle="--", linewidth=1.1, zorder=2)
    ax.set_xlabel("test instance"); ax.set_ylabel("iteration reduction (%)")
    ax.set_title("per-instance iteration reduction")
    clean_axes(ax)

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
    plt.close(fig)
    print(f"saved {out_path}")


def plot_training_loss(history, out_path, subtitle=""):
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
    plt.close(fig)
    print(f"saved {out_path}")


def plot_instance(res_dr, res_learned, j_star: float, tol: float, title: str, out_path: str) -> None:
    """Four panels for one instance: residual of x^k, residual of the shadow
    iterate xbar^k, relative optimality gap of xbar^k, and dist(xbar^k, X_loc)."""
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.4))

    ax = axes[0]
    ax.semilogy(range(1, len(res_dr.residual) + 1), res_dr.residual,
                label="plain DR", color=C_DR, linewidth=2)
    ax.semilogy(range(1, len(res_learned.residual) + 1), res_learned.residual,
                label="DR + learned", color=C_LEARNED, linewidth=2)
    ax.axhline(tol, color=INK_MUTED, linestyle="--", linewidth=1.1, label="tolerance")
    ax.set_xlabel("iteration k"); ax.set_ylabel(r"$\|x^{k+1}-x^k\|$")
    ax.set_title("residual"); ax.legend(frameon=False, fontsize=8.5)
    clean_axes(ax)

    ax = axes[1]
    ax.semilogy(range(1, len(res_dr.shadow_residual) + 1), res_dr.shadow_residual,
                label="plain DR", color=C_DR, linewidth=2)
    ax.semilogy(range(1, len(res_learned.shadow_residual) + 1), res_learned.shadow_residual,
                label="DR + learned", color=C_LEARNED, linewidth=2)
    ax.axhline(tol, color=INK_MUTED, linestyle="--", linewidth=1.1, label="tolerance")
    ax.set_xlabel("iteration k"); ax.set_ylabel(r"$\|\bar{x}^{k+1}-\bar{x}^k\|$")
    ax.set_title("residual (shadow / feasible iterate)"); ax.legend(frameon=False, fontsize=8.5)
    clean_axes(ax)

    ax = axes[2]
    j_denom = max(abs(j_star), 1e-9)
    gap_dr = np.abs(np.asarray(res_dr.cost) - j_star) / j_denom + 1e-12
    gap_l = np.abs(np.asarray(res_learned.cost) - j_star) / j_denom + 1e-12
    ax.semilogy(range(1, len(gap_dr) + 1), gap_dr, label="plain DR", color=C_DR, linewidth=2)
    ax.semilogy(range(1, len(gap_l) + 1), gap_l, label="DR + learned", color=C_LEARNED, linewidth=2)
    ax.set_xlabel("iteration k"); ax.set_ylabel(r"$|J(\bar{x}^k) - J^\star| / |J^\star|$")
    ax.set_title("relative optimality gap"); ax.legend(frameon=False, fontsize=8.5)
    clean_axes(ax)

    ax = axes[3]
    viol_dr = np.asarray(res_dr.violation) + 1e-12
    viol_l = np.asarray(res_learned.violation) + 1e-12
    ax.semilogy(range(1, len(viol_dr) + 1), viol_dr, label="plain DR", color=C_DR, linewidth=2)
    ax.semilogy(range(1, len(viol_l) + 1), viol_l, label="DR + learned", color=C_LEARNED, linewidth=2)
    ax.set_xlabel("iteration k"); ax.set_ylabel(r"$\mathrm{dist}(\bar{x}^k, X^{\mathrm{loc}})$")
    ax.set_title("local constraint violation"); ax.legend(frameon=False, fontsize=8.5)
    clean_axes(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.text(0.5, 0.965, title, ha="center", fontsize=10, color=INK_MUTED, style="italic")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
