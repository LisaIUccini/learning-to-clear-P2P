"""
regen_density_plots_svg.py
============================

Reconstructs 6 already-published PNGs (plot1b/c/d_pct_home_market_*,
plot2b/c/d_pct_dstar_gres_*) as true vector SVGs, from the SAME underlying
per-instance data (re-extracted from the original run logs -- no new
training/inference). Verified: per-density mean/median/min/max recomputed
here match the existing PNGs' values exactly (see chat), so this is a
faithful reconstruction, not a new analysis.

Usage: python3 regen_density_plots_svg.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import plot_style as ps

DENSITIES = [0.3, 0.5, 0.8, 1.0]

HOME_LOGS = {
    0.3: "runs/cross_density/seed0_density0.3_drslope/live.log",
    0.5: "runs/cross_density/seed0_density0.5_gamma6_unroll305_scale10_asymmetric/live.log",
    0.8: "runs/cross_density/seed0_density0.8_gamma8_unroll750_rho9925_drslope/live.log",
    1.0: "runs/cross_density/seed0_density1.0_seed0_drslope/live.log",
}
GRES_LOGS = {
    0.3: "runs/cross_density/seed0_density0.3_dstar_gres_scale10/live.log",
    0.5: "runs/cross_density/seed0_density0.5_dstar_gres_scale10/live.log",
    0.8: "runs/cross_density/seed0_density0.8_drslope_dstar_gres_scale10/live.log",
    1.0: "runs/cross_density/seed0_density1.0_seed0_dstar_gres_scale10/live.log",
}


def extract_pct(path: str) -> np.ndarray:
    text = open(path).read()
    return np.array([float(x) for x in re.findall(r'\(([+-][0-9.]+)%\)', text)])


def ci95(vals: np.ndarray) -> float:
    n = len(vals)
    se = vals.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return t * se


def plot_line(logs: dict, title_suffix: str, out_path: str) -> None:
    means = [extract_pct(logs[d]).mean() for d in DENSITIES]
    mins = [extract_pct(logs[d]).min() for d in DENSITIES]
    maxs = [extract_pct(logs[d]).max() for d in DENSITIES]

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.fill_between(DENSITIES, mins, maxs, color=ps.C_LEARNED, alpha=0.25, zorder=2)
    ax.plot(DENSITIES, means, color=ps.C_LEARNED, marker="o", linewidth=2.5,
            markersize=7, label="mean", zorder=3)
    ax.set_title(f"Mean improvement vs density -- {title_suffix} (n=30 each)\n"
                 f"shaded band = observed min-max range")
    ax.set_xlabel("density (sparsity)")
    ax.set_ylabel("iteration-count reduction (%)")
    ax.set_xticks(DENSITIES)
    ax.legend(loc="upper right", frameon=False)
    ps.clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_ci(logs: dict, title_suffix: str, out_path: str) -> None:
    means, cis = [], []
    for d in DENSITIES:
        vals = extract_pct(logs[d])
        means.append(vals.mean())
        cis.append(ci95(vals))
    means = np.array(means); cis = np.array(cis)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.fill_between(DENSITIES, means - cis, means + cis, color=ps.C_LEARNED, alpha=0.25, zorder=2)
    ax.plot(DENSITIES, means, color=ps.C_LEARNED, marker="o", linewidth=2.5,
            markersize=7, label="mean", zorder=3)
    ax.set_title(f"Mean improvement vs density -- {title_suffix} (n=30 each)\n"
                 f"shaded band = 95% confidence interval on the mean")
    ax.set_xlabel("density (sparsity)")
    ax.set_ylabel("iteration-count reduction (%)")
    ax.set_xticks(DENSITIES)
    ax.legend(loc="upper right", frameon=False)
    ps.clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_iqr(logs: dict, title_suffix: str, out_path: str) -> None:
    medians, q25, q75 = [], [], []
    for d in DENSITIES:
        vals = extract_pct(logs[d])
        medians.append(np.median(vals))
        q25.append(np.percentile(vals, 25))
        q75.append(np.percentile(vals, 75))

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.fill_between(DENSITIES, q25, q75, color=ps.C_LEARNED, alpha=0.25, zorder=2)
    ax.plot(DENSITIES, medians, color=ps.C_LEARNED, marker="o", linewidth=2.5,
            markersize=7, label="median", zorder=3)
    ax.set_title(f"Median improvement vs density -- {title_suffix} (n=30 each)\n"
                 f"shaded band = interquartile range (25th-75th percentile)")
    ax.set_xlabel("density (sparsity)")
    ax.set_ylabel("iteration-count reduction (%)")
    ax.set_xticks(DENSITIES)
    ax.legend(loc="upper right", frameon=False)
    ps.clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main():
    out_dir = "runs/cross_density"
    plot_line(HOME_LOGS, "home market test", os.path.join(out_dir, "plot1b_pct_home_market_line.svg"))
    plot_ci(HOME_LOGS, "home market test", os.path.join(out_dir, "plot1c_pct_home_market_ci.svg"))
    plot_iqr(HOME_LOGS, "home market test", os.path.join(out_dir, "plot1d_pct_home_market_iqr.svg"))
    plot_line(GRES_LOGS, "D*/G_res +/-10% test", os.path.join(out_dir, "plot2b_pct_dstar_gres_line.svg"))
    plot_ci(GRES_LOGS, "D*/G_res +/-10% test", os.path.join(out_dir, "plot2c_pct_dstar_gres_ci.svg"))
    plot_iqr(GRES_LOGS, "D*/G_res +/-10% test", os.path.join(out_dir, "plot2d_pct_dstar_gres_iqr.svg"))
    print("saved 6 SVGs to", out_dir)


if __name__ == "__main__":
    import os
    main()
