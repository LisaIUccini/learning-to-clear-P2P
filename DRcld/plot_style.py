"""
plot_style.py
==============

Shared look for every figure this project produces (`demo.py`'s
`plot_results`/`plot_training_loss`, `diagnostics.py`'s `plot_instance`):
seaborn's "pastel" categorical palette, thin recessive gridlines, no
top/right spines, and a fixed color identity used everywhere a series
means the same thing --

    C_DR       plain DR              (always this color, every plot)
    C_LEARNED  DR + learned          (always this color, every plot)
    C_WIN      per-instance win  (speed-up >= 1x)
    C_LOSS     per-instance loss (speed-up <  1x)
    C_TRAIN    training-loss curve
    C_VAL      validation-loss curve

Import this before creating any figure -- it configures rcParams as a
side effect of import.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
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
    """Pad shorter curves with their own final value (converged = stays
    small) so every run contributes at every iteration index, then return
    the per-iteration median and interquartile band."""
    max_len = max(len(c) for c in curves)
    padded = np.stack([np.concatenate([c, np.full(max_len - len(c), c[-1])]) for c in curves])
    return np.median(padded, axis=0), np.percentile(padded, 25, axis=0), np.percentile(padded, 75, axis=0)


def stats_from(test_res: dict) -> dict:
    sp = np.asarray(test_res["speedup"])
    dr = np.asarray(test_res["dr"]); learned = np.asarray(test_res["learned"])
    pct = (dr - learned) / dr * 100.0     # per-instance iteration reduction, in %
    return {
        "mean": sp.mean(), "median": np.median(sp), "min": sp.min(), "max": sp.max(),
        "pct_mean": pct.mean(), "pct_median": np.median(pct), "pct_min": pct.min(), "pct_max": pct.max(),
        "win_rate": (sp >= 1.0).mean(), "n_win": int((sp >= 1.0).sum()), "n": len(sp),
    }
