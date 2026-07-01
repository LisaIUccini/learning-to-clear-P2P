"""
plot_style.py
=============

Reusable, paper-quality matplotlib styling (IEEE-conference look:
serif fonts, mathtext, restrained grid, tight vector export).

Usage
-----
    from plot_style import set_paper_style, save_figure
    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ...
    save_figure(fig, "residual")     # writes residual.pdf and residual.png
"""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# A calm, colour-blind-friendly palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#000000"]


def set_paper_style(base_fontsize: int = 9) -> None:
    """Apply a consistent paper-ready rcParams profile (no external LaTeX)."""
    mpl.rcParams.update({
        # fonts
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize - 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        # lines / markers
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        # axes / grid
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "axes.axisbelow": True,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.6,
        # export
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # editable text in the PDF
        "ps.fonttype": 42,
    })


def save_figure(fig, name: str, formats=("pdf", "png")) -> None:
    """Save `fig` under `name` in several formats (vector first)."""
    for fmt in formats:
        fig.savefig(f"{name}.{fmt}")


# single- and double-column widths (inches) for convenience
COL_WIDTH = 3.4
DOUBLE_WIDTH = 7.0