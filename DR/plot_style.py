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


def _backend_is_interactive() -> bool:
    """True if the active matplotlib backend can pop up a window.

    In a headless session (e.g. SSH without X-forwarding) matplotlib falls
    back to 'agg', which is non-interactive -> we detect that and skip show().
    """
    import matplotlib
    non_interactive = {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}
    return matplotlib.get_backend().lower() not in non_interactive


def show_or_save(fig, name: str, formats=("pdf", "png")) -> None:
    """Show the figure if a display is available, then save it.

    - interactive backend (display present): plt.show() blocks until you
      close the window, then the figure is saved;
    - headless (no display / SSH without X): showing is skipped and the
      figure is saved directly. No crash in either case.
    """
    import matplotlib.pyplot as plt
    if _backend_is_interactive():
        try:
            plt.show()
        except Exception as exc:                     # pragma: no cover
            print(f"(could not open a window: {exc})")
    else:
        print("(no interactive display detected -> saving without showing)")
    save_figure(fig, name, formats)
    print(f"saved {name}." + " and ".join(formats).replace(" and ", ", "))


def show_or_save_many(pairs, formats=("pdf", "png")) -> None:
    """Same as `show_or_save` but for several figures at once.

    `pairs` is a list of (figure, name). A single plt.show() pops up all
    open windows (blocks until you close them), then every figure is saved.
    Headless-safe: showing is skipped when no display is available.
    """
    import matplotlib.pyplot as plt
    if _backend_is_interactive():
        try:
            plt.show()
        except Exception as exc:                     # pragma: no cover
            print(f"(could not open windows: {exc})")
    else:
        print("(no interactive display detected -> saving without showing)")
    for fig, name in pairs:
        save_figure(fig, name, formats)
        print(f"saved {name}.pdf, png")


# single- and double-column widths (inches) for convenience
COL_WIDTH = 3.4
DOUBLE_WIDTH = 7.0


# ----------------------------------------------------------------------
# Lightweight graph drawing (no networkx dependency)
# ----------------------------------------------------------------------
def _undirected_edges(neighbors):
    return [(n, m) for n in neighbors for m in neighbors[n] if m > n]


def spring_layout(neighbors, N, iterations: int = 300, seed: int = 0):
    """Fruchterman-Reingold force-directed layout, pure numpy.

    Returns an (N, 2) array of node positions normalised to [0, 1]^2.
    Fine for small/medium graphs (O(N^2) per iteration).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.3, size=(N, 2))
    edges = _undirected_edges(neighbors)
    k = 1.0 / np.sqrt(max(N, 1))            # ideal edge length
    for it in range(iterations):
        disp = np.zeros((N, 2))
        # repulsion between all node pairs
        for i in range(N):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = 1.0
            disp[i] += ((k * k / dist**2)[:, None] * delta).sum(0)
        # attraction along edges
        for (u, v) in edges:
            delta = pos[u] - pos[v]
            dist = max(np.linalg.norm(delta), 1e-3)
            f = (dist / k) * delta / dist
            disp[u] -= f
            disp[v] += f
        # cooling
        t = 0.1 * (1.0 - it / max(iterations, 1))
        length = np.linalg.norm(disp, axis=1)
        length[length < 1e-6] = 1.0
        pos += (disp.T / length * np.minimum(length, t)).T
    span = pos.max(0) - pos.min(0)
    span[span < 1e-9] = 1.0
    return (pos - pos.min(0)) / span


def draw_network(ax, neighbors, N, pos=None, seed: int = 0,
                 node_size: float = 60, node_color: str = "#0072B2",
                 edge_color: str = "0.6", labels: bool = False):
    """Draw a graph on `ax` (nodes + undirected edges), no axes/frame."""
    if pos is None:
        pos = spring_layout(neighbors, N, seed=seed)
    for (u, v) in _undirected_edges(neighbors):
        ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                color=edge_color, lw=0.7, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, color=node_color,
               edgecolors="white", linewidths=0.6, zorder=2)
    if labels:
        for i in range(N):
            ax.annotate(str(i), (pos[i, 0], pos[i, 1]), fontsize=6,
                        ha="center", va="center", color="white", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    return pos