import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

from energy_network import EnergyNetwork


def _save_or_show(fig, save_path: str = None):
    """Save figure to disk or show interactively, then close."""
    if save_path is not None:
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname if dirname else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_loss(loss_history: dict, save_path: str = None):
    """
    Plot residuals for pure vs learned algorithm.

    Parameters
    ----------
    results   : output of run_comparison (or averaged results dict)
    save_path : if given, save figure here; otherwise show interactively
    """
    plt.style.use("seaborn-v0_8-pastel")
    fig, ax = plt.subplots(figsize=(12, 5))

    iters = range(len(loss_history))

    ax.semilogy(iters, loss_history, marker="o", label="loss history")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.set_title("Loss History Training")
    ax.legend()

    plt.xticks(range(len(loss_history)))
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_results(results: dict, save_path: str = None):
    """
    Plot residuals for pure vs learned algorithm.

    Parameters
    ----------
    results   : output of run_comparison (or averaged results dict)
    save_path : if given, save figure here; otherwise show interactively
    """
    plt.style.use("seaborn-v0_8-pastel")
    fig, ax = plt.subplots(figsize=(12, 5))

    res_pure = results["residuals_pure"]
    iters = range(len(res_pure))

    ax.semilogy(iters, res_pure,                     marker="o", label="Pure")
    ax.semilogy(iters, results["residuals_learned"],  marker="*", label="Learned + GNN")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals")
    ax.legend()

    plt.xticks(range(len(res_pure)))
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_results_std(results: dict, std: dict = None, save_path: str = None):
    """
    Plot residuals with optional shaded standard deviation band.

    Parameters
    ----------
    results   : dict with mean residual sequences
    std       : dict with std sequences (same keys), or None
    save_path : if given, save figure here; otherwise show interactively
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    iters = np.arange(1, len(results["residuals_pure"]) + 1)

    def _plot(key, color, label):
        mean = np.array(results[key])
        ax.semilogy(iters, mean, label=label, color=color)
        if std is not None:
            s  = np.array(std[key])
            lo = np.clip(mean - s, 1e-12, None)
            hi = mean + s
            ax.fill_between(iters, lo, hi, alpha=0.2, color=color)

    _plot("residuals_pure",    "steelblue",  "Pure")
    _plot("residuals_learned", "darkorange", "Learned + GNN")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|r^k\|^2$")
    ax.set_title("Residuals")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_fig(net: EnergyNetwork, save_path: str = "graph_visualization.png"):
    """
    Visualize the energy network state.
 
    Draws directed trade flows q_{mn} on each edge, color-coded by saturation,
    and displays D, G, Q, G_res for each prosumer.
 
    Parameters
    ----------
    net       : EnergyNetwork — current state to visualize
    save_path : path to save the figure (default: "graph_visualization.png")
    """
    N = net.N
    prosumers = net.prosumers

    # Circular layout
    """
    pos = {
        i: np.array([np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)])
        for i in range(N)
    }
    """
    pos = {
        0: np.array([0.0,  0.866]),
        1: np.array([-1.0, 0.0 ]),
        2: np.array([ 1.0, 0.0 ]),
    }

    # Extract node-level quantities
    D     = {n: float(prosumers[n].D)     for n in range(N)}
    G     = {n: float(prosumers[n].G)     for n in range(N)}
    G_res = {n: float(prosumers[n].G_res) for n in range(N)}
    Q     = {n: sum(float(prosumers[n].z[m]) for m in prosumers[n].neighbors)
             for n in range(N)}

    # Extract edge-level quantities
    q = {
        (m, n): float(prosumers[n].z[m])
        for (m, n) in net.edge_list
    }
    k = {
        (m, n): float(prosumers[n].k[m])
        for (m, n) in net.edge_list
    }

    # ------------------------------------------------------------------ #
    # Figure setup
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.8, 1.8)

    cmap_pos    = "#2a702f"   # green — flow well below capacity
    cmap_neg    = "#f87171"   # red   — flow at/near capacity
    node_radius = 0.14

    # ------------------------------------------------------------------ #
    # Edges: arcs with arrows and q flow labels
    # ------------------------------------------------------------------ #
    drawn_edges = set()
    for (m, n) in net.edge_list:
        if (n, m) in drawn_edges:
            continue
        drawn_edges.add((m, n))

        p0, p1  = pos[m], pos[n]
        d_unit  = (p1 - p0) / np.linalg.norm(p1 - p0)
        perp    = np.array([-d_unit[1], d_unit[0]]) * 0.06
        midpoint = (p0 + p1) / 2

        for sign, (src, dst) in [(1, (m, n)), (-1, (n, m))]:
            val_show = q[(src, dst)]
            color    = cmap_pos if val_show < (k[(src, dst)] - 0.2) else cmap_neg
            off      = sign * perp

            d       = pos[dst] - pos[src]
            d       = d / np.linalg.norm(d)
            p_start = pos[src] + d * node_radius + off
            p_end   = pos[dst] - d * node_radius + off

            ax.annotate(
                "",
                xy=p_end, xytext=p_start,
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.8,
                    mutation_scale=18,
                    connectionstyle="arc3,rad=0.0",
                ),
            )

            mid = midpoint + off * 3
            ax.text(
                mid[0], mid[1],
                f"q[({src},{dst})]={val_show:+.2f}",
                ha="center", va="center",
                fontsize=8, color=color, fontfamily="monospace",
                path_effects=[pe.withStroke(linewidth=2, foreground="#ffffff")],
            )

    # ------------------------------------------------------------------ #
    # Nodes: circles with prosumer info
    # ------------------------------------------------------------------ #
    for n, p in pos.items():
        circle = plt.Circle(p, node_radius, color="#fafafa", zorder=3,
                            linewidth=1, ec="#5e5e92")
        ax.add_patch(circle)

        ax.text(p[0], p[1] + 0.02, f"{n}",
                ha="center", va="center",
                fontsize=11, color="#5e5e92",
                fontfamily="monospace", zorder=4)

        offset_y = -0.30 if n != 0 else 0.30
        ax.text(
            p[0], p[1] + offset_y,
            (f"D{n}={D[n]:.2f}  G{n}={G[n]:.2f}  "
             f"Q{n}={Q[n]:.2f}  \u0394G{n}={G_res[n]:.2f}\n"
             f"  {D[n]:.2f} = {G[n]:.2f} + {G_res[n]:.2f} + {Q[n]:.2f}"),
            ha="center", va="center",
            fontsize=8, color="#5e5e92", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff",
                      ec="#5e5e92", lw=1),
            zorder=5,
        )

    # ------------------------------------------------------------------ #
    # Legend
    # ------------------------------------------------------------------ #
    legend_elements = [
        mpatches.Patch(color=cmap_pos, label="q[(m,n)] < k[(m,n)]"),
        mpatches.Patch(color=cmap_neg, label="q[(m,n)] = k[(m,n)]"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=2,
              facecolor="#ffffff", edgecolor="#5e5e92", labelcolor="#5e5e92",
              fontsize=8, framealpha=0.9)

    plt.tight_layout()
    _save_or_show(fig, save_path)