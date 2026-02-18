import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx

def plot_fig(N,D,G,q,Q, G_res):
    # ── Layout nodi (triangolo equilatero) ─────────────────────────────────────────
    
    pos = {
        0: np.array([0.0,  0.866]),
        1: np.array([-1.0, 0.0 ]),
        2: np.array([ 1.0, 0.0 ]),
    }
    """
    pos = {
    i: np.array([np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N)])
    for i in range(N)
    }
    """
    # ── Figura ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.7, 1.7)

    # ── Archi con frecce e label del flusso q ─────────────────────────────────────
    cmap_pos = "#2a702f"   # blu  → flusso positivo
    cmap_neg = "#f87171"   # rosso → flusso negativo
    node_radius = 0.14

    drawn_edges = set()
    for (m, n), val in q.items():
        if (n, m) in drawn_edges:
            continue
        drawn_edges.add((m, n))

        p0, p1 = pos[m], pos[n]

        # versore direzione m→n e perpendicolare (calcolati una volta per coppia)
        d_unit = (p1 - p0) / np.linalg.norm(p1 - p0)
        perp = np.array([-d_unit[1], d_unit[0]]) * 0.06

        for sign, (src, dst) in [(1, (m, n)), (-1, (n, m))]:
            val_show = q[(src, dst)]
            color = cmap_pos if val_show >= 0 else cmap_neg
            off = sign * perp  # le due frecce stanno su lati opposti

            # direzione src→dst (può essere m→n o n→m)
            d = pos[dst] - pos[src]
            d = d / np.linalg.norm(d)

            # accorcia partenza e arrivo del raggio del nodo
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
            mid = (pos[src] + pos[dst]) / 2 + off * 3
            ax.text(
                mid[0], mid[1],
                f"q[({src},{dst})]={val_show:+.2f}",
                ha="center", va="center",
                fontsize=8, color=color, fontfamily="monospace",
                path_effects=[pe.withStroke(linewidth=2, foreground="#ffffff")],
            )

    # ── Nodi ──────────────────────────────────────────────────────────────────────
    
    for n, p in pos.items():
        circle = plt.Circle(p, node_radius, color="#fafafa", zorder=3, linewidth=1,
                            ec="#5e5e92")
        ax.add_patch(circle)

        ax.text(p[0], p[1] + 0.02, f"{n}", ha="center", va="center",
                fontsize=11, color="#5e5e92",
                fontfamily="monospace", zorder=4)

        # badge D e G sotto/sopra il nodo
        offset_y = -0.30 if n != 0 else 0.30
        ax.text(p[0], p[1] + offset_y,
                f"D{n}={D[n]:.2f}  G{n}={G[n]:.2f}  Q{n}={Q[n]:.2f}  \u0394G{n}={G_res[n]:.2f} \n  {D[n]:.2f} = {G[n]:.2f} + {G_res[n]:.2f} + {Q[n]:.2f}",
                ha="center", va="center",
                fontsize=8, color="#5e5e92", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#5e5e92", lw=1),
                zorder=5)

    # ── Legenda ───────────────────────────────────────────────────────────────────
    
    legend_elements = [
        mpatches.Patch(color=cmap_pos, label="q[(m,n)] > 0  n buys from m"),
        mpatches.Patch(color=cmap_neg, label="q[(m,n)] < 0  n sells to m"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=2,
            facecolor="#ffffff", edgecolor="#5e5e92", labelcolor="#5e5e92",
            fontsize=8, framealpha=0.9)
    
    #ax.set_title("Grafo dei flussi  —  D, G per nodo", color="white",fontsize=14, fontfamily="monospace", pad=14)

    plt.tight_layout()
    plt.savefig("graph_visualization.png",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

def plot_convergence(residuals):
    plt.figure(figsize=(6,4))
    plt.semilogy(residuals, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Residual VI ||x - Pi_X(x - γF(x))||')
    plt.title('Convergence Forward-Backward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_convergence.png",dpi=150, bbox_inches="tight")


