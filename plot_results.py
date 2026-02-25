import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

def plot_fig(N,D,G,q,Q, G_res, k, namefig="graph_visualization.png"):
    # # Node layout (equilateral or equidistant triangle)
    
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
    
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.7, 1.7)

    # Arcs with arrows and q flow labels
    cmap_pos = "#2a702f"   
    cmap_neg = "#f87171"   # red -> flow equal to maximum capacity
    node_radius = 0.14

    drawn_edges = set()
    for (m, n), val in q.items():
        if (n, m) in drawn_edges:
            continue
        drawn_edges.add((m, n))

        p0, p1 = pos[m], pos[n]
        d_unit = (p1 - p0) / np.linalg.norm(p1 - p0)
        perp = np.array([-d_unit[1], d_unit[0]]) * 0.06

        for sign, (src, dst) in [(1, (m, n)), (-1, (n, m))]:
            val_show = q[(src, dst)]
            color = cmap_pos if val_show < (k[(src,dst)]-0.2) else cmap_neg
            off = sign * perp 

            d = pos[dst] - pos[src]
            d = d / np.linalg.norm(d)

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

    # plot prosumers and related information    
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

    # Legend    
    legend_elements = [
        mpatches.Patch(color=cmap_pos, label="q[(m,n)] < k[(m,n)]"),
        mpatches.Patch(color=cmap_neg, label="q[(m,n)] = k[(m,n)] "),
    ]
    ax.legend(handles=legend_elements, loc="lower center", ncol=2,
            facecolor="#ffffff", edgecolor="#5e5e92", labelcolor="#5e5e92",
            fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(namefig,
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    #plt.show()

"""
plot figure convergence as residual VI
"""
def plot_convergence(residuals, residuals_l, namefig="plot_convergence.png", title="Convergence of Forward-Backward"):
    plt.style.use("seaborn-v0_8-pastel")
    plt.figure(figsize=(6,4))
    plt.semilogy(residuals, marker='o')
    plt.semilogy(residuals_l, marker='*')
    plt.xlabel('Iteration')
    plt.ylabel('Residual VI ||x - Pi_X(x - γF(x))||')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(namefig,dpi=150, bbox_inches="tight")
    #plt.show()


