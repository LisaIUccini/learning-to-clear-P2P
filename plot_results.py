import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
import solveVI_dynamic_DR as solveVI
from energy_network import EnergyNetwork

def plot_fig(N,D,G,q,Q, G_res, k, namefig="graph_visualization.png"):
    # # Node layout (equilateral or equidistant triangle)
    """
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
    plt.savefig(f"plot/{namefig}",
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    #plt.show()

"""
plot figure convergence as residual VI
"""
def plot_convergence(residuals, residuals_l, legend1="", legend2="", namefig="plot_convergence.png", title="Convergence of Forward-Backward"):
    plt.style.use("seaborn-v0_8-pastel")
    plt.figure(figsize=(6,4))
    plt.semilogy(residuals, marker='o', label=legend1)
    plt.semilogy(residuals_l, marker='*', label=legend2)
    plt.xlabel('Iteration')
    plt.ylabel('Residual VI ||x - Pi_X(x - γF(x))||')
    plt.legend(loc='best', fontsize=9, frameon=False)
    plt.title(title)
    plt.xticks(range(len(residuals)))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plot/{namefig}",dpi=150, bbox_inches="tight")
    #plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plot convergenza
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergenceDR(result: solveVI.DRResult, title: str = ""):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(result.residuals, color='steelblue', linewidth=2)
    ax.axhline(y=result.residuals[-1], color='tomato', linestyle='--',
               linewidth=1, label=f'finale = {result.residuals[-1]:.1e}')
    ax.set_xlabel("Iterazione $k$")
    ax.set_ylabel(r"$\|r(u^k)\|$  (scala log)")
    ax.set_title(f"Convergenza Douglas-Rachford"
                 f"{' — ' + title if title else ''}")
    ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def check_and_report_v2(net: EnergyNetwork,
                        result: solveVI.DRResult,
                        tol_feas: float = 1e-3):
    print("\n" + "=" * 58)
    print("  Risultati — Market Energy Game v2 (OSQP)")
    print("=" * 58)
    print(f"  Convergenza : {'✓' if result.converged else '✗'}  "
          f"({result.n_iter} iter,  ‖r‖ = {result.residuals[-1]:.2e})")

    violations = []
    for n in range(net.N):
        D   = net.D[n].item()
        G   = net.G[n].item()
        Q_n = sum(net.q[(m, n)].item() for m in net.neighbors.get(n, []))
        bal = D - (net.G_res[n].item() + G + Q_n)
        if abs(bal) > tol_feas:
            violations.append(f"Nodo {n}: bilanciamento = {bal:.4f}")
        if D < net.D_min[n].item() - tol_feas or D > net.D_max[n].item() + tol_feas:
            violations.append(f"Nodo {n}: D={D:.3f} fuori bounds")
        if G < net.G_min[n].item() - tol_feas or G > net.G_max[n].item() + tol_feas:
            violations.append(f"Nodo {n}: G={G:.3f} fuori bounds")
    seen = set()
    for n in range(net.N):
        for m in net.neighbors.get(n, []):
            e = (min(m, n), max(m, n))
            if e in seen: continue
            seen.add(e)
            rec = net.q[(m, n)].item() + net.q[(n, m)].item()
            if rec > tol_feas:
                violations.append(f"Arco ({m},{n}): reciprocità = {rec:.4f}")

    if violations:
        print(f"\n  ⚠  {len(violations)} violazioni:")
        for v in violations: print(f"     {v}")
    else:
        print(f"\n  ✓  Tutti i vincoli soddisfatti")

    print(f"\n{'─'*58}")
    print(f"  {'n':>3}  {'D*':>5}  {'D':>6}  {'G':>6}  {'ΔG':>5}  {'Q_n':>7}  {'Π_n':>7}")
    print(f"{'─'*58}")
    SW = 0.0
    for n in range(net.N):
        D   = net.D[n].item(); G = net.G[n].item()
        Q_n = sum(net.q[(m, n)].item() for m in net.neighbors.get(n, []))
        U_n = -net.a_tilde[n].item() * (D - net.D_star[n].item()) ** 2 + 1.0
        C_n = 0.5 * net.a[n].item() * G ** 2 + net.b[n].item() * G + 1.0
        Ct  = sum(net.c[(n, m)] * net.q[(m, n)].item() for m in net.neighbors.get(n, []))
        Pi  = U_n - C_n - Ct
        SW += Pi
        print(f"  {n:>3}  {net.D_star[n].item():>5.2f}  {D:>6.3f}  {G:>6.3f}  "
              f"{net.G_res[n].item():>5.2f}  {Q_n:>7.3f}  {Pi:>7.4f}")
    print(f"{'─'*58}")
    #print(f"  Social Welfare = {SW:.4f}")
    return violations, SW




