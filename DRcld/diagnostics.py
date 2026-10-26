"""
diagnostics.py
==============

Per-instance convergence diagnostics: for a single market instance, plot
plain DR against DR + learned perturbation on four panels --

    1. residual (auxiliary iterate x^k)  vs. iteration k   (log y)
    2. residual (shadow/feasible iterate xbar^k) vs. iteration k  (log y)
       -- x^k need not be a feasible market allocation at all; xbar^k =
       proj_X^rec(x^k) is, and it is xbar^k that is proven to converge to
       the true v-GNE. The two can diverge in behaviour (e.g. x^k can sit
       on a long shared plateau while xbar^k is still moving, or vice
       versa), so seeing them side by side is diagnostic in its own right.
    3. RELATIVE optimality gap |J(xbar^k) - J*| / |J*| vs. iteration k
       (log y; J* from `equilibrium.solve_equilibrium_qp`, the same for
       every v-GNE since the game is a potential game). Normalised by
       |J*| (2026-10-14, was the raw absolute gap before) so the gap is
       comparable across instances/densities whose J* differs by orders
       of magnitude -- an absolute gap of 0.01 means very different
       things at J*=0.1 vs J*=100.
    4. constraint violation dist(xbar^k, X_loc) vs. iteration k (log y;
       `Network.dist_to_Xloc`, added 2026-10-14) -- xbar^k is always
       X^rec-feasible by construction (balance + reciprocity), but not
       necessarily inside the LOCAL box constraints yet; this tracks how
       far it still is, the other half of what "xbar^k is a true v-GNE"
       requires.

-- and save one 4-panel figure per instance, so results can be inspected
instance by instance instead of only through aggregate statistics.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import C_DR, C_LEARNED, INK_MUTED, clean_axes


def plot_instance(res_dr, res_learned, j_star: float, tol: float, title: str,
                   out_path: str) -> None:
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


def save_instance_plots(specs, model, dr_cfg, device, seed, out_dir: str,
                         run_pair_fn) -> None:
    """`run_pair_fn(spec, model, dr_cfg, device, seed, record=True)` must
    return `(res_dr, res_learned)` (see demo.py's `run_pair`). Saves, per
    instance, into `out_dir` (created if needed):
      - one 4-panel PNG (`plot_instance`)
      - one JSON with the RAW numbers behind every panel (user-requested
        2026-10-14: "tutto nel file log cosi posso analizzare e creare
        altri plot se necessito" -- a rendered PNG alone can't be
        reprocessed; the JSON carries k_DR/k_learned and the full
        per-iteration residual/shadow-residual/relative-gap/violation
        trajectories for both plain DR and DR+learned, on the shadow
        iterate xbar^k where the module docstring says so)."""
    import json as json_mod
    from equilibrium import solve_equilibrium_qp
    import market as market_mod

    os.makedirs(out_dir, exist_ok=True)
    for i, spec in enumerate(specs):
        res_dr, res_learned = run_pair_fn(spec, model, dr_cfg, device,
                                           seed=seed * 7919 + i, record=True)
        net = market_mod.build_network(spec, device=device)
        j_star = solve_equilibrium_qp(net)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        title = (f"instance {i + 1}  N={spec.N}  DR {res_dr.n_iter} it | learned {res_learned.n_iter} it "
                  f"| x{sp:.2f} ({pct:+.1f}%)")
        base = os.path.join(out_dir, f"instance_{i + 1:02d}_N{spec.N}")
        plot_instance(res_dr, res_learned, j_star, dr_cfg.tol, title, base + ".png")

        j_denom = max(abs(j_star), 1e-9)
        def _rel_gap(res):
            return (np.abs(np.asarray(res.cost) - j_star) / j_denom).tolist()
        data = {
            "instance": i + 1, "N": spec.N, "j_star": j_star, "tol": dr_cfg.tol,
            "k_DR": res_dr.n_iter, "k_learned": res_learned.n_iter,
            "speedup": sp, "pct_reduction": pct,
            "dr": {"residual": res_dr.residual, "shadow_residual": res_dr.shadow_residual,
                   "cost": res_dr.cost, "relative_gap": _rel_gap(res_dr),
                   "violation": res_dr.violation},
            "learned": {"residual": res_learned.residual, "shadow_residual": res_learned.shadow_residual,
                        "cost": res_learned.cost, "relative_gap": _rel_gap(res_learned),
                        "violation": res_learned.violation},
        }
        with open(base + ".json", "w") as f:
            json_mod.dump(data, f)
    print(f"saved {len(specs)} per-instance diagnostic plots + raw-trajectory JSON files to {out_dir}/")
