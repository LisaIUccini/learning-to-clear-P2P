"""
example_dr_density_sweep.py
===========================

Study how the Douglas--Rachford iteration count changes with graph density.
For every requested density, generate N_NETWORKS independent market networks,
solve each one, save all raw results to CSV, and plot iterations vs density.

Place this file in the same directory as:
    market_generator.py
    dr_solver.py
    plot_style.py

Run:
    python example_dr_density_sweep.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from market_generator import MarketConfig, MarketGenerator
from dr_solver import DRConfig, DRSolver
from plot_style import set_paper_style, show_or_save_many


# ============================== CONFIG ================================
N = 10
GRAPH_MODEL = "sparse"

# Eleven values from tree-like (0) to complete (1).
# Change this array if you want a finer or different sweep.
DENSITIES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

# Number of independently generated networks for each density.
N_NETWORKS = 1

GAMMA = 1.5
LAM = 1.0
TOL = 1e-6
MAX_ITER = 5000

# The same replicate seeds are reused at every density. This gives a paired,
# reproducible comparison across density values while still generating
# N_NETWORKS distinct instances at each density.
BASE_SEED = 42

# Plot against either the generator parameter or the measured edge density.
# Valid values: "requested" or "realized".
X_AXIS = "requested"

FIG_SIZE = (6.6, 4.0)
OUTPUT_DIR = Path("density_sweep_results")
OUTPUT_NAME = f"dr_iterations_vs_density_N{N}"
CSV_NAME = f"dr_density_sweep_N{N}.csv"
# =====================================================================


def build_network(density: float, seed: int):
    """Generate one market/network instance."""
    config = MarketConfig(graph_model=GRAPH_MODEL, density=float(density))
    return MarketGenerator(config).sample(N, seed=seed)


def solve_network(net):
    """Run Douglas--Rachford with the same settings used in example_dr.py."""
    config = DRConfig(
        gamma=GAMMA,
        lam=LAM,
        n_iter=MAX_ITER,
        stop_tol=TOL,
    )
    return DRSolver(config).solve(net, record=True, write_back=True)


def realized_edge_density(net) -> float:
    """Return the usual undirected graph density 2|E| / (N(N-1))."""
    if net.N < 2:
        return 0.0
    n_edges = sum(len(adj) for adj in net.neighbors.values()) / 2.0
    return float(2.0 * n_edges / (net.N * (net.N - 1)))


def run_experiment() -> list[dict]:
    """Run all density/replicate combinations and return raw records."""
    records: list[dict] = []
    total = len(DENSITIES) * N_NETWORKS
    run_index = 0

    for density in DENSITIES:
        for replicate in range(N_NETWORKS):
            run_index += 1
            seed = BASE_SEED 

            net = build_network(float(density), seed)
            result = solve_network(net)

            residual_history = np.asarray(result.history["residual"], dtype=float)
            final_residual = (
                float(residual_history[-1]) if residual_history.size else np.nan
            )

            record = {
                "requested_density": float(density),
                "realized_density": realized_edge_density(net),
                "replicate": replicate,
                "seed": seed,
                "iterations": int(result.n_iter),
                "converged": bool(result.converged),
                "final_residual": final_residual,
            }
            records.append(record)

            status = "converged" if result.converged else "MAX_ITER"
            print(
                f"[{run_index:3d}/{total}] "
                f"density={density:.2f}, replicate={replicate + 1:02d}, "
                f"seed={seed}, realized={record['realized_density']:.3f}, "
                f"iterations={result.n_iter}, {status}"
            )

    return records


def save_csv(records: list[dict], path: Path) -> None:
    """Save one row per network/solver run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "requested_density",
        "realized_density",
        "replicate",
        "seed",
        "iterations",
        "converged",
        "final_residual",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"saved {path}")


def density_summary(records: list[dict]) -> list[dict]:
    """Compute descriptive statistics for every requested density."""
    summary: list[dict] = []

    for density in DENSITIES:
        group = [
            row
            for row in records
            if np.isclose(row["requested_density"], float(density))
        ]
        iterations = np.asarray([row["iterations"] for row in group], dtype=float)
        realized = np.asarray([row["realized_density"] for row in group], dtype=float)
        converged = np.asarray([row["converged"] for row in group], dtype=bool)

        summary.append(
            {
                "requested_density": float(density),
                "mean_realized_density": float(np.mean(realized)),
                "mean_iterations": float(np.mean(iterations)),
                "std_iterations": float(np.std(iterations, ddof=1))
                if len(iterations) > 1
                else 0.0,
                "median_iterations": float(np.median(iterations)),
                "q25_iterations": float(np.percentile(iterations, 25)),
                "q75_iterations": float(np.percentile(iterations, 75)),
                "convergence_rate": float(np.mean(converged)),
            }
        )

    return summary


def print_summary(summary: list[dict]) -> None:
    """Print a compact table to the terminal."""
    print("\nSummary by requested density")
    print(
        "density  actual_mean  iterations_mean±std  median [Q1,Q3]  converged"
    )
    for row in summary:
        print(
            f"{row['requested_density']:7.2f}  "
            f"{row['mean_realized_density']:11.3f}  "
            f"{row['mean_iterations']:9.1f} ± {row['std_iterations']:7.1f}  "
            f"{row['median_iterations']:7.1f} "
            f"[{row['q25_iterations']:.1f}, {row['q75_iterations']:.1f}]  "
            f"{100.0 * row['convergence_rate']:8.1f}%"
        )


def iterations_figure(records: list[dict], summary: list[dict]):
    """Scatter all runs and overlay mean ± one standard deviation."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    x_key = "requested_density" if X_AXIS == "requested" else "realized_density"
    summary_x_key = (
        "requested_density" if X_AXIS == "requested" else "mean_realized_density"
    )

    # Small deterministic horizontal jitter makes overlapping replicates visible.
    x_values = np.asarray([row[x_key] for row in records], dtype=float)
    y_values = np.asarray([row["iterations"] for row in records], dtype=float)
    converged = np.asarray([row["converged"] for row in records], dtype=bool)
    rng = np.random.default_rng(BASE_SEED)
    spacing = np.min(np.diff(np.unique(np.sort(x_values)))) if len(np.unique(x_values)) > 1 else 0.1
    jitter = rng.uniform(-0.06 * spacing, 0.06 * spacing, size=len(records))

    ax.scatter(
        x_values[converged] + jitter[converged],
        y_values[converged],
        s=24,
        alpha=0.55,
        label="single converged runs",
        zorder=2,
    )

    if np.any(~converged):
        ax.scatter(
            x_values[~converged] + jitter[~converged],
            y_values[~converged],
            s=38,
            marker="x",
            linewidths=1.2,
            label=f"not converged (cap={MAX_ITER})",
            zorder=3,
        )

    x_summary = np.asarray([row[summary_x_key] for row in summary], dtype=float)
    mean_iterations = np.asarray(
        [row["mean_iterations"] for row in summary], dtype=float
    )
    std_iterations = np.asarray(
        [row["std_iterations"] for row in summary], dtype=float
    )

    ax.errorbar(
        x_summary,
        mean_iterations,
        yerr=std_iterations,
        marker="o",
        capsize=3,
        linewidth=1.6,
        label=r"mean $\pm$ std",
        zorder=4,
    )

    if X_AXIS == "requested":
        ax.set_xlabel("generator density parameter (0 = tree, 1 = complete)")
    else:
        ax.set_xlabel(r"realized edge density $2|E|/[N(N-1)]$")

    ax.set_ylabel("Douglas--Rachford iterations")
    ax.set_title(
        rf"Iterations vs graph density ($N={N}$, {N_NETWORKS} networks/density)"
    )
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def convergence_rate_figure(summary: list[dict]):
    """Diagnostic plot: fraction of runs that converged before MAX_ITER."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    x_key = "requested_density" if X_AXIS == "requested" else "mean_realized_density"
    x_values = np.asarray([row[x_key] for row in summary], dtype=float)
    rates = 100.0 * np.asarray(
        [row["convergence_rate"] for row in summary], dtype=float
    )

    ax.plot(x_values, rates, marker="o")
    ax.set_ylim(-2, 102)
    ax.set_ylabel("converged runs (%)")
    if X_AXIS == "requested":
        ax.set_xlabel("generator density parameter (0 = tree, 1 = complete)")
    else:
        ax.set_xlabel(r"mean realized edge density $2|E|/[N(N-1)]$")
    ax.set_title(rf"Convergence rate before $k={MAX_ITER}$")
    fig.tight_layout()
    return fig


def main() -> None:
    if GRAPH_MODEL != "sparse":
        raise ValueError("This sweep is intended for GRAPH_MODEL='sparse'.")
    if X_AXIS not in {"requested", "realized"}:
        raise ValueError("X_AXIS must be either 'requested' or 'realized'.")
    if N_NETWORKS < 1:
        raise ValueError("N_NETWORKS must be at least 1.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = run_experiment()
    summary = density_summary(records)
    save_csv(records, OUTPUT_DIR / CSV_NAME)
    print_summary(summary)

    set_paper_style()
    figures = [
        (
            iterations_figure(records, summary),
            str(OUTPUT_DIR / OUTPUT_NAME),
        ),
        (
            convergence_rate_figure(summary),
            str(OUTPUT_DIR / f"dr_convergence_rate_N{N}"),
        ),
    ]
    show_or_save_many(figures)


if __name__ == "__main__":
    main()