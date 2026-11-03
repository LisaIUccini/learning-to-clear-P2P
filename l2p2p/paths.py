"""Filesystem layout of the project. Every script resolves paths from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repository root
RESULTS = ROOT / "results"
OFFICIAL = RESULTS / "official"

DR_GNN = OFFICIAL / "dr_gnn"            # d{density}/: phi_w.pt, train.log, instance_plots/
OOD = OFFICIAL / "ood"                  # d{density}_level{20,30,40}/, d{density}_pm1/
L2WS_CENTRALIZED = OFFICIAL / "l2ws" / "centralized"   # d{density}_{loss}/
L2WS_DISTRIBUTED = OFFICIAL / "l2ws" / "distributed"   # d{density}_{loss}/
A2DR = OFFICIAL / "a2dr"                # d{density}/results.json, calibration/
PAPER_DATA = OFFICIAL / "paper_data"    # extra evaluations read by paper figures/tables

PAPER = ROOT / "paper"
PAPER_FIGURES = PAPER / "paper_figs_official"
PAPER_TABLES = PAPER / "paper_tables_official"


def density_tag(density: float) -> str:
    """`0.8 -> "d0.8"`, `1.0 -> "d1.0"` (the naming used by every results folder)."""
    return f"d{density}"
