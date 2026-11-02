"""Regenerate every paper figure and table from the saved results.

    python3 paper/make_all.py

Figures go to paper_figs_official/, tables to paper_tables_official/
(or to $L2P2P_PAPER_OUT/figures and /tables).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "fig1_methods_comparison.py", "fig1_methods_comparison_3dens.py", "fig1_noreg.py",
    "fig2b_stopping_tolerance.py", "fig2_gap_levels.py", "fig2_gap_levels_2methods.py",
    "fig3_difficulty_gain.py", "fig4_ood_robustness.py", "fig4_ood_robustness_3dens.py",
    "tabA_comparison.py", "tabC_ood.py", "tabD_market_plausibility.py", "tabE_hyperparameters.py",
    "tabF_stopping_quality.py", "tabT4_timing.py",
]


def main():
    for script in SCRIPTS:
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True, cwd=HERE)


if __name__ == "__main__":
    main()
