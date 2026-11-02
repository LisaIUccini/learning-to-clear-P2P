# Paper figures and tables

Updated 2026-11-02. Outputs used by the paper: `paper_figs_official/` and
`paper_tables_official/`. `python3 paper/make_all.py` regenerates all of them
from `results/official/` (nothing is recomputed); `tests/test_paper.py` checks
they come out identical.

**Status:** ✅ ok · ⚠️ to review

## Figures (`paper_figs_official/`)

| File | Script | Reads | Status |
|---|---|---|---|
| `fig1_methods_comparison` | `fig1_methods_comparison.py` | dr_gnn, l2ws, a2dr | ✅ |
| `fig1_methods_comparison_3dens` | `fig1_methods_comparison_3dens.py` | same | ✅ |
| `fig1_methods_comparison_noreg`, `..._3dens_noreg` | `fig1_noreg.py` | same, without L2WS-reg | ✅ |
| `fig2b_stopping_tolerance` | `fig2b_stopping_tolerance.py` | dr_gnn residuals | ✅ (perturbed stopping rule) |
| `fig2_gap_levels` | `fig2_gap_levels.py` | dr_gnn, l2ws/centralized, paper_data/fig2_data | ⚠️ first-hit |
| `fig2_gap_levels_2methods` | `fig2_gap_levels_2methods.py` | dr_gnn | ⚠️ first-hit |
| `fig3_difficulty_gain` | `fig3_difficulty_gain.py` | dr_gnn | ✅ |
| `fig4_ood_robustness`, `..._3dens` | `fig4_ood_robustness*.py` | dr_gnn, ood | ✅ |

## Tables (`paper_tables_official/`)

| File | Script | Reads | Status |
|---|---|---|---|
| `tabA_comparison.tex` | `tabA_comparison.py` | dr_gnn, l2ws, a2dr | ✅ |
| `tabC_ood.tex` | `tabC_ood.py` | dr_gnn, ood | ✅ |
| `tabD_market_plausibility.tex` | `tabD_market_plausibility.py` | paper_data/table_d_data | ✅ |
| `tabE_hyperparameters.tex`, `tabE2_...` | `tabE_hyperparameters.py` | dr_gnn logs, a2dr | ✅ |
| `tabF_stopping_quality.tex` (Table 5) | `tabF_stopping_quality.py` | dr_gnn, paper_data/table_ek_data | ⚠️ e^k column |
| `tabT4_timing.tex` (Table 2) | `tabT4_timing.py` | paper_data/table_t4_data | ✅ |

`results/official/paper_data/` is produced by `python3 -m scripts.paper_data
{trajectories, plausibility, timing, train-time, stopping-quality}`.

## Open issues (they change numbers of the paper: decide before regenerating)

1. **e^k column of Table 5.** The saved `table_ek_data` came from a script with a
   copied DR loop that rebuilt the GNN with `scale=1.0` (the checkpoints of
   densities 0.4/0.8/1.0 use 1.8) and did not update the `prev_residual` /
   `stall_ema` features; even its plain-DR values do not match the official
   runs. `scripts.paper_data stopping-quality` computes it on the real code:
   at density 0.8, 4.3e-4 instead of 9.7e-6. Not regenerated yet.
2. **Timing data of Table 2.** Measured by a copy of the same loop (same
   `scale=1.0` issue). Per-call times hardly depend on it; `scripts.paper_data
   timing`/`train-time` now time the real code. Not regenerated yet.
3. **Gap figures (`fig2_gap_levels*`).** They use the first iteration with
   |gap| <= eps; the gap crosses zero (the shadow iterate is not
   balance-feasible), so |gap| dips transiently. A "stays below eps until the
   stop" criterion is more meaningful.
4. **Stopping rule.** With the unperturbed rule the median reduction drops (density
   0.8: 75.4% -> 58.8%, 1.0: 45.8% -> 29.0%) and the accuracy at termination returns
   to DR's level. Figures 1/2b and Tables A/F use the perturbed (official) rule.
   See `results/reeval_unperturbed_stop/20261028_2006/REPORT.md`.
