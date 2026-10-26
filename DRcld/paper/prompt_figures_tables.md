# Task: generate all figures and tables for Section 5 (Numerical Results) and the appendix

You are working in the repository of the paper "Learning to Clear Peer-to-peer Electricity Markets" (L4DC submission). All experiments have finished. Your job is to write **one reproducible script** that reads the saved results and produces every figure and LaTeX table listed below, then verify the outputs.

**Do not rerun any training or experiment.** Only read saved results. If something needed is missing, do not invent numbers: produce a clearly marked placeholder, list the missing item in the final report, and ask me before running anything new. The only exception is the timing measurement in T4, which is cheap; ask before running it anyway.

---

## 0. Before writing code

1. **Explore the results directory** and list what is available: per-instance iteration counts (e.g. `data.json` / `data_with05.json` with fields `dr`, `learned`, `speedup`, `pct`), residual and gap trajectories (`.npz`), grid-search logs for γ, sanity-check outputs, configs, logs with training times. Report the structure you found before plotting anything.
2. **Locate the paper's LaTeX source.** Find the document class/template, the font package (e.g. `times`, `mathptmx`, Computer Modern), the base font size, and `\textwidth` (compile a tiny test with `\showthe\textwidth` if needed). Figures must match these exactly.
3. Write a short mapping table (in the final report) from **method key in the data → display name in the paper**. Use the names below unless the data forces otherwise.

## 1. Global conventions (apply to every output)

### Methods, names, colors, markers
Use the Okabe–Ito colorblind-safe palette. Filled markers = centralized, hollow markers = distributed. Keep the mapping identical across all figures.

| Key | Display name | Color | Type |
|---|---|---|---|
| plain DR | DR | `#000000` (black / gray) | baseline |
| DR+GNN | Learned (ours) | `#0072B2` blue | distributed |
| L2WS-reg (dist.) | L2WS-reg (dist.) | `#E69F00` orange, hollow | distributed |
| L2WS-asym (dist.) | L2WS-asym (dist.) | `#009E73` green, hollow | distributed |
| L2WS-reg (cent.) | L2WS-reg (cent.) | `#E69F00` orange, filled | centralized |
| L2WS-asym (cent.) | L2WS-asym (cent.) | `#009E73` green, filled | centralized |
| A2DR | A2DR | `#CC79A7` pink, filled | centralized |
| L2WS-fpres (cent./dist.) | L2WS-fp-residual | `#D55E00` vermillion | appendix only |

Put the display names, colors and markers in a single dictionary at the top of the script.

### Counting rules (critical: check them in the code that produced the data)
- Iterations = communication rounds.
- **L2WS (distributed)**: iteration count = DR iterations after the warm start **+ L** message-passing rounds. Verify whether the saved counts already include L; if not, add L and state it in the report.
- **A2DR**: count **every application of the DR operator T_DR**, including the evaluations used by the safeguard, not only outer iterations. Verify in the code which one was saved.
- **Centralized L2WS**: DR iterations after the warm start (note in the caption that it requires a coordinator).
- Instances that hit the cap (4000 iterations) are counted as 4000 and **reported** (number per setting).

### Statistics
- Relative reduction per instance: `Delta_i = (k_DR_i - k_i) / k_DR_i`, in percent.
- Report **median and IQR (25th–75th percentile)**; mean only in the full appendix table.
- Win-rate = fraction of instances with `Delta_i > 0` (strict).
- Worst case = min over instances of `Delta_i`.
- Spearman correlation via `scipy.stats.spearmanr`, with p-value.
- Never round before computing statistics; round only when printing (1 decimal for %, integers for iterations).

### Figure style
- Matplotlib with `text.usetex=True` (or the pgf backend) and the **same font family and size as the paper** (axis labels = base font size of the paper, tick labels 1–2 pt smaller).
- Width: `\textwidth` for multi-panel figures, `0.48\textwidth` for single panels. Set the figure size in inches from the measured `\textwidth`.
- Save **vector PDF** (`.pdf`) for the paper and a PNG preview (`.png`, 200 dpi) for checking.
- Notation: δ as `$\delta$`, reduction axis label "iteration reduction vs. DR (\%)", iteration axis label "iteration $k$ (= communication rounds)".
- Dashed horizontal line at 0 in every reduction plot.
- No titles inside figures (captions go in LaTeX); panel labels "(a) $\delta = 0.2$" etc. below panels, as in the current figures.
- Light grid, no top/right spines, consistent y-limits across panels of the same figure.

### Table style
- LaTeX `booktabs`, `\small`, no vertical rules.
- Cells: `median [Q1, Q3]` for reductions, plus win-rate where indicated.
- **Bold** the best value per column **among distributed methods**; mark centralized references with a dagger `$^\dagger$` and separate them with `\midrule` and a row label "Centralized references".
- Each table is written to its own `.tex` file containing only the `tabular` (the `table` environment and caption are in the paper), plus a commented draft caption at the top of the file.

### Output layout
```
paper_figs/      figA_convergence.pdf, figB_difficulty.pdf, figC_ood.pdf, ... (+ .png previews)
paper_tables/    tabA_comparison.tex, tab_gap_levels.tex, ...
paper_numbers.json   every number that appears in any figure/table, keyed by output and setting
make_paper_figures.py  single entry point: `python make_paper_figures.py --results <dir> --out <dir>`
```
The script must be deterministic and rerunnable; no hard-coded numbers.

---

## 2. Main-text outputs

### T-A. Comparison table (Section 5.2) — `tabA_comparison.tex`
- Rows: methods, in two blocks.
  - *Distributed*: Learned (ours), L2WS-reg (dist.), L2WS-asym (dist.).
  - *Centralized references*: L2WS-reg (cent.), L2WS-asym (cent.), A2DR.
- Columns: densities δ ∈ {0.2, 0.4, 0.5, 0.6, 0.8, 1.0}.
- Cell: median reduction % [IQR], and win-rate on a second line or in a separate small column.
- Add one column "extra rounds" (0 for Learned, L for L2WS dist., "global" for A2DR / centralized L2WS).
- In-distribution test sets only.
- Also print the DR median iteration count per density in a first row ("DR iterations").

### F-A. Convergence curves (Section 5.2) — `figA_convergence.pdf`
- Two panels side by side: one sparse density (δ = 0.2) and one dense density (δ = 0.8); if another pair is more representative, propose it in the report.
- y-axis: **relative optimality gap** `|J(x̄^k) − J*| / |J(x̄^0) − J*|` (log scale), computed on the shadow iterates with J* from the centralized QP, **without the constants d_n and b̃_n**. If gap trajectories are not available, fall back to the fixed-point residual `||x^{k+1} − x^k||` and say so in the report and in the axis label.
- Curves: median over test instances with IQR band, for DR, Learned (ours), and L2WS-reg (dist.). For L2WS, shift the curve right by L rounds.
- Vertical dashed line at k = K (end of the training horizon), labeled.
- Horizontal dotted lines at gap levels 1e-2, 1e-3, 1e-4.

### F-M. Iterations vs. accuracy, in the style of Moret et al. (2018, Fig. 1) — `figM_gap_levels.pdf`
- Reference: Moret, Baroche, Sorin, Pinson, "Negotiation algorithms for peer-to-peer electricity markets: computational properties", PSCC 2018, Fig. 1 (number of iterations required to reach different levels of accuracy).
- x-axis: relative optimality gap ε on a log scale, **decreasing from left to right** (from 1e0 to 1e-5), tick labels as powers of ten.
- y-axis: iterations (= communication rounds) required to reach gap ≤ ε **with local-constraint violation ≤ 1e-4** (first hitting time), computed on a fine grid of ε (e.g. 30 log-spaced values).
- Curves: median over test instances with IQR band, for DR, Learned (ours), L2WS-reg (dist., including the L rounds), and A2DR (counting every application of T_DR). Optionally the centralized L2WS as dashed lines.
- Instances that never reach a level within the cap are excluded from that point and their number is reported; stop drawing a curve when more than half of the instances have not reached the level.
- Main-text version: two panels (δ = 0.2 and δ = 0.8, or the pair proposed for F-A). Appendix version `figApp_gap_levels.pdf`: 2×3 panels, all densities.
- F-M and F-A contain related information; produce both so that I can choose one for the main text.

### F-B. Difficulty vs. gain (Section 5.2) — `figB_difficulty.pdf`
- Single panel, `0.48\textwidth`.
- One point per in-distribution test instance, for Learned (ours) only.
- x: DR iteration count (log scale); y: Delta_i (%); color by δ (sequential colormap or 6 distinct colors, with legend "$\delta = 0.2$" ...).
- Print to console and to `paper_numbers.json`: Spearman ρ_s and p-value overall and per density.

### F-C. Robustness to distribution shifts (Section 5.3) — `figC_ood.pdf`
- 2×3 panels, one per density.
- In each panel, x categories: in-dist., ±20%, ±30%, ±40%, ±1 prosumer.
- For each category, **one box plot** for Learned (ours) only. Box = IQR, line = median, whiskers = 1.5 IQR, **outliers shown** as small dots. Use one color per test set, consistent with the existing OOD figure (in-dist., ±20%, ±30%, ±40%, ±1 prosumer).
- Above or below each box, small text with the win-rate is optional; put win-rates in the table T-C instead if the figure gets crowded.

### T-C. OOD summary (Section 5.3, or appendix if space is short) — `tabC_ood.tex`
- Rows: test set (ID, ±20, ±30, ±40, ±1 prosumer), for Learned (ours) only.
- Columns: densities. Cell: median [IQR] and win-rate; also worst case Delta.

---

## 3. Appendix outputs

- **T-gap. Iterations to optimality-gap levels** — `tab_gap_levels.tex`: for each density and method (DR, Learned, L2WS-reg dist., A2DR, centralized L2WS), median [IQR] iterations to reach gap ≤ 1e-2, 1e-3, 1e-4 **with local-constraint violation ≤ 1e-4**; number of instances that never reach the level within the cap.
- **F-full. Full method comparison** — `figApp_methods.pdf`: 2×3 panels (one per density), **box plots** in the same style as F-C (box = IQR, line = median, whiskers = 1.5 IQR, outliers shown), one box per method including L2WS-fp-residual. Distributed methods on the left, a thin vertical separator, centralized references on the right; hollow boxes for distributed, filled (semi-transparent) for centralized. Same y-limits in all panels.
- **F-methods (main-text candidate)** — `figB2_methods.pdf`: same as F-full but without L2WS-fp-residual, at `\textwidth`. Produce it so that I can choose between this figure and table T-A for Section 5.2.
- **T-full. Full results** — `tabApp_full.tex`: per density × method × test set: mean, median, IQR, win-rate, worst case, number of capped instances.
- **F-γ. DR step-size grid search** — `figApp_gamma.pdf`: median DR iterations vs γ (log x), one line per density, chosen γ marked; instances at the cap shown with a distinct marker.
- **T-calib. Plausibility of the generated markets** — `tabApp_calibration.tex`: per density: feasible %, average buyers/sellers, median demand curtailment, dumping %, congested links %. Use the sanity-check outputs of the **current** generator (`market.py`, c ∈ U(0.05, 0.18)); if they exist only for the old/recalibrated generator, do not use them and report it.
- **T-hyper. Hyperparameters** — `tabApp_hyper.tex`: read from the config actually used for the final runs (not defaults): d_h, MLP layers, output scaling, ρ rule, slow channel (ρ_slow, weight), K per density, loss weights (α_behind, α_ahead, β), iteration weights mode, optimizer, LR schedule, gradient clipping, epochs; L2WS: L, architecture, LR, epochs; A2DR: memory, regularization η; DR: γ per density, λ, bisection steps, tolerance, cap.
- **T-seeds** (only if several base markets per density exist) — per seed median reduction and win-rate.
- **Ablations** (only if the runs exist): ρ variants, slow channel on/off — small line or bar plots.

### T4. Timing (ask before running)
If no timing logs exist, propose a short script that measures on ~5 test instances per density: mean wall-clock time of the local prox step and of one Φ_w evaluation, per prosumer and iteration; and read training time per base market from logs. Output `tabApp_timing.tex`.

---

## 4. Verification (mandatory before reporting)

1. Open every PNG preview and check: fonts readable at print size, no overlapping labels, legends not covering data, consistent colors across figures, δ notation correct.
2. Cross-check at least 3 numbers per table against the raw data by an independent computation (print both).
3. Check the counting rules (L rounds, A2DR operator applications) by reading the code that generated the data, and state what you found.
4. Check that every figure's y-limits include all whiskers/outliers.
5. Confirm that no number in any output was typed by hand.

## 5. Final report (in chat)

- Data structure found, method-key → display-name mapping.
- List of produced files, one line each.
- Key numbers: per density, median reduction and win-rate of Learned vs best distributed competitor vs best centralized reference; Spearman ρ_s.
- Anything missing or suspicious (missing densities, capped instances, mismatched counting, gap trajectories unavailable), with what you did about it.
- A ready-to-paste LaTeX snippet for each main-text figure: `\begin{figure}...\includegraphics[width=...]{...}\caption{TBD}\label{...}\end{figure}`.
