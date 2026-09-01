# Experiment index

**2026-09-01 update (topology decomposition: node-only vs edge-only vs combined)**: the combined
+/-10% topology test (below) confounds node-count and edge-count changes. Isolated them: node-only
(`market.sample_one_node_topology_dataset`, exactly +/-1 node) and edge-only (reused
`sample_common_scale_topology_dataset` at pct=0.05, which -- verified -- never changes N for
N0=10, so it's pure edge churn). Same 5 checkpoints, pure inference.

| density | combined (+/-10%) | node-only (+/-1) | edge-only (+/-5%) |
|---|---|---|---|
| 0.3 | 97% / 2.28x | 87% / 2.06x | **97% / 2.47x** |
| 0.5 | 63% / 1.31x | 80% / 1.52x | **93% / 1.52x** |
| 0.8 | 70% / 1.32x | 67% / 1.20x | **90% / 1.56x** |
| 1.0 seed=0 | 67% / 1.34x | 73% / 1.46x | **77% / 1.64x** |
| 1.0 seed=1 | 70% / 1.30x | 67% / 1.21x | **77% / 1.54x** |

Edge-only is well-tolerated everywhere (strongest of the 3 variants at every density). Node-only
is intermediate. Combined is worse than either alone at every density except 0.3 -- looks like a
genuine interaction effect, not one dominant axis. CAVEAT: the combined test's shrink mechanism
(BFS-prefix subgraph) differs from node-only's shrink (single node removal) -- not a fully clean
decomposition on the shrink side; growth mechanism is identical across all 3. See
`project_l2p2p_paper.md` in the auto memory for the full write-up. Runs:
`runs/cross_density/*_topology_scale10` / `*_topology_onenode` / `*_topology_edge05`.

**2026-09-01 update (zero-shot validation completed, two axes)**: closed the D*/G_res gap for
0.3 and 1.0 (both seeds), then ran a brand-new TOPOLOGY zero-shot axis (graph node+edge count
shared +/-10% factor, connectivity-preserving, `market.sample_common_scale_topology_dataset`)
across all 5 current-best checkpoints. Pure inference, no retraining.

| density | D*/G_res win-rate / mean | topology win-rate / mean |
|---|---|---|
| 0.3 | 100% / 2.42x | 97% / 2.28x |
| 0.5 | 100% / 1.46-1.57x | **63% / 1.31x** |
| 0.8 | 100% / 1.34x | **70% / 1.32x** |
| 1.0 seed=0 | 100% / 1.85x | **67% / 1.34x** |
| 1.0 seed=1 | 100% / 1.68x | **70% / 1.30x** |

**Conclusion**: parameter-shift (D*/G_res) generalisation is uniformly strong (100% everywhere).
Topology-shift generalisation is strong ONLY at density=0.3; every other density lands at
63-70% win-rate / ~1.3x mean -- consistently weaker than every other validation axis tested this
session, though still >1x mean (not a total failure the way `dr_slope` on density=0.5 was). See
`project_l2p2p_paper.md` in the auto memory for the full table and an untested hypothesis (sparser
base graphs suffer a smaller RELATIVE structural disruption from the same +/-10% factor). Runs:
`runs/cross_density/*_dstar_gres_scale10` and `runs/cross_density/*_topology_scale10`.

**2026-08-31 update (dr_slope campaign)**: `omega_mode="dr_slope"` (instance-adaptive
iteration weighting, see `train.asymmetric_loss`) beat the previous best recipe on every
density except 0.5, where it fails outright (self-inflicted overshoot plateau, unresolved
as of this writing -- full mechanism in `feedback_gamma_variance_and_loss_cap.md` point 3c
in the auto memory). Current best per density/seed, all in `runs/cross_density/`:

| density | recipe | win-rate | mean |
|---|---|---|---|
| 0.3 | `seed0_density0.3_drslope` | 100% | 2.96x |
| 0.5 | `seed0_density0.5_gamma6_unroll305_scale10_asymmetric` (uniform, NOT dr_slope) | 100% | 1.81x |
| 0.8 | `seed0_density0.8_gamma8_unroll750_rho9925_drslope` | 100% | 1.83x |
| 1.0 seed=0 | `seed0_density1.0_seed0_drslope` | 100% | 2.09x |
| 1.0 seed=1 | `seed1_density1.0_seed1_drslope` | 100% | 1.79x |

Zero-shot D*/G_res validation exists for density=0.5 (uniform recipe, 1.46-1.57x/97-100%,
`*_dstar_gres_scale10`) and density=0.8 (dr_slope recipe, `seed0_density0.8_drslope_dstar_gres_scale10`,
1.34x/100%) -- **NOT yet run for density=0.3 or density=1.0 under dr_slope**, a gap in the
validation checklist. The old (now-superseded) `runs/density_1.0/` folder and its zero-shot
run were archived to `_archive/session_dr_slope_campaign/density_1.0_pre_drslope/` along
with every beaten/failed dr_slope attempt from this campaign; nothing was deleted.

**2026-08-31 earlier reorganisation**: `runs/` was cleaned up to keep only the current best
result per density/seed at the top level. Every run this file references by name
that isn't directly under `runs/` anymore has been moved, unmodified, into
`runs/_archive/pre_session_history/` (everything below, predating the 2026-08-28
density=1.0 investigation) or `runs/_archive/session_density_debug/` (superseded
attempts from the density 1.0/0.5/0.8 cross-density debugging session, see
`project_l2p2p_paper.md` / `feedback_gamma_variance_and_loss_cap.md` in the auto
memory for the full story of what was tried and why each was superseded). Nothing
was deleted.

Each folder is one full train+evaluate run: `demo_run.log` (console output),
`phi_w.pt` (trained weights), `results.png` (4-panel summary),
`instance_plots/` (per-instance diagnostics, where generated). All numbers
are on the 12-instance in-cluster test set unless noted.

| folder | loss | architecture | dataset | win-rate | mean / median x |
|---|---|---|---|---|---|
| `eq15_loss` | Eq. 15 (absolute) | single channel | plain cluster | 58% (7/12) | 1.08x / 1.21x |
| `relative_reweighted` | relative (reweighted) | single channel | plain cluster | worse than Eq.15 | 0.93x / 0.94x |
| `hinge_v1` | hinge, eta=0.15 | single channel + gate | plain cluster | 75% (9/12) | 1.05x / 1.08x |
| `hinge_v2_gate` | hinge, eta=0.15 | + conservative gate init | plain cluster | 75% (9/12) | 1.04x / 1.07x |
| `hinge_v3_eta025` | hinge, eta=0.25 | single channel + gate | plain cluster | 58% (7/12) | 1.05x / 1.11x |
| `hinge_v4_longhorizon_FAILED` | hinge, eta=0.15 | rho=0.97, unroll=400 | plain cluster | 33% (4/12) | 0.91x / 0.96x |
| `hinge_v5_final` | hinge, eta=0.15 | scale=0.4 (gentler) | plain cluster | **75% (9/12)** | 1.05x / 1.09x |
| `a1_moderate_horizon` | hinge, eta=0.15 | rho=0.97, unroll=250, 40 epochs | plain cluster | 25% (3/12) | 0.90x / 0.92x |
| `a2b1_dual_channel_curriculum` | hinge, eta=0.15 | dual channel (fast+slow) | B1 curriculum (hard-oversampled) | 8% (1/12) | 0.54x / 0.36x |
| `asymmetric_longunroll` | asymmetric (w_behind=2,w_ahead=1) | single channel, unroll=350 (rho unchanged=0.95) | plain cluster | 67% (8/12) | 1.07x / 1.06x (max 1.60x) |
| `asymmetric_4to1` | asymmetric (w_behind=4,w_ahead=1) | single channel, unroll=350 | plain cluster | 75% (9/12) | 1.04x / 1.05x (max 1.29x) |
| `fixed_market` | hinge, eta=0.15 (identical to hinge_v5_final) | identical to hinge_v5_final | **ZERO diversity** (1 market, only x0 varies) | **92% (11/12)** | **1.40x / 1.48x** (max 1.72x) |

**Best validated result: `hinge_v5_final`** (72% win-rate in-cluster+out-of-cluster
combined when both were evaluated; 75% in-cluster alone). All 18/18
(`hinge_v5_final`) and 18/18 (`eq15_loss`) reported iteration counts were
cross-checked against the exact QP equilibrium (cost gap <= 1e-4) --
genuine convergence, not a stopping-rule artifact. `a2b1_dual_channel_curriculum`
was spot-checked too: also genuine convergence, just a much worse (though
safe) policy.

`demo.py`'s current default config reproduces `hinge_v5_final`'s setup
(dual-channel architecture present in `gnn.py`, but `scale_slow=0.0`
neutralises the slow channel exactly, so it contributes nothing regardless
of what it learns). Run `python3 demo.py <run_name>` to write a fresh
`runs/<run_name>/`.

## What was learned about extending the correction's reach

Two independent attempts to let the correction influence more of a slow
instance's trajectory both made win-rate substantially WORSE, not better,
compared to the short-decay baseline:

- **A1** (single channel, just a slower fixed decay + far more training):
  win-rate collapsed 72%->25%. Ruled out "it was just undertrained" as the
  explanation -- 2x the epochs did not recover it.
- **A2+B1** (a second, separately-gated slow channel + a training set
  oversampling hard instances so the slow channel gets gradient signal):
  win-rate collapsed further, to 8%. Diagnosis: the curriculum imbalance
  (12/24 training instances near the hardest end) taught the network an
  almost-instance-independent policy -- most held-out instances converged
  to a suspiciously narrow, common iteration count (~1040-1060) regardless
  of their true difficulty (ranging 269-3442 under plain DR), instead of a
  policy that discriminates fast from slow.

Both are real, reproducible negative results, not noise -- see the
`train.py` module docstring for `hinge_loss`'s note, and this file's table
above for the numbers. The mechanism-level conclusion: a short, protective
decay window is not a limitation to engineer around so much as it is
*load-bearing* for the win-rate this project is optimising for, at least
with the training budget and dataset used here.

## Chasing average magnitude (not just win-rate): asymmetric loss

`hinge_loss` has a structural property worth being explicit about: once an
iteration is `eta` fraction ahead of plain DR, it gets EXACTLY ZERO
gradient (relu is flat past the margin) -- so it has no incentive to ever
produce a bigger win than the margin asks for. That's why win-rate went up
under it but mean speed-up stayed modest: the loss never asked for more.

`asymmetric_loss` (`w_behind * relu(rho-rho_dr) - w_ahead * relu(rho_dr-rho)`,
see train.py) removes the cap: continuous, uncapped reward for beating DR
by more, continuous penalty for lagging by more, weighted 2:1 toward
penalising lagging (to bias toward reliability). A floored log-ratio
version was designed first and REJECTED before any training run -- checked
numerically, a frozen trajectory scored -5.71 vs only -2.27 for a genuine
10x speed-up, i.e. freezing was rewarded over 2x more than honest fast
convergence, reopening the exact exploit that sank the very first loss
design in this project. `asymmetric_loss` closes that by construction
(freezing's reward is bounded by rho_dr itself, not a tunable floor):
checked the same way, frozen scores -0.30 vs -0.27 for 10x -- no
meaningful gap to exploit.

Combined with extending the training unroll 200->350 while leaving `rho`
(the decay rate) UNCHANGED at 0.95 -- letting training observe more of the
consequences of an early intervention without ever giving the correction
itself more time to interfere with a late plateau, unlike A1/A2+B1 which
changed the decay too -- result: mean/median 1.07x/1.06x (up from
1.05x/1.09x), max 1.60x (up from 1.30x), but win-rate DOWN to 67% (8/12,
from 75%). Real progress on magnitude, but at a win-rate cost the
asymmetry weighting (2:1) did not fully avoid -- validated 12/12 genuine
equilibria via QP. `hinge_v5_final` remains the win-rate-best reference;
this run is kept as the magnitude-leaning alternative and as the base to
retune the w_behind:w_ahead ratio from.

Follow-up: pushing the ratio to 4:1 (`asymmetric_4to1`) recovered win-rate back to 75% but also gave back almost all the extra magnitude (1.04x/1.05x, max 1.29x) -- essentially converging back onto `hinge_v5_final`'s own numbers. Diminishing/no further returns from asymmetry tuning alone on the diverse dataset; see the `fixed_market` finding below for what actually moved both numbers at once.

## THE key diagnostic: is diversity the bottleneck? Yes.

`fixed_market/` retrains the EXACT `hinge_v5_final` recipe (architecture,
loss, hyperparameters -- nothing changed) on a dataset where the graph and
every market parameter are IDENTICAL across all 25 training and 12 test
"instances" -- the only thing that varies, for both, is the random draw
of x^0. Result:

| | win-rate | mean / median | max |
|---|---|---|---|
| `hinge_v5_final` (diverse cluster, +/-10%) | 75% (9/12) | 1.05x / 1.09x | 1.30x |
| `fixed_market` (zero diversity) | **92% (11/12)** | **1.40x / 1.48x** | 1.72x |

Same weights architecture, same loss, same every hyperparameter. The only
change is removing cross-instance diversity from the learning problem.
Validated 12/12 genuine equilibria via QP.

Conclusion: the network was never failing to learn a strong correction --
it was being asked to find ONE shared policy that works across a whole
DISTRIBUTION of different graphs and parameter sets, which is a
fundamentally harder target than "accelerate DR on this specific market".
Give it one market to specialise in and it gets both dramatically higher
win-rate AND dramatically higher average magnitude at once -- the
trade-off every loss redesign on the diverse dataset kept running into
(hinge vs. eq15, eta tuning, asymmetric weighting) turns out not to be a
fundamental property of the correction mechanism, it was a symptom of the
task being too broad for a single shared policy trained this way.

The one regression (`fixed_market` instance 1, x0.75) is on the fastest-
converging starting point (196 iterations, the quickest of the 12) --
consistent with the SAME "already-fast trajectories are the riskiest to
touch" pattern seen on the diverse dataset throughout, suggesting that
specific failure mode is about proximity to convergence, not market
diversity, and may persist even with a fully specialised model.

Practical implication: for a REAL deployment where the same physical grid
clears repeatedly (a very plausible setting -- the grid topology and
typical parameter ranges do not change clearing to clearing), training one
specialised model per market (or per narrow cluster, closer to what the
+/-10% cluster experiments already approximate) looks like the highest-
leverage remaining lever, more so than further loss/architecture tuning on
a maximally diverse training set.

## Decomposing diversity further: topology vs. parameters, and a surprise

`config.py` now holds every tunable parameter in one place, and
`market.sample_common_scale_dataset` implements what was actually asked
for here: ONE shared scale factor (1+/-pct) applied to the WHOLE
parameter set per instance, graph exactly fixed -- not the earlier
`sample_param_only_dataset` (every individual parameter independently
jittered, kept archived as `param_variation_independent_per_param`, 42%
win-rate -- a materially LARGER perturbation in aggregate than "each
value within pct" suggests, since ~100+ independent draws rarely cancel).

| | win-rate | mean / median | max |
|---|---|---|---|
| `fixed_market` (zero diversity, trained AND tested on it) | 92% (11/12) | 1.40x / 1.48x | 1.72x |
| `common_scale` (RETRAINED on +/-10% coherent shifts) | 75% (9/12) | 1.18x / 1.20x | 1.50x |
| `fixed_market_on_common_scale` (fixed_market weights, ZERO-SHOT, no retraining) | **92% (11/12)** | **1.41x / 1.46x** | 1.75x |

All 12/12 genuine equilibria via QP. The surprise: the model that NEVER
saw parameter variation generalises to +/-10% coherent shifts essentially
AS WELL AS -- slightly better, even -- as its home performance, and
BETTER than a model deliberately retrained on that exact variation.
Retraining on the (mild, coherent) diversity made things worse, not
better, mirroring the same "diversity tax" pattern seen at the full
cluster scale, just smaller.

Likely mechanism: every GNN feature is already a RELATIVE quantity (
position within the local box, balance residual normalised by box width,
cost coefficients normalised against a FIXED reference range) rather than
an absolute one. A coherent scale-up/down of the whole market changes the
absolute numbers but leaves these relative quantities close to unchanged,
so a policy specialised on one instance's relative structure transfers
almost for free to a proportionally-rescaled version of it -- while
retraining on the family instead re-introduces a (smaller) version of the
"one shared policy across several targets" cost that hurt every diverse-
dataset experiment. This suggests topology change (not parameter
rescaling) is the more expensive kind of diversity to generalise across,
though that has not been isolated as cleanly as this comparison.

Zero-shot on the topology axis, and on the much harsher independent-per-
parameter axis, were also checked directly:

| | win-rate | mean / median | max |
|---|---|---|---|
| `fixed_market_on_topology_scale` (fixed_market weights, ZERO-SHOT, graph +/-10% shared factor) | 83% (10/12) | 1.21x / 1.21x | 1.68x |
| `fixed_market_on_independent_params` (fixed_market weights, ZERO-SHOT, EVERY param jittered independently +/-10%) | 33% (4/12) | 0.91x / 0.83x | 2.05x |

Confirms topology shift is handled almost as well as parameter shift
zero-shot (83% vs 92%), while independent-per-parameter jitter (NOT a
shared factor -- see the correction at the top of this section) is a much
larger effective perturbation and genuinely defeats zero-shot transfer.

## Five-test follow-up: bigger samples (n=30) + topology-axis retraining

Extends the diagnostic above along three directions at once, each in its
own `runs/` folder with a `NOTES.md`: (1) re-run the two zero-shot cross-
axis checks and the zero-diversity home-market check on a bigger, disjoint
30-instance test set for a more statistically robust read than the
original 12; (2) actually TRAIN (not just zero-shot evaluate) on the
topology-scale axis, mirroring what `common_scale` already did for the
parameter axis, so both axes now have a matched zero-shot-vs-retrained
comparison. Same `config.py` recipe throughout (architecture, loss,
hyperparameters unchanged); only the dataset/weights differ.

| folder | test | win-rate | mean / median | max |
|---|---|---|---|---|
| `fixed_market_bigtest` | fixed_market weights, ZERO-SHOT, home market (zero diversity), n=30 | 97% (29/30) | 1.41x / 1.48x | 1.72x |
| `fixed_market_on_common_scale_n30` | fixed_market weights, ZERO-SHOT, params +/-10% shared factor, n=30 | 97% (29/30) | 1.40x / 1.46x | 1.75x |
| `fixed_market_on_topology_scale_n30` | fixed_market weights, ZERO-SHOT, graph +/-10% shared factor, n=30 | 83% (25/30) | 1.24x / 1.27x | 1.68x |
| `common_scale_retrain_n30` | RETRAINED on params +/-10% shared factor, n=30 | 77% (23/30) | 1.18x / 1.20x | 1.56x |
| `topology_scale` | RETRAINED on graph +/-10% shared factor, n=30 | 77% (23/30) | 1.13x / 1.11x | 1.67x |

All three n=30 zero-shot numbers replicate the original n=12 read almost
exactly (home market 92%->97%, common-scale 92%->97%, topology-scale
83%->83%) -- the original 12-instance results were not a small-sample
artefact. QP-validated: 27/30, 30/30, 28/30, 30/30, 30/30 genuine
equilibria respectively (tol 1e-4) -- every "failure" is a borderline miss
(`gap_learned` in the 1.0-1.2e-4 range) where plain DR's own gap is
~1e-6, i.e. an artefact of DR's residual-based stopping rule (tol=1e-5 on
the iterate, not on cost) landing right at the QP check's boundary, not a
sign of convergence to the wrong point (`validate_five_tests.py`).

Both retraining results reproduce the "diversity tax" pattern seen
throughout this project: retraining on either axis of mild, coherent
+/-10% diversity UNDERPERFORMS the zero-shot fixed_market model on that
same axis (common-scale: 77%/1.18x retrained vs. 97%/1.40x zero-shot;
topology-scale: 77%/1.13x retrained vs. 83%/1.24x zero-shot). The gap is
much larger on the parameter axis (97%->77%, -20pp) than the topology axis
(83%->77%, -6pp) -- consistent with the "relative GNN features" mechanism
proposed above: retraining "costs" more where zero-shot transfer was
already excellent (parameters) than where it was already the harder case
(topology), since there's more headroom to give up. Practical takeaway
unchanged: for markets that stay physically similar clearing-to-clearing,
deploying the zero-diversity specialist as-is beats deliberately
retraining it on the anticipated shift, on both diversity axes tested so
far.

## fixed_market_2x: does more training/data on the same market help, or cause pace collapse?

`fixed_market_2x` doubles both training signal dimensions on the SAME
zero-diversity market from `fixed_market`: 50 training instances (up from
25) and 40 epochs (up from 20), same graph (density 0.3), same random-x0
initialisation, evaluated on a fresh 30-instance test set. Two zero-shot
cross-axis checks (common-scale, topology-scale, n=30 each) were then
re-run with these new weights, mirroring the earlier fixed_market
cross-axis checks.

| test | 1x (`fixed_market`) | 2x (`fixed_market_2x`) |
|---|---|---|
| home market (n=30) | 92%/97%\*, 1.40x-1.41x / 1.46x-1.48x | 93%, 1.51x / 1.50x |
| zero-shot common-scale (n=30) | 92%, 1.41x / 1.46x | **97%**, 1.52x / 1.48x |
| zero-shot topology-scale (n=30) | 83%, 1.24x / 1.27x | **70%**, 1.25x / 1.28x |

\*the two numbers are `fixed_market_bigtest` (97%) and the original 12-instance
`fixed_market` result (92%) -- both zero-shot/home reads of the 1x model.

More training on the home market helped there (93% vs 92%, and notably
higher magnitude: 1.51x vs 1.40x) and even improved zero-shot transfer to
the parameter axis (97% vs 92%). But it made zero-shot transfer to the
TOPOLOGY axis clearly WORSE (83%->70% win-rate, -13pp; magnitude flat).

**Mechanism, confirmed two independent ways:**

1. **Per-instance iteration counts.** In `fixed_market` (1x), the model's
   own iteration count visibly tracks each instance's difficulty (e.g.
   DR=545->learned=376, DR=196->learned=261 -- a real spread, 261-376). In
   `fixed_market_2x`, the learned iteration count collapses to a near-
   constant band (275-282) across ALL 30 instances regardless of DR's own
   range (196-560) -- visible directly in the results.png scatter panel as
   a flat horizontal band of points instead of a diagonal-ish spread.
   Zero-shot on common-scale (same graph) reproduces the same narrow band
   (264-290); zero-shot on topology-scale (graph actually changes size)
   does NOT collapse nearly as tightly (277-429), which is exactly why
   that axis's win-rate suffers most -- a near-constant absolute iteration
   budget stops being a good match once the market's own natural iteration
   scale shifts with topology.

2. **QP validation margins got thinner.** Cross-checking final cost gaps
   against the exact QP equilibrium (tol 1e-4) found only 7/30, 13/30,
   18/30 "genuine" on the three fixed_market_2x tests respectively --
   MUCH lower than the ~27-30/30 seen throughout the rest of this project.
   Critically, every "failure" has `gap_dr` ~1e-6 (DR itself is exact) but
   `gap_learned` landing consistently at 1.0-1.3e-4 -- just outside the
   cutoff, and NOT a wrong equilibrium (relative error ~0.001%). This is
   the same borderline band seen occasionally elsewhere in this project
   (DR's own tol=1e-5 residual criterion doesn't exactly guarantee a
   cost gap under 1e-4), but pace collapse means MOST instances now land
   in that thin margin instead of just 2-5/30 -- because a near-constant
   stopping length no longer benefits from the per-instance "natural
   overshoot past the tolerance" that varying, adaptive stopping points
   used to provide. Reported win-rates/speedups are still real (n_iter
   counts are exact and independent of this check), but this model
   converges with a visibly thinner safety margin than every prior run.
   (`validate_fixed_market_2x.py`)

**Conclusion**: more training data/epochs on a single fixed market is a
double-edged lever. It sharpens both raw performance AND overfitting to
that market's specific absolute iteration scale -- good news if the
deployed market truly never changes shape, a real risk if any topology
drift is expected, where the more conservative `fixed_market` (1x) model
generalises more reliably despite slightly lower home-market numbers.

## Clean ablation: epochs alone (25 instances fixed) vs. epochs+instances together

Follow-up to `fixed_market_2x` above, which changed epochs AND training
instance count together (20->40 epochs, 25->50 instances), so it couldn't
say which of the two actually drives pace collapse. `fixed_market_epochs10`,
`_epochs30`, `_epochs40` hold instances fixed at 25 (matching the original
`fixed_market`) and vary ONLY epochs, all evaluated on the same fresh
30-instance test set (`fixed_market_bigtest` supplies the n=30 read for the
20-epoch point, since it's a zero-shot re-evaluation of those exact weights).

| epochs | instances | win-rate | mean/median | min/max | learned-iteration range |
|---|---|---|---|---|---|
| 10 | 25 | 80% (24/30) | 1.26x/1.32x | 0.73/1.61 | 267-408 (delta 141) |
| 20 | 25 | 97% (29/30) | 1.41x/1.48x | 0.75/1.72 | 261-376 (delta 115) |
| 30 | 25 | 97% (29/30) | 1.44x/1.47x | 0.74/1.85 | 266-337 (delta 71) |
| 40 | 25 | 97% (29/30) | **1.53x/1.54x** | 0.73/2.01 | 266-300 (delta 34) |
| 40 | 50 (`fixed_market_2x`) | 93% (28/30) | 1.51x/1.50x | 0.71/2.01 | 275-282 (delta 7) |

Clean, monotonic result with instances held fixed: win-rate jumps 80%->97%
between 10 and 20 epochs (10 epochs is simply under-trained) then plateaus
at 97% through 30 and 40; mean/median magnitude keeps climbing the whole
way (1.26x -> 1.53x). Meanwhile the per-instance learned-iteration spread
narrows monotonically with epochs alone (141 -> 115 -> 71 -> 34), even
with instance count held constant -- so epochs alone, independent of
training-instance count, are sufficient to drive pace collapse. Adding
more instances on top (`fixed_market_2x`'s 50 vs. 25) narrows the spread
further still (34 -> 7) and is also where win-rate finally dips (97% ->
93%) -- consistent with both epochs and instance repetition pushing the
same direction, with the combination compounding faster than either alone.

Practical reading: on THIS single fixed market, more epochs is close to
a free lunch for raw performance up to ~30-40 (magnitude keeps improving,
win-rate saturates rather than drops) -- the actual cost of pace collapse
only shows up when testing generalisation OFF this exact market (see
`fixed_market_2x_on_topology_scale`: 83%->70% win-rate vs. the 1x/20-epoch
model, and the QP-margin thinning in the section above). Whether the same
epoch-only ablation reproduces that same zero-shot topology-scale
degradation progressively (10/30/40 vs. 20 epochs) has not yet been
tested -- would need three more cheap zero-shot cross-checks, one per
epoch count, mirroring `fixed_market_2x_on_topology_scale`'s setup.

**QP validation of the epochs-only ablation** (`validate_epoch_ablation.py`,
tol 1e-4, home-market n=30 each) confirms the same margin-thinning pattern
completely independent of instance count:

| epochs | genuine equilibria | worst gap |
|---|---|---|
| 10 | 30/30 | 8.78e-06 (comfortable) |
| 30 | 29/30 | 1.02e-04 (right at the edge) |
| 40 | 15/30 | 1.18e-04 (half the instances at the edge) |

Monotonic with epochs alone, matching the learned-iteration-spread finding
exactly -- two independent measurements (n_iter spread AND QP cost-gap
margin) now agree that epoch count alone, with training-instance count
held fixed, is sufficient to drive pace collapse.

## Confirming the ablation causally: zero-shot generalisation at epochs=30

`fixed_market_epochs30_on_common_scale` / `_on_topology_scale` (n=30 each,
zero-shot, NO retraining, weights = `fixed_market_epochs30/phi_w.pt`)
close the loop: does the epochs-only pace collapse actually cost
generalisation progressively, the way `fixed_market_2x` (epochs+instances
together) showed for the topology axis?

| model (home 25 instances) | common-scale zero-shot | topology-scale zero-shot |
|---|---|---|
| 20 epochs (`fixed_market`) | 97% (29/30), 1.40x/1.46x | 83% (25/30), 1.24x/1.27x |
| 30 epochs | 97% (29/30), 1.43x/1.46x | **77%** (23/30), 1.23x/1.26x |
| 40 epochs + 50 instances (`fixed_market_2x`) | 97% (29/30), 1.52x/1.48x | **70%** (21/30), 1.25x/1.28x |

Clean, monotonic confirmation: topology-scale win-rate degrades
progressively with the SAME epoch-driven pace collapse measured two other
ways above (iteration-spread narrowing, QP-margin thinning) -- 83% -> 77%
-> 70%, tracking epochs 20 -> 30 -> 40(+more instances) exactly. Meanwhile
common-scale zero-shot stays flat at 97% throughout -- the parameter axis
genuinely does not care about pace collapse (consistent with the "GNN
features are relative quantities" mechanism proposed earlier: a coherent
parameter rescaling doesn't change the market's natural iteration-count
scale, so a rigid absolute-pace policy still matches; a topology change
does shift that scale, so it doesn't). QP-validated: 28/30 genuine on
both epochs=30 cross-checks (worst gaps 1.07-1.08e-4, the same familiar
borderline band, not wrong equilibria).

**Practical recommendation**: 30 epochs is the sensible default over 40 --
it captures nearly all of 40 epochs' extra home-market magnitude (1.43x
vs. 1.53x mean) while giving up much less topology-axis robustness (77%
vs. 70%, a smaller drop from the 20-epoch baseline's 83%) and keeping a
noticeably healthier QP safety margin (29/30 vs. 15/30 at home). 20 epochs
remains the safest choice if any topology drift at deployment is a real
possibility; 10 epochs is simply under-trained and not recommended.

## Combined-scale: both axes shifting together

`fixed_market_on_combined_scale` / `fixed_market_epochs30_on_combined_scale`
(zero-shot, n=30, `market.sample_common_scale_combined_dataset`): BOTH the
graph and the parameters vary per instance, each via its OWN
independently-drawn shared factor in [1-pct, 1+pct] (two separate draws,
not one factor driving both -- topology and parameter scale are
conceptually different kinds of shift). New nodes from topology growth
draw fresh primitives, same as the topology-only dataset.

| weights | win-rate | mean/median | min/max | QP genuine |
|---|---|---|---|---|
| `fixed_market` (20 epochs) | 87% (26/30) | 1.27x / 1.22x | 0.87x / 1.76x | 28/30 |
| `fixed_market_epochs30` | 83% (25/30) | 1.30x / 1.22x | 0.86x / 1.95x | 28/30 |

Combined-scale tracks the topology-scale axis closely (87%/83% here vs.
83%/77% for topology-scale alone at the same two checkpoints) rather than
compounding multiplicatively with the parameter axis (which alone stays
at 97% throughout) -- topology is the bottleneck; adding a coherent
parameter shift on top of an already-present topology shift costs little
extra. Implemented via `_perturb_primitives`'s new `factor` argument
(apply one pre-drawn shared factor to every inherited value, instead of
each value drawing its own independent factor in [1-pct,1+pct]) plus
`_perturb_topology_shared_factor`, reused from the topology-only dataset.

## Density 0.7: does the whole picture replicate on a denser graph?

Same recipe as `fixed_market_epochs30` (25 train instances, 30 epochs,
n_test=30, x0 random) but starting from a base market with graph density
0.7 instead of 0.3 (34 edges vs. 20 on N0=10) -- `density_0.7/epochs30`
plus its three zero-shot cross-axis checks, all using the new `density`
CLI parameter added to `demo_fixed_market.py` and the three
`demo_fixed_market_on_*.py` scripts (defaults to `config.MARKET_DENSITY`
if omitted, so every existing call site is unaffected). Density runs now
live under `runs/density_<value>/<name>/` to keep the flat `runs/`
listing from growing unbounded as more densities get tested -- existing
density-0.3 runs are untouched (still flat in `runs/` directly) since
retroactively moving ~25 folders was judged not worth the churn/risk for
runs already cross-referenced everywhere in this file.

| test | density 0.3 | density 0.7 |
|---|---|---|
| home | 97% (29/30), 1.44x/1.47x | 90% (27/30), 1.20x/1.20x |
| common-scale (zero-shot) | 97% (29/30), 1.43x/1.46x | 90% (27/30), 1.21x/1.22x |
| topology-scale (zero-shot) | 77% (23/30), 1.23x/1.26x | 80% (24/30), 1.23x/1.15x |
| combined-scale (zero-shot) | 83% (25/30), 1.30x/1.22x | **67% (20/30)**, 1.08x/1.08x |

QP-validated: 30/30 genuine on all four density-0.7 checks, comfortable
margins throughout (5.5e-5 to 8.7e-5) -- notably NO sign of the QP-margin
thinning seen in the epochs ablation at density 0.3, consistent with this
being a single (non-ablated) 30-epoch checkpoint rather than a pace-
collapse-prone higher-epoch one (`validate_density07.py`).

Two findings: (1) absolute performance is lower across the board at
density 0.7 -- a denser graph changes DR's own dynamics enough that the
same architecture/loss/training-budget accelerates it less effectively;
(2) the topology axis is nearly unchanged between densities (77% vs 80%)
but the COMBINED axis is NOT -- at density 0.3 combined-scale (83%)
tracked topology-scale (77%) closely, i.e. the harder axis dominated and
adding the parameter shift cost little extra; at density 0.7 combined-scale
(67%) is clearly WORSE than either axis alone (80% topology, 90% common),
i.e. the two shifts compound here rather than one dominating. The
"combining shifts is no worse than the harder axis alone" conclusion from
density 0.3 does NOT generalise to density 0.7 -- whether shifts compound
multiplicatively or are dominated by the harder axis appears to depend on
the base market's own structure, not just on which two axes are combined.

### Density 0.7: 30 vs. 40 epochs

| test | 30 epochs | 40 epochs |
|---|---|---|
| home | 90% (27/30), 1.20x/1.20x | 90% (27/30), 1.25x/1.27x |
| common-scale | 90% (27/30), 1.21x/1.22x | 93% (28/30), 1.21x/1.22x |
| topology-scale | 80% (24/30), 1.23x/1.15x | 77% (23/30), 1.20x/1.13x |
| combined-scale | 67% (20/30), 1.08x/1.08x | 67% (20/30), 1.08x/1.09x |

QP-validated: 30/30 on every 30-epoch check (margins 5.5-8.7e-5); at 40
epochs still 30/30 on home/common-scale/topology-scale (margins 8.8-9.5e-5)
but 29/30 on combined-scale (worst gap 1.12e-4). Same directional
pace-collapse signal as density 0.3 (topology-scale 80%->77% with more
epochs, QP margins drift up slightly) but far milder here -- nothing like
density 0.3's 40-epoch collapse to 15/30 genuine on home. Whether pace
collapse's severity scales with graph density (denser graph = milder
collapse for the same epoch budget) is a plausible reading but not yet
tested cleanly (would need the same epochs=10 point at density 0.7 for a
fair comparison).

## Density 0.5: a much sharper pace collapse, and why

`density_0.5/epochs30` (same recipe: 25 train instances, 30 epochs,
n_test=30, x0 random) performs dramatically worse than either 0.3 or 0.7:

| test | win-rate | mean/median | learned-iteration range |
|---|---|---|---|
| home | **57% (17/30)** | 0.96x / 1.03x | 759-801 (delta 42) |
| common-scale (zero-shot) | 57% (17/30) | 0.96x / 1.02x | -- |
| topology-scale (zero-shot) | 77% (23/30) | 1.21x / 1.06x | -- |
| combined-scale (zero-shot) | 73% (22/30) | 1.06x / 1.07x | -- |

Notably topology-scale and combined-scale BEAT home here -- the reverse
of every other density tested, where shifting off the training market
always cost win-rate. QP-validated: 30/30 genuine on all four axes
(margins 5.9e-6 to 9.9e-5) -- the poor numbers are real convergence, not
a validation artefact.

**Root cause, confirmed two ways.** First, the per-instance log shows the
most extreme pace collapse seen anywhere in this project: learned
iteration counts sit in a 759-801 band (delta=42) despite plain DR ranging
319-903 (delta=584) across the SAME 30 test instances -- tighter collapse
than density 0.3 needed 40 epochs (delta=34) or even `fixed_market_2x`'s
50-instances-plus-40-epochs (delta=7 only with much more training). Here
it happens at just 30 epochs.

Second, and this is the actual explanation: the home market's own
difficulty distribution over random x0 is sharply skewed. Sampling 60
fresh x0 draws and running plain DR on each:

| density | shape | range | mean/median |
|---|---|---|---|
| 0.3 | unimodal, smooth spread | 237-610 | 397/388 |
| 0.5 | bimodal: small fast cluster + dominant slow cluster, with a real gap (625->691, +66, vs. neighbouring gaps of a few units) | 307-921 | 753/798 |
| | fraction "fast" (<600): 15%; fraction "slow" (600-1000): 85% | | |

At density 0.3, every training x0 draws from a smooth continuum, so
gradient signal is available across the whole difficulty range throughout
training -- consistent with needing a much longer training budget (40
epochs, or +more instances) before collapse sets in. At density 0.5, ~85%
of training x0 draws land in the dominant "slow" mode; the rare "fast"
15% supplies too little, too infrequent gradient to keep the network
discriminating between the two regimes, so it collapses onto a policy
tuned for the majority mode (~790 iterations, close to that mode's own
median of 798) almost immediately. At test time the same ~85/15 split
recurs, so the rigid ~790-iteration intervention is a near-miss "modest
win" on the majority (1.02-1.13x) and a near-catastrophe on the minority
(as low as 0.41x) -- never the large wins seen elsewhere, because the
policy isn't actually adaptive, it is just close to the right answer most
of the time by construction.

This also explains why zero-shot topology-scale/combined-scale BEAT home
here: perturbing the graph draws a materially different market whose own
x0-difficulty distribution is presumably less skewed (not measured
directly), so the rigid ~790 policy gets more chances to land favourably
than it does against home's specific 85/15 split.

**Why "more training" will not fix this** (directly tested against the
user's own proposed fixes): more epochs and a larger training-instance
count BOTH already have a measured direction in this project -- both
monotonically NARROW the learned-iteration spread (the epochs ablation
above; `fixed_market_2x`'s 50-vs-25-instance comparison) -- i.e. both
deepen exactly this kind of collapse, they don't fix it. The problem here
is a skewed training DISTRIBUTION, not an undertrained network; feeding
more of the same skewed distribution predictably reinforces the majority-
tuned policy rather than teaching the minority regime.

**Not yet tried**: fewer epochs (mirrors the density-0.3 epochs=10 result,
which stayed genuinely adaptive at the cost of being undertrained);
deliberately rebalancing/oversampling the underrepresented "fast" x0 draws
during training (higher risk -- `a2b1_dual_channel_curriculum` already
showed that curriculum reweighting in this codebase can backfire badly,
though that attempt oversampled the HARD end, the opposite direction from
what's proposed here); shrinking `GNN_SCALE` (gentler default amplitude,
previously reduced disturbance to losing instances at density 0.3 without
costing win-rate).

## Density 0.5: does more training (bigger dataset + more epochs) fix it? Partially, and not as predicted.

Two follow-ups to the density-0.5 diagnosis above, both requested directly
to test whether the "more training deepens collapse" precedent (from
density 0.3's epochs ablation and `fixed_market_2x`) actually holds here
too.

**eta=0.10 vs. eta=0.15** (25 instances, 30 epochs, otherwise identical):
interrupted before full artifacts saved (only the evaluation printout
survived), but that printout is a real, reproducible number: win-rate
57%->67% (20/30), mean/median 0.97x/1.04x (basically flat vs. 0.96x/1.03x
at eta=0.15). A lower margin requirement did measurably help SOME
borderline instances cross from loss to win, without hurting magnitude --
consistent with the mechanism proposed (a less demanding target gives the
network more room to apply a calibrated, non-maximal intervention). Not
yet re-run to completion with saved weights/plots for further testing.

**n_train=50, epochs=40, eta=0.15 (density_0.5/epochs40_train50)**: win-rate
57% (17/30) -- IDENTICAL to the 25-instance/30-epoch baseline, mean/median
0.97x/1.02x (also flat). By total win-rate alone, more training changed
nothing. But the per-instance breakdown tells a different, more nuanced
story than the density-0.3 precedent predicted:

| instance | DR iters | learned (25 instances, 30 epochs) | learned (50 instances, 40 epochs) |
|---|---|---|---|
| 1 | 509 | 785 (x0.65) | 691 (x0.74) |
| 8 | 455 | 786 (x0.58) | 667 (x0.68) |
| 17 | 342 | 759 (x0.45) | 606 (x0.56) |
| 30 | 319 | 786 (x0.41) | 608 (x0.52) |

Every one of the four "fast" minority instances moved measurably AWAY
from the collapsed majority-mode value (~790) with more training, not
further into it -- the overall learned-iteration spread widened from
759-801 (delta=42) to 606-816 (delta=210). This is the OPPOSITE direction
from density 0.3's epochs/instances ablation, where more training always
narrowed the spread. None of the four crossed into a genuine win, and the
flat win-rate implies some previously-winning "slow" instances likely
moved the other way, roughly cancelling out in the aggregate count.
QP-validated: 30/30 genuine, comfortable margin (5.9e-6) -- no sign of the
margin-thinning seen in density 0.3's high-epoch collapse either.

**Revised understanding**: the "more training -> more collapse" rule from
density 0.3 is NOT universal -- it held there (smooth, unimodal x0-
difficulty distribution) but does not hold the same way on density 0.5's
sharply bimodal distribution, where more training instead nudged the
minority regime's behaviour to partially differentiate from the majority,
without (yet) enough separation to flip any losses into wins. Whether
substantially more training (well beyond 50/40) would eventually complete
that separation, or whether it plateaus here, is untested. The eta=0.10
result suggests the loss's margin parameter may be a more direct/effective
lever for this specific failure mode than training volume.

## Density 0.5: fixing the root cause -- DR's own gamma, not GNN training

Follow-up to the skewed-distribution diagnosis above: rather than tuning
GNN-side levers (epochs, dataset size, eta) to cope with the skew, a cheap
gamma sweep for PLAIN DR alone (no training, `validate_density05_gamma_sweep*.py`,
30 fresh x0 draws per gamma) searched for a gamma that fixes the skew at
its source.

| gamma | mean | median | stdev | cv (stdev/mean) | min-max |
|---|---|---|---|---|---|
| 1.0 | 2641 | 2789 | 628 | 0.238 | 630-3298 |
| 4.0 (current default) | 763 | 793 | 144 | 0.188 | 307-921 |
| 8.0 | 560 | 562 | 50 | 0.090 | 473-634 |
| 10.0 | 567 | 561 | 25 | 0.044 | 534-618 |
| 12.0 | 652 | 650 | 11 | 0.017 | 632-682 |
| **14.0** | 761 | 763 | **8.9** | **0.012** | 742-773 |
| 16.0 | 880 | 880 | 18 | 0.020 | 851-936 |
| 20.0 | 1075 | 1071 | 16 | 0.015 | 1051-1108 |
| 24.0 | 1318 | 1318 | 27 | 0.020 | 1261-1369 |
| 30.0+ | -- | -- | -- | -- | DOES NOT CONVERGE (30/30 hit n_iter cap) |

Clear U-shaped trade-off: gamma~8-10 minimises absolute iteration count;
gamma=14 minimises relative spread (cv=0.012, essentially every x0 draw
converges in nearly the same number of steps) while keeping almost the
SAME mean as the current gamma=4.0 default (761 vs. 763) -- i.e. gamma=14
eliminates the skew at no cost to average DR speed. Gamma>=30 breaks DR's
own convergence entirely (reflection step too aggressive, pure numerical
instability, nothing to do with the GNN).

**Retrained fixed_market with gamma=14 (all else at baseline: 25 instances,
30 epochs, eta=0.15):**

| | win-rate | mean/median | min/max | QP |
|---|---|---|---|---|
| gamma=4.0 (original baseline) | 57% (17/30) | 0.96x/1.03x | **0.41x**/1.13x | 30/30 |
| gamma=14.0 only | 63% (19/30) | 1.01x/1.00x | **0.98x**/1.03x | 30/30 (worst gap 9.17e-5) |

The catastrophic failure mode is GONE (worst case 0.41x -> 0.98x) just
from fixing gamma -- confirms the skewed-distribution diagnosis was
correct and directly fixable at the DR level. But win-rate only rose
modestly (57%->63%) and magnitude stayed flat (~1.0x): with DR now itself
near-uniform, there is much less "room" left for the GNN to accelerate
further, and the hinge loss's zero-gradient-past-margin property (see the
loss section above) caps any incentive to push harder once roughly at
parity.

**Also recalibrating unroll/rho to gamma=14's own horizon** (`GNN_RHO`
0.95->0.975, `TRAIN_UNROLL` 200->400, chosen so the perturbation's active
window covers a similar FRACTION of the market's typical DR length as the
original 200/0.95 pair did for density 0.3's ~388-iteration median --
previously only ~25% of a ~761-iteration trajectory was inside rho=0.95's
effective window, now ~50% is):

| | win-rate | mean/median | min/max | QP |
|---|---|---|---|---|
| gamma=14 only (unroll=200, rho=0.95) | 63% (19/30) | 1.01x/1.00x | 0.98x/1.03x | 30/30 |
| gamma=14 + unroll=400 + rho=0.975 | **83% (25/30)** | 1.01x/1.02x | 0.95x/1.05x | 30/30 (worst gap 9.22e-5) |

Recalibrating the correction's active window to match the market's own
timescale recovered most of the win-rate gap to density 0.3/0.7 (83%,
comparable to topology-scale zero-shot's 77-83% range seen elsewhere)
without reintroducing the historical "extend the horizon" failure mode
(A1: 72%->25%, `fixed_market_longhorizon`) -- the difference being those
attempts extended an ALREADY-adequate window on a different market; here
200 was a genuine, measurable mismatch (only 25% coverage) being corrected
to a better-matched one, not an arbitrary extension.

**Open question flagged directly by the user**: win-rate improved a lot,
but average MAGNITUDE (mean speed-up) is still only ~1.01x -- barely above
break-even, even though win-rate is 83%. Two compounding reasons: (1)
gamma=14 was chosen to minimise SPREAD (cv), not iteration count -- gamma
8-10 achieves a notably lower mean (560-567) with only mildly worse cv
(0.04-0.09), leaving more physical "room" for a genuine speedup rather
than DR already sitting near whatever floor gamma=14's dynamics allow; (2)
hinge_loss's margin gives zero gradient once eta=0.15 is met at a given
step and the per-instance numbers suggest the network is not even
consistently reaching that 15% margin here, consistent with there being
little slack left to extract at gamma=14 specifically. Not yet tested:
retraining at gamma=8 or gamma=10 instead of 14, trading a little of the
spread reduction for more raw headroom.

## Density 0.5: pushing magnitude further -- gamma, unroll/rho, and scale together

Full progression of fixes applied to density=0.5 (all n=30, all QP-validated
genuine, all `demo_fixed_market.py` with its new `density`/`eta`/`gamma`/
`unroll`/`gnn_rho`/`gnn_scale` CLI overrides):

| config | win-rate | mean/median | min/max | QP |
|---|---|---|---|---|
| gamma=4.0 (original) | 57% (17/30) | 0.96x/1.03x | 0.41x/1.13x | 30/30 |
| gamma=14, unroll=200 (default) | 63% (19/30) | 1.01x/1.00x | 0.98x/1.03x | 30/30 |
| gamma=14, unroll=400, rho=0.975 | 83% (25/30) | 1.01x/1.02x | 0.95x/1.05x | 30/30 |
| gamma=14, unroll=400, rho=0.975, eta=0.25, 40 epochs | 73% (22/30) | 1.01x/1.01x | 0.94x/1.04x | -- |
| gamma=10, unroll=280, rho=0.964 | 87% (26/30) | 1.04x/1.03x | 0.99x/1.14x | -- |
| gamma=10, unroll=280, rho=0.964, **scale=0.7** | 97% (29/30) | 1.05x/1.04x | 1.00x/1.14x | 30/30 (8.74e-5) |
| gamma=10, unroll=280, rho=0.964, **scale=1.0** | 97% (29/30) | 1.06x/1.05x | 0.99x/1.14x | 30/30 (9.17e-5) |

Raising `GNN_SCALE` (perturbation amplitude ceiling) from the project
default 0.4 to 0.7 gave a genuine, validated jump: win-rate 87%->97%,
never a loss (min 0.99x->1.00x). But pushing further to 1.0 bought almost
nothing more (1.05x->1.06x mean, max STAYS AT 1.14x across all three scale
values 0.4/0.7/1.0) -- diminishing returns already reached. Scale was a
real, worthwhile lever (unlike eta, where higher hurt) but has a visible
ceiling around 0.7-1.0 for this market.

**Bottom line, stated plainly**: every lever tested (gamma retuning,
unroll/rho recalibration to the market's own horizon, scale increase) improved
RELIABILITY substantially and validatedly (57%->97% win-rate, worst case
0.41x->0.99x) but average MAGNITUDE has plateaued around 1.05-1.06x mean,
1.14x max -- nowhere near density 0.3's 1.40-1.53x. The max speed-up being
essentially IDENTICAL (1.13-1.14x) across gamma values (10 and 14) and
across scale values (0.4, 0.7, 1.0) is itself informative: it did not move
with any lever tried, suggesting this specific market's dynamics (at this
seed) may have a real, structural ceiling on how much a bounded local
perturbation can shortcut its convergence, distinct from the reliability
question (which every lever DID move). Not yet isolated which structural
property sets that ceiling, or whether it is specific to this exact
density-0.5/seed-0 instance vs. a general density-0.5 property.

## Density 1.0 (complete graph): same methodology, same ceiling

Applied the full systematic recipe (gamma sweep -> unroll/rho match ->
scale increase) to density=1.0 (N0=10, 45/45 edges, complete graph)
directly from the start, using lessons from density 0.5.

**Gamma sweep** (30 fresh x0 draws per gamma, no training): even the
default gamma=4.0 is already fairly uniform here (cv=0.094, much better
than density 0.5's 0.188 at the same gamma) -- the complete graph's dense
coupling appears intrinsically more homogeneous across x0 draws. Optimum
at gamma=10 (cv=0.010, mean=746.6, min-max 734-769 -- an extremely tight
band).

**Baseline** (gamma=10, unroll=373, rho=0.9729, scale=0.7, eta=0.15, 25
instances, 30 epochs): **100% win-rate (30/30), min=1.00x** -- zero losses
across every test instance, the cleanest result in this entire density
exploration. mean=1.03x, median=1.03x, max=1.08x. QP: 24/30 genuine
(worst gap 1.16e-4, familiar borderline band, not a wrong equilibrium).

**Two follow-up levers tested, both confirming what density 0.5 already
showed, this time on a market with a very different structure:**

| config | win-rate | mean/median | min/max | QP |
|---|---|---|---|---|
| baseline (hinge, eta=0.15) | **100% (30/30)** | 1.03x/1.03x | **1.00x**/1.08x | 24/30 |
| eta=0.25 | 80% (24/30) | 1.01x/1.01x | 0.97x/1.05x | 27/30 |
| asymmetric_loss (w_behind=2, w_ahead=1) | 93% (28/30) | 1.02x/1.02x | 0.97x/1.05x | 21/30 |

Raising eta cost win-rate (100%->80%) for no magnitude gain, exactly as
at density 0.5 -- confirms this is not density-specific. More
significantly: `asymmetric_loss` (chosen specifically to remove
`hinge_loss`'s zero-gradient-past-margin cap, hypothesised as THE
magnitude bottleneck once reliability risk was fixed) made things WORSE
across every metric, not better -- win-rate down, mean/max down, losses
reappeared. The hypothesis that hinge's margin cap was suppressing
achievable magnitude is REJECTED by this experiment.

**Revised conclusion**: across two densities (0.5, 1.0), two different
seeds (density 0.5 seed=0 and seed=1), and now multiple loss functions
(hinge at two eta values, asymmetric), the maximum achievable speed-up
has stayed confined to roughly 1.05-1.31x depending on the specific
market instance, regardless of which training-side lever was pulled
(gamma, unroll/rho, scale, epochs, eta, loss function). Only the training
DATA'S underlying market seems to move the ceiling (density 0.5 seed=1
reached max=1.31x vs. seed=0's 1.14-1.17x) -- suggesting the ceiling is a
property of the specific market's own optimisation landscape (how much a
bounded, one-message-per-iteration local perturbation can actually
shortcut ITS convergence), not a hyperparameter that can be tuned away.
Density 0.3's 1.40-1.53x results, obtained with the ORIGINAL untouched
gamma=4.0/unroll=200/rho=0.95/scale=0.4 recipe, may reflect that specific
market's own dynamics already being well-matched to those defaults by
chance, rather than evidence that comparable magnitude is achievable
everywhere with enough tuning.

## Cross-seed generality, both densities: the ceiling is market-instance-specific, not density-specific

Same systematic recipe (gamma sweep -> unroll/rho match -> scale=0.7,
eta=0.15) applied to a SECOND seed at density=1.0 (seed=1, gamma=10 again
optimal there too: cv=0.019 at gamma=10/12, mean=864.6, unroll=431,
rho=0.9765) -- completing a 2x2 grid across {density 0.5, density 1.0} x
{seed 0, seed 1}:

| density | seed | win-rate | mean/median | min/max |
|---|---|---|---|---|
| 0.5 | 0 (+100 epochs) | 100% (30/30) | 1.07x/1.06x | 1.02x/1.17x |
| 0.5 | 1 (+100 epochs) | 70% (21/30) | 1.06x/1.07x | 0.93x/1.27x |
| 1.0 | 0 | **100% (30/30)** | 1.03x/1.03x | **1.00x**/1.08x |
| 1.0 | 1 | 60% (18/30) | 1.01x/1.01x | 0.95x/1.06x |

Both densities show the SAME split: seed=0 gives excellent, loss-free
results (100% win-rate both times); seed=1 gives clearly worse results
both times (70% and 60% win-rate, losses reappear). QP-validated: 23/30
genuine on density 1.0 seed=1 (worst gap 1.14e-4, familiar borderline
band, not a wrong equilibrium).

This rules out density itself as the deciding factor -- it is NOT "0.5 is
hard, 1.0 is easy" or vice versa. Seed=0 happens to produce, at both
densities, a market whose gamma-tuned DR dynamics the trained correction
handles perfectly; seed=1 produces, at both densities, a market where the
same methodology (gamma sweep, unroll/rho matched to the new median,
scale=0.7) still leaves real losses on the table. The half-median
unroll heuristic and gamma-sweep procedure generalise mechanically (both
seed=1 sweeps found a sensible, still-improving gamma), but do not
guarantee loss-free reliability the way they did for seed=0 -- some
additional, not-yet-identified property of the specific market instance
matters beyond graph density and a well-chosen gamma/unroll/rho/scale.
Whether that property is systematically discoverable (and whether density
0.3's seed=0 owes its exceptional 1.40-1.53x result to the same kind of
instance-specific luck) remains open.
