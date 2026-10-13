# cross_density/realistic_params_fixed_x0_v5

**Test**: train directly on PARAMETER diversity (D*, G_res, D_min, D_max, G_min, G_max) with a
SHARED x0 and FIXED topology (density=1.0, same graph as every other experiment), instead of the
usual x0-only diversity. x0 is treated as an input the user supplies to start the algorithm, not
something that varies at deployment -- evaluation uses this SAME shared x0 throughout (each
instance's own box-projection of it), no random-x0 test. Realistic (household-prosumer, kW-scale)
base market -- see `experiments/build_realistic_training_dataset.py`; G_min is now >=0 by
construction (was negative by default before), matching the literature this codebase's market model
is based on (Belgioioso et al., arXiv:2107.13444, where dispatchable generation is non-negative by
definition).

**Perturbation scheme ("level + width", v5)**: superseded v1-v4's independent D_min/D_max (and
G_min/G_max) factors, which could squeeze a node's box down to ~1/11th of its base width while x0
stayed fixed -- leaving x0 up to ~9.7 box-half-widths outside its own box for 72% of the v3/v4
training set (measured), silently violating gnn.py's documented h_D/h_G in [-1,1] invariant (true in
every other experiment, where x0 is always freshly drawn inside whatever box is active). Here, D*
and the D-box move TOGETHER under one "level" factor (+/-20%, D*'s relative position
inside its box preserved exactly -- no collapse, no re-clipping needed), the D-box's WIDTH is
jittered separately by a narrower "width" factor (+/-8%, a more stable/contractual
property than the demand level); same for G_min/G_max (move together) vs the G-box's width. G_res
(no box of its own) is perturbed independently by its own level factor. The box can never
collapse/invert by construction. `adjust_x0_to_spec` then projects the ONE shared x0 onto each
instance's own box (clip D, G; q/kappa untouched, kappa is never perturbed) -- a no-op whenever
already inside (the common case now), else the smallest possible correction.

**Dataset**: 75 FIXED training instances (slot 0 always the unperturbed original network,
the other 74 perturbed once and reused every epoch) and 30 FIXED test instances
(slot 0 = original, unperturbed) -- a real, concrete, reproducible dataset, not regenerated per
epoch. Infeasible or negative draws are discarded and resampled (not kept). Loss asymmetry:
w_ahead=1.0, w_behind=2.0.

**Results**: mean 1.25x, median 1.23x,
min 1.06x, max 1.52x, win-rate
100%
(30/30).
x0 needed adjusting in 70/75 train and 28/30 test instances.

**Question answered**: does removing the box-collapse/x0-out-of-box artifact (v1-v4's dominant,
noisy source of "heterogeneity") with an economically-motivated level+width perturbation, plus a
sane per-instance x0 projection, let the model generalise better across the 6-quantity perturbation
space than v1-v4's ~1.15-1.21x/plateaued ceiling?
