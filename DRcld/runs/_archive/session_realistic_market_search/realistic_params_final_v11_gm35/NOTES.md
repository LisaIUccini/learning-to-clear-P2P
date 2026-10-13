# realistic_params_final_v11_gm35

**Test**: train directly on PARAMETER diversity (D*, G_res, D_min, D_max, G_min, G_max) with a
SHARED x0 and FIXED topology (density=1.0, same graph as every other experiment), instead of the
usual x0-only diversity. x0 is treated as an input the user supplies to start the algorithm, not
something that varies at deployment -- evaluation uses this SAME shared x0 throughout (each
instance's own box-projection of it), no random-x0 test. Realistic (household-prosumer, kW-scale)
base market -- see `experiments/build_realistic_training_dataset.py`; G_min is now >=0 by
construction (was negative by default before), matching the literature this codebase's market model
is based on (Belgioioso et al., arXiv:2107.13444, where dispatchable generation is non-negative by
definition).

**Perturbation scheme ("level + width")**: supersedes v1-v4's independent D_min/D_max (and
G_min/G_max) factors, which could squeeze a node's box down to ~1/11th of its base width -- a real
bug (box collapse). Here, D* and the D-box move TOGETHER under one "level" factor
(+/-20%, D*'s relative position inside its box preserved exactly -- no collapse, no
re-clipping needed), the D-box's WIDTH is jittered separately by a narrower "width" factor
(+/-8%, a more stable/contractual property than the demand level); same for G_min/G_max
(move together) vs the G-box's width. G_res (no box of its own) is perturbed independently by its
own level factor. The box can never collapse/invert by construction.

**x0 (finalised 2026-09-10)**: the ONE shared x0 is used RAW/UNCLIPPED for every instance -- x0's
feasibility relative to any instance's box is NOT a requirement of the algorithm (DR's own
`local_prox` clamps back onto the box on the very first step regardless), and this scheme's box
can't collapse, so the un-clipped excursion stays bounded (measured: h_D roughly in [-2.4, 1.8]) --
an earlier version of this script clipped x0 into each instance's box, which was correct but never
actually necessary; removed for a simpler, more defensible design.

**Dataset**: 25 FIXED training instances (slot 0 always the unperturbed original network,
the other 24 perturbed once and reused every epoch) and 30 FIXED test instances
(slot 0 = original, unperturbed) -- a real, concrete, reproducible dataset, not regenerated per
epoch. Infeasible or negative draws are discarded and resampled (not kept). Loss asymmetry:
w_ahead=1.4, w_behind=6.0.

**Calibration**: unroll=307, gnn_rho=0.9674 -- freshly re-derived from THIS dataset's own
plain-DR iteration distribution (fixed, unclipped x0) via the project's standing rule
(unroll=mean_DR/2, rho=1-10/unroll), not reused from any other market/dataset.

**Results**: mean 1.21x, median 1.19x,
min 1.11x, max 1.41x, win-rate
100%
(30/30).

**Question answered**: with box-collapse ruled out by construction, x0 kept simple (no clip), and
rho/unroll properly calibrated on this exact dataset, does the model train cleanly (no self-inflicted
plateau) and match or beat the earlier v5 result (1.25x/100%, which used the same scheme but with an
x0 clip and rho/unroll calibrated on an earlier, slightly different check)?
