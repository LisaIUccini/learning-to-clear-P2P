# cross_density/realistic_params_fixed_x0

**Test**: train directly on PARAMETER diversity (D*, G_res, D_min, D_max, G_min, G_max) with
FIXED x0 and FIXED topology (density=1.0, same graph as every other experiment), instead of the
usual x0-only diversity. x0 is treated as an input the user supplies to start the algorithm, not
something that varies at deployment -- evaluation uses this SAME fixed x0 throughout, no random-x0
test. Realistic (household-prosumer, kW-scale) base market -- see
`experiments/build_realistic_training_dataset.py`; G_min is now >=0 by construction (was negative
by default before), matching the literature this codebase's market model is based on (Belgioioso
et al., arXiv:2107.13444, where dispatchable generation is non-negative by definition).

**Dataset**: 25 FIXED training instances (slot 0 always the unperturbed original network,
the other 24 perturbed once and reused every epoch) and 30 FIXED test instances
(slot 0 = original, unperturbed) -- a real, concrete, reproducible dataset, not regenerated per
epoch. Each perturbed instance is a direct multiplicative perturbation of the SAME base market's
D*/G_res (+/-15%, one shared factor per quantity per instance) and D_min/D_max/G_min/G_max
(+/-10%) -- graph, costs, and capacities held exactly fixed. x0 is the SAME single draw
(seed=0) for every training instance, every epoch, and every test instance.

**Results**: mean 1.19x, median 1.18x,
min 1.07x, max 1.34x, win-rate
100%
(30/30).

**Question answered**: does training directly on parameter diversity (instead of x0 diversity),
with x0 fixed throughout (as the user supplies it), produce a model that generalises well across
the 6-quantity perturbation space?
