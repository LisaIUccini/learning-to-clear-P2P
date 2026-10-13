# cross_density/realistic_params_fixed_x0_v4

**Test**: train directly on PARAMETER diversity (D*, G_res, D_min, D_max, G_min, G_max) with
FIXED x0 and FIXED topology (density=1.0, same graph as every other experiment), instead of the
usual x0-only diversity. x0 is treated as an input the user supplies to start the algorithm, not
something that varies at deployment -- evaluation uses this SAME fixed x0 throughout, no random-x0
test. Realistic (household-prosumer, kW-scale) base market -- see
`experiments/build_realistic_training_dataset.py`; G_min is now >=0 by construction (was negative
by default before), matching the literature this codebase's market model is based on (Belgioioso
et al., arXiv:2107.13444, where dispatchable generation is non-negative by definition).

**Dataset**: 75 FIXED training instances (slot 0 always the unperturbed original network,
the other 74 perturbed once and reused every epoch) and 30 FIXED test instances
(slot 0 = original, unperturbed) -- a real, concrete, reproducible dataset, not regenerated per
epoch. Each perturbed instance is a direct multiplicative perturbation of the SAME base market's
D*, G_res, D_min, D_max, G_min, G_max, all six by ONE UNIFORM +/-20% factor (independent
draw per quantity, shared across nodes) -- graph, costs, and capacities held exactly fixed.
Infeasible draws are discarded and resampled (not kept). Loss asymmetry: w_ahead=1.0,
w_behind=2.0 (up from the project's usual w_ahead=1.0, since this checkpoint already
reaches 100% win-rate with room to trade some of that safety margin for larger magnitude). x0 is
the SAME single draw (seed=0) for every training instance, every epoch, and every
test instance.

**Results**: mean 1.19x, median 1.21x,
min 1.02x, max 1.33x, win-rate
100%
(30/30).

**Question answered**: does training directly on parameter diversity (instead of x0 diversity),
with x0 fixed throughout (as the user supplies it), produce a model that generalises well across
the 6-quantity perturbation space?
