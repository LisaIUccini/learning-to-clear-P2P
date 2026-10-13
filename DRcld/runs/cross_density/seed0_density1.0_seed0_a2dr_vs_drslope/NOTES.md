# cross_density/seed0_density1.0_seed0_a2dr_vs_drslope

**Test**: four-way comparison on the SAME market as `seed0_density1.0_seed0_drslope`
(density=1.0, N0=10, seed=0, DR
gamma=4.0), 30 test instances, each algorithm run from the
SAME random x0 per instance (seed = TEST_SEED*7919+i, this project's usual
convention): plain DR (distributed, `algorithm.run_dr`), A2DR (Anderson
accelerated DR, `a2dr.py` -- Algorithm 3.1 of Fu/Zhang/Boyd 2020,
MODEL-FREE, no training at all, eta=3e-05, mmax=5), DR + the trained
GNN correction from `seed0_density1.0_seed0_drslope/phi_w.pt` (rho=0.9667, scale=1.0),
and centralised DR (`dr_centralized.py` -- the SAME DR map, run by one
coordinator holding the whole state; validated to match plain DR's
iteration count within +/-2 iterations on this market).

**Why eta=3e-05, not the paper's own default (1e-8)**: a regularization
sweep found the paper's default oscillates for 1000+ iterations before
eventually converging (0.3-0.9x, i.e. SLOWER than plain DR) -- this DR's
local prox is solved by 20-40 bisection steps under box/capacity
constraints (a highly kinked, nonsmooth map), unlike the paper's own
closed-form/single-Newton test problems, which were also equilibrated
first (Sec. 5); this project's DR has neither. Raising eta (the (3.4)
regularization strength) to eta~3e-5-1e-4 removes the oscillation and
gives a clean, stable speed-up; eta>=0.1 over-regularizes back toward
plain DR's own 1.0x. mmax=5 was similarly chosen from a sweep on
this exact market (mmax=3: 1.47x mean; mmax=5: 1.56x; mmax=8: 1.55x --
diminishing returns past 5).

**Result, in ITERATIONS** (what a shared-memory/simulated run of each
would see -- centralised DR ties plain DR by construction, same map):

| method | mean | median | min | max | win-rate |
|---|---|---|---|---|---|
| Centralised DR | 1.00x | 1.00x | 1.00x | 1.00x | 60% |
| A2DR (no training) | 1.53x | 1.51x | 1.37x | 1.77x | 100% |
| DR + learned (dr_slope) | 2.09x | 2.11x | 1.46x | 2.60x | 100% |

**Result, in TOTAL COMMUNICATION ROUNDS** (what a REAL P2P deployment
would see -- trading-graph diameter=1): plain DR and DR+learned
cost 1 round/iteration; A2DR costs 3 rounds/iteration [1
plain-DR round + a gather + a broadcast for the (3.4) least-squares
step's global inner products, each <= diameter hops]; centralised DR
costs a FIXED `2*diameter` = 2 rounds TOTAL, independent of
iteration count (one gather of every prosumer's private data + one final
broadcast), because nothing needs to travel between iterations once the
coordinator holds the whole state:

| method | mean total rounds | requires |
|---|---|---|
| Centralised DR | **2** | every prosumer reveals private data to 1 coordinator |
| DR + learned (dr_slope) | 299 | nothing extra -- fully distributed, one-hop only |
| Plain DR (distributed) | 623 | fully distributed, one-hop only (the baseline) |
| A2DR (no training) | 1,238 | a global gather+broadcast EVERY iteration |

**Honest headline**: there is a real, three-way trade-off, not a single
ranking. Centralised DR is dramatically cheaper in communication (a fixed
handful of rounds vs. hundreds) but only because every prosumer gives up
exactly the privacy/decentralisation a distributed algorithm exists to
protect -- it is not a fair peer in the SAME category as the other three.
Among the methods that DO stay fully distributed (plain DR, A2DR,
DR+learned): DR+learned is the cheapest AND fastest; A2DR, despite winning
on raw iteration count (1.53x), is actually the MOST
expensive of all four once its non-local reduction/broadcast is honestly
counted -- worse than even staying with plain DR. A sparser/less-connected
market (diameter>1, more realistic for a real P2P grid) would only widen
every one of these gaps.

`comparison_bars.png`: per-instance iteration counts, all four methods,
sorted by plain-DR difficulty (centralised DR's bars sit almost exactly on
plain DR's, by construction). `speedup_scatter.png`: per-instance A2DR
speed-up vs. learned speed-up in iterations (y=x line splits which method
wins where). `total_communication_rounds.png`: mean total communication
rounds, all four methods, log scale -- the real headline picture.
