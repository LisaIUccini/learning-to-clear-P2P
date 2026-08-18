"""
verify_scale_amplification.py
===============================

Cheap (no retraining) test of whether giving the ALREADY-TRAINED cost-gap
model MORE room to act -- deploying at a higher amplitude than the
validated DEPLOY_SCALE=1.0 -- makes the perturbation more effective
(closer to J*, faster convergence) or just more erratic (the gate is not
very selective yet, per the d^k diagnostic, so amplification could cut
both ways).

Reuses the SAME 30-instance fresh test set (same 5 references, same seed)
as verify_costgap_30.py, so all scale results are directly comparable to
that run's DEPLOY_SCALE=1.0 numbers (mean=2.055, median=1.471, std=2.855,
min=0.370, max=16.696, regressed=7/30).

The summability bound stays finite for ANY scale (it just grows linearly)
-- Theorem 1's guarantee is unaffected regardless of what these numbers
show, this is purely an empirical question about the TRAINED policy's
behaviour at a larger amplitude, not a safety question.

    python verify_scale_amplification.py
"""

import numpy as np

from market_generator import MarketConfig, ClusteredDatasetConfig, build_clustered_datasets
from training import Trainer, TrainConfig, compare, Reporter

NODES, DENSITY, GRAPH_MODELS, SEED = 10, 0.7, ("sparse",), 42
N_REFERENCES = 5
N_TEST_PER_REF = 6
PCT = 0.10

GAMMA, LAM, BISECT = 1.0, 1.0, 25
TOL, MAX_ITER = 1e-6, 20000

MODEL_COST = "FixedGraphN10D07test_clustered_costgap/clustered_costgap_phi_theta.pt"
SCALES_TO_TEST = [1.0, 1.5, 2.0]  # 1.0 reproduces verify_costgap_30.py's number (sanity check)


def summarize(name, speedups):
    s = np.asarray(speedups, dtype=float)
    print(f"\n{name}")
    print(f"  mean={s.mean():.3f}  median={np.median(s):.3f}  std={s.std():.3f}  "
         f"min={s.min():.3f}  max={s.max():.3f}  regressed={int((s < 1.0).sum())}/{len(s)}")
    return s


def main():
    market_cfg = MarketConfig(graph_model=GRAPH_MODELS[0], density=DENSITY)
    cfg = ClusteredDatasetConfig(
        n_references=N_REFERENCES, n_train_per_ref=1,
        n_test_per_ref=N_TEST_PER_REF, pct=PCT, N=NODES, density=DENSITY,
        graph_model=GRAPH_MODELS[0], seed=SEED)
    _, test_ds, refs = build_clustered_datasets(cfg, market_cfg)
    print(f"fresh test set: {len(test_ds)} instances (same as verify_costgap_30.py)")

    tcfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT)
    rep = Reporter(True)

    results = {}
    for scale in SCALES_TO_TEST:
        print(f"\n=== DEPLOY_SCALE = {scale} ===")
        gnn = Trainer.load(MODEL_COST)   # fresh load each time -- gnn.cfg.scale is mutated
        gnn.cfg.scale = scale
        res = compare(test_ds, gnn, tcfg, tol=TOL, max_iter=MAX_ITER, reporter=rep)
        results[scale] = summarize(f"scale={scale} (30 instances)", res["speedup"])

    print("\n=== SUMMARY across scales ===")
    print(f"{'metric':<12}" + "".join(f"{'scale='+str(s):>14}" for s in SCALES_TO_TEST))
    for label, fn in [("mean", np.mean), ("median", np.median), ("std", np.std),
                      ("min", np.min), ("max", np.max),
                      ("regressed", lambda a: int((a < 1.0).sum()))]:
        row = [fn(results[s]) for s in SCALES_TO_TEST]
        print(f"{label:<12}" + "".join(f"{v:>14.3f}" for v in row))


if __name__ == "__main__":
    main()
