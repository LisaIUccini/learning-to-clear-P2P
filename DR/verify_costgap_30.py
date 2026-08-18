"""
verify_costgap_30.py
=====================

Confirmatory check for the cost-gap-to-J* result in example_clustered_costgap.py:
on the original 10-instance clustered test set, 5/10 instances FLIPPED
category (win<->regression) between the with- and without-cost-gap models --
a high flip rate on a small sample is exactly the "might be noise" signal
documented repeatedly in failed_experiments_log.md (pattern 6).

Re-evaluates BOTH already-trained models (no retraining -- this is purely
a measurement question, not a training one) on a FRESH, larger, 30-instance
test set drawn from the SAME 5 references (same seed=42, same graph) as
example_clustered.py / example_clustered_costgap.py, so the comparison
stays apples-to-apples (same test instances for both models, matching
scale to the project's other "large sample" checks).

    python verify_costgap_30.py
"""

import numpy as np

from market_generator import MarketConfig, ClusteredDatasetConfig, build_clustered_datasets
from training import Trainer, TrainConfig, compare, Reporter

NODES, DENSITY, GRAPH_MODELS, SEED = 10, 0.7, ("sparse",), 42
N_REFERENCES = 5
N_TEST_PER_REF = 6              # 5 x 6 = 30 test instances (fresh draws,
                                # same references, same rng-stream position
                                # convention as the original 10-instance test)
PCT = 0.10

GAMMA, LAM, BISECT = 1.0, 1.0, 25
TOL, MAX_ITER = 1e-6, 20000

MODEL_NO_COST = "FixedGraphN10D07test_clustered/clustered_phi_theta.pt"
MODEL_COST = "FixedGraphN10D07test_clustered_costgap/clustered_costgap_phi_theta.pt"


def summarize(name, speedups):
    s = np.asarray(speedups, dtype=float)
    print(f"\n{name}")
    print(f"  mean={s.mean():.3f}  median={np.median(s):.3f}  std={s.std():.3f}  "
         f"min={s.min():.3f}  max={s.max():.3f}  regressed={int((s < 1.0).sum())}/{len(s)}")
    return s


def main():
    market_cfg = MarketConfig(graph_model=GRAPH_MODELS[0], density=DENSITY)
    cfg = ClusteredDatasetConfig(
        n_references=N_REFERENCES, n_train_per_ref=1,  # train_ds unused here
        n_test_per_ref=N_TEST_PER_REF, pct=PCT, N=NODES, density=DENSITY,
        graph_model=GRAPH_MODELS[0], seed=SEED)
    _, test_ds, refs = build_clustered_datasets(cfg, market_cfg)
    print(f"fresh test set: {len(test_ds)} instances, {N_REFERENCES} references "
         f"x {N_TEST_PER_REF} perturbations, same graph/references as training")

    tcfg = TrainConfig(gamma=GAMMA, lam=LAM, prox_bisect=BISECT)
    rep = Reporter(True)

    print("\n=== model WITHOUT cost-gap term ===")
    gnn_no_cost = Trainer.load(MODEL_NO_COST)
    gnn_no_cost.cfg.scale = 1.0
    res_no_cost = compare(test_ds, gnn_no_cost, tcfg, tol=TOL, max_iter=MAX_ITER, reporter=rep)
    s1 = summarize("WITHOUT cost-gap (30 fresh instances)", res_no_cost["speedup"])

    print("\n=== model WITH cost-gap term (gamma_cost=1.0) ===")
    gnn_cost = Trainer.load(MODEL_COST)
    gnn_cost.cfg.scale = 1.0
    res_cost = compare(test_ds, gnn_cost, tcfg, tol=TOL, max_iter=MAX_ITER, reporter=rep)
    s2 = summarize("WITH cost-gap (30 fresh instances)", res_cost["speedup"])

    print("\n=== SUMMARY (30 fresh instances, same test set for both) ===")
    print(f"{'metric':<12} {'no cost-gap':>12} {'with cost-gap':>14}")
    for label, a, b in [
        ("mean", s1.mean(), s2.mean()), ("median", np.median(s1), np.median(s2)),
        ("std", s1.std(), s2.std()), ("min", s1.min(), s2.min()),
        ("max", s1.max(), s2.max()),
        ("regressed", int((s1 < 1.0).sum()), int((s2 < 1.0).sum())),
    ]:
        print(f"{label:<12} {a:>12.3f} {b:>14.3f}")


if __name__ == "__main__":
    main()
