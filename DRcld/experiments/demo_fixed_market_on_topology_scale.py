"""
demo_fixed_market_on_topology_scale.py
=========================================

Zero-shot generalisation check, the TOPOLOGY counterpart of
`demo_fixed_market_on_dstar_gres_scale.py`: NO retraining. Loads an
already-trained checkpoint and runs INFERENCE ONLY on a test set where
market PARAMETERS stay exactly fixed and only the GRAPH varies -- one
shared factor in [1-pct,1+pct] per instance applied to node count AND
edge count together (`market.sample_common_scale_topology_dataset`).
Node/edge changes are connectivity-preserving (BFS-prefix to shrink,
random attachment to grow, extra edges added/removed to hit the target
edge count without disconnecting the graph) -- see
`market._perturb_topology_shared_factor`. Genuinely new nodes/edges
(when the graph grows) get freshly-drawn primitives from `cfg`'s ranges;
inherited nodes/edges keep their base values exactly (pct=0 jitter).

Usage: python3 demo_fixed_market_on_topology_scale.py <run_name>
       <weights_path> <base_seed> <density> <dr_gamma> <gnn_rho>
       <gnn_scale> [n_test] [pct] [mode]

`mode` is "pct" (default, `market.sample_common_scale_topology_dataset`,
node AND edge count both scaled by one shared +/-pct factor) or "one_node"
(`market.sample_one_node_topology_dataset`, node count changes by EXACTLY
+/-1 per instance, no independent edge-count scaling -- isolates the
node-count axis from the edge-churn confound in "pct" mode; `pct` is
ignored in this mode).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod
from diagnostics import save_instance_plots
from demo import run_pair, evaluate, plot_results, write_run_notes


def main():
    run_name = sys.argv[1]
    weights_path = sys.argv[2]
    base_seed = int(sys.argv[3])
    density = float(sys.argv[4])
    dr_gamma = float(sys.argv[5])
    gnn_rho = float(sys.argv[6])
    gnn_scale = float(sys.argv[7])
    n_test = int(sys.argv[8]) if len(sys.argv) > 8 else 30
    pct = float(sys.argv[9]) if len(sys.argv) > 9 else 0.10
    mode = sys.argv[10] if len(sys.argv) > 10 else "pct"
    assert mode in ("pct", "one_node")

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (weights={weights_path}  base_seed={base_seed}  "
          f"density={density}  dr_gamma={dr_gamma}  gnn_rho={gnn_rho}  gnn_scale={gnn_scale}  "
          f"n_test={n_test}  pct={pct}  mode={mode})")

    device = torch.device("cpu")
    mc = market.MarketConfig(density=density)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)

    print("=" * 70)
    print(f"loading {weights_path} -- NO retraining, inference only")
    print("=" * 70)
    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=gnn_rho, scale=gnn_scale,
        rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
    blob = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    base_nb, base_prim, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    print(f"  base market: N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}")

    if mode == "pct":
        print(f"\ndataset: market parameters EXACTLY fixed for inherited nodes/edges; graph varies via ONE "
              f"shared +/-{pct*100:.0f}% factor applied to node count AND edge count together")
        test_specs = market.sample_common_scale_topology_dataset(
            base_nb, base_prim, mc, n_instances=n_test, pct=pct, seed=config.TRAIN_SEED + 4)
    else:
        print(f"\ndataset: market parameters EXACTLY fixed for inherited nodes/edges; graph varies by "
              f"EXACTLY +/-1 node per instance (coin flip), no independent edge-count scaling")
        test_specs = market.sample_one_node_topology_dataset(
            base_nb, base_prim, mc, n_instances=n_test, seed=config.TRAIN_SEED + 4)
    sizes = [spec.N for spec in test_specs]
    print(f"  instance sizes: min={min(sizes)}  max={max(sizes)}  (base N={base_spec.N})")

    print("\nevaluating: plain DR vs DR + Phi_w, zero-shot (topology shifted, no retraining)")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="topology-scale")

    print("\nrecording residual curves...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    if mode == "pct":
        subtitle = (f"weights: {os.path.basename(os.path.dirname(weights_path))}  |  tested on: graph shared "
                    f"+/-{pct*100:.0f}% node+edge factor, params fixed, n={n_test} (zero-shot)")
        dataset_desc = f"""The GRAPH
itself varies via ONE shared random factor in [{1-pct:.2f}, {1+pct:.2f}] per instance,
applied to node count AND edge count together (`market.sample_common_scale_topology_dataset`):
shrinking takes a random connected BFS-prefix induced subgraph, growing attaches
each new node to a uniformly random existing node, then edges are added or removed
(never disconnecting the graph) to hit the target edge count implied by the same
factor. Genuinely new nodes/edges (when the graph grows) get freshly-drawn
primitives from this market's own parameter ranges."""
        question_desc = (f"the GRAPH itself (node count and/or edge count) drifts by up to {pct*100:.0f}%, "
                          f"with market parameters held fixed")
    else:
        subtitle = (f"weights: {os.path.basename(os.path.dirname(weights_path))}  |  tested on: graph "
                    f"+/-1 node (coin flip), no edge-count scaling, params fixed, n={n_test} (zero-shot)")
        dataset_desc = """The GRAPH changes by
EXACTLY +/-1 node per instance (coin flip, `market.sample_one_node_topology_dataset`):
growing attaches the one new node to a uniformly random existing node (a single
new edge); shrinking removes one uniformly random node whose removal keeps the
graph connected (retried on a different node if the first pick is a cut vertex).
No independent edge-count scaling -- every other node/edge is untouched, isolating
the node-count axis from the `sample_common_scale_topology_dataset` edge-churn
confound. The one new node (on growth) gets freshly-drawn primitives from this
market's own parameter ranges."""
        question_desc = "the GRAPH gains or loses EXACTLY one node, with market parameters held fixed"
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: zero-shot inference (NO retraining). Weights: `{weights_path}`.

**Dataset**: base market density={density}, base_seed={base_seed} (N={base_spec.N},
edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}). Market parameters
held EXACTLY fixed for every node/edge inherited from the base graph. {dataset_desc}
Instance sizes ranged {min(sizes)}-{max(sizes)} nodes (base N={base_spec.N}).

**Question answered**: how does a model trained only on x0-diversity (this exact
market/topology, unperturbed) perform when {question_desc}?
Directly comparable to the D*/G_res zero-shot test (parameter-axis) to see which
diversity axis the model handles better.
""")


if __name__ == "__main__":
    main()
