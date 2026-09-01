# cross_density/seed0_density0.3_topology_onenode

**Test**: zero-shot inference (NO retraining). Weights: `runs/cross_density/seed0_density0.3_drslope/phi_w.pt`.

**Dataset**: base market density=0.3, base_seed=0 (N=10,
edges=20). Market parameters
held EXACTLY fixed for every node/edge inherited from the base graph. The GRAPH changes by
EXACTLY +/-1 node per instance (coin flip, `market.sample_one_node_topology_dataset`):
growing attaches the one new node to a uniformly random existing node (a single
new edge); shrinking removes one uniformly random node whose removal keeps the
graph connected (retried on a different node if the first pick is a cut vertex).
No independent edge-count scaling -- every other node/edge is untouched, isolating
the node-count axis from the `sample_common_scale_topology_dataset` edge-churn
confound. The one new node (on growth) gets freshly-drawn primitives from this
market's own parameter ranges.
Instance sizes ranged 9-11 nodes (base N=10).

**Question answered**: how does a model trained only on x0-diversity (this exact
market/topology, unperturbed) perform when the GRAPH gains or loses EXACTLY one node, with market parameters held fixed?
Directly comparable to the D*/G_res zero-shot test (parameter-axis) to see which
diversity axis the model handles better.
