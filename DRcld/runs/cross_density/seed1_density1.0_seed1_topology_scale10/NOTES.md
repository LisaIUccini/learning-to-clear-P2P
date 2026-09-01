# cross_density/seed1_density1.0_seed1_topology_scale10

**Test**: zero-shot inference (NO retraining). Weights: `runs/cross_density/seed1_density1.0_seed1_drslope/phi_w.pt`.

**Dataset**: base market density=1.0, base_seed=1 (N=10,
edges=45). Market parameters
held EXACTLY fixed for every node/edge inherited from the base graph. The GRAPH
itself varies via ONE shared random factor in [0.90, 1.10] per instance,
applied to node count AND edge count together (`market.sample_common_scale_topology_dataset`):
shrinking takes a random connected BFS-prefix induced subgraph, growing attaches
each new node to a uniformly random existing node, then edges are added or removed
(never disconnecting the graph) to hit the target edge count implied by the same
factor. Genuinely new nodes/edges (when the graph grows) get freshly-drawn
primitives from this market's own parameter ranges. Instance sizes ranged
9-11 nodes (base N=10).

**Question answered**: how does a model trained only on x0-diversity (this exact
market/topology, unperturbed) perform when the GRAPH itself (node count and/or
edge count) drifts by up to 10%, with market parameters held fixed?
Directly comparable to the D*/G_res zero-shot test (parameter-axis) to see which
diversity axis the model handles better.
