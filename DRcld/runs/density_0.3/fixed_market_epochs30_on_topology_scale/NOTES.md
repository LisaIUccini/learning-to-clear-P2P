# fixed_market_epochs30_on_topology_scale

**Test**: zero-shot inference (NO retraining). Weights: `runs/fixed_market_epochs30/phi_w.pt`
(trained on ZERO diversity of any kind -- only x0 varied).

**Dataset**: market PARAMETERS EXACTLY fixed; the GRAPH varies via ONE shared
random factor in [1-0.1, 1+0.1] applied to node count AND edge
count together (`market.sample_common_scale_topology_dataset`), the topology
analogue of `sample_common_scale_dataset`'s single shared parameter factor.
30 test instances, disjoint draws from the training data.

**Question answered**: how well does a model trained on a single fixed market
generalise, with no retraining, to that same market after a coherent up-to-
10% shift in its TOPOLOGY (node/edge count), parameters held fixed?
Directly comparable to `fixed_market_on_common_scale` (the parameter-axis
counterpart) to see which diversity axis the model handles better.
