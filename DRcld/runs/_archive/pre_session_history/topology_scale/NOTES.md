# topology_scale

**Test**: full retraining (centralised, self-supervised, loss_mode='hinge')
from scratch, using the exact `config.py` recipe validated on zero
diversity (`fixed_market`: 92% win-rate, 1.40x/1.48x mean/median).

**Dataset**: market PARAMETERS EXACTLY fixed (identical to the `fixed_market`
reference for every inherited node) for both training and test sets. The
GRAPH varies via ONE shared random factor in [1-0.1, 1+0.1] applied to
node count AND edge count together (`market.sample_common_scale_topology_dataset`)
-- the topology analogue of `sample_common_scale_dataset`'s single shared
parameter factor. 25 training instances, 30 test instances
(disjoint seed draws).

**Question answered**: how well does the network learn when it is TRAINED
(not just zero-shot evaluated) on a market whose TOPOLOGY coherently shifts
by up to 10% per instance, parameters held fixed? Directly comparable
to `fixed_market_on_topology_scale` (the zero-shot version of this same test
distribution, no retraining) and to `common_scale` (the parameter-axis
counterpart of this same retraining experiment) to see which diversity axis
benefits more from retraining vs. zero-shot transfer.
