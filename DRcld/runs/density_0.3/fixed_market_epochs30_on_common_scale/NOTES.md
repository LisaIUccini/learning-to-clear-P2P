# fixed_market_epochs30_on_common_scale

**Test**: zero-shot inference (NO retraining). Weights: `runs/fixed_market_epochs30/phi_w.pt`
(trained on ZERO diversity of any kind -- only x0 varied).

**Dataset**: graph EXACTLY fixed (identical to the `fixed_market` reference market);
market parameters vary via ONE shared random factor in [1-0.1, 1+0.1]
applied to the WHOLE parameter set together, per instance (`market.sample_common_scale_dataset`).
30 test instances, disjoint draws from the training data.

**Question answered**: how well does a model trained on a single fixed market
generalise, with no retraining, to that same market after a coherent up-to-
10% shift in every parameter?
