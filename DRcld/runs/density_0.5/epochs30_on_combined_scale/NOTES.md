# density_0.5/epochs30_on_combined_scale

**Test**: zero-shot inference (NO retraining). Weights: `runs/density_0.5/epochs30/phi_w.pt`
(trained on ZERO diversity of any kind -- only x0 varied).

**Dataset**: BOTH the graph and the market parameters vary per instance, each
via its OWN independently-drawn shared random factor in [1-0.1, 1+0.1]
(`market.sample_common_scale_combined_dataset`) -- two separate draws, not
one factor driving both (topology and parameter scale are conceptually
different kinds of shift). 30 test instances, disjoint draws from
the training data.

**Question answered**: does combining two individually-survivable coherent
shifts (parameters alone: ~97% zero-shot; topology alone: ~70-83% zero-shot,
depending on how much the base model was trained) compound into a harder
generalisation problem, or is the degradation dominated by whichever axis
is already the harder one? Directly comparable to
`fixed_market_on_common_scale` and `fixed_market_on_topology_scale` (the
single-axis zero-shot checks) using the SAME weights.
