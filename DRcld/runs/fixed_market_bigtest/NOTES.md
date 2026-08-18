# fixed_market_bigtest

**Test**: zero-shot inference (NO retraining). Weights: `runs/fixed_market/phi_w.pt`
(trained on ZERO diversity of any kind -- only x0 varied).

**Dataset**: the identical spec used to train `fixed_market` (same graph,
same every parameter -- NO perturbation applied at all), repeated 30
times. The only source of variation across instances is the random draw
of the initial state x^0 (`market.sample_base` + per-instance seeding via
`config.TEST_SEED`), exactly as in `demo_fixed_market.py`'s own test set,
just with more instances for a more statistically robust read.

**Question answered**: re-tests the fixed_market model on a bigger, freshly
drawn sample of starting points from its OWN training market, to check
whether the original 12-instance result (92% win-rate, 1.40x/1.48x mean/
median) holds up with more instances or was partly a small-sample artefact.
