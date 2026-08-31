# common_scale_retrain_n30

**Test**: full retraining (centralised, self-supervised, loss_mode='hinge')
from scratch, using the exact `config.py` recipe validated on zero
diversity (`fixed_market`: 92% win-rate, 1.40x/1.48x mean/median).

**Dataset**: graph EXACTLY fixed (identical to the `fixed_market` reference
market) for both training and test sets. Market parameters vary via ONE
shared random factor in [1-0.1, 1+0.1] applied to the WHOLE
parameter set together, per instance (`market.sample_common_scale_dataset`)
-- NOT independent per-parameter jitter (see `runs/param_variation_independent_per_param`
for that much harsher, worse-performing variant).
25 training instances, 30 test instances (disjoint seed draws).

**Question answered**: how well does the network learn when it is TRAINED
(not just zero-shot evaluated) on a market that coherently shifts by up to
10% per instance? Directly comparable to `fixed_market_on_common_scale`
(the zero-shot version of this same test distribution, no retraining) to see
whether retraining on this mild coherent diversity helps or hurts relative
to just deploying the zero-diversity model as-is.
