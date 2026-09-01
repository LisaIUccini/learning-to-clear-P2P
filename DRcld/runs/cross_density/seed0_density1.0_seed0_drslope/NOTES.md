# cross_density/seed0_density1.0_seed0_drslope

**Test**: full training from scratch (centralised, self-supervised,
loss_mode='asymmetric'), architecture/loss/hyperparameters from
`config.py` except epochs (overridden to 60 here), graph density
(overridden to 1.0 here), eta (overridden to 0.15 here, config
default is 0.15), DR's own gamma (overridden to 4.0
here, config default is 4.0 -- train_cfg.gamma is derived from
dr_cfg.gamma, so the differentiable rollout during training automatically
matches whatever gamma is used at inference), training unroll length
(overridden to 300 here, config default is 200), the
perturbation's decay rate GNN_RHO (overridden to 0.9667 here, config
default is 0.95), the perturbation's amplitude GNN_SCALE
(overridden to 1.0 here, config default is 0.4), and
the loss itself (overridden to 'asymmetric' here, config default is
'hinge', w_behind=2.0, w_ahead=1.0).

**Dataset**: ONE fixed market (graph density 1.0, N0=10,
seed=0) --
graph and every parameter IDENTICAL across all
"instances". The only source of variation, for both training (25
instances) and test (30 instances), is the random draw of the
initial state x^0 per (instance, epoch).

**Question answered**: does more training signal (more repeats of the same
market, more epochs) improve win-rate/magnitude further on top of the
zero-diversity baseline (`runs/fixed_market`: 25 train instances,
20 epochs, 92% win-rate, 1.40x/1.48x mean/median on n=12), or does
the zero-diversity setting already saturate with the original budget?
