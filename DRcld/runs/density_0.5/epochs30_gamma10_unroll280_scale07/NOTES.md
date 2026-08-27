# density_0.5/epochs30_gamma10_unroll280_scale07

**Test**: full training from scratch (centralised, self-supervised,
loss_mode='hinge'), architecture/loss/hyperparameters from
`config.py` except epochs (overridden to 30 here), graph density
(overridden to 0.5 here), eta (overridden to 0.15 here, config
default is 0.15), DR's own gamma (overridden to 10.0
here, config default is 4.0 -- train_cfg.gamma is derived from
dr_cfg.gamma, so the differentiable rollout during training automatically
matches whatever gamma is used at inference), training unroll length
(overridden to 280 here, config default is 200), the
perturbation's decay rate GNN_RHO (overridden to 0.964 here, config
default is 0.95), and the perturbation's amplitude GNN_SCALE
(overridden to 0.7 here, config default is 0.4).

**Dataset**: ONE fixed market (graph density 0.5, N0=10,
seed=0) -- graph and every parameter IDENTICAL across all
"instances". The only source of variation, for both training (25
instances) and test (30 instances), is the random draw of the
initial state x^0 per (instance, epoch).

**Question answered**: does more training signal (more repeats of the same
market, more epochs) improve win-rate/magnitude further on top of the
zero-diversity baseline (`runs/fixed_market`: 25 train instances,
20 epochs, 92% win-rate, 1.40x/1.48x mean/median on n=12), or does
the zero-diversity setting already saturate with the original budget?
