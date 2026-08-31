# fixed_market_2x

**Test**: full training from scratch (centralised, self-supervised,
loss_mode='hinge'), architecture/loss/hyperparameters from
`config.py` except epochs (overridden to 40 here).

**Dataset**: ONE fixed market (graph density 0.3, N0=10,
seed=0) -- graph and every parameter IDENTICAL across all
"instances". The only source of variation, for both training (50
instances) and test (30 instances), is the random draw of the
initial state x^0 per (instance, epoch).

**Question answered**: does more training signal (more repeats of the same
market, more epochs) improve win-rate/magnitude further on top of the
zero-diversity baseline (`runs/fixed_market`: 25 train instances,
20 epochs, 92% win-rate, 1.40x/1.48x mean/median on n=12), or does
the zero-diversity setting already saturate with the original budget?
