# l2ws_sweep_L2_d0.6

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=2
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.6, base_seed=3, gamma=4.0, training unroll k=321,
hidden_dim=32, n_rounds=2, lr=0.003, loss_mode=regression.

**Results**: mean 1.28x, median 1.23x,
min 0.96x, max 1.63x, win-rate
90%.
Mean % reduction: 20.1%, median 18.6%.
