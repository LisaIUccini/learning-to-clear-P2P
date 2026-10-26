# l2ws_N10_d0.8_seed0_distributed_regression

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.8, base_seed=0, gamma=4.0, training unroll k=1159,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=regression.

**Results**: mean 1.54x, median 1.59x,
min 1.00x, max 2.05x, win-rate
100%.
Mean % reduction: 31.8%, median 37.2%.
