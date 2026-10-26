# l2ws_N10_d0.5_seed1_distributed_regression

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.5, base_seed=1, gamma=6.0, training unroll k=570,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=regression.

**Results**: mean 1.08x, median 1.09x,
min 0.80x, max 1.52x, win-rate
57%.
Mean % reduction: 4.4%, median 7.8%.
