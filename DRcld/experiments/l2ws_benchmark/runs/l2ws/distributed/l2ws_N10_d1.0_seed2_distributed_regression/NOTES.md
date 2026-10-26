# l2ws_N10_d1.0_seed2_distributed_regression

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=1.0, base_seed=2, gamma=6.0, training unroll k=1354,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=regression.

**Results**: mean 2.22x, median 2.18x,
min 0.67x, max 3.90x, win-rate
83%.
Mean % reduction: 40.8%, median 54.2%.
