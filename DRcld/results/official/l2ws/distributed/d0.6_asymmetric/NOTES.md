# l2ws_N10_d0.6_seed3_distributed_asymmetric

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.6, base_seed=3, gamma=4.0, training unroll k=321,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=asymmetric.

**Results**: mean 1.03x, median 0.99x,
min 0.77x, max 1.32x, win-rate
47%.
Mean % reduction: 1.3%, median -1.0%.
