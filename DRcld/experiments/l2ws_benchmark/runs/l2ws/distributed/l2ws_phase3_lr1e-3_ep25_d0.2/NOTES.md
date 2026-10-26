# l2ws_phase3_lr1e-3_ep25_d0.2

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.2, base_seed=3, gamma=4.0, training unroll k=46,
hidden_dim=32, n_rounds=3, lr=0.001, loss_mode=regression.

**Results**: mean 1.90x, median 1.82x,
min 0.90x, max 3.50x, win-rate
93%.
Mean % reduction: 42.7%, median 45.0%.
