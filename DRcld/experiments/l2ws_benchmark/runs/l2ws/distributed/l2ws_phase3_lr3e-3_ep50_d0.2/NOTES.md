# l2ws_phase3_lr3e-3_ep50_d0.2

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.2, base_seed=3, gamma=4.0, training unroll k=46,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=regression.

**Results**: mean 3.58x, median 3.43x,
min 1.67x, max 6.22x, win-rate
100%.
Mean % reduction: 69.4%, median 70.8%.
