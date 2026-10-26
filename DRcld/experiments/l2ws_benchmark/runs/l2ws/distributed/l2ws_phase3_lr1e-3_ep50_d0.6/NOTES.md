# l2ws_phase3_lr1e-3_ep50_d0.6

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.6, base_seed=3, gamma=4.0, training unroll k=321,
hidden_dim=32, n_rounds=3, lr=0.001, loss_mode=regression.

**Results**: mean 1.28x, median 1.25x,
min 0.95x, max 1.62x, win-rate
90%.
Mean % reduction: 20.3%, median 19.9%.
