# l2ws_phase3_lr1e-2_ep50_d0.6

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.6, base_seed=3, gamma=4.0, training unroll k=321,
hidden_dim=32, n_rounds=3, lr=0.01, loss_mode=regression.

**Results**: mean 1.12x, median 1.06x,
min 0.83x, max 1.43x, win-rate
80%.
Mean % reduction: 8.6%, median 6.0%.
