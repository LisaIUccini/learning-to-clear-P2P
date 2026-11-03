# l2ws_N10_d0.8_seed0_distributed_asymmetric

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.8, base_seed=0, gamma=4.0, training unroll k=1159,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=asymmetric.

**Results**: mean 1.52x, median 1.64x,
min 0.94x, max 2.14x, win-rate
93%.
Mean % reduction: 30.4%, median 39.1%.
