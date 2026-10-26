# l2ws_N10_d0.8_seed0_distributed_fpresidual

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.8, base_seed=0, gamma=4.0, training unroll k=1159,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=fp_residual.

**Results**: mean 0.92x, median 0.96x,
min 0.57x, max 1.21x, win-rate
47%.
Mean % reduction: -14.2%, median -3.8%.
