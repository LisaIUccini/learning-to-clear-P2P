# l2ws_N10_d0.2_seed3_distributed_fpresidual

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.2, base_seed=3, gamma=4.0, training unroll k=46,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=fp_residual.

**Results**: mean 0.90x, median 0.86x,
min 0.42x, max 1.62x, win-rate
37%.
Mean % reduction: -20.6%, median -15.9%.
