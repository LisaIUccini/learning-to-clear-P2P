# l2ws_N10_d0.5_seed1_distributed_fpresidual

**Distributed L2WS benchmark** (prototype, 2026-10-16) -- message-passing GNN warm-start
predictor (`distributed_model.py`) instead of the centralised `WarmStartMLP`, L=3
one-time communication rounds instead of reading every node's private data into one global MLP.

density=0.5, base_seed=1, gamma=6.0, training unroll k=570,
hidden_dim=32, n_rounds=3, lr=0.003, loss_mode=fp_residual.

**Results**: mean 0.97x, median 1.00x,
min 0.72x, max 1.39x, win-rate
50%.
Mean % reduction: -6.0%, median -0.3%.
