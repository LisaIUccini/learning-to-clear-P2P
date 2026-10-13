# zeroshot_dstar_gres_v4

**Test**: zero-shot, INFERENCE ONLY (no training) -- loads the checkpoint from
`realistic_market_x0_diversity_v4_wbehind6` (x0-diversity, ONE fixed realistic market,
w_behind=6.0/w_ahead=1.4, home-market result 1.77x/100%, mean 43.5% iteration reduction) and
evaluates it on a dataset where D_star and G_res are perturbed by +/-20% (one shared factor each
per instance, box bounds D_min/D_max/G_min/G_max held EXACTLY fixed --
`sample_dstar_gres_only_dataset`), x0 FRESH RANDOM per test instance (matching this checkpoint's
own training convention, via `demo.evaluate`/`demo.run_pair`).

**Question**: does the x0-diversity-trained checkpoint (which only ever saw ONE market during
training) generalize zero-shot to a market with genuinely different D*/G_res, with no retraining?

**Architecture note**: `demo_train_realistic_market_x0_diversity.py`'s GNNConfig call only
overrides rho/scale -- rho_slow/scale_slow fall back to GNNConfig's own dataclass defaults
(0.99/0.4), so the slow channel WAS active throughout v4's training (not skipped). Reconstructed
identically here (gnn_rho=0.973, gnn_scale=1.0, gnn_rho_slow=0.99, gnn_scale_slow=0.4) for a
faithful zero-shot deployment.

**Results**: mean 1.33x, median 1.32x, min 0.85x, max 2.38x, win-rate 80% (24/30).

**Comparison**:
| | mean speedup | mean % reduction | win-rate |
|---|---|---|---|
| home market (v4_wbehind6, x0-diversity only) | 1.77x | 43.5% | 100% |
| zero-shot, D*/G_res +/-20% (this test) | 1.33x | ~25% | 80% |

**Interpretation**: real, non-catastrophic degradation. The model still delivers a genuine average
improvement with zero retraining, but 6/30 instances now lose to plain DR (down to 0.85x) -- the
x0-diversity-only checkpoint does not fully hold up when the underlying market itself shifts, which
is expected since it only ever saw ONE fixed market during training (all its "practice" was on
varying x0 within that one landscape, never on varying the landscape itself).
