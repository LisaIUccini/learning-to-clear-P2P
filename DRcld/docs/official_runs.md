# Provenance of the official runs

The results in `results/official/` were produced, before the refactoring of
November 2026, by scripts that took every parameter as a positional argument.
The commands below were recovered from the session logs; `l2p2p/official.py`
encodes the same values, and `tests/test_regression.py` checks that the
current code reproduces the saved results.

## DR+GNN (`experiments/train_official_pilot.py`)

Arguments: density, rho_variant, run_name, n_train, n_test, epochs, base_seed,
n_nodes, gamma, unroll, rho10, rho5 (unused), scale_slow, rho_slow, scale,
c_scheme and curvature_regime (both ignored with generator=old), generator,
c_range, omega_mode, omega_power, lr_final_frac, w_ahead, w_behind.

```
0.2 10 official_N10_d0.2_seed3_replicate        50 30 25 3 10 4.0  426 0.9765 0.9661 0.0 0.99 1.0 seller_discount literature old 0.05,0.18 fixed    1.5 0.01
0.4 10 official_N10_d0.4_seed3_replicate        50 30 25 3 10 4.0  318 0.9686 0.9661 0.4 0.99 1.8 seller_discount literature old 0.05,0.18 fixed    1.5 0.01 1.6 6.0
0.5 10 official_N10_d0.5_seed1_replicate        50 30 25 1 10 6.0  570 0.9825 0.9661 0.0 0.99 1.0 seller_discount literature old 0.05,0.18 fixed    1.5 0.01 1.0 2.0
0.6 10 official_N10_d0.6_seed3_replicate        50 30 25 3 10 4.0  321 0.9688 0.9661 0.0 0.99 1.0 seller_discount literature old 0.05,0.18 fixed    1.5 0.01 1.0 2.0
0.8 10 official_N10_d0.8_seed0_replicate_fixed  50 30 25 0 10 4.0 1159 0.9914 0.9661 0.4 0.99 1.8 seller_discount literature old 0.05,0.18 dr_slope 0.0 0.01 1.6 6.0
1.0 10 official_N10_d1.0_seed2_replicate        50 30 25 2 10 6.0 1354 0.9926 0.9661 0.4 0.99 1.8 seller_discount literature old 0.05,0.18 dr_slope 0.0 0.01 1.6 10.0
```

(density 0.2: w_ahead/w_behind omitted, i.e. the defaults 1.0/2.0.) The
retraining check of 2026-11-02 reran the density-0.6 command and obtained the
official `phi_w.pt` bit for bit.

Calibration of gamma and K: K is half the mean number of plain-DR iterations
on 40 calibration instances and rho = 1 - 10/K. For densities 0.2, 0.4 and 0.6
that calibration dates from an earlier pipeline (2026-10-07) and is below the
DR iteration counts of the current test sets; gamma was chosen by a grid search.
Neither calibration is part of the current code: both values are fixed inputs.

## Out-of-distribution tests

`experiments/demo_zeroshot_levelwidth_oldgen.py <weights> <density> <base_seed>
<gamma> <rho> <scale> <rho_slow> <scale_slow> 30 <pct_level> 0.08 0.05,0.18 <out>`
with pct_level in {0.20, 0.30, 0.40}, and
`experiments/demo_zeroshot_node_pm1_oldgen.py <weights> <density> <rho> <scale>
30 <gamma> <rho_slow> <scale_slow> <base_seed> <out> 0.05,0.18`, with the
values of the DR+GNN table above. The model is now rebuilt from the
configuration stored in its checkpoint (same values).

## L2WS

Centralized (`train_l2ws.py`): `<density> <name> 50 30 50 <base_seed> 10 <gamma>
<unroll> 64 1 3e-3 0.05,0.18 <loss>`, loss weights at their defaults
(gamma_mono 2, w_behind 2, w_ahead 1, dr_slope). Distributed
(`train_l2ws_distributed.py`): `<density> <name> 50 30 50 <base_seed> 10
<gamma> <unroll> 32 3 3e-3 0.05,0.18 <loss> 0 2 2 dr_slope 0.0 true residual`.
The unroll is the DR+GNN K, except 46 at density 0.2. The fp_residual loss
was also run but is not used by the paper and was removed.

## A2DR

`calibrate_eta_mmax.py` over eta in {1e-8, 1e-6, 3e-5, 1e-4} and mmax in
{3, 5, 10} on 10 calibration instances; the chosen setting maximises the mean
speed-up among those converging on every instance (density 1.0: none does,
(3e-5, 5) was chosen by hand). `compare_a2dr_vs_official.py` then ran with the
chosen (eta, mmax) on the 30 test instances.
