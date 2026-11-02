# Learning to Clear Peer-to-Peer Electricity Markets

Code of the paper: a distributed Douglas–Rachford (DR) market-clearing
iteration augmented with a learned, summable correction parametrised by a
graph neural network (DR+GNN), its offline training, and the benchmarks of
Section 5 (learned warm starts, Anderson-accelerated DR).

Python 3.10, CPU only. Every command below runs from this folder.

```
pip install -r requirements.txt
python3 -m pytest                      # ~3 min: regression + parity + paper tests
```

## Layout

```
l2p2p/                  library
  market.py             random base markets (paper Table 1) and Network construction
  scenarios.py          instances derived from a base market: level/width shifts, +/-1 prosumer
  network.py            Prosumer (one node of Algorithm 1) and Network (vectorised mirror for training)
  fixed_point.py        the base operator (DouglasRachford) and the iteration loop (run_dr)
  gnn.py                the learned correction Phi_w and its deployment on the prosumers
  training.py           unrolled rollout, asymmetric loss (Eq. 16), Trainer
  evaluation.py         DR vs accelerated method from the same x0, per-instance records
  a2dr.py               Anderson-accelerated DR (benchmark)
  l2ws/                 learned warm-start benchmark (centralised MLP, distributed GNN)
  equilibrium.py        exact v-GNE through one QP (reference J*, market statistics)
  official.py           configuration of every official run, per density
  constants.py, paths.py, plotting.py, runlog.py
scripts/                command-line entry points (one per experiment, see below)
  analysis/             re-evaluation with the unperturbed stopping rule (2026-10-28)
paper/                  figure and table scripts, outputs in paper_figs_official/, paper_tables_official/
results/official/       the official results (models, logs, per-instance records)
tests/                  regression, parity and paper tests
docs/official_runs.md   provenance of the official runs
```

## Replacing Douglas–Rachford

`fixed_point.py` separates the base operator from the learned correction.
Another distributed fixed-point method can be used by implementing the two
methods of `DouglasRachford`:

- `local_update(prosumer) -> (D, G, {m: q_m})`: one node's update, from its own
  state and the trades received in this round (inference, `run_fixed_point`);
- `centralised_update(net, x, xbar) -> x`: the same map for the whole network,
  differentiable (training, `training.rollout`).

`run_fixed_point(net, operator, cfg)` then runs it with or without a deployed
correction.

## Reproducing the results

| what | command | time |
|---|---|---|
| figures and tables, from the saved results | `python3 paper/make_all.py` | 1 min |
| DR+GNN, one density (train + test) | `python3 -m scripts.train_dr_gnn --density 0.8` | 0.5–3 h |
| DR+GNN, test only (saved model) | `python3 -m scripts.train_dr_gnn --density 0.8 --eval-only` | minutes |
| out-of-distribution tests | `python3 -m scripts.eval_ood --density 0.8 --shift level20` (`level30`, `level40`, `pm1`) | minutes |
| L2WS benchmark | `python3 -m scripts.train_l2ws --density 0.8 --variant centralized --loss asymmetric` (`--eval-only` to test only) | hours |
| A2DR benchmark | `python3 -m scripts.run_a2dr calibrate --density 0.8`, then `compare` | hours |
| extra data of some figures/tables | `python3 -m scripts.paper_data trajectories` (`plausibility`, `timing`, `train-time`, `stopping-quality`) | varies |
| everything, stage by stage | `python3 -m scripts.reproduce {inference,training,paper-data,paper} [--dry-run]` | |

Each script writes into its folder of `results/official/` unless `--out`
is given. Training is deterministic on CPU: retraining density 0.6 with the
original script reproduced the official weights bit for bit.

The hyperparameters of every run are in `l2p2p/official.py`. The DR step size
and the training horizon K are fixed inputs there: they were calibrated
earlier, partly with a previous pipeline, and that calibration is not part of
this code (see `docs/official_runs.md`).
