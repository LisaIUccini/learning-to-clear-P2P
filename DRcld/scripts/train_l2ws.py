"""Learned-warm-start (L2WS) benchmark for one density, variant and loss.

    python3 -m scripts.train_l2ws --density 0.8 --variant centralized --loss asymmetric
    python3 -m scripts.train_l2ws --density 0.8 --variant distributed --loss regression
    python3 -m scripts.train_l2ws --density 0.8 --variant centralized --loss asymmetric --eval-only

centralized: MLP over all market parameters (needs a coordinator);
distributed: message-passing GNN with L = 3 rounds, added to its iteration count.
Writes to results/official/l2ws/{variant}/d{density}_{loss}/ (or --out):
h_w.pt / h_w_dist.pt, train.log, training_loss.png, results.png, NOTES.md
and, for the centralized variant, instance_plots/.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from l2p2p import constants as C
from l2p2p import market, official, paths
from l2p2p.evaluation import evaluate, summarize, write_notes
from l2p2p.l2ws.centralized_model import L2WSPredictor, ThetaNormalizer, WarmStartConfig, WarmStartMLP, \
    build_theta, theta_dim
from l2p2p.l2ws.distributed_model import DistributedPredictor, DistributedWarmStartConfig, \
    DistributedWarmStartGNN, NodeEdgeNormalizer
from l2p2p.l2ws.training import LOSSES, fit_warm_start, run_pair
from l2p2p.plotting import plot_results, plot_training_loss
from l2p2p.runlog import tee_stdout
from l2p2p.training import TrainConfig

VARIANTS = ("centralized", "distributed")
CHECKPOINT = {"centralized": "h_w.pt", "distributed": "h_w_dist.pt"}


def output_dir(variant, density, loss):
    root = paths.L2WS_CENTRALIZED if variant == "centralized" else paths.L2WS_DISTRIBUTED
    return root / f"{paths.density_tag(density)}_{loss}"


def build_predictor(variant, base, train_specs, device):
    """A fresh, untrained predictor. Both constructions draw networks from
    torch's global random state before initialising the weights; that order
    is part of what reproduces the official runs."""
    if variant == "centralized":
        out_dim = market.build_network(base, device=device).dim
        normalizer = ThetaNormalizer([build_theta(s) for s in train_specs])
        D_mid = (base.D_min + base.D_max) / 2.0
        G_mid = (base.G_min + base.G_max) / 2.0
        init_point = torch.from_numpy(np.concatenate(
            [D_mid, G_mid, np.zeros(out_dim - 2 * base.N)]).astype(np.float32))
        cfg = official.L2WS_CENTRALIZED
        mlp = WarmStartMLP(theta_dim(base), out_dim,
                           WarmStartConfig(hidden_dim=cfg["hidden_dim"], n_hidden_layers=cfg["n_hidden_layers"]),
                           init_point=init_point)
        return L2WSPredictor(mlp, normalizer, device=device), mlp
    normalizer = NodeEdgeNormalizer(train_specs, device=device)
    cfg = official.L2WS_DISTRIBUTED
    gnn = DistributedWarmStartGNN(DistributedWarmStartConfig(
        hidden_dim=cfg["hidden_dim"], n_rounds=cfg["n_rounds"], shared_weights=cfg["shared_weights"],
        update_type=cfg["update_type"]))
    return DistributedPredictor(gnn, normalizer, device=device), gnn


def train(variant, density, loss, out_dir, n_train=official.N_TRAIN, epochs=official.L2WS_EPOCHS,
          unroll=None, n_val=official.N_VAL):
    """Train a predictor and save its checkpoint + training_loss.png. The
    validation loss is only monitored: `n_val` does not change the weights."""
    d = official.DENSITY[density]
    cfg = official.L2WS_CENTRALIZED if variant == "centralized" else official.L2WS_DISTRIBUTED
    unroll = unroll or d.l2ws_unroll
    device = torch.device("cpu")
    torch.manual_seed(C.TRAIN_SEED)
    train_specs, val_specs, _ = official.datasets(density, n_train=n_train, n_val=n_val, n_test=1)
    print(f"L2WS {variant}: density={density}  loss_mode={loss}  gamma={d.gamma}  unroll={unroll}  "
          f"n_train={len(train_specs)}  epochs={epochs}  base_seed={d.base_seed}  "
          + "  ".join(f"{k}={v}" for k, v in cfg.items()), flush=True)
    predictor, module = build_predictor(variant, train_specs[0], train_specs, device)
    dr_cfg = official.dr_config(density)
    train_cfg = TrainConfig(gamma=d.gamma, lam=C.DR_LAMBDA, bisect=C.DR_BISECT_STEPS, unroll=unroll)
    history = fit_warm_start(predictor, module, train_specs, val_specs, train_cfg, epochs, cfg["lr"],
                             grad_clip=C.TRAIN_GRAD_CLIP, seed=C.TRAIN_SEED, device=device, loss_mode=loss,
                             dr_cfg=dr_cfg, gamma_mono=cfg["gamma_mono"], w_behind=cfg["w_behind"],
                             w_ahead=cfg["w_ahead"], omega_mode=cfg["omega_mode"], omega_power=cfg["omega_power"])
    predictor.save(os.path.join(out_dir, CHECKPOINT[variant]))
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))
    return predictor


def evaluate_predictor(variant, density, predictor, out_dir, n_test=official.N_TEST):
    """Cold-start DR vs warm-started DR on the test set. The centralized
    variant also saves instance_plots/ (read by the paper figures)."""
    dr_cfg = official.dr_config(density)
    _, _, test_specs = official.datasets(density, n_train=1, n_test=n_test)
    label = f"l2ws/d{density}" if variant == "centralized" else f"l2ws-dist/d{density}"
    records = os.path.join(out_dir, "instance_plots") if variant == "centralized" else None
    print("\nevaluating: cold-start plain DR vs plain DR warm-started by h_w(theta)", flush=True)
    print("-" * 70)
    test_res = evaluate(test_specs, predictor, dr_cfg, torch.device("cpu"), seed=C.TEST_SEED, label=label,
                        pair_runner=run_pair, method_name="L2WS", records_dir=records)
    st = summarize(test_res)
    print(f"  mean % reduction (per-instance average): mean={st['pct_mean']:.1f}%  "
          f"median={st['pct_median']:.1f}%", flush=True)
    plot_results(test_res, st, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"),
                 subtitle=f"L2WS ({variant}), density={density}  |  tested on {len(test_specs)}")
    write_notes(out_dir, f"""\
# L2WS benchmark ({variant}), density {density}

Produced by scripts/train_l2ws.py with the configuration in l2p2p/official.py.
Baseline: cold-start plain DR from a fresh random x0; the warm start replaces x0.

Results: mean reduction {st['pct_mean']:.1f}%, median {st['pct_median']:.1f}%, win-rate {st['win_rate']:.0%}.
""")
    return test_res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", type=float, required=True, choices=official.DENSITIES)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--loss", required=True, choices=LOSSES)
    ap.add_argument("--out", help="output folder (default: the official one)")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the checkpoint already in the folder")
    ap.add_argument("--n-train", type=int, default=official.N_TRAIN)
    ap.add_argument("--n-test", type=int, default=official.N_TEST)
    ap.add_argument("--epochs", type=int, default=official.L2WS_EPOCHS)
    ap.add_argument("--unroll", type=int)
    args = ap.parse_args()

    out_dir = args.out or str(output_dir(args.variant, args.density, args.loss))
    os.makedirs(out_dir, exist_ok=True)
    with tee_stdout(os.path.join(out_dir, "eval.log" if args.eval_only else "train.log")):
        if args.eval_only:
            path = os.path.join(out_dir, CHECKPOINT[args.variant])
            predictor = (L2WSPredictor.load(path, official.base_market(args.density))
                         if args.variant == "centralized" else DistributedPredictor.load(path))
        else:
            predictor = train(args.variant, args.density, args.loss, out_dir, n_train=args.n_train,
                              epochs=args.epochs, unroll=args.unroll)
        evaluate_predictor(args.variant, args.density, predictor, out_dir, n_test=args.n_test)
        print(f"\nDONE -- {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
