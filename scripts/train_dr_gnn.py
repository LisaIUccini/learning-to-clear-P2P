"""Train Phi_w for one density and evaluate it on the in-distribution test set.

    python3 -m scripts.train_dr_gnn --density 0.8               # train + evaluate
    python3 -m scripts.train_dr_gnn --density 0.8 --eval-only   # evaluate the saved phi_w.pt

Writes to results/official/dr_gnn/d{density}/ (or --out): phi_w.pt,
train.log, training_loss.png, results.png, NOTES.md and instance_plots/
(one JSON + PNG per test instance; the paper figures read the JSONs).
The --n-*/--epochs/--unroll overrides exist for quick checks only.
"""

from __future__ import annotations

import argparse
import os
import statistics as stats

import torch

from l2p2p import constants as C
from l2p2p import official
from l2p2p.evaluation import evaluate, summarize, write_notes
from l2p2p.gnn import PerturbationGNN
from l2p2p.plotting import plot_results, plot_training_loss
from l2p2p.runlog import tee_stdout
from l2p2p.training import TrainConfig, Trainer


def train_model(density, out_dir, n_train=official.N_TRAIN, epochs=official.EPOCHS, unroll=None,
                n_val=official.N_VAL):
    """Train Phi_w and save phi_w.pt + training_loss.png into `out_dir`. The
    validation loss is only monitored: `n_val` does not change the weights."""
    d = official.DENSITY[density]
    unroll = unroll or d.unroll
    device = torch.device("cpu")
    # Seeds the initialisation of Phi_w below; Trainer.fit reseeds before training.
    torch.manual_seed(C.TRAIN_SEED)
    dr_cfg = official.dr_config(density)
    train_cfg = TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect, unroll=unroll,
                            gamma_mono=C.TRAIN_GAMMA_MONO, epochs=epochs,
                            w_behind=d.w_behind, w_ahead=d.w_ahead, lr=C.TRAIN_LR,
                            lr_final_frac=official.LR_FINAL_FRAC, seed=C.TRAIN_SEED,
                            omega_mode=d.omega_mode, omega_power=d.omega_power)
    train_specs, val_specs, _ = official.datasets(density, n_train=n_train, n_val=n_val, n_test=1)

    print(f"training Phi_w: density={density}  gamma*={d.gamma}  unroll={unroll}  gnn_rho={d.rho}  "
          f"scale={d.scale}  n_train={len(train_specs)}  n_val={len(val_specs)}  epochs={epochs}  "
          f"base_seed={d.base_seed}  omega_mode={d.omega_mode}  omega_power={d.omega_power}  "
          f"w_ahead={d.w_ahead}  w_behind={d.w_behind}  lr_final_frac={official.LR_FINAL_FRAC}", flush=True)
    print(f"slow channel: scale_slow={d.scale_slow}  rho_slow={official.RHO_SLOW}", flush=True)
    model = PerturbationGNN(official.gnn_config(density))
    trainer = Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=val_specs, device=device)
    trainer.save(os.path.join(out_dir, "phi_w.pt"))
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"))
    return model


def evaluate_model(density, model, out_dir, n_test=official.N_TEST):
    """Plain DR vs DR+Phi_w on the test set; writes instance_plots/, results.png, NOTES.md."""
    d = official.DENSITY[density]
    dr_cfg = official.dr_config(density)
    _, _, test_specs = official.datasets(density, n_train=1, n_test=n_test)
    print("\nevaluating: plain DR vs DR + Phi_w, in-distribution test set", flush=True)
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, torch.device("cpu"), seed=C.TEST_SEED,
                        label=f"d{density}", records_dir=os.path.join(out_dir, "instance_plots"))
    st = summarize(test_res)
    print(f"  mean % reduction (per-instance average): mean={st['pct_mean']:.1f}%  "
          f"median={st['pct_median']:.1f}%", flush=True)
    subtitle = f"density={density}, gamma={d.gamma}, rho={d.rho}  |  tested on {len(test_specs)} instances"
    plot_results(test_res, st, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)
    write_notes(out_dir, f"""\
# DR+GNN, density {density}

Produced by scripts/train_dr_gnn.py with the configuration in l2p2p/official.py.
Test set: {len(test_specs)} in-distribution instances, fresh random x0 per instance.

Results: mean speed-up {stats.mean(test_res['speedup']):.2f}x, median {stats.median(test_res['speedup']):.2f}x,
mean reduction {st['pct_mean']:.1f}%, median {st['pct_median']:.1f}%, win-rate {st['win_rate']:.0%}.
""")
    return test_res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", type=float, required=True, choices=official.DENSITIES)
    ap.add_argument("--out", help="output folder (default: the official one)")
    ap.add_argument("--eval-only", action="store_true", help="evaluate the phi_w.pt already in the folder")
    ap.add_argument("--n-train", type=int, default=official.N_TRAIN)
    ap.add_argument("--n-test", type=int, default=official.N_TEST)
    ap.add_argument("--epochs", type=int, default=official.EPOCHS)
    ap.add_argument("--unroll", type=int, help="training horizon K (default: the official one)")
    args = ap.parse_args()

    out_dir = args.out or str(official.dr_gnn_dir(args.density))
    os.makedirs(out_dir, exist_ok=True)
    log_name = "eval.log" if args.eval_only else "train.log"
    with tee_stdout(os.path.join(out_dir, log_name)):
        if args.eval_only:
            model = official.load_model(os.path.join(out_dir, "phi_w.pt"))
        else:
            model = train_model(args.density, out_dir, n_train=args.n_train, epochs=args.epochs,
                                unroll=args.unroll)
        evaluate_model(args.density, model, out_dir, n_test=args.n_test)
        print(f"\nDONE -- {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
