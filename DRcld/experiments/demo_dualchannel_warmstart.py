"""
demo_dualchannel_warmstart.py
==============================

Test the slow channel WITHOUT risking the already-validated fast channel:
warm-start from a known-good fast-only checkpoint (its slow-channel heads
are still at random init -- scale_slow=0 during that run gave them zero
gradient), FREEZE the shared encoder (phi_msg, phi_upd) and the fast
channel's own heads, and train ONLY the slow channel's heads
(phi_node_slow, phi_edge_slow, phi_gate_slow) with a longer unroll so
training can actually observe the plateau region the fast channel can't
reach. Because the two channels are purely ADDITIVE and the fast channel
is frozen, its contribution is mathematically identical to the baseline
checkpoint's -- the slow channel can only ADD to it, never corrupt it.

Usage: python3 demo_dualchannel_warmstart.py <run_name> <checkpoint_path>
       <base_seed> <density> <gamma> <unroll> <rho_fast> <scale_fast>
       <rho_slow> <scale_slow> [epochs] [n_train] [n_test] [eta]
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import sys

import numpy as np
import torch

import config
import market
import algorithm
import gnn as gnn_mod
import train as train_mod
from diagnostics import save_instance_plots
from demo import run_pair, evaluate, plot_results, plot_training_loss, write_run_notes


def main():
    run_name = sys.argv[1]
    ckpt_path = sys.argv[2]
    base_seed = int(sys.argv[3])
    density = float(sys.argv[4])
    dr_gamma = float(sys.argv[5])
    unroll = int(sys.argv[6])
    rho_fast = float(sys.argv[7])
    scale_fast = float(sys.argv[8])
    rho_slow = float(sys.argv[9])
    scale_slow = float(sys.argv[10])
    epochs = int(sys.argv[11]) if len(sys.argv) > 11 else 30
    n_train = int(sys.argv[12]) if len(sys.argv) > 12 else config.N_TRAIN_INSTANCES
    n_test = int(sys.argv[13]) if len(sys.argv) > 13 else config.N_TEST_INSTANCES
    eta = float(sys.argv[14]) if len(sys.argv) > 14 else config.TRAIN_ETA

    out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"writing all artifacts to {out_dir}/  (warm-start from {ckpt_path})")
    print(f"  base_seed={base_seed}  density={density}  gamma={dr_gamma}  unroll={unroll}  "
          f"rho_fast={rho_fast}  scale_fast={scale_fast}  rho_slow={rho_slow}  "
          f"scale_slow={scale_slow}  epochs={epochs}  n_train={n_train}  n_test={n_test}  eta={eta}")

    device = torch.device("cpu")
    torch.manual_seed(config.TRAIN_SEED)

    mc = market.MarketConfig(density=density)
    dr_cfg = algorithm.DRConfig(gamma=dr_gamma, lam=config.DR_LAMBDA,
                                 n_iter=config.DR_MAX_ITER, tol=config.DR_TOLERANCE,
                                 bisect=config.DR_BISECT_STEPS)
    train_cfg = train_mod.TrainConfig(gamma=dr_cfg.gamma, lam=dr_cfg.lam, bisect=dr_cfg.bisect,
                                       unroll=unroll, gamma_mono=config.TRAIN_GAMMA_MONO,
                                       eta=eta, epochs=epochs, w_behind=config.TRAIN_W_BEHIND,
                                       w_ahead=config.TRAIN_W_AHEAD, lr=config.TRAIN_LR,
                                       grad_clip=config.TRAIN_GRAD_CLIP, seed=config.TRAIN_SEED,
                                       loss_mode="hinge")

    _, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=base_seed)
    print(f"  N={base_spec.N}  edges={sum(len(v) for v in base_spec.neighbors.values()) // 2}")
    train_specs = [base_spec] * n_train
    test_specs = [base_spec] * n_test

    model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
        mc, rho=rho_fast, scale=scale_fast, rho_slow=rho_slow, scale_slow=scale_slow))
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=True)
    print(f"  loaded checkpoint: missing={missing} unexpected={unexpected}")

    trainable_prefixes = ("phi_node_slow", "phi_edge_slow", "phi_gate_slow")
    n_frozen, n_trainable = 0, 0
    for name, p in model.named_parameters():
        if name.startswith(trainable_prefixes):
            p.requires_grad_(True)
            n_trainable += p.numel()
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"  frozen params: {n_frozen}   trainable (slow-channel-only) params: {n_trainable}")

    print(f"\ncontinuing training: ONLY slow channel, everything else frozen "
          f"(loss_mode={train_cfg.loss_mode!r})")
    trainer = train_mod.Trainer(model, train_cfg)
    history = trainer.fit(train_specs, val_specs=test_specs[:5], device=device)

    print("\nevaluating: plain DR vs DR + Phi_w (frozen-fast + trained-slow), fully distributed inference")
    print("-" * 70)
    test_res = evaluate(test_specs, model, dr_cfg, device, seed=config.TEST_SEED, label="warmstart")

    print("\nrecording residual curves on the test set...")
    curves_dr, curves_learned = [], []
    for i, spec in enumerate(test_specs):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=config.TEST_SEED * 7919 + i, record=True)
        curves_dr.append(np.asarray(res_dr.residual))
        curves_learned.append(np.asarray(res_learned.residual))
    test_res["curves_dr"] = curves_dr
    test_res["curves_learned"] = curves_learned

    subtitle = (f"warm-start from {os.path.basename(os.path.dirname(ckpt_path))}, fast channel FROZEN, "
                f"slow channel trained: rho_slow={rho_slow}, scale_slow={scale_slow}, unroll={unroll}, "
                f"{n_train} instances, {epochs} epochs  |  tested on: same market, n={n_test}")
    plot_results(test_res, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)
    plot_training_loss(history, out_path=os.path.join(out_dir, "training_loss.png"), subtitle=subtitle)

    print("\ngenerating per-instance diagnostic plots...")
    save_instance_plots(test_specs, model, dr_cfg, device, seed=config.TEST_SEED,
                         out_dir=os.path.join(out_dir, "instance_plots"), run_pair_fn=run_pair)

    weights_path = os.path.join(out_dir, "phi_w.pt")
    trainer.save(weights_path)
    print(f"saved trained weights to {weights_path}")

    write_run_notes(out_dir, f"""\
# {run_name}

**Test**: warm-start dual-channel experiment. Model initialised from
`{ckpt_path}` (fast channel + shared encoder already trained, slow-channel
heads at random init since scale_slow=0 gave them zero gradient in that
run). Shared encoder (phi_msg, phi_upd) and fast-channel heads (phi_node,
phi_edge, phi_gate) FROZEN (requires_grad=False) for this run -- only the
slow-channel heads (phi_node_slow, phi_edge_slow, phi_gate_slow) are
trained. Because the two channels are additive and the fast channel's
weights are unchanged, its own contribution is mathematically identical to
the checkpoint's; only the slow channel's ADDITIVE contribution differs.

**Config**: density={density}, base_seed={base_seed}, gamma={dr_gamma},
unroll={unroll}, rho_fast={rho_fast}, scale_fast={scale_fast},
rho_slow={rho_slow}, scale_slow={scale_slow}, epochs={epochs}, eta={eta},
{n_train} train instances, {n_test} test instances.
""")


if __name__ == "__main__":
    main()
