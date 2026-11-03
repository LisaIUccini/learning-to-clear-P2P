"""Out-of-distribution test of an official Phi_w, without retraining (paper Sec. 5.3).

    python3 -m scripts.eval_ood --density 0.8 --shift level20   # level shift +/-20% (also level30, level40)
    python3 -m scripts.eval_ood --density 0.8 --shift pm1       # one prosumer more or less

Writes live.log (one line per instance, parsed by the paper scripts) and,
for level shifts, results.png into results/official/ood/d{density}_{shift}/
(or --out).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from l2p2p import constants as C
from l2p2p import official, paths
from l2p2p.evaluation import print_speedup_summary, run_pair, summarize
from l2p2p.plotting import plot_results
from l2p2p.runlog import tee_stdout
from l2p2p.scenarios import draw_level_width, sample_one_node_changes

SHIFTS = ("level20", "level30", "level40", "pm1")
LEVEL_SEED = C.TEST_SEED + 4
PM1_SEED = C.TEST_SEED + 7


def level_shift_pairs(base, model, dr_cfg, device, n_test, pct_level, max_retries=30):
    """Slot 0 is the base market; slots 1..n_test-1 are level+width draws,
    redrawn when infeasible or when plain DR does not converge within the cap
    (so a capped DR count never enters the statistics)."""
    results = [run_pair(base, model, dr_cfg, device, seed=LEVEL_SEED * 7919, record=True)]
    n_infeasible = n_nonconverged = 0
    for i in range(1, n_test):
        for attempt in range(max_retries):
            spec = draw_level_width(base, LEVEL_SEED, i, attempt, pct_level, official.OOD_PCT_WIDTH)
            if spec is None:
                n_infeasible += 1
                continue
            res_dr, res_learned = run_pair(spec, model, dr_cfg, device,
                                           seed=LEVEL_SEED * 7919 + i * 131 + attempt, record=True)
            if not res_dr.converged:
                n_nonconverged += 1
                print(f"  ... instance {i}: draw discarded, DR did not converge "
                      f"(hit {res_dr.n_iter} iters), retrying (attempt {attempt + 1})", flush=True)
                continue
            results.append((res_dr, res_learned))
            print(f"  [ood] {i:2d}/{n_test - 1} kept (attempt {attempt + 1})  "
                  f"DR {res_dr.n_iter:5d} it  |  learned {res_learned.n_iter:5d} it", flush=True)
            break
        else:
            raise RuntimeError(f"instance {i}: no feasible+converged draw found in {max_retries} retries "
                               f"at pct_level={pct_level}")
    print(f"  {n_infeasible} infeasible and {n_nonconverged} DR-non-converged draws discarded", flush=True)
    return results


def run_level_shift(density, model, dr_cfg, device, n_test, pct_level, out_dir):
    base = official.base_market(density)
    results = level_shift_pairs(base, model, dr_cfg, device, n_test, pct_level)
    print("-" * 70)
    dr_iters, learned_iters, speedups = [], [], []
    for i, (res_dr, res_learned) in enumerate(results):
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        dr_iters.append(res_dr.n_iter); learned_iters.append(res_learned.n_iter); speedups.append(sp)
        print(f"  [ood] {i + 1:2d}/{len(results)}  DR {res_dr.n_iter:5d} it  |  "
              f"learned {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)
    print_speedup_summary("ood", speedups)
    test_res = {"dr": dr_iters, "learned": learned_iters, "speedup": speedups,
                "curves_dr": [np.asarray(r.residual) for r, _ in results],
                "curves_learned": [np.asarray(l.residual) for _, l in results]}
    st = summarize(test_res)
    print(f"  [ood] mean % reduction (per-instance average): mean={st['pct_mean']:.1f}%  "
          f"median={st['pct_median']:.1f}%", flush=True)
    subtitle = (f"OOD test: level+width +/-{pct_level:.0%}/+/-{official.OOD_PCT_WIDTH:.0%}, "
                f"density={density}, n={n_test}")
    plot_results(test_res, st, dr_cfg.tol, out_path=os.path.join(out_dir, "results.png"), subtitle=subtitle)


def run_one_node_change(density, model, dr_cfg, device, n_test):
    base = official.base_market(density)
    specs, ops = sample_one_node_changes(base, official.market_config(density), n_test, seed=PM1_SEED)
    print(f"  {len(specs)}/{n_test} valid +/-1 instances found "
          f"({ops.count('add')} add, {ops.count('remove')} remove)", flush=True)
    print("-" * 70)
    speedups, pcts = [], []
    for i, (op, spec) in enumerate(zip(ops, specs)):
        res_dr, res_learned = run_pair(spec, model, dr_cfg, device, seed=C.TEST_SEED * 7919 + i, record=False)
        sp = res_dr.n_iter / max(res_learned.n_iter, 1)
        pct = (res_dr.n_iter - res_learned.n_iter) / max(res_dr.n_iter, 1) * 100.0
        speedups.append(sp); pcts.append(pct)
        print(f"  [{op:6s}]  N={spec.N}  DR {res_dr.n_iter:5d} it  |  "
              f"learned {res_learned.n_iter:5d} it  |  x{sp:5.2f} ({pct:+.1f}%)", flush=True)
    print_speedup_summary("pm1", speedups)
    print(f"  [pm1] mean % reduction (per-instance average): mean={np.mean(pcts):.1f}%  "
          f"median={np.median(pcts):.1f}%", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", type=float, required=True, choices=official.DENSITIES)
    ap.add_argument("--shift", required=True, choices=SHIFTS)
    ap.add_argument("--n-test", type=int, default=official.N_TEST)
    ap.add_argument("--out", help="output folder (default: the official one)")
    args = ap.parse_args()

    out_dir = args.out or str(paths.OOD / f"{paths.density_tag(args.density)}_{args.shift}")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cpu")
    dr_cfg = official.dr_config(args.density)
    model = official.load_official_model(args.density)
    with tee_stdout(os.path.join(out_dir, "live.log")):
        print(f"OOD test ({args.shift}) of {official.dr_gnn_dir(args.density)}/phi_w.pt  "
              f"(density={args.density}  n_test={args.n_test}  dr_gamma={dr_cfg.gamma}  "
              f"DR_MAX_ITER={dr_cfg.n_iter})", flush=True)
        if args.shift == "pm1":
            run_one_node_change(args.density, model, dr_cfg, device, args.n_test)
        else:
            pct_level = int(args.shift[len("level"):]) / 100
            run_level_shift(args.density, model, dr_cfg, device, args.n_test, pct_level, out_dir)


if __name__ == "__main__":
    main()
