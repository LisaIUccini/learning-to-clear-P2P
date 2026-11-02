"""Regression tests: the code reproduces the official results and the
outputs of the pre-refactoring scripts.

- inference: the first test instance of every density gives the same
  iteration counts and bit-identical trajectories as the saved official runs;
- training, OOD tests and benchmarks: reduced runs (3 training instances,
  1 epoch, K = 12, 1-2 test instances) give bit-identical weights and the same
  iteration counts as reference runs of the original scripts
  (tests/reference/*.json, digests only).

All the expensive work runs once, in parallel, in the `jobs` fixture.
"""

import contextlib
import io
import json
import os
import re
import tempfile
from multiprocessing import Pool

import pytest
import torch

from digests import checkpoint_digest, json_digest, state_dict_digest
from l2p2p import constants as C
from l2p2p import official
from l2p2p.evaluation import run_pair
from scripts import eval_ood, run_a2dr, train_dr_gnn, train_l2ws

REF_DR_GNN = json.load(open("tests/reference/mini_dr_gnn.json"))
REF_BENCH = json.load(open("tests/reference/mini_benchmarks.json"))
D = 0.2                                  # density of the reduced runs
MINI = dict(n_train=3, epochs=1, unroll=12, n_val=1)
L2WS_CASES = [("centralized", "asymmetric"), ("centralized", "regression"),
              ("distributed", "asymmetric"), ("distributed", "regression")]


def _pairs(text, pattern):
    return [[int(a), int(b)] for a, b in re.findall(pattern, text)]


def _official_first_instance(density, tmp):
    _, _, test_specs = official.datasets(density, n_train=1, n_test=1)
    res_dr, res_l = run_pair(test_specs[0], official.load_official_model(density), official.dr_config(density),
                             torch.device("cpu"), seed=C.TEST_SEED * 7919, record=True)
    return {"k": (res_dr.n_iter, res_l.n_iter),
            "dr": (res_dr.residual, res_dr.cost, res_dr.violation),
            "learned": (res_l.residual, res_l.cost, res_l.violation)}


def _dr_gnn_mini(density, evaluate, out_dir):
    model = train_dr_gnn.train_model(density, out_dir, **MINI)
    out = {"weights": state_dict_digest(os.path.join(out_dir, "phi_w.pt"))}
    if evaluate:
        train_dr_gnn.evaluate_model(density, model, out_dir, n_test=1)
        out["instance_01"] = json_digest(os.path.join(out_dir, "instance_plots", "instance_01_N10.json"))
    return out


def _ood(shift, out_dir):
    """(DR, DR+GNN) iteration counts of the first two OOD instances, from the printed log."""
    model = official.load_official_model(D)
    dr_cfg = official.dr_config(D)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        if shift == "pm1":
            eval_ood.run_one_node_change(D, model, dr_cfg, torch.device("cpu"), n_test=2)
        else:
            eval_ood.run_level_shift(D, model, dr_cfg, torch.device("cpu"), n_test=2, pct_level=0.20,
                                     out_dir=out_dir)
    pattern = (r"N=\d+\s+DR\s+(\d+)\s+it\s+\|\s+learned\s+(\d+)\s+it" if shift == "pm1"
               else r"\[ood\]\s+\d+/2\s+DR\s+(\d+)\s+it\s+\|\s+learned\s+(\d+)\s+it")
    return _pairs(log.getvalue(), pattern)


def _l2ws(variant, loss, out_dir):
    predictor = train_l2ws.train(variant, D, loss, out_dir, **MINI)
    res = train_l2ws.evaluate_predictor(variant, D, predictor, out_dir, n_test=1)
    return {"checkpoint": checkpoint_digest(os.path.join(out_dir, train_l2ws.CHECKPOINT[variant])),
            "first_pair": [res["dr"][0], res["learned"][0]]}


def _run(job):
    """One job in a worker process: single thread, own temporary folder, quiet."""
    name, fn, args = job
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        return name, fn(*args, tmp)


def _a2dr_compare(tmp):
    return run_a2dr.compare(D, tmp, n_test=1)


def _a2dr_calibrate(tmp):
    return run_a2dr.calibrate(D, 1e-4, 3, os.path.join(tmp, "calib.json"), n_val=1)


@pytest.fixture(scope="module")
def jobs():
    todo = [(f"official_{d}", _official_first_instance, (d,)) for d in official.DENSITIES]
    todo += [("dr_gnn_d02", _dr_gnn_mini, (0.2, True)), ("dr_gnn_d08", _dr_gnn_mini, (0.8, False))]
    todo += [("ood_level20", _ood, ("level20",)), ("ood_pm1", _ood, ("pm1",))]
    todo += [(f"l2ws_{v}_{l}", _l2ws, (v, l)) for v, l in L2WS_CASES]
    todo += [("a2dr_compare", _a2dr_compare, ()), ("a2dr_calibrate", _a2dr_calibrate, ())]
    with Pool(len(todo)) as pool:
        return dict(pool.map(_run, todo))


@pytest.mark.parametrize("density", official.DENSITIES)
def test_inference_matches_official_runs(jobs, density):
    saved = json.load(open(official.dr_gnn_dir(density) / "instance_plots" / "instance_01_N10.json"))
    got = jobs[f"official_{density}"]
    assert got["k"] == (saved["k_DR"], saved["k_learned"])
    for key in ("dr", "learned"):
        residual, cost, violation = got[key]
        assert residual == saved[key]["residual"]
        assert cost == saved[key]["cost"]
        assert violation == saved[key]["violation"]


@pytest.mark.parametrize("tag", ["d02", "d08"])
def test_dr_gnn_mini_training(jobs, tag):
    """0.2: fixed iteration weights, no slow channel; 0.8: dr_slope weights, slow channel."""
    assert jobs[f"dr_gnn_{tag}"]["weights"] == REF_DR_GNN[f"dr_gnn_{tag}"]["weights"]


def test_dr_gnn_mini_evaluation_record(jobs):
    assert jobs["dr_gnn_d02"]["instance_01"] == REF_DR_GNN["dr_gnn_d02"]["instances"]["instance_01_N10.json"]


@pytest.mark.parametrize("shift,ref", [("level20", "ood_level20_d0.2"), ("pm1", "ood_pm1_d0.2")])
def test_ood(jobs, shift, ref):
    assert jobs[f"ood_{shift}"] == REF_BENCH[ref][:2]


@pytest.mark.parametrize("variant,loss", L2WS_CASES)
def test_l2ws_mini_training(jobs, variant, loss):
    """Identical checkpoint (weights and normaliser statistics) and first test instance."""
    got, ref = jobs[f"l2ws_{variant}_{loss}"], REF_BENCH[f"l2ws_{variant}_{loss}"]
    assert got["checkpoint"] == ref["checkpoint"]
    assert got["first_pair"] == ref["iters"][0]


def test_a2dr_compare(jobs):
    ref = REF_BENCH["a2dr_compare_d0.2"]
    got = jobs["a2dr_compare"]
    for key in ("dr_iters", "a2dr_iters", "learned_iters"):
        assert got[key] == ref[key][:1]
    assert (got["trading_graph_diameter"], got["a2dr_rounds_per_iter"]) == \
        (ref["trading_graph_diameter"], ref["a2dr_rounds_per_iter"])


def test_a2dr_calibrate(jobs):
    ref = REF_BENCH["a2dr_calibrate_d0.2_eta1e-4_mmax3"]
    got = jobs["a2dr_calibrate"]
    assert (got["dr_iters"], got["a2dr_iters"]) == (ref["dr_iters"][:1], ref["a2dr_iters"][:1])
