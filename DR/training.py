"""
training.py
===========

Centralised training of the perturbation model Phi_theta.

Why centralised
---------------
Back-propagation has to flow through the whole unrolled trajectory, so the
optimiser needs the entire state at once. Training is therefore a one-off,
offline, centralised procedure. What comes out of it is a single set of
weights theta, which is then copied to every prosumer (`gnn.deploy`) and
used in the fully distributed loop. Nothing about deployment is centralised.

The rollout used here MIRRORS the distributed one exactly
---------------------------------------------------------
Same order, same information, same amplitude schedule:

    v^k   = scale * rho^k * Phi_theta(x^k)     <- features from the PRE-update
    xbar  = proj_{X^rec}(x)                       state, i.e. exactly what a
    refl  = 2*xbar - x                            prosumer holds after line 3
    z     = prox_{gamma J}(refl)
    x^{k+1} = x + lam*(z - xbar) + v^k

The only difference is bookkeeping: here the per-node prox is evaluated by
the vectorised, differentiable `EnergyNetwork.prox_local`, whereas each
prosumer uses its own `Prosumer.local_prox`. The two compute the same
quantity -- a property that is verified numerically, because if it drifted
we would be training one system and deploying another.

Loss (self-supervised, no labels)
---------------------------------
The target is simply "make the residual fall fast". Writing
r_k = ||xbar^{k+1} - xbar^k||, the default objective is

    L = (1/K) * sum_k log(r_k + eps)

The logarithm makes the loss scale-free: residuals span many orders of
magnitude along a trajectory, and without it the first iterations would
dominate the gradient entirely. No knowledge of the equilibrium is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import time as _time

import numpy as np
import torch

from energy_network import EnergyNetwork
from gnn import PerturbationGNN, deploy, undeploy


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class TrainConfig:
    # --- solver settings the model is trained against ---
    # Tune the BASELINE first: accelerating a badly tuned DR proves nothing.
    gamma: float = 16.0
    lam: float = 1.8
    prox_bisect: int = 30

    # --- unrolling ---
    unroll: int = 30            # K: iterations differentiated through
    rho: float = 0.9            # decay of the perturbation (summability)
    scale: float = 1.0          # amplitude

    # --- optimisation ---
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    loss_type: str = "log_ratio"      # log_ratio | log_residual | rel_residual | final
    eps: float = 1e-12

    # --- misc ---
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True

    def __post_init__(self):
        if self.loss_type not in ("log_ratio", "log_residual",
                                  "rel_residual", "final"):
            raise ValueError(f"unknown loss_type {self.loss_type!r}")
        if not (0.0 < self.rho < 1.0):
            raise ValueError("rho must lie in (0,1)")


# ======================================================================
# Reporting
# ======================================================================
class Reporter:
    """Structured console log: which phase, on what, with which settings.

    Set `enabled=False` to silence it entirely.
    """

    WIDTH = 68

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._t0 = _time.time()
        self._phase_t0 = self._t0

    # --- primitives ---
    def _p(self, text: str = "") -> None:
        if self.enabled:
            print(text, flush=True)

    def rule(self, char: str = "-") -> None:
        self._p(char * self.WIDTH)

    def phase(self, index: int, total: int, title: str) -> None:
        self._phase_t0 = _time.time()
        self._p("")
        self.rule("=")
        self._p(f"  L2Play  |  phase {index}/{total}  |  {title.upper()}")
        self.rule("=")

    def kv(self, key: str, value: str) -> None:
        self._p(f"  {key:<14}: {value}")

    def line(self, text: str) -> None:
        self._p(f"  {text}")

    def done(self, text: str = "") -> None:
        dt = _time.time() - self._phase_t0
        extra = f" | {text}" if text else ""
        self._p(f"  {'done':<14}: {dt:.1f}s{extra}")

    def elapsed(self) -> float:
        return _time.time() - self._t0

    # --- domain-specific blocks ---
    def dataset(self, name: str, ds) -> None:
        s = ds.summary()
        self.kv(name, f"{s['instances']} instances | N={s['N'][0]}..{s['N'][1]} "
                      f"| undirected edges {s['undirected_edges'][0]}.."
                      f"{s['undirected_edges'][1]}")

    def setup(self, gnn, cfg: "TrainConfig", n_instances: int) -> None:
        """Everything the run depends on, printed once."""
        k99 = int(np.ceil(np.log(0.01) / np.log(cfg.rho)))
        self.kv("baseline DR", f"gamma={cfg.gamma}  lambda={cfg.lam}  "
                               f"bisect={cfg.prox_bisect}")
        self.kv("model", f"d_h={gnn.cfg.hidden_dim} | {gnn.n_parameters()} "
                         f"parameters (independent of N and |E|)")
        self.kv("features", f"node (D,G) | edge "
                            f"{'(q_mn,q_nm,sum)' if gnn.cfg.rich_edge_features else '(q_nm received)'}"
                            f" | normalise={gnn.cfg.normalize}")
        self.kv("unrolling", f"K={cfg.unroll} steps | rho={cfg.rho:.2f} "
                             f"(amplitude below 1% after k={k99})")
        self.kv("summability", "guaranteed by construction: tanh output x rho^k")
        self.kv("loss", f"{cfg.loss_type}"
                        + ("  (0 = same as DR, negative = faster)"
                           if cfg.loss_type == "log_ratio" else ""))
        self.kv("optimiser", f"Adam lr={cfg.lr:g} clip={cfg.grad_clip} | "
                             f"{cfg.epochs} epochs x {n_instances} instances")
        if cfg.unroll < k99:
            self.line(f"note: K={cfg.unroll} < {k99}, so training does not "
                      f"cover the whole window where the")
            self.line(f"      perturbation is active. Consider raising unroll.")

    def epoch(self, i: int, total: int, train: float,
              val: Optional[float], dt: float) -> None:
        msg = f"  epoch {i:3d}/{total} | train {train:+.4f}"
        if val is not None:
            msg += f" | val {val:+.4f}"
        msg += f" | {dt:.1f}s"
        self._p(msg)


# ======================================================================
# Differentiable rollout (centralised mirror of the distributed loop)
# ======================================================================
def rollout(net: EnergyNetwork, gnn: Optional[PerturbationGNN],
            cfg: TrainConfig, x0: Optional[torch.Tensor] = None,
            n_iter: Optional[int] = None, differentiable: bool = True
            ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Unroll the DR iteration, optionally perturbed by `gnn`.

    Returns the final auxiliary iterate and the list of shadow residuals
    r_k = ||xbar^{k+1} - xbar^k||, kept as tensors so they can carry
    gradients.
    """
    K = cfg.unroll if n_iter is None else n_iter
    x = net.flatten() if x0 is None else x0
    if not differentiable:
        x = x.detach()
    xbar = net.project_reciprocity(x)

    residuals: List[torch.Tensor] = []
    ctx = torch.enable_grad() if differentiable else torch.no_grad()
    with ctx:
        for k in range(K):
            # line 8: evaluated on the PRE-update state, as the prosumers do
            v = None
            if gnn is not None:
                v = (cfg.scale * (cfg.rho ** k)) * gnn(net, x)

            refl = 2.0 * xbar - x                                  # line 5
            z = net.prox_local(cfg.gamma, refl,
                               n_bisect=cfg.prox_bisect)           # line 6
            x_new = x + cfg.lam * (z - xbar)                       # line 7
            if v is not None:
                x_new = x_new + v                                  # line 8

            xbar_new = net.project_reciprocity(x_new)              # line 4 (next)
            residuals.append(torch.linalg.vector_norm(xbar_new - xbar))
            x, xbar = x_new, xbar_new

    return x, residuals


def needs_baseline(cfg: TrainConfig) -> bool:
    """True if the loss compares the run against plain (unperturbed) DR."""
    return cfg.loss_type in ("log_ratio", "rel_residual")


def rollout_loss(residuals: List[torch.Tensor], cfg: TrainConfig,
                 baseline: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
    """Self-supervised objective built from the residual trajectory.

    log_ratio (default)
        mean_k [ log(r_k) - log(r_k^DR) ], where r^DR is the residual of the
        SAME rollout without the perturbation (detached). Directly readable:

             0.00   the learned model changes nothing
            < 0     residuals smaller than plain DR  ->  faster
            -0.69   residual roughly halved at every unrolled step

        Being a ratio it is scale-free, so instances of different size
        contribute comparably, and its gradient does not blow up as the
        residuals shrink.

    log_residual
        mean_k log(r_k). Also valid, and NEGATIVE simply because residuals
        are smaller than 1 -- but its value depends on the scale of the
        instance, so it cannot be compared across instances.

    rel_residual / final
        mean_k r_k / r_k^DR, and the log of the last residual.
    """
    r = torch.stack(residuals)
    if cfg.loss_type == "log_residual":
        return torch.log(r + cfg.eps).mean()
    if cfg.loss_type == "final":
        return torch.log(r[-1] + cfg.eps)
    if baseline is None:
        raise ValueError(f"loss_type={cfg.loss_type!r} needs a baseline rollout")
    b = torch.stack(baseline).detach()
    if cfg.loss_type == "log_ratio":
        return (torch.log(r + cfg.eps) - torch.log(b + cfg.eps)).mean()
    return (r / (b + cfg.eps)).mean()


# ======================================================================
# Trainer
# ======================================================================
class Trainer:
    """Trains Phi_theta on a MarketDataset by unrolling the iteration."""

    def __init__(self, gnn: PerturbationGNN, cfg: Optional[TrainConfig] = None):
        self.gnn = gnn
        self.cfg = cfg if cfg is not None else TrainConfig()
        # keep the model and the solver settings consistent
        self.gnn.cfg.rho = self.cfg.rho
        self.gnn.cfg.scale = self.cfg.scale
        self.opt = torch.optim.Adam(gnn.parameters(), lr=self.cfg.lr,
                                    weight_decay=self.cfg.weight_decay)
        self.history: Dict[str, List[float]] = {"loss": [], "val": []}

    # ------------------------------------------------------------------
    def _instance_loss(self, net: EnergyNetwork,
                       need_baseline: bool) -> torch.Tensor:
        x0 = net.flatten().detach()
        base = None
        if need_baseline:
            _, base = rollout(net, None, self.cfg, x0=x0.clone(),
                              differentiable=False)
        _, res = rollout(net, self.gnn, self.cfg, x0=x0.clone(),
                         differentiable=True)
        return rollout_loss(res, self.cfg, baseline=base)

    # ------------------------------------------------------------------
    def fit(self, dataset, val_dataset=None,
            reporter: Optional[Reporter] = None) -> Dict[str, List[float]]:
        """Train for `cfg.epochs` passes over `dataset` (CENTRALISED)."""
        cfg = self.cfg
        rep = reporter if reporter is not None else Reporter(cfg.verbose)
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)
        need_base = needs_baseline(cfg)

        rep.setup(self.gnn, cfg, len(dataset))
        rep.line("")

        for epoch in range(cfg.epochs):
            t0 = _time.time()
            self.gnn.train()
            total = 0.0
            for i in range(len(dataset)):
                net = dataset.build(i, device=device)
                self.opt.zero_grad()
                loss = self._instance_loss(net, need_base)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.gnn.parameters(),
                                                   cfg.grad_clip)
                self.opt.step()
                total += float(loss)
            mean_loss = total / max(len(dataset), 1)
            self.history["loss"].append(mean_loss)

            v = None
            if val_dataset is not None:
                v = self.evaluate_loss(val_dataset)
                self.history["val"].append(v)
            rep.epoch(epoch + 1, cfg.epochs, mean_loss, v, _time.time() - t0)
        return self.history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_loss(self, dataset) -> float:
        self.gnn.eval()
        device = torch.device(self.cfg.device)
        tot = 0.0
        for i in range(len(dataset)):
            net = dataset.build(i, device=device)
            x0 = net.flatten()
            base = None
            if needs_baseline(self.cfg):
                _, base = rollout(net, None, self.cfg, x0=x0.clone(),
                                  differentiable=False)
            _, res = rollout(net, self.gnn, self.cfg, x0=x0.clone(),
                             differentiable=False)
            tot += float(rollout_loss(res, self.cfg, baseline=base))
        return tot / max(len(dataset), 1)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.gnn.state_dict(),
                    "gnn_config": self.gnn.cfg.__dict__,
                    "train_config": self.cfg.__dict__}, path)

    @staticmethod
    def load(path: str) -> PerturbationGNN:
        from gnn import GNNConfig
        blob = torch.load(path, map_location="cpu")
        gnn = PerturbationGNN(GNNConfig(**blob["gnn_config"]))
        gnn.load_state_dict(blob["state_dict"])
        gnn.eval()
        return gnn


# ======================================================================
# End-to-end evaluation: plain DR vs L2Play, in the DISTRIBUTED solver
# ======================================================================
def iterations_to_tolerance(dataset, gnn: Optional[PerturbationGNN],
                            cfg: TrainConfig, tol: float = 1e-6,
                            max_iter: int = 20000, device=None) -> List[int]:
    """Run the distributed solver on every instance and count iterations.

    This is the honest end-to-end measurement: the learned model is queried
    by each prosumer through its own LocalAgent, exactly as in deployment.
    """
    from dr_solver import DRSolver, DRConfig

    solver = DRSolver(DRConfig(gamma=cfg.gamma, lam=cfg.lam, n_iter=max_iter,
                               stop_tol=tol, prox_bisect=cfg.prox_bisect))
    out: List[int] = []
    for i in range(len(dataset)):
        net = dataset.build(i, device=device)
        if gnn is not None:
            deploy(gnn, net)          # one LocalAgent per prosumer
        else:
            undeploy(net)
        res = solver.solve(net, record=False, write_back=False)
        out.append(res.n_iter if res.converged else max_iter)
    return out



def residual_curves(dataset, gnn: Optional[PerturbationGNN], cfg: TrainConfig,
                    tol: float = 1e-6, max_iter: int = 20000,
                    device=None) -> List["np.ndarray"]:
    """Residual history of every instance, run in the DISTRIBUTED solver.

    For each instance the run starts from the same x0 whether or not the
    learned model is deployed, so the two families of curves are directly
    comparable. Curves have different lengths (instances converge at
    different iterations); `plot_style.median_band` aligns them.
    """
    from dr_solver import DRSolver, DRConfig

    solver = DRSolver(DRConfig(gamma=cfg.gamma, lam=cfg.lam, n_iter=max_iter,
                               stop_tol=tol, prox_bisect=cfg.prox_bisect))
    curves = []
    for i in range(len(dataset)):
        net = dataset.build(i, device=device)
        x0 = net.flatten().clone()
        if gnn is not None:
            deploy(gnn, net)
        else:
            undeploy(net)
        res = solver.solve(net, x0=x0, record=True, write_back=False)
        curves.append(np.asarray(res.history["residual"]))
    return curves

def compare(dataset, gnn: PerturbationGNN, cfg: TrainConfig,
            tol: float = 1e-6, max_iter: int = 20000,
            reporter: Optional[Reporter] = None) -> Dict[str, object]:
    """Iterations of plain DR vs L2Play over a dataset (DISTRIBUTED solver)."""
    rep = reporter if reporter is not None else Reporter(False)
    rep.kv("mode", "each prosumer: communicate -> DR lines 4-7 -> "
                   "own model (line 8)")
    rep.kv("stopping", f"shadow residual < {tol:g} (cap {max_iter} iterations)")
    rep.line("")
    base = iterations_to_tolerance(dataset, None, cfg, tol, max_iter)
    learned = iterations_to_tolerance(dataset, gnn, cfg, tol, max_iter)
    for i, (b, l) in enumerate(zip(base, learned)):
        rep.line(f"instance {i + 1:3d}/{len(base)} | DR {b:6d} it | "
                 f"L2Play {l:6d} it | x{b / max(l, 1):.2f}")
    speedup = [b / max(l, 1) for b, l in zip(base, learned)]
    n = len(speedup)
    return {
        "dr": base,
        "l2play": learned,
        "speedup": speedup,
        "mean_speedup": sum(speedup) / max(n, 1),
        "median_dr": sorted(base)[n // 2] if n else 0,
        "median_l2play": sorted(learned)[n // 2] if n else 0,
    }