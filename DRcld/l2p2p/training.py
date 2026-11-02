"""Offline, centralised training of the learned correction Phi_w (paper App. B).

Back-propagating through an unrolled trajectory needs the whole network
state at once, so training runs on the vectorised mirror of Algorithm 1
(`rollout`: `Network.prox_local` / `project_reciprocity` and
`PerturbationGNN.forward_centralised`). What comes out is one set of weights,
deployed to every prosumer and evaluated through `forward_local` at
inference time (`gnn.deploy`); `tests/test_parity.py` checks the two paths
agree.

Loss (`asymmetric_loss`, paper Eq. 16): the residuals of the corrected
trajectory are compared, iteration by iteration, with those of plain DR from
the same x^0; lagging behind DR is penalised with slope `w_behind`, being
ahead is rewarded with slope `w_ahead < w_behind`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from . import market
from .fixed_point import DouglasRachford
from .gnn import PerturbationGNN
from .network import STALL_EMA_DECAY


@dataclass
class TrainConfig:
    gamma: float = 1.5          # DR step size (same as the deployed iteration)
    lam: float = 1.0
    bisect: int = 40
    unroll: int = 150           # K: iterations differentiated through per instance
    gamma_mono: float = 2.0     # weight of the monotonicity penalty (beta, Eq. 16)
    w_behind: float = 2.0       # slope of the penalty for lagging behind plain DR
    w_ahead: float = 1.0        # slope of the reward for being ahead of plain DR
    omega_mode: str = "fixed"   # iteration weights: "fixed" -> (k/K)^omega_power,
                                # "dr_slope" -> adapted to the plain-DR trajectory
    omega_power: float = 0.0    # 0 = uniform weights ("fixed" mode only)
    epochs: int = 25
    lr: float = 3e-3
    lr_final_frac: Optional[float] = None  # cosine-anneal lr to lr * lr_final_frac; None = constant
    grad_clip: float = 1.0
    seed: int = 0
    eps: float = 1e-9


def rollout(net, gnn: Optional[PerturbationGNN], cfg: TrainConfig,
            x0: Optional[torch.Tensor] = None, differentiable: bool = True
            ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Unroll `cfg.unroll` iterations of Algorithm 1 (plain DR if `gnn` is None).

    Returns (x_final, residuals) with residuals[k] = ||x^{k+1} - x^k||,
    measured after the correction is added (what the stopping rule sees).
    """
    operator = DouglasRachford(cfg.gamma, cfg.lam, cfg.bisect)
    x = (net.flatten() if x0 is None else x0)
    if not differentiable:
        x = x.detach()
    xbar = net.project_reciprocity(x)
    mem = gnn.init_mem(net.N, net.device) if gnn is not None else None
    prev_res = torch.zeros(net.N, device=net.device) if gnn is not None else None
    stall = torch.zeros(net.N, device=net.device) if gnn is not None else None

    residuals: List[torch.Tensor] = []
    ctx = torch.enable_grad() if differentiable else torch.no_grad()
    with ctx:
        for k in range(cfg.unroll):
            x_new = operator.centralised_update(net, x, xbar)          # lines 6-10

            if gnn is not None:
                direction_fast, direction_slow, mem = gnn.forward_centralised(
                    net, x, mem, prev_res, stall)                      # line 11
                amp_fast = gnn.cfg.scale * gnn.decay(k)
                amp_slow = gnn.cfg.scale_slow * gnn.decay_slow(k)
                x_new = x_new + amp_fast * direction_fast + amp_slow * direction_slow  # line 12
                prev_res = torch.sqrt(net.per_node_move_sq(x_new, x).detach() + 1e-12)
                stall = STALL_EMA_DECAY * stall + (1.0 - STALL_EMA_DECAY) * prev_res

            residuals.append(torch.linalg.vector_norm(x_new - x))
            xbar = net.project_reciprocity(x_new)
            x = x_new

    return x, residuals


def asymmetric_loss(residuals: List[torch.Tensor], residuals_dr: List[torch.Tensor],
                    scale: torch.Tensor, gamma_mono: float, eps: float,
                    w_behind: float = 2.0, w_ahead: float = 1.0,
                    omega_power: float = 0.0, omega_mode: str = "fixed") -> torch.Tensor:
    """Paper Eq. 16. With a^k = rho^k - rho^k_DR (normalised residuals):

        L = mean_k[ omega_k (w_behind relu(a^k) - w_ahead relu(-a^k)) ]
            + gamma_mono * mean_k[ relu(rho^{k+1} - rho^k) ]

    The reward for being ahead is bounded by rho^k_DR itself, so freezing the
    iteration is not a cheap optimum. `w_behind != w_ahead` is essential: with
    equal slopes the detached DR term drops out of the gradient.

    omega_mode="dr_slope" weights each iteration by how slowly plain DR's
    log-residual decreases there (larger weight on DR's slow phase, ~10x cap),
    normalised to mean 1.
    """
    denom = torch.linalg.vector_norm(scale) + eps
    rho = torch.stack(residuals) / denom
    rho_dr = (torch.stack(residuals_dr) / denom).detach()
    a = rho - rho_dr
    per_step = w_behind * torch.relu(a) - w_ahead * torch.relu(-a)
    if omega_mode == "dr_slope":
        log_rho_dr = torch.log(rho_dr + eps)
        log_decrease = torch.clamp(log_rho_dr[:-1] - log_rho_dr[1:], min=0.0)
        typical = log_decrease.mean().clamp(min=eps)
        omega_local = typical / (log_decrease + 0.1 * typical)   # capped ~10x typical at a true plateau
        still_converging = rho_dr[:-1] > 2.0 * rho_dr[-1]
        omega_local = torch.where(still_converging, omega_local, torch.ones_like(omega_local))
        omega_local = torch.cat([omega_local, omega_local[-1:]])
        omega = omega_local / omega_local.mean()
        decay = (omega * per_step).mean()
    elif omega_power != 0.0:
        K = rho.shape[0]
        k_frac = torch.arange(1, K + 1, dtype=rho.dtype, device=rho.device) / K
        omega = k_frac ** omega_power
        omega = omega / omega.mean()
        decay = (omega * per_step).mean()
    else:
        decay = per_step.mean()
    mono = (torch.relu(rho[1:] - rho[:-1]).mean()
            if rho.numel() > 1 else torch.zeros((), dtype=rho.dtype))
    return decay + gamma_mono * mono


class Trainer:
    def __init__(self, gnn: PerturbationGNN, cfg: Optional[TrainConfig] = None):
        self.gnn = gnn
        self.cfg = cfg or TrainConfig()
        self.opt = torch.optim.Adam(gnn.parameters(), lr=self.cfg.lr)
        self.sched = None
        if self.cfg.lr_final_frac is not None:
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=self.cfg.epochs, eta_min=self.cfg.lr * self.cfg.lr_final_frac)
        self.history = {"train": [], "val": []}

    def _instance_loss(self, net, differentiable: bool) -> torch.Tensor:
        cfg = self.cfg
        x0 = net.flatten().detach()
        _, residuals = rollout(net, self.gnn, cfg, x0=x0, differentiable=differentiable)
        _, residuals_dr = rollout(net, None, cfg, x0=x0, differentiable=False)
        scale = net.physical_scale()
        return asymmetric_loss(residuals, residuals_dr, scale, cfg.gamma_mono, cfg.eps,
                               cfg.w_behind, cfg.w_ahead, cfg.omega_power, cfg.omega_mode)

    def fit(self, train_specs, val_specs=None, device=None, verbose: bool = True):
        """One Adam step per training instance and epoch. x^0 is redrawn for
        every (instance, epoch) from a seeded generator (paper App. B)."""
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        device = device or torch.device("cpu")

        for epoch in range(cfg.epochs):
            self.gnn.train()
            total = 0.0
            for i, spec in enumerate(train_specs):
                gen = torch.Generator(device=device)
                gen.manual_seed(cfg.seed * 1_000_003 + epoch * 7_919 + i)
                net = market.build_network(spec, device=device, rng=gen)

                self.opt.zero_grad()
                loss = self._instance_loss(net, differentiable=True)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), cfg.grad_clip)
                self.opt.step()
                total += loss.item()
            if self.sched is not None:
                self.sched.step()
            train_loss = total / max(len(train_specs), 1)
            self.history["train"].append(train_loss)

            val_loss = None
            if val_specs:
                val_loss = self.evaluate(val_specs, device=device)
                self.history["val"].append(val_loss)

            if verbose:
                msg = f"  epoch {epoch + 1:3d}/{cfg.epochs} | train {train_loss:+.4f}"
                if val_loss is not None:
                    msg += f" | val {val_loss:+.4f}"
                print(msg, flush=True)
        return self.history

    @torch.no_grad()
    def evaluate(self, specs, device=None) -> float:
        """Mean loss over `specs`; x^0 comes from torch's global random state
        (monitoring only: it does not affect the trained weights)."""
        self.gnn.eval()
        device = device or torch.device("cpu")
        total = 0.0
        for spec in specs:
            net = market.build_network(spec, device=device)
            total += float(self._instance_loss(net, differentiable=False))
        return total / max(len(specs), 1)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.gnn.state_dict(), "cfg": self.gnn.cfg.__dict__}, path)
