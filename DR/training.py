"""
training.py
============

Centralised training of the perturbation model Phi_theta.

Why centralised
---------------
Back-propagation has to flow through the whole unrolled trajectory, so the
optimiser needs the entire state at once. Training is therefore a one-off,
offline, centralised procedure. What comes out of it is a single set of
weights theta, copied to every prosumer (`gnn.deploy`) and used in the
fully distributed loop (`dr_solver.DRSolver`). Nothing about deployment is
centralised.

The rollout MIRRORS the distributed one exactly
------------------------------------------------
Same order, same information, same amplitude schedule as `Prosumer.dr_step`:

    v^k    = scale * rho^k * Phi_theta(x^k)     <- features from the PRE-update
    xbar   = proj_{X^rec}(x)                       state, i.e. exactly what a
    refl   = 2*xbar - x                            prosumer holds after line 3
    z      = prox_{gamma J}(refl)
    x^{k+1} = x + lam*(z - xbar) + v^k

The only difference is bookkeeping: here the per-node prox is evaluated by
the vectorised, differentiable `EnergyNetwork.prox_local`, whereas each
prosumer uses its own `Prosumer.local_prox`; `gnn.py`'s smoke test cross
-checks the two agree numerically. If they ever drifted we would be
training one system and deploying another.

Loss: strictly self-supervised -- no baseline rollout, no precomputed target
-------------------------------------------------------------------------------
Three earlier designs were tried; the first two failed end-to-end, the
third worked but was set aside on principle, which is why this one looks
the way it does.

Attempt 1: minimise the consecutive-step residual itself,
r_k = ||xbar^{k+1} - xbar^k||, relative to plain DR on the same instance
and initial state: L = mean_k [log(r_k+eps) - log(r_k^DR+eps)]. EXPLOITABLE:
a direction that briefly nearly FREEZES the trajectory (x^{k+1} ~ x^k)
drives r_k -> 0 and log(r_k) -> -infinity, an unbounded reward, while the
cost of that stall (the unperturbed tail has to cover the distance it
didn't) is a bounded, ordinary log-residual increase later. Measured
end-to-end: mean speed-up 0.73x -- SLOWER than plain DR -- even though the
unrolled training loss looked excellent.

Attempt 2: same idea, averaged only over the tail k in [k_start, K), past
the amplitude's decay horizon. Helped (0.73x -> 1.23x) but plateaued far
short of a reliable, uniform improvement: gains concentrated on
already-slow outlier instances (up to 2.16x) while about a third of the
ordinary, fast-converging instances got WORSE (0.61x-0.99x). The root
cause survives windowing: r_k measures how far ONE step moved, not how
close the point is to the solution -- loosely correlated at best.

Attempt 3: anchor to a precomputed near-equilibrium xbar* (plain DR run to
convergence once per instance, cheap, cached) and minimise
d_k/d_k^DR + beta*relu(d_k/d_k^DR - 1), d_k = ||xbar^k - xbar*||. This
worked well in testing, but it IS a target-based signal -- xbar* plays the
role of a label, even though it costs nothing extra to obtain and comes
from the very algorithm being accelerated, not an external oracle. Set
aside to keep training strictly self-supervised: no reference solution,
precomputed or otherwise, enters the loss at all.

This design (attempt 4, current) uses only the perturbed trajectory's own
r_k sequence and a STATIC, zero-computation per-instance scale (the same
`gnn.physical_scale` used to size the perturbation itself -- half the D/G
box, the trade capacity kappa -- no DR steps of any kind needed to obtain
it). Writing rho_k = r_k / (||physical_scale|| + eps):

    L = mean_k [ rho_k ]  +  gamma_mono * mean_k [ relu(rho_{k+1,aux} - rho_{k,aux}) ]

The first term rewards small (normalised) steps -- without a log, so
driving one rho_k to zero contributes at most 0 to the mean: a bounded,
modest benefit, not an exploitable unbounded one. The second term uses a
real property of plain DR (an averaged operator's own residual sequence is
naturally non-increasing) -- but on rho_aux, the AUXILIARY iterate's
residual, not the shadow one r_k above. This matters: the shadow residual
is NOT guaranteed monotone even for plain, unperturbed DR (measured: ~1 in
6 steps increases on a typical instance -- it is a derived quantity, one
extra projection on top of the iterate the theory actually covers), so an
earlier revision of this loss that penalised ITS increases was partly
penalising ordinary DR behaviour, not just perturbation damage. Using
rho_aux fixes that: an increase there really is evidence the perturbation
broke DR's own guaranteed property, since plain DR would not have produced
one (up to the bisection prox's own numerical noise).

Revision history on top of attempt 4, kept here rather than deleted so the
same dead ends are not re-walked:
  - `loss_late_weight` (weighting the first term towards later steps, to
    push on "stay near zero" for already-fast instances): REMOVED. It was
    a patch for the "never worse than DR" problem, and turned out to be
    part of what made long-horizon training collapse onto one generic,
    instance-independent pace instead of an instance-adaptive policy (see
    "ATTEMPTS TO EXTEND REACH" in example_l2play.py) -- disabling it alone
    did not fix that collapse either, so it was just complexity with no
    longer any justification, not a working safeguard.

Attempt 5 (TRIED AND REVERTED): both terms measured on a residual computed
BEFORE v^k instead of after -- d^k = ||z^k-xbar^k||, the classic DR/KM
fixed-point residual (line 6's prox output minus line 4's shadow). The
motivation was real: both r_k and the auxiliary residual above are
measured AFTER v^k is added (x^{k+1} already includes it), so in principle
the network could choose v^k ~= -lam*(z^k-xbar^k) to cancel DR's own step
and "freeze" the trajectory, shrinking the measured residual without
genuine progress -- the same failure mode as attempt 1, just not gamed via
an unbounded log reward this time. d^k cannot be touched by v^k at the
same index at all, closing that door, and has a clean theoretical
reading (d^k=0 iff x^k is a DR fixed point; metric subregularity gives
dist(x^k,Fix) <= kappa*d^k for this problem's polyhedral/quadratic
structure, so it is a legitimate distance surrogate, not just a heuristic).
Despite the sound theory, it measured WORSE end-to-end on two independent
datasets: 2.12x->1.45x (30-instance varying-graph set, regressions
7/30->13/30) and 1.23x->1.10x (a denser fixed graph, regressions
2/10->6/10). Best-guess reason: precisely because d^k excludes v^k at the
same index, v^k gets NO direct gradient signal from the current step's
loss term -- only an indirect, diluted one through v^k's effect on later
steps' state. The theoretically cleaner, exploit-proof signal turned out
to be a practically weaker training signal for the actual decision
variable, and that cost outweighed the benefit of closing a vulnerability
gradient descent may not have been exploiting from this initialisation
anyway. Reverted to the post-v^k residuals above.

Attempt 6 (TRIED AND REVERTED): rather than replacing the post-v^k
residuals with d^k (attempt 5), ADD d^k as a third term on top of them,
so the decay term keeps giving v^k a direct gradient signal while d^k
guards against the same freezing exploit. Tested two ways on the fixed
graph N10D07 dataset, both with gamma=1.0 (no change) but two different
gamma_mono settings, all else (rho=0.9, unroll=400, 40 epochs, lr=3e-3)
identical to the validated recipe:
  - gamma_mono=0 (drop the monotonicity penalty entirely, on the theory
    that a genuinely convergent policy should have monotonicity emerge on
    its own rather than being forced) + w_fix=1: WORSE (1.23x->1.13x mean
    on the original 10-instance test set, regressions 2/10->5/10, with the
    single best instance collapsing from 1.79x to 0.45x -- slower than
    plain DR, using more than double DR's own iteration count).
  - gamma_mono=2 (kept, UNCHANGED) + w_fix=1, re-tested on a larger
    30-instance test set (to rule out the 10-instance set just being too
    noisy -- a single instance flipping moved the mean by ~0.1-0.15x in
    both experiments above): STILL worse, 1.23x->1.11x.
Conclusion: the monotonicity penalty is not merely blocking useful
aggressive trajectories (the original hypothesis) -- it is doing real
stabilising work. Removing or diluting it, even while adding a
theoretically well-motivated extra term (d^k, which IS exploit-proof by
construction and has a clean theoretical reading -- see attempt 5), costs
more in variance across instances than it gains. Reverted; `rollout` no
longer computes d^k at all.

Truncated BPTT (`TrainConfig.tbptt_segment`, `Trainer._fit_instance_tbptt`)
addresses a DIFFERENT problem than the loss shape: training rho/unroll
large enough to reach a late plateau requires differentiating through
thousands of recurrent steps, and doing that in one single backward pass
(no matter how the loss is shaped) washes out the gradient signal for
"do this specific thing on THIS instance, early on", leaving only a
generic late-horizon behaviour that is easy to learn regardless of loss
weighting. Splitting the same long horizon into short, truncated windows
(state persists across windows, gradient does not) keeps every individual
backward pass as well-conditioned as the short-unroll regime that is known
to work, while still exposing the network to the full horizon over the
course of training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import time as _time

import numpy as np
import torch

from energy_network import EnergyNetwork
from gnn import PerturbationGNN, GNNConfig, perturbation_vector, deploy, undeploy


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class TrainConfig:
    # --- solver settings the model is trained against ---
    # Tune the BASELINE first: accelerating a badly tuned DR proves nothing.
    gamma: float = 1.5
    lam: float = 1.0
    prox_bisect: int = 30

    # --- unrolling ---
    # K should still comfortably exceed the amplitude's decay horizon (see
    # the note printed by Reporter.setup), so the loss also sees how the
    # trajectory behaves once the perturbation has switched off, not only
    # while it is active.
    unroll: int = 120           # K: iterations differentiated through
    rho: float = 0.9            # decay of the perturbation (summability;
                                # used only when decay == "geometric")
    scale: float = 1.0          # amplitude, as a fraction of the local
                                # physical range (see gnn.PerturbationGNN)
    decay: str = "geometric"    # "geometric" | "polynomial" -- see
                                # gnn.GNNConfig's `decay` docstring. Both
                                # are fixed, theta-independent, summable
                                # sequences, so either preserves Theorem 1;
                                # "polynomial" decays far more slowly,
                                # trading a smaller per-step amplitude for
                                # a much longer horizon of non-negligible
                                # influence (see documentation.txt sec. 10,
                                # "DIAGNOSTIC").
    poly_p: float = 1.5         # exponent for "polynomial" (> 1 required)

    # Truncated BPTT: if set, `unroll` is covered in consecutive windows of
    # this length instead of one single differentiable rollout. The state
    # (x, recurrent memory, previous residual) still flows forward across
    # the WHOLE `unroll` horizon, but gradients never cross a window
    # boundary (detached there) and the optimiser steps once per window.
    # Needed whenever `unroll` is long enough that a single backward pass
    # through the whole thing washes out per-instance signal (see the
    # "ATTEMPTS TO EXTEND REACH" note in example_l2play.py) -- None (the
    # default) reproduces the original single-shot behaviour exactly.
    tbptt_segment: Optional[int] = None

    # --- loss (strictly self-supervised, see module docstring) ---
    # rho_k = r_k / ||physical_scale||, a per-instance normalisation that
    # needs zero DR computation (static box/capacity information only).
    # gamma_mono weights the penalty for any INCREASE in rho_k step-to-step,
    # computed on the AUXILIARY iterate's residual (the one DR's own theory
    # guarantees is non-increasing -- verified numerically; the shadow
    # residual used for the decay term below is NOT guaranteed monotone,
    # not even for plain unperturbed DR, so penalising its increases would
    # partly penalise normal DR behaviour instead of perturbation damage).
    # loss_start_frac restricts the loss to k in [loss_start_frac*unroll,
    # unroll); 0 (default) uses the whole window -- safe here, rho_k cannot
    # be gamed by freezing (no log, no baseline to race against).
    loss_gamma_mono: float = 2.0
    loss_start_frac: float = 0.0

    # --- optimisation ---
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eps: float = 1e-9

    # --- misc ---
    seed: int = 0
    device: str = "cpu"
    verbose: bool = True

    def __post_init__(self):
        if self.decay not in ("geometric", "polynomial"):
            raise ValueError('decay must be "geometric" or "polynomial"')
        if self.decay == "geometric" and not (0.0 < self.rho < 1.0):
            raise ValueError("rho must lie in (0,1)")
        if self.decay == "polynomial" and not (self.poly_p > 1.0):
            raise ValueError("poly_p must be > 1 for summability (Riemann zeta)")
        if self.unroll <= 0:
            raise ValueError("unroll must be positive")
        if not (0.0 <= self.loss_start_frac < 1.0):
            raise ValueError("loss_start_frac must lie in [0,1)")
        if self.tbptt_segment is not None and not (0 < self.tbptt_segment <= self.unroll):
            raise ValueError("tbptt_segment must lie in (0, unroll]")

    @property
    def loss_start(self) -> int:
        """First index (0-based) of the residual list that enters the loss."""
        return int(self.loss_start_frac * self.unroll)


# ======================================================================
# Reporting (plain, structured console log)
# ======================================================================
class Reporter:
    """Minimal structured console log. Set `enabled=False` to silence it."""

    WIDTH = 68

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._t0 = _time.time()
        self._phase_t0 = self._t0

    def _p(self, text: str = "") -> None:
        if self.enabled:
            print(text, flush=True)

    def rule(self, char: str = "-") -> None:
        self._p(char * self.WIDTH)

    def phase(self, index: int, total: int, title: str) -> None:
        self._phase_t0 = _time.time()
        self._p("")
        self.rule("=")
        self._p(f"  phase {index}/{total}  |  {title.upper()}")
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

    def dataset(self, name: str, ds) -> None:
        s = ds.summary()
        self.kv(name, f"{s['instances']} instances | N={s['N'][0]}..{s['N'][1]} "
                      f"| undirected edges {s['undirected_edges'][0]}.."
                      f"{s['undirected_edges'][1]}")

    def setup(self, gnn: PerturbationGNN, cfg: "TrainConfig", n_instances: int) -> None:
        if cfg.decay == "geometric":
            k99 = int(np.ceil(np.log(0.01) / np.log(cfg.rho)))
            decay_desc = f"rho={cfg.rho:.2f}"
            summ_desc = "x physical scale x rho^k"
        else:
            k99 = int(np.ceil(100.0 ** (1.0 / cfg.poly_p) - 1.0))
            decay_desc = f"poly_p={cfg.poly_p:.2f}"
            summ_desc = "x physical scale x (k+1)^-poly_p"
        self.kv("baseline DR", f"gamma={cfg.gamma}  lambda={cfg.lam}  "
                               f"bisect={cfg.prox_bisect}")
        self.kv("model", f"d_h={gnn.cfg.hidden_dim} | {gnn.n_parameters()} "
                         f"parameters (independent of N and |E|)")
        self.kv("unrolling", f"K={cfg.unroll} steps | {cfg.decay} decay, "
                             f"{decay_desc} (amplitude below 1% after k={k99})")
        if cfg.tbptt_segment:
            n_seg = int(np.ceil(cfg.unroll / cfg.tbptt_segment))
            self.kv("tbptt", f"truncated BPTT: {n_seg} windows of "
                            f"{cfg.tbptt_segment} steps (state persists, "
                            f"gradient does not cross window boundaries)")
        self.kv("summability", f"guaranteed by construction: tanh output {summ_desc}")
        self.kv("loss", f"self-referential: normalised residual + "
                       f"gamma_mono={cfg.loss_gamma_mono:g} monotonicity "
                       f"(on the auxiliary residual), window "
                       f"k=[{cfg.loss_start},{cfg.unroll})")
        self.kv("no target", "no baseline rollout, no precomputed equilibrium -- "
                             "strictly self-supervised")
        self.kv("optimiser", f"Adam lr={cfg.lr:g} clip={cfg.grad_clip} | "
                             f"{cfg.epochs} epochs x {n_instances} instances")
        if cfg.unroll < k99:
            self.line(f"note: K={cfg.unroll} < {k99}, training does not cover "
                      f"the whole window where the perturbation is still")
            self.line(f"      active at more than 1% amplitude -- consider "
                      f"raising `unroll`.")

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
            n_iter: Optional[int] = None, differentiable: bool = True,
            mem0=None, prev_res0: Optional[torch.Tensor] = None,
            k_start: int = 0
            ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor],
                      object, torch.Tensor]:
    """Unroll the DR iteration, optionally perturbed by `gnn`.

    Mirrors `Prosumer.dr_step` line by line (shadow -> reflection -> prox
    -> relaxed update -> perturbation), but vectorised over the whole
    network via `EnergyNetwork.prox_local` / `project_reciprocity`, and
    with `gnn.direction` playing the role of every prosumer's own
    `gnn_agent.perturbation` call, evaluated on the SAME pre-update state.

    Returns `(x, residuals, residuals_aux, mem, prev_res)`:
      - `residuals`     : shadow residuals r_k = ||xbar^{k+1}-xbar^k||
      - `residuals_aux` : auxiliary residuals ||x^{k+1}-x^k|| -- the
                          quantity DR's own theory guarantees is
                          (near-exactly) non-increasing
    both measured AFTER v^k is added on line 8, and kept as tensors so
    they can carry gradients back to theta when `differentiable=True`.

    (Two revisions of this loss were tried and reverted -- see training.py's
    module docstring for the full writeup ("attempt 5" and "attempt 6"),
    kept there rather than deleted so the same dead ends are not re-walked.)

    `mem0`/`prev_res0`/`k_start` let a caller resume a previously-run
    prefix instead of always starting "cold" at k=0 -- used by truncated
    BPTT (`Trainer._fit_instance_tbptt`) to cover a long horizon as several
    short, well-conditioned windows instead of one long one. Passing none
    of them reproduces the original single-shot behaviour exactly: fresh
    memory, previous residual zero, amplitude schedule starting at k=0.
    """
    K = cfg.unroll if n_iter is None else n_iter
    x = net.flatten() if x0 is None else x0
    if not differentiable:
        x = x.detach()
    xbar = net.project_reciprocity(x)

    if gnn is not None:
        mem = gnn.init_mem(net) if mem0 is None else mem0
        prev_res = (torch.zeros(net.N, device=net.device) if prev_res0 is None
                   else prev_res0)
    else:
        mem, prev_res = None, None

    residuals: List[torch.Tensor] = []
    residuals_aux: List[torch.Tensor] = []
    ctx = torch.enable_grad() if differentiable else torch.no_grad()
    with ctx:
        for local_k in range(K):
            k = k_start + local_k
            # line 8: evaluated on the PRE-update state, as every prosumer does
            v = None
            if gnn is not None:
                v, mem = perturbation_vector(gnn, net, x, k, mem, prev_res)

            refl = 2.0 * xbar - x                                  # line 5
            z = net.prox_local(cfg.gamma, refl,
                               n_bisect=cfg.prox_bisect)           # line 6
            x_new = x + cfg.lam * (z - xbar)                       # line 7
            if v is not None:
                x_new = x_new + v                                  # line 8

            xbar_new = net.project_reciprocity(x_new)              # line 4 (next)
            residuals.append(torch.linalg.vector_norm(xbar_new - xbar))
            residuals_aux.append(torch.linalg.vector_norm(x_new - x))

            if gnn is not None:
                D_old, G_old, Q_old = net.get_DGQ(x)
                D_new, G_new, Q_new = net.get_DGQ(x_new)
                per_node_sq = ((D_new - D_old) ** 2 + (G_new - G_old) ** 2
                              + net._node_sum((Q_new - Q_old) ** 2))
                prev_res = torch.sqrt(per_node_sq + 1e-12)

            x, xbar = x_new, xbar_new

    return x, residuals, residuals_aux, mem, prev_res


def stability_loss(residuals: List[torch.Tensor], residuals_aux: List[torch.Tensor],
                   scale: torch.Tensor, gamma_mono: float = 2.0,
                   eps: float = 1e-9, start: int = 0) -> torch.Tensor:
    """Strictly self-referential loss -- no baseline rollout, no target.

        rho_k     = r_k     / (||scale|| + eps)      (shadow residual)
        rho_k_aux = r_k_aux / (||scale|| + eps)      (auxiliary residual)
        L = mean_k[rho_k]  +  gamma_mono * mean_k[relu(rho_{k+1,aux}-rho_{k,aux})]

    `scale` is `gnn.PerturbationGNN.physical_scale(net)`, a static,
    zero-computation per-instance normalisation. See the module docstring
    for why this replaces the two consecutive-residual designs (both
    exploitable) and the equilibrium-anchored one (not strictly
    self-supervised).

    The decay term uses the SHADOW residual because that is exactly what
    the deployment stopping rule checks against `tol` -- minimising it is
    minimising the thing that actually decides "are we done". The
    monotonicity term uses the AUXILIARY residual instead: that is the
    quantity DR's own convergence theory guarantees is non-increasing, so
    an increase there is real evidence the perturbation broke that
    property. The shadow residual is NOT guaranteed monotone -- not even
    for plain, unperturbed DR (verified: ~1 in 6 steps increases on a
    typical instance) -- so penalising ITS increases would partly punish
    ordinary DR behaviour instead of perturbation-induced damage.

    (An "attempt 9" adding a third, pre-perturbation d^k term on top of
    this loss -- with and without dropping gamma_mono to let monotonicity
    emerge on its own -- was tried and reverted: worse on both a 10- and a
    30-instance fixed-graph test set. See `rollout`'s docstring for the
    full numbers and writeup.)
    """
    denom = torch.linalg.vector_norm(scale) + eps

    r = torch.stack(residuals[start:])
    rho = r / denom
    decay = rho.mean()

    ra = torch.stack(residuals_aux[start:])
    rho_aux = ra / denom
    if rho_aux.numel() > 1:
        mono = torch.relu(rho_aux[1:] - rho_aux[:-1]).mean()
    else:
        mono = torch.zeros((), dtype=rho.dtype, device=rho.device)

    return decay + gamma_mono * mono


# ======================================================================
# Trainer
# ======================================================================
class Trainer:
    """Trains Phi_theta on a MarketDataset by unrolling the DR iteration."""

    def __init__(self, gnn: PerturbationGNN, cfg: Optional[TrainConfig] = None):
        self.gnn = gnn
        self.cfg = cfg if cfg is not None else TrainConfig()
        # keep the deployed amplitude schedule consistent with training
        self.gnn.cfg.rho = self.cfg.rho
        self.gnn.cfg.scale = self.cfg.scale
        self.gnn.cfg.decay = self.cfg.decay
        self.gnn.cfg.poly_p = self.cfg.poly_p
        self.opt = torch.optim.Adam(gnn.parameters(), lr=self.cfg.lr,
                                    weight_decay=self.cfg.weight_decay)
        self.history: Dict[str, List[float]] = {"loss": [], "val": []}

    # ------------------------------------------------------------------
    def _instance_loss(self, net: EnergyNetwork,
                       differentiable: bool = True) -> torch.Tensor:
        """Single-shot loss over the whole `cfg.unroll` window.

        Used for validation/evaluation always, and for training whenever
        `cfg.tbptt_segment` is None (short-unroll regime, unchanged from
        the original design). For long-horizon TRAINING see
        `_fit_instance_tbptt` instead -- backpropagating through the whole
        window at once is exactly the failure mode truncated BPTT exists
        to avoid.
        """
        cfg = self.cfg
        x0 = net.flatten().detach()
        _, res, res_aux, _, _ = rollout(net, self.gnn, cfg, x0=x0,
                                        differentiable=differentiable)
        scale = self.gnn.physical_scale(net)
        return stability_loss(res, res_aux, scale, cfg.loss_gamma_mono, cfg.eps,
                              start=cfg.loss_start)

    # ------------------------------------------------------------------
    def _fit_instance_tbptt(self, net: EnergyNetwork) -> float:
        """One instance's worth of TRUNCATED BPTT training.

        Covers the whole `cfg.unroll` horizon as consecutive windows of
        `cfg.tbptt_segment` steps. State (x, recurrent memory, previous
        residual) flows forward across windows; gradients do not -- each
        window gets its own zero_grad/backward/step, and the state handed
        to the next window is `.detach()`-ed first. Returns the mean loss
        across windows (float, for reporting only).
        """
        cfg = self.cfg
        seg = cfg.tbptt_segment
        total_k = cfg.unroll
        scale = self.gnn.physical_scale(net)

        x = net.flatten().detach()
        mem = self.gnn.init_mem(net)
        prev_res = torch.zeros(net.N, device=net.device)

        k_start = 0
        losses: List[float] = []
        while k_start < total_k:
            n = min(seg, total_k - k_start)
            self.opt.zero_grad()
            x_new, res, res_aux, mem_new, prev_res_new = rollout(
                net, self.gnn, cfg, x0=x, n_iter=n, differentiable=True,
                mem0=mem, prev_res0=prev_res, k_start=k_start)
            loss = stability_loss(res, res_aux, scale, cfg.loss_gamma_mono, cfg.eps)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), cfg.grad_clip)
            self.opt.step()
            losses.append(float(loss))

            x = x_new.detach()
            mem = mem_new.detach()
            prev_res = prev_res_new.detach()
            k_start += n

        return sum(losses) / max(len(losses), 1)

    # ------------------------------------------------------------------
    def fit(self, dataset, val_dataset=None,
            reporter: Optional[Reporter] = None) -> Dict[str, List[float]]:
        """Train for `cfg.epochs` passes over `dataset` (CENTRALISED).

        A fresh random initial state is drawn for every (instance, epoch)
        pair -- reproducibly, from `cfg.seed` -- so the model is not
        overfit to one arbitrary starting point per instance; it only ever
        sees genuinely different states along genuinely different
        trajectories of the SAME market. With `cfg.tbptt_segment` None
        (default), one rollout+step per instance per epoch, exactly as
        before; otherwise several short windows per instance (truncated
        BPTT, see `_fit_instance_tbptt`), each with its own step.
        """
        cfg = self.cfg
        rep = reporter if reporter is not None else Reporter(cfg.verbose)
        torch.manual_seed(cfg.seed)
        device = torch.device(cfg.device)

        rep.setup(self.gnn, cfg, len(dataset))
        rep.line("")

        for epoch in range(cfg.epochs):
            t0 = _time.time()
            self.gnn.train()
            total = 0.0
            for i in range(len(dataset)):
                seed = cfg.seed * 1_000_003 + epoch * 7_919 + i
                net = dataset.build(i, device=device, seed=seed)
                if cfg.tbptt_segment:
                    total += self._fit_instance_tbptt(net)
                else:
                    self.opt.zero_grad()
                    loss = self._instance_loss(net, differentiable=True)
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
        """Same loss, on fixed (non-epoch-varying) initial states, no grad."""
        self.gnn.eval()
        device = torch.device(self.cfg.device)
        tot = 0.0
        for i in range(len(dataset)):
            net = dataset.build(i, device=device)
            tot += float(self._instance_loss(net, differentiable=False))
        return tot / max(len(dataset), 1)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"state_dict": self.gnn.state_dict(),
                    "gnn_config": self.gnn.cfg.__dict__,
                    "train_config": self.cfg.__dict__}, path)

    @staticmethod
    def load(path: str) -> PerturbationGNN:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        gnn = PerturbationGNN(GNNConfig(**blob["gnn_config"]))
        gnn.load_state_dict(blob["state_dict"])
        gnn.eval()
        return gnn


# ======================================================================
# End-to-end evaluation: plain DR vs DR+Phi_theta, in the DISTRIBUTED solver
# ======================================================================
def iterations_to_tolerance(dataset, gnn: Optional[PerturbationGNN],
                            cfg: TrainConfig, tol: float = 1e-6,
                            max_iter: int = 20000, device=None) -> List[int]:
    """Run the distributed solver on every instance and count iterations.

    This is the honest end-to-end measurement: the learned model is
    queried by each prosumer through its own `LocalAgent`, exactly as in
    deployment -- `training.rollout` is used ONLY to compute gradients,
    never to report the speed-up.
    """
    from dr_solver import DRSolver, DRConfig

    solver = DRSolver(DRConfig(gamma=cfg.gamma, lam=cfg.lam, n_iter=max_iter,
                               stop_tol=tol, prox_bisect=cfg.prox_bisect))
    out: List[int] = []
    for i in range(len(dataset)):
        net = dataset.build(i, device=device)
        if gnn is not None:
            deploy(gnn, net)
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


def histories(dataset, gnn: Optional[PerturbationGNN], cfg: TrainConfig,
             tol: float = 1e-6, max_iter: int = 20000,
             device=None) -> List[Dict[str, List[float]]]:
    """Full per-iteration diagnostics of every instance (DISTRIBUTED).

    Same DR vs DR+Phi_theta convention as `residual_curves` (same x0 for
    both), but returns the whole `DRResult.history` dict per instance --
    residual, residual_aux, cost, balance, reciprocity, reciprocity_raw --
    for figures that need more than the plain shadow residual (e.g. an
    extended z-residual or a cost-gap-to-optimum plot).
    """
    from dr_solver import DRSolver, DRConfig

    solver = DRSolver(DRConfig(gamma=cfg.gamma, lam=cfg.lam, n_iter=max_iter,
                               stop_tol=tol, prox_bisect=cfg.prox_bisect))
    out = []
    for i in range(len(dataset)):
        net = dataset.build(i, device=device)
        x0 = net.flatten().clone()
        if gnn is not None:
            deploy(gnn, net)
        else:
            undeploy(net)
        res = solver.solve(net, x0=x0, record=True, write_back=False)
        out.append(res.history)
    return out


def optimal_costs(dataset, cfg: Optional[TrainConfig] = None, device=None) -> List[float]:
    """Exact J* per instance via the centralised QP (see equilibrium_qp.py).

    Every prosumer's cost is separable (no cross terms), so this GNEP is a
    potential game and its v-GNE coincide with the minimisers of one
    convex QP over all constraints at once -- solved exactly, in
    milliseconds, instead of the tens of thousands of (sometimes
    non-converging) DR iterations the previous long-plain-DR-rollout
    estimate needed. `cfg` is accepted for backward-compatible call sites
    but unused (the QP needs no DR hyperparameters at all).
    """
    from equilibrium_qp import solve_equilibrium_qp

    out = []
    for i in range(len(dataset)):
        net = dataset.build(i, device=device)
        _, J_star = solve_equilibrium_qp(net)
        out.append(J_star)
    return out


def compare(dataset, gnn: PerturbationGNN, cfg: TrainConfig,
            tol: float = 1e-6, max_iter: int = 20000,
            reporter: Optional[Reporter] = None) -> Dict[str, object]:
    """Iterations of plain DR vs DR+Phi_theta over a dataset (DISTRIBUTED)."""
    rep = reporter if reporter is not None else Reporter(False)
    rep.kv("mode", "each prosumer: communicate -> DR lines 4-7 -> "
                   "own model (line 8)")
    rep.kv("stopping", f"shadow residual < {tol:g} (cap {max_iter} iterations)")
    rep.line("")
    base = iterations_to_tolerance(dataset, None, cfg, tol, max_iter)
    learned = iterations_to_tolerance(dataset, gnn, cfg, tol, max_iter)
    for i, (b, l) in enumerate(zip(base, learned)):
        rep.line(f"instance {i + 1:3d}/{len(base)} | DR {b:6d} it | "
                 f"learned {l:6d} it | x{b / max(l, 1):.2f}")
    speedup = [b / max(l, 1) for b, l in zip(base, learned)]
    n = len(speedup)
    return {
        "dr": base,
        "learned": learned,
        "speedup": speedup,
        "mean_speedup": sum(speedup) / max(n, 1),
        "median_dr": sorted(base)[n // 2] if n else 0,
        "median_learned": sorted(learned)[n // 2] if n else 0,
    }
