"""
train.py
========

Centralised training of the perturbation GNN Phi_w (paper Sec. III-C).

Back-propagation through an unrolled trajectory needs the whole network
state at once, so training runs centralised -- `rollout` below mirrors
`Prosumer.dr_step` line for line, but vectorised over the graph via
`Network.prox_local` / `project_reciprocity` and `gnn.forward_centralised`.
Only training is centralised: what comes out is one set of weights w,
broadcast to every prosumer (`gnn.deploy`) and run through
`gnn.forward_local` at inference (algorithm.py), never through this module.

Three losses (`Trainer` defaults to `hinge_loss`, the only one whose
objective is "beat plain DR on (nearly) every instance" rather than "beat
it on average") -- none use a target equilibrium; the two relative ones
additionally use the SAME instance's own plain-DR trajectory as a
reference, computed fresh each step (no grad, cheap, never a precomputed
label)
--------------------------------------------------------------------------
`stability_loss` -- Eq. 15 of the paper, with uniform iteration weights
omega_k = 1:

    r^k       = ||x^{k+1} - x^k||                (post-perturbation, the
                                                    quantity DR's own theory
                                                    -- an averaged operator's
                                                    residual sequence -- is
                                                    guaranteed non-increasing)
    rho^k     = r^k / (||sigma|| + eps)           sigma = physical reference
                                                    scale (gnn.physical_scale)
    L = mean_k[rho^k] + gamma_mono * mean_k[relu(rho^{k+1} - rho^k)]

This rewards a small step in an ABSOLUTE sense. It has no notion of how
fast plain DR itself would have solved this particular instance, so it
cannot tell "this instance is structurally slow, push hard" apart from
"this instance is already fast, stay out of the way" -- gradient descent
on it optimises the same target everywhere. Measured end to end this gives
a real but uneven, only mildly positive speed-up (see demo.py's run log).

`relative_loss` -- uses the SAME instance's unperturbed (plain-DR)
trajectory, from the SAME x^0, to REWEIGHT which iterations matter,
instead of training on every iteration equally:

    rho^k     = ||x^{k+1}-x^k||     / (||sigma||+eps)   (perturbed, w/ grad)
    rho^k_dr  = ||x^{k+1}_dr-x^k_dr|| / (||sigma||+eps)  (plain DR, no grad)
    w^k       = rho^k_dr / (mean_j[rho^j_dr] + eps)      (detached weights)
    L = mean_k[w^k * rho^k] + gamma_mono * mean_k[relu(rho^{k+1}-rho^k)]

NOTE on a mistake worth recording: an earlier version of this loss used
mean_k[rho^k - rho^k_dr] (an ADDITIVE detached baseline) instead. Since
rho^k_dr does not depend on the network's weights, subtracting it shifts
the loss's VALUE but leaves its GRADIENT with respect to the weights
IDENTICAL to plain `stability_loss` (constants vanish under
differentiation) -- Adam then takes the exact same steps regardless of
which loss is "used", which is exactly what happened: two training runs
with supposedly different losses produced bit-identical final weights.
A detached baseline can only change the training DYNAMICS if it enters
NONLINEARLY (or as a non-constant per-step weight) rather than as a flat
additive offset -- multiplying rather than subtracting is what makes this
version actually different from `stability_loss`. Concretely, w^k puts
most of the training signal on iterations where plain DR itself is still
moving a lot, and lets it fall away on the tail once DR has essentially
converged, where there is nothing left to improve. This directly targets
the difference in objective raised earlier: it can now tell "this instance
needs help right now" from "leave this alone".

This costs one extra plain-DR rollout per training step (cheap, no
gradient). It is not strictly self-supervised (the plain-DR rollout on the
current x^0 enters training), though it never enters inference or acts as
a precomputed target/label -- deployment safety (Theorem 1) is unaffected
either way.

MEASURED RESULT: worse than `stability_loss`, not better. On the same
cluster dataset and gamma, `relative_loss` gave mean/median speed-up
0.93x/0.94x in-cluster and 0.90x/0.93x out-of-cluster -- `stability_loss`
(Eq. 15) gave 1.08x/1.21x and 1.11x/1.02x. Best-guess reason: w^k is
smallest exactly on the tail, once plain DR's own residual has nearly
vanished -- but the tail is precisely where the discrete "iterations to
tolerance" outcome is decided, so starving it of gradient signal lets the
learned correction behave carelessly right where it matters most. `Trainer`
therefore defaults to `loss_mode="absolute"` (Eq. 15); `"relative"` is kept
available and documented as a negative result, not a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from gnn import PerturbationGNN


@dataclass
class TrainConfig:
    gamma: float = 1.5          # DR step size (same as deployment DRConfig.gamma)
    lam: float = 1.0
    bisect: int = 40
    unroll: int = 150           # K: iterations differentiated through per instance
    gamma_mono: float = 2.0     # monotonicity-penalty weight (Eq. 15)
    loss_mode: str = "asymmetric"  # "asymmetric" (uncapped, bounded reward -- targets
                                    #   win-rate AND magnitude at once, see module docstring)
                                # | "hinge" (margin vs. plain DR, targets win-rate only, capped magnitude)
                                # | "log_ratio" (uncapped but exploitable via its floor -- NOT recommended)
                                # | "absolute" (Eq. 15, validated but uneven)
                                # | "relative" (reweighted, measured WORSE -- see module docstring)
    eta: float = 0.15           # required fractional margin over plain DR ("hinge" mode)
    floor: float = 1e-3         # reward floor, normalised units ("log_ratio" mode, not recommended)
    w_behind: float = 2.0       # penalty weight for lagging plain DR ("asymmetric" mode)
    w_ahead: float = 1.0        # reward weight for beating plain DR ("asymmetric" mode)
    omega_power: float = 0.0    # iteration-weight exponent omega_k=(k/K)^omega_power, normalised
                                # to mean 1 ("asymmetric" mode, omega_mode="fixed"). 0 = uniform
                                # (paper Eq. 16 default). >0 up-weights LATE iterations -- see
                                # module docstring's note on a uniform average letting a cheap
                                # early lead pay for a costly late plateau that a uniform mean
                                # doesn't penalise proportionally to how much it actually costs in
                                # total iterations.
    omega_mode: str = "fixed"  # "fixed" (omega_power's k/K schedule) | "dr_slope" (instance-
                                # adaptive: weight by the reference plain-DR trajectory's own
                                # local log-decay rate instead of raw iteration index -- see
                                # asymmetric_loss's docstring; meant to transfer better to shifted
                                # markets since it isn't tied to a fixed iteration index)
    epochs: int = 25
    lr: float = 3e-3
    grad_clip: float = 1.0
    seed: int = 0
    eps: float = 1e-9


def rollout(net, gnn: Optional[PerturbationGNN], cfg: TrainConfig,
            x0: Optional[torch.Tensor] = None, differentiable: bool = True
            ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Unroll `cfg.unroll` DR iterations, perturbed by `gnn` if given.

    Returns (x_final, residuals), residuals[k] = ||x^{k+1}-x^k|| measured
    AFTER the perturbation is added (matches what a real deployment's own
    stopping rule would see).
    """
    import network as network_mod

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
            refl = 2.0 * xbar - x                                   # line 5,7
            z = net.prox_local(cfg.gamma, refl, n_bisect=cfg.bisect)  # line 9
            x_new = x + cfg.lam * (z - xbar)                         # line 10

            if gnn is not None:
                direction_fast, direction_slow, mem = gnn.forward_centralised(
                    net, x, mem, prev_res, stall)                      # line 11
                amp_fast = gnn.cfg.scale * gnn.decay(k)
                amp_slow = gnn.cfg.scale_slow * gnn.decay_slow(k)
                x_new = x_new + amp_fast * direction_fast + amp_slow * direction_slow  # line 12
                prev_res = torch.sqrt(net.per_node_move_sq(x_new, x).detach() + 1e-12)
                d = network_mod.STALL_EMA_DECAY
                stall = d * stall + (1.0 - d) * prev_res

            residuals.append(torch.linalg.vector_norm(x_new - x))
            xbar = net.project_reciprocity(x_new)
            x = x_new

    return x, residuals


def stability_loss(residuals: List[torch.Tensor], scale: torch.Tensor,
                    gamma_mono: float, eps: float) -> torch.Tensor:
    """Eq. 15 with omega_k = 1 -- see module docstring."""
    denom = torch.linalg.vector_norm(scale) + eps
    rho = torch.stack(residuals) / denom
    decay = rho.mean()
    mono = (torch.relu(rho[1:] - rho[:-1]).mean()
            if rho.numel() > 1 else torch.zeros((), dtype=rho.dtype))
    return decay + gamma_mono * mono


def relative_loss(residuals: List[torch.Tensor], residuals_dr: List[torch.Tensor],
                   scale: torch.Tensor, gamma_mono: float, eps: float) -> torch.Tensor:
    """Baseline-REWEIGHTED loss -- see module docstring. `residuals_dr` must
    come from a `differentiable=False` rollout (no grad; it only supplies
    per-step IMPORTANCE WEIGHTS, never a training target to fit -- an
    additive baseline here would have zero effect on the gradient, see the
    module docstring's note)."""
    denom = torch.linalg.vector_norm(scale) + eps
    rho = torch.stack(residuals) / denom
    rho_dr = (torch.stack(residuals_dr) / denom).detach()
    w = rho_dr / (rho_dr.mean() + eps)
    decay = (w * rho).mean()
    mono = (torch.relu(rho[1:] - rho[:-1]).mean()
            if rho.numel() > 1 else torch.zeros((), dtype=rho.dtype))
    return decay + gamma_mono * mono


def hinge_loss(residuals: List[torch.Tensor], residuals_dr: List[torch.Tensor],
               scale: torch.Tensor, gamma_mono: float, eps: float,
               eta: float = 0.15) -> torch.Tensor:
    """Margin-hinge loss against the plain-DR baseline -- targets a
    MAJORITY/ALL-instances win rate directly, rather than a positive
    average.

        rho^k     = ||x^{k+1}-x^k||     / (||sigma||+eps)   (perturbed, w/ grad)
        rho^k_dr  = ||x^{k+1}_dr-x^k_dr|| / (||sigma||+eps)  (plain DR, no grad)
        target^k  = (1 - eta) * rho^k_dr                     (detached)
        L = mean_k[ relu(rho^k - target^k) ] + gamma_mono * mean_k[relu(rho^{k+1}-rho^k)]

    Every iteration k with rho^k already at or below `target^k` (i.e. at
    least an `eta` fraction faster than plain DR at that same step)
    contributes EXACTLY ZERO to the loss and gets ZERO gradient: there is
    no reward for "beating DR by even more" once the margin is met, but
    also no way to avoid the penalty except by genuinely reaching it, so
    "do nothing everywhere" (rho^k = rho^k_dr) is NOT a zero-loss solution
    -- it costs eta * rho^k_dr at every step, which is what keeps the
    network from collapsing to the always-safe no-op instead of actually
    trying to help. This is a HARD requirement (falling short by anything
    less than `eta` still costs something) rather than the earlier
    reweighting attempt's soft one, and unlike an unweighted mean it does
    not evaporate on the tail: at the tail rho^k_dr is small too, so
    target^k is small, and any lag there (the learned trajectory not
    having converged as fast as DR right when it matters for "iterations
    to tolerance") still produces a proportional penalty -- exactly where
    the reweighted version lost its signal instead.

    MEASURED RESULT: this is the one that actually moved win-rate, not just
    the average. Same cluster dataset/gamma as the other two: `stability_loss`
    (Eq. 15) win-rate 61% overall (7/12 in-cluster, 4/6 out-of-cluster);
    `hinge_loss` (eta=0.15) 72% overall (9/12, 4/6) -- with a LOWER mean
    speed-up (1.05x vs 1.08x) than Eq. 15, confirming it is trading average
    magnitude for reliability, exactly as designed. Two follow-up levers
    were tried and did NOT improve on this: raising eta to 0.25 (more
    aggressive corrections) dropped win-rate to 58% despite a higher median
    speed-up -- more margin demanded produced bigger, riskier interventions,
    not more careful ones; and extending the correction's active horizon
    (rho 0.95->0.97, unroll 200->400, meant to let training observe the
    late-trajectory plateau diagnosed below) dropped win-rate to 33% --
    active for longer means MORE, not less, disturbance to that plateau.
    Shrinking `GNNConfig.scale` (0.7->0.4) at eta=0.15 held win-rate at 72%
    with a slightly better median (1.09x) and strictly less disturbance to
    the instances that still lose -- kept as the final configuration.

    The residual ~28% is not noise: the SAME handful of instances lose under
    every variant above. Direct trajectory inspection on one of them (N=10,
    plain DR 446 iterations) shows the learned trajectory strictly AHEAD of
    plain DR through k~200 (e.g. k=170: 0.032 vs 0.010), then BOTH runs hit
    a long, nearly flat plateau around residual~0.0055 from k~220 to k~340+
    -- plain DR happens to break out around k~360, the learned run is still
    stuck there past k=440 (finishes at 640). The correction has fully
    decayed to ~0 by k~150-200 in this configuration, so it cannot be
    "wrong" during the plateau itself -- it just leaves the trajectory at a
    different point than plain DR would have reached by k~200, and this
    particular instance's plateau-exit timing is highly sensitive to that
    starting point. Training only ever unrolls to k=200-400, so it never
    observes the plateau or its exit at all for instances whose real
    horizon is 1000-3400+ iterations -- there is no gradient signal telling
    the network which early trajectories land favourably relative to a
    plateau it has never seen. This looks like a genuine limit of a SINGLE
    SHARED policy trained this way, not a bug: closing it further would need
    either training signal that actually reaches these late regions (tried
    once, made things worse -- see above) or an instance-adaptive amplitude
    that is bigger risk to design well than what remained in scope here.
    """
    denom = torch.linalg.vector_norm(scale) + eps
    rho = torch.stack(residuals) / denom
    rho_dr = (torch.stack(residuals_dr) / denom).detach()
    target = (1.0 - eta) * rho_dr
    decay = torch.relu(rho - target).mean()
    mono = (torch.relu(rho[1:] - rho[:-1]).mean()
            if rho.numel() > 1 else torch.zeros((), dtype=rho.dtype))
    return decay + gamma_mono * mono


def log_ratio_loss(residuals: List[torch.Tensor], residuals_dr: List[torch.Tensor],
                    scale: torch.Tensor, gamma_mono: float, eps: float,
                    floor: float = 1e-3) -> torch.Tensor:
    """Floored log-ratio against the plain-DR baseline -- unlike
    `hinge_loss`, this has NO margin and NO cutoff: every iteration is
    continuously rewarded for beating DR (more reward the further ahead)
    and continuously penalised for lagging it (more penalty the further
    behind), so it asks for higher win-rate AND higher average magnitude
    at once, instead of trading one for the other.

        rho^k     = ||x^{k+1}-x^k||     / (||sigma||+eps)   (perturbed, w/ grad)
        rho^k_dr  = ||x^{k+1}_dr-x^k_dr|| / (||sigma||+eps)  (plain DR, no grad)
        L = mean_k[ log(rho^k+floor) - log(rho^k_dr+floor) ]
            + gamma_mono * mean_k[relu(rho^{k+1}-rho^k)]

    Why a FLOOR, not raw log(rho^k): raw log(rho^k) is exactly the first
    loss design this project tried (see `stability_loss`'s sibling
    write-up in the module docstring history) and it was abandoned for a
    concrete reason -- log(rho^k) -> -infinity as a trajectory nearly
    freezes (rho^k -> 0), an UNBOUNDED reward for a state that need not be
    anywhere near the true equilibrium, measured to actively make things
    worse end-to-end. Adding a fixed floor bounds that reward.

    NOT RECOMMENDED, CAUGHT BEFORE TRAINING: a FIXED floor does not scale
    with rho^k_dr, which itself spans several orders of magnitude over a
    trajectory (~0.7 down to ~1e-2), so its relative effect is wildly
    inconsistent -- checked numerically (no training needed) with
    rho^k_dr=0.3, floor=1e-3: a frozen trajectory scores -5.71, a
    genuinely 10x-faster one only -2.27. Freezing is rewarded MORE than
    2x as strongly as a real, large, honest speed-up -- precisely the
    exploit the floor was meant to close, reopened by the floor being too
    small relative to typical rho^k_dr early in a trajectory (and, by the
    same argument, too large relative to rho^k_dr near the tail, where it
    would instead flatten genuine differences). `asymmetric_loss` below
    fixes this a different way -- bounding the reward for beating DR by
    something that scales with rho^k_dr itself (not a tunable constant)
    -- and is what `Trainer` actually uses; kept here as a documented
    dead end, the same way `relative_loss` was.
    """
    denom = torch.linalg.vector_norm(scale) + eps
    rho = torch.stack(residuals) / denom
    rho_dr = (torch.stack(residuals_dr) / denom).detach()
    decay = (torch.log(rho + floor) - torch.log(rho_dr + floor)).mean()
    mono = (torch.relu(rho[1:] - rho[:-1]).mean()
            if rho.numel() > 1 else torch.zeros((), dtype=rho.dtype))
    return decay + gamma_mono * mono


def asymmetric_loss(residuals: List[torch.Tensor], residuals_dr: List[torch.Tensor],
                     scale: torch.Tensor, gamma_mono: float, eps: float,
                     w_behind: float = 2.0, w_ahead: float = 1.0,
                     omega_power: float = 0.0, omega_mode: str = "fixed") -> torch.Tensor:
    """Asymmetrically-weighted LINEAR difference against the plain-DR
    baseline -- continuous and uncapped like `log_ratio_loss` (no margin,
    no cutoff: still rewarded for beating DR by more, still penalised for
    lagging by more), but the reward for beating DR is bounded by
    rho^k_dr ITSELF rather than by a tunable floor, closing the freezing
    exploit that sank `log_ratio_loss` without needing one.

        a^k  = rho^k - rho^k_dr                         (rho^k has grad)
        L = mean_k[ w_behind * relu(a^k) - w_ahead * relu(-a^k) ]
            + gamma_mono * mean_k[relu(rho^{k+1}-rho^k)]

    Bounded reward, by construction, not by a floor: as rho^k -> 0
    (frozen), a^k -> -rho^k_dr, so the reward -> -w_ahead * rho^k_dr, a
    FINITE value set by how big DR's own step was -- not an artificial
    constant. Checked numerically before training (rho^k_dr=0.3,
    w_behind=2, w_ahead=1): frozen scores -0.30, a genuine 10x-faster
    trajectory scores -0.27 -- freezing is barely better than genuinely
    converging fast, not ~2.5x better the way it was under
    `log_ratio_loss`'s floor. `w_behind > w_ahead` (2:1 here) penalises
    falling behind DR more than it rewards pulling ahead, biasing toward
    the win-rate side of the win-rate/magnitude trade-off while still
    giving uncapped credit for genuine speed-ups on the instances where
    that is safely achievable.

    NOTE: using w_behind == w_ahead here would make this loss
    mathematically IDENTICAL to `stability_loss` in its gradient (the
    relu-difference collapses to the plain signed difference rho^k -
    rho^k_dr, and rho^k_dr is a detached constant, so it drops out of the
    gradient entirely -- exactly the mistake documented in
    `relative_loss`'s note). The asymmetry is not a tuning nicety here,
    it is the entire mechanism by which this loss differs from
    `stability_loss` at all.
    """
    denom = torch.linalg.vector_norm(scale) + eps
    rho = torch.stack(residuals) / denom
    rho_dr = (torch.stack(residuals_dr) / denom).detach()
    a = rho - rho_dr
    per_step = w_behind * torch.relu(a) - w_ahead * torch.relu(-a)
    if omega_mode == "dr_slope":
        # Instance-ADAPTIVE alternative to the fixed k-based schedule below:
        # weight each step by how much SLOWER than typical the REFERENCE
        # plain-DR trajectory is progressing there, in relative/geometric
        # terms (log-space, so this is invariant to the absolute size of
        # rho_dr -- a plateau near the end of a long trajectory and one near
        # the start both get flagged the same way, unlike a fixed k-schedule
        # which only knows ABSOLUTE position). A genuine plateau (rho_dr
        # barely moving relative to itself) gets up-weighted; a steep,
        # healthy decay region gets down-weighted. Because this reads the
        # SAME instance's own rho_dr fresh every call, it follows wherever
        # that instance's plateau actually is, instead of assuming it is
        # always at the same iteration index -- meant to generalise better
        # to shifted markets than `omega_power`'s fixed schedule (see
        # runs/cross_density density=0.8 zero-shot D*/G_res result, which
        # dropped to 67% win-rate under the fixed schedule).
        log_rho_dr = torch.log(rho_dr + eps)
        log_decrease = torch.clamp(log_rho_dr[:-1] - log_rho_dr[1:], min=0.0)
        typical = log_decrease.mean().clamp(min=eps)
        omega_local = typical / (log_decrease + 0.1 * typical)   # capped ~10x typical at a true plateau
        # EXCLUDE the "already converged" tail from up-weighting: if this x0
        # draw's plain-DR trajectory simply finishes before the unroll window
        # ends (common on markets with wide x0-heterogeneity, e.g.
        # density=0.5), rho_dr goes flat there because there is nothing left
        # to do, not because of a genuine plateau -- without this guard that
        # trivial flat tail gets the SAME up-weighting as a real plateau,
        # which pulls training's attention away from the genuine difficulty
        # (confirmed: broke density=0.5 outright, 1.81x/100% -> 0.82x/0%,
        # while density=0.8's own unroll was short enough relative to its
        # mean that few instances hit this case). "Already converged" =
        # within 2x of this trajectory's own final value.
        still_converging = rho_dr[:-1] > 2.0 * rho_dr[-1]
        omega_local = torch.where(still_converging, omega_local, torch.ones_like(omega_local))
        omega_local = torch.cat([omega_local, omega_local[-1:]])
        omega = omega_local / omega_local.mean()
        decay = (omega * per_step).mean()
    elif omega_power != 0.0:
        K = rho.shape[0]
        k_frac = torch.arange(1, K + 1, dtype=rho.dtype, device=rho.device) / K
        omega = k_frac ** omega_power
        omega = omega / omega.mean()          # keep overall loss scale comparable to uniform
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
        self.history = {"train": [], "val": []}

    def _instance_loss(self, net, differentiable: bool, fixed_x0: Optional[torch.Tensor] = None
                        ) -> torch.Tensor:
        cfg = self.cfg
        if fixed_x0 is not None:
            net.unflatten(fixed_x0)
        x0 = net.flatten().detach()
        _, residuals = rollout(net, self.gnn, cfg, x0=x0, differentiable=differentiable)
        scale = self.gnn.physical_scale(net)
        if cfg.loss_mode == "absolute":
            return stability_loss(residuals, scale, cfg.gamma_mono, cfg.eps)
        elif cfg.loss_mode == "relative":
            _, residuals_dr = rollout(net, None, cfg, x0=x0, differentiable=False)
            return relative_loss(residuals, residuals_dr, scale, cfg.gamma_mono, cfg.eps)
        elif cfg.loss_mode == "hinge":
            _, residuals_dr = rollout(net, None, cfg, x0=x0, differentiable=False)
            return hinge_loss(residuals, residuals_dr, scale, cfg.gamma_mono, cfg.eps, cfg.eta)
        elif cfg.loss_mode == "log_ratio":
            _, residuals_dr = rollout(net, None, cfg, x0=x0, differentiable=False)
            return log_ratio_loss(residuals, residuals_dr, scale, cfg.gamma_mono, cfg.eps, cfg.floor)
        elif cfg.loss_mode == "asymmetric":
            _, residuals_dr = rollout(net, None, cfg, x0=x0, differentiable=False)
            return asymmetric_loss(residuals, residuals_dr, scale, cfg.gamma_mono, cfg.eps,
                                    cfg.w_behind, cfg.w_ahead, cfg.omega_power, cfg.omega_mode)
        raise ValueError(f"unknown loss_mode {cfg.loss_mode!r}")

    def fit(self, train_specs=None, val_specs=None, device=None, verbose: bool = True,
            fixed_x0: Optional[torch.Tensor] = None, resample_specs_fn=None):
        """`fixed_x0`, if given (a flattened [D,G,Q] tensor, `Network.flatten`
        layout), OVERRIDES the usual per-(instance,epoch) random x0 draw with
        this SAME tensor for every instance/epoch -- used when the diversity
        axis under training is the market's own parameters (D*/G_res/box
        bounds), not x0. `unflatten` sets state directly with no feasibility
        check against that instance's own (possibly-shifted) box; DR handles
        an out-of-box start fine (`local_prox` clamps back onto the box on
        the very first step), so this is safe, just not a feasible x0 for
        every perturbed instance -- by design here, not an oversight.

        `resample_specs_fn`, if given (a zero-arg callable returning a list
        of MarketSpecs, typically closing over its own per-call rng/seed),
        is called ONCE PER EPOCH to get a FRESH `train_specs` list for that
        epoch -- mirrors what the usual per-(instance,epoch) random x0 draw
        already does implicitly for x0-diversity training (25 "slots" x 60
        epochs = 1500 distinct draws, not 25): when `fixed_x0` removes that
        free per-epoch diversity, resampling the SPEC list instead restores
        the same order of magnitude of distinct training examples seen over
        a full run, rather than looping the same fixed small list every
        epoch. Mutually exclusive with a static `train_specs` list (pass
        exactly one of the two)."""
        import market as market_mod
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        device = device or torch.device("cpu")
        assert (train_specs is None) != (resample_specs_fn is None), \
            "pass exactly one of train_specs (static list) or resample_specs_fn (resampled per epoch)"

        for epoch in range(cfg.epochs):
            self.gnn.train()
            total = 0.0
            epoch_specs = resample_specs_fn() if resample_specs_fn is not None else train_specs
            for i, spec in enumerate(epoch_specs):
                if fixed_x0 is not None:
                    net = market_mod.build_network(spec, device=device)
                else:
                    gen = torch.Generator(device=device)
                    gen.manual_seed(cfg.seed * 1_000_003 + epoch * 7_919 + i)
                    net = market_mod.build_network(spec, device=device, rng=gen)

                self.opt.zero_grad()
                loss = self._instance_loss(net, differentiable=True, fixed_x0=fixed_x0)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), cfg.grad_clip)
                self.opt.step()
                total += loss.item()
            train_loss = total / max(len(epoch_specs), 1)
            self.history["train"].append(train_loss)

            val_loss = None
            if val_specs:
                val_loss = self.evaluate(val_specs, device=device, fixed_x0=fixed_x0)
                self.history["val"].append(val_loss)

            if verbose:
                msg = f"  epoch {epoch + 1:3d}/{cfg.epochs} | train {train_loss:+.4f}"
                if val_loss is not None:
                    msg += f" | val {val_loss:+.4f}"
                print(msg, flush=True)
        return self.history

    @torch.no_grad()
    def evaluate(self, specs, device=None, fixed_x0: Optional[torch.Tensor] = None) -> float:
        import market as market_mod
        self.gnn.eval()
        device = device or torch.device("cpu")
        total = 0.0
        for spec in specs:
            net = market_mod.build_network(spec, device=device)
            total += float(self._instance_loss(net, differentiable=False, fixed_x0=fixed_x0))
        return total / max(len(specs), 1)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.gnn.state_dict(), "cfg": self.gnn.cfg.__dict__}, path)
