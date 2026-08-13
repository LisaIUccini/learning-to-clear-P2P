"""
prosumer.py
===========

Model of a single prosumer in a peer-to-peer (P2P) energy market.

This module contains **only the model** of one market participant:
its fixed physical parameters, its decision variables, its local cost
and gradient, and the minimal communication buffer needed to reason
about the reciprocity coupling with neighbours.

The prosumer is also the COMPUTING UNIT of the distributed algorithm:
it runs the Douglas-Rachford steps on its own data (see `dr_step`).
There is no central solver -- `DRSolver` only schedules the rounds.

----------------------------------------------------------------------
Market model (from Le Cadre et al. 2020, as used in the L2Play paper)
----------------------------------------------------------------------

The distribution network is a connected graph G = (N, E). Each node
n in N is a prosumer with a set of neighbours Omega_n (the peers it may
trade with). A prosumer controls:

    D_n >= 0        flexible demand
    G_n >= 0        controllable local generation (flexibility activation)
    q_{mn} in R     bilateral trade with neighbour m in Omega_n
                    q_{mn} > 0: n buys from m ;  q_{mn} < 0: n sells to m

and is subject to a fixed renewable injection Delta G_n >= 0 (RES output,
here called `G_res`) which is a parameter, not a decision.

Local feasible set  X^loc_n :
    D_min_n <= D_n <= D_max_n                         (demand box)
    G_min_n <= G_n <= G_max_n                         (generation box)
    q_{mn}  <= kappa_{mn}          for all m           (link capacity)
    D_n = G_n + Delta G_n + Q_n,   Q_n = sum_m q_{mn}  (energy balance)

Reciprocity coupling  X^rec  (couples neighbours, handled at the
network level):
    q_{mn} + q_{nm} <= 0           for all edges (m, n)

Local cost  J_n(x_n) = C_n(G_n) + C~_n(q_n) - U_n(D_n) :
    C_n(G_n)  = 1/2 a_n G_n^2 + b_n G_n            (+ const d_n, dropped)
    C~_n(q_n) = sum_m c_{nm} q_{mn}
    U_n(D_n)  = - a~_n (D_n - D*_n)^2              (+ const b~_n, dropped)

so, dropping the additive constants (they change neither the gradient
nor the equilibrium):

    J_n = a~_n (D_n - D*_n)^2 + 1/2 a_n G_n^2 + b_n G_n + sum_m c_{nm} q_{mn}

with gradient (the local block of the game pseudo-gradient F):

    dJ_n/dD_n     = 2 a~_n (D_n - D*_n)
    dJ_n/dG_n     = a_n G_n + b_n
    dJ_n/dq_{mn}  = c_{nm}
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Union

import torch

Number = Union[float, int, torch.Tensor]


def _as_scalar(value: Number, device: torch.device,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalise a python number or tensor to a 0-dim tensor on `device`."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(value), device=device, dtype=dtype)


class Prosumer:
    """A single prosumer / market participant.

    Parameters
    ----------
    node_id : int
        Global index of this prosumer in the network.
    neighbors : sequence of int
        Indices of the peers this prosumer can trade with (Omega_n).
        Stored sorted; this fixes the local ordering of `q`.
    D_min, D_max : number
        Lower / upper bound on flexible demand D_n.
    G_min, G_max : number
        Lower / upper bound on controllable generation G_n.
    G_res : number
        Fixed renewable injection Delta G_n (>= 0), not optimised.
    a_tilde : number
        Curvature a~_n of the demand-usage benefit (> 0).
    a, b : number
        Generation-cost coefficients (a_n > 0).
    D_star : number
        Target demand D*_n.
    c : dict {m: number}
        Per-unit trading cost c_{nm} for each neighbour m.
    k : dict {m: number}
        Link capacity kappa_{mn} for each neighbour m.
    device : torch.device, optional
        Defaults to CPU.

    Notes
    -----
    Decision variables D, G, q are stored as 0-dim tensors so that they
    can carry gradients later (needed for GNN training). The state is
    initialised inside the feasible boxes by `init_state`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        node_id: int,
        neighbors: Sequence[int],
        D_min: Number,
        D_max: Number,
        G_min: Number,
        G_max: Number,
        G_res: Number,
        a_tilde: Number,
        a: Number,
        b: Number,
        D_star: Number,
        c: Dict[int, Number],
        k: Dict[int, Number],
        device: Optional[torch.device] = None,
        init: bool = True,
    ):
        self.node_id = int(node_id)
        self.neighbors: List[int] = sorted(int(m) for m in neighbors)
        self.device = device if device is not None else torch.device("cpu")

        # --- fixed scalar parameters (0-dim tensors) ---
        self.D_min = _as_scalar(D_min, self.device)
        self.D_max = _as_scalar(D_max, self.device)
        self.G_min = _as_scalar(G_min, self.device)
        self.G_max = _as_scalar(G_max, self.device)
        self.G_res = _as_scalar(G_res, self.device)
        self.a_tilde = _as_scalar(a_tilde, self.device)
        self.a = _as_scalar(a, self.device)
        self.b = _as_scalar(b, self.device)
        self.D_star = _as_scalar(D_star, self.device)

        # --- per-neighbour parameters ---
        self.c = {m: _as_scalar(c[m], self.device) for m in self.neighbors}
        self.k = {m: _as_scalar(k[m], self.device) for m in self.neighbors}

        self._zero = torch.zeros((), device=self.device)

        # --- decision variables (filled by init_state) ---
        self.D: torch.Tensor = self._zero.clone()
        self.G: torch.Tensor = self._zero.clone()
        self.q: Dict[int, torch.Tensor] = {m: self._zero.clone()
                                           for m in self.neighbors}

        # --- communication buffer ---
        # q_received[m] = q_{nm}, i.e. neighbour m's trade decision toward n.
        # Needed to evaluate the reciprocity coupling q_{mn} + q_{nm} <= 0.
        self.q_received: Dict[int, torch.Tensor] = {m: self._zero.clone()
                                                    for m in self.neighbors}

        # --- residual bookkeeping for the distributed iteration ---
        self._prev_shadow = None
        self.residual_sq = torch.zeros((), device=self.device)
        self.shadow_residual_sq = torch.zeros((), device=self.device)

        # --- on-board copy of the trained perturbation model (line 8) ---
        # None -> plain DR. Set by gnn.deploy() after centralised training.
        # `_round` is this node's OWN count of dr_step calls since the last
        # reset: purely local bookkeeping (no message needed) that stays in
        # lockstep with every other node because the scheduler calls
        # dr_step exactly once per synchronous round. It drives the
        # decaying amplitude rho^_round of the learned perturbation.
        self.gnn_agent = None
        self._round = 0
        # own recurrent memory (gnn.PerturbationGNN's GRUCell state) and the
        # local move of the PREVIOUS iteration -- both purely private state,
        # lazily sized to gnn_agent's hidden_dim on first use (see
        # local_gnn_state), reset alongside everything else below.
        self._mem = None
        self._prev_residual = torch.zeros((), device=self.device)

        if init:
            self.init_state()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def n_neighbors(self) -> int:
        return len(self.neighbors)

    @property
    def local_dim(self) -> int:
        """Dimension of the local decision vector x_n = (D, G, q)."""
        return 2 + self.n_neighbors

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    def init_state(self, generator: Optional[torch.Generator] = None) -> "Prosumer":
        """Sample D, G, q inside their box constraints (uniform).

        `q` is drawn in [-kappa, kappa]; note the model only *constrains*
        q from above (q <= kappa), the negative range is physical (selling)
        and simply gives a symmetric, feasible starting point.
        Balance / reciprocity are *not* enforced here — they are restored
        by the algorithm's projection steps.
        """
        def _rand():
            return torch.rand((), device=self.device, generator=generator)

        self.D = self.D_min + (self.D_max - self.D_min) * _rand()
        self.G = self.G_min + (self.G_max - self.G_min) * _rand()
        self.q = {m: (2.0 * _rand() - 1.0) * self.k[m] for m in self.neighbors}
        self.reset_residuals()
        return self

    # ------------------------------------------------------------------
    # Vector view of q  (ordered by self.neighbors)
    # ------------------------------------------------------------------
    def q_vector(self) -> torch.Tensor:
        """Return q as a 1-D tensor ordered by `self.neighbors`, shape (k,)."""
        if not self.neighbors:
            return torch.zeros(0, device=self.device)
        return torch.stack([self.q[m] for m in self.neighbors])

    def set_q_vector(self, vec: torch.Tensor) -> None:
        """Set q from a 1-D tensor ordered by `self.neighbors`."""
        for i, m in enumerate(self.neighbors):
            self.q[m] = vec[i]

    def flatten(self) -> torch.Tensor:
        """Local state vector x_n = [D, G, q[m0], q[m1], ...], shape (2+k,)."""
        return torch.cat([self.D.reshape(1), self.G.reshape(1), self.q_vector()])

    def unflatten(self, x: torch.Tensor) -> None:
        """Inverse of `flatten`: set D, G, q from a (2+k,) vector."""
        self.D = x[0]
        self.G = x[1]
        for i, m in enumerate(self.neighbors):
            self.q[m] = x[2 + i]

    # ------------------------------------------------------------------
    # Local quantities
    # ------------------------------------------------------------------
    def net_import(self) -> torch.Tensor:
        """Q_n = sum_{m in Omega_n} q_{mn}  (0-dim tensor)."""
        if not self.neighbors:
            return self._zero.clone()
        return self.q_vector().sum()

    def local_cost(self) -> torch.Tensor:
        """J_n(x_n) with additive constants dropped (0-dim tensor)."""
        cost = (
            self.a_tilde * (self.D - self.D_star) ** 2
            + 0.5 * self.a * self.G ** 2
            + self.b * self.G
        )
        if self.neighbors:
            cost = cost + torch.stack(
                [self.c[m] * self.q[m] for m in self.neighbors]
            ).sum()
        return cost

    # --- pseudo-gradient blocks (local part of F) ---
    def grad_D(self) -> torch.Tensor:
        return 2.0 * self.a_tilde * (self.D - self.D_star)

    def grad_G(self) -> torch.Tensor:
        return self.a * self.G + self.b

    def grad_q(self, m: int) -> torch.Tensor:
        return self.c[m]

    def pseudo_gradient(self) -> torch.Tensor:
        """Local block of F, aligned with `flatten`: shape (2+k,)."""
        parts = [self.grad_D().reshape(1), self.grad_G().reshape(1)]
        if self.neighbors:
            parts.append(torch.stack([self.grad_q(m) for m in self.neighbors]))
        else:
            parts.append(torch.zeros(0, device=self.device))
        return torch.cat(parts)

    # ------------------------------------------------------------------
    # Communication (one-hop)
    # ------------------------------------------------------------------
    def send_q_to(self, other: "Prosumer", training: bool = False) -> None:
        """Push this node's decision q_{other} into `other`'s receive buffer.

        After the call: other.q_received[self.node_id] = self.q[other.node_id].
        With training=False the value is detached (true distributed setting);
        with training=True the gradient link across nodes is preserved.
        """
        val = self.q[other.node_id]
        other.q_received[self.node_id] = val if training else val.detach().clone()

    # ==================================================================
    # DISTRIBUTED DOUGLAS-RACHFORD  (Algorithm 1, lines 4-8)
    # ------------------------------------------------------------------
    # Everything below runs on this prosumer alone. The ONLY external
    # information used is `q_received`, filled by the one-hop message
    # exchange of line 3 (EnergyNetwork.communicate).
    # ==================================================================
    def shadow_q(self) -> Dict[int, torch.Tensor]:
        """Line 4: shadow projection of the trades onto X^rec.

            qbar_{mn} = q_{mn} - max(0, q_{mn} + q_{nm}) / 2

        Requires a fresh `q_received`. D and G are left untouched, since
        the reciprocity set only constrains the trades: the full shadow of
        this node is therefore (D, G, qbar).
        """
        return {m: self.q[m] - torch.relu(self.q[m] + self.q_received[m]) / 2.0
                for m in self.neighbors}

    def local_prox(self, gamma: float, rD: torch.Tensor, rG: torch.Tensor,
                   rq: Dict[int, torch.Tensor], n_bisect: int = 60):
        """Line 6: this prosumer's own prox, solved locally.

            argmin_{xi in X^loc_n}  gamma * J_n(xi) + 1/2 ||xi - r||^2

        The objective is separable in (D, G, q) and the only coupling
        inside X^loc_n is this node's energy balance
        D = G + G_res + sum_m q_{mn}. Dualising that single equality with a
        scalar multiplier nu gives closed forms

            D(nu) = clip( (2 gamma a~ D* + rD + nu) / (2 gamma a~ + 1) )
            G(nu) = clip( (-gamma b + rG - nu) / (gamma a + 1) )
            q_m(nu) = min( rq_m - gamma c_m - nu , kappa_m )

        and the balance residual is monotone increasing in nu, so nu* is
        found by a scalar bisection performed by this node. No communication.
        """
        g = float(gamma)
        denomD = 2.0 * g * self.a_tilde + 1.0
        denomG = g * self.a + 1.0

        def D_of(nu):
            return torch.clamp((2.0 * g * self.a_tilde * self.D_star + rD + nu)
                               / denomD, self.D_min, self.D_max)

        def G_of(nu):
            return torch.clamp((-g * self.b + rG - nu) / denomG,
                               self.G_min, self.G_max)

        def q_of(nu):
            return {m: torch.minimum(rq[m] - g * self.c[m] - nu, self.k[m])
                    for m in self.neighbors}

        # Search loop below only needs the SUM of q_m(nu), not the individual
        # values -- precompute rq/c/k as vectors once so `balance(nu)` is a
        # single vectorised op per bisection call instead of rebuilding and
        # stacking a Python dict ~80+n_bisect times. The per-neighbour dict
        # is only materialised once, at the end, via q_of(nu) (identical
        # computation to before, so the returned values are unchanged).
        if self.neighbors:
            rq_vec = torch.stack([rq[m] for m in self.neighbors])
            c_vec = torch.stack([self.c[m] for m in self.neighbors])
            k_vec = torch.stack([self.k[m] for m in self.neighbors])

            def q_sum(nu):
                return torch.minimum(rq_vec - g * c_vec - nu, k_vec).sum()
        else:
            def q_sum(nu):
                return torch.zeros((), device=self.device)

        def balance(nu):
            return D_of(nu) - G_of(nu) - self.G_res - q_sum(nu)

        lo = -torch.ones((), device=self.device)
        hi = torch.ones((), device=self.device)
        for _ in range(80):                       # bracket the root
            need_lo = bool(balance(lo) > 0)
            need_hi = bool(balance(hi) < 0)
            if not need_lo and not need_hi:
                break
            if need_lo:
                lo = lo * 2.0 - 1.0
            if need_hi:
                hi = hi * 2.0 + 1.0
        for _ in range(n_bisect):                 # bisect
            mid = 0.5 * (lo + hi)
            if bool(balance(mid) > 0):
                hi = mid
            else:
                lo = mid
        nu = 0.5 * (lo + hi)
        return D_of(nu), G_of(nu), q_of(nu)

    def local_gnn_state(self):
        """Assemble the strictly local input of the perturbation model.

        Contains only this prosumer's own state, own fixed cost/box
        parameters, its own recurrent memory and previous-residual signal,
        and the q_{nm} received during line 3 -- nothing belonging to a
        non-neighbour. Built BEFORE the DR update, so the learned map
        consumes exactly the information already on hand right after the
        communication round: no second exchange.
        """
        from gnn import LocalNodeState
        if self._mem is None:
            self._mem = self.gnn_agent.gnn.init_local_mem(device=self.device)
        return LocalNodeState(
            node_id=self.node_id, neighbors=list(self.neighbors),
            D=self.D, G=self.G, D_star=self.D_star, G_res=self.G_res,
            D_min=self.D_min, D_max=self.D_max,
            G_min=self.G_min, G_max=self.G_max,
            a_tilde=self.a_tilde, a=self.a, b=self.b,
            q_own={m: self.q[m] for m in self.neighbors},
            q_recv={m: self.q_received[m] for m in self.neighbors},
            c={m: self.c[m] for m in self.neighbors},
            kappa={m: self.k[m] for m in self.neighbors},
            mem=self._mem, prev_residual=self._prev_residual,
        )

    def dr_step(self, gamma: float, lam: float = 1.0,
                n_bisect: int = 60) -> None:
        """One full local iteration: lines 4 to 8 of Algorithm 1.

        Call `EnergyNetwork.communicate()` first, so that `q_received`
        holds this round's q_{nm} values. If `self.gnn_agent` is None the
        method runs plain DR (lines 4 to 7).
        """
        if len(self.q_received) != self.n_neighbors:
            raise RuntimeError(
                f"prosumer {self.node_id}: {len(self.q_received)} of "
                f"{self.n_neighbors} messages available. Run "
                "EnergyNetwork.communicate() before dr_step()."
            )

        # --- line 8 is EVALUATED HERE, on the pre-update state -----------
        # The features must be the ones available right after line 3; the
        # value is applied after the DR update, as in Algorithm 1. `_round`
        # is this node's OWN count of completed dr_step calls -- it equals
        # the global iteration index k with no communication needed, since
        # every node is called exactly once per synchronous round.
        v = None
        new_mem = None
        if self.gnn_agent is not None:
            v_D, v_G, v_q, new_mem = self.gnn_agent.perturbation(
                self.local_gnn_state(), self._round)
            v = (v_D, v_G, v_q)

        # --- line 4: shadow of the CURRENT iterate (exact, uses received q)
        qbar = self.shadow_q()
        self._update_shadow_residual(qbar)          # monitoring only

        # --- line 5: reflection (D, G unchanged by proj_{X^rec})
        rD, rG = self.D, self.G
        rq = {m: 2.0 * qbar[m] - self.q[m] for m in self.neighbors}

        # --- line 6: local prox
        zD, zG, zq = self.local_prox(gamma, rD, rG, rq, n_bisect=n_bisect)

        # --- line 7 of Algorithm 1: fully-relaxed DR update -------------
        # The Douglas-Rachford recursion is
        #     x^{k+1} = x^k + lambda (z^k - xbar^k)
        # on EVERY coordinate (Eq. 21 of the paper). The reciprocity
        # resolvent J_{gamma B} = proj_{X^rec} leaves (D, G) untouched, so
        # their shadow equals the iterate itself, Dbar^k = D^k and
        # Gbar^k = G^k, and the relaxation reads
        #     D^{k+1} = D^k + lambda (zD - D^k)      (Dbar = D)
        #     G^{k+1} = G^k + lambda (zG - G^k)      (Gbar = G)
        #     q^{k+1} = q^k + lambda (zq - qbar).
        # At lambda = 1 this collapses to D^{k+1} = zD, G^{k+1} = zG, the
        # special case that was previously hard-coded. Relaxing every block
        # keeps the iterate consistent with Theorem 1 for ANY lambda in
        # (0, 2), not only lambda = 1.
        D_new = self.D + lam * (zD - self.D)
        G_new = self.G + lam * (zG - self.G)
        q_new = {m: self.q[m] + lam * (zq[m] - qbar[m]) for m in self.neighbors}

        # --- line 8: add the learned, summable perturbation ---
        if v is not None:
            v_D, v_G, v_q = v
            D_new = D_new + v_D
            G_new = G_new + v_G
            q_new = {m: q_new[m] + v_q[m] for m in self.neighbors}

        # local part of the auxiliary fixed-point residual ||x^{k+1}-x^k||
        parts = [(D_new - self.D) ** 2, (G_new - self.G) ** 2]
        parts += [(q_new[m] - self.q[m]) ** 2 for m in self.neighbors]
        self.residual_sq = torch.stack(parts).sum()

        # --- commit ---
        self.D, self.G, self.q = D_new, G_new, q_new
        self._round += 1
        if new_mem is not None:
            self._mem = new_mem
            self._prev_residual = torch.sqrt(self.residual_sq + 1e-12)

    def _update_shadow_residual(self, qbar: Dict[int, torch.Tensor]) -> None:
        """This node's share of ||xbar^k - xbar^{k-1}||^2 (monitoring).

        The shadow of the current iterate is exactly (D, G, qbar); it is
        available only after the messages of this round have arrived, so
        the shadow residual is reported with a one-iteration lag. No extra
        communication is needed.
        """
        if self._prev_shadow is not None:
            pD, pG, pq = self._prev_shadow
            parts = [(self.D - pD) ** 2, (self.G - pG) ** 2]
            parts += [(qbar[m] - pq[m]) ** 2 for m in self.neighbors]
            self.shadow_residual_sq = torch.stack(parts).sum()
        self._prev_shadow = (self.D, self.G, dict(qbar))

    def reset_residuals(self) -> None:
        """Forget the residual bookkeeping and restart the round counter.

        Call before a new run: `_round` must start at 0 so the learned
        perturbation's amplitude schedule rho^_round restarts too, and the
        recurrent memory / previous-residual signal must restart from
        "no history yet" (zeros) along with it.
        """
        self._prev_shadow = None
        self.residual_sq = torch.zeros((), device=self.device)
        self.shadow_residual_sq = torch.zeros((), device=self.device)
        self._round = 0
        self._mem = None
        self._prev_residual = torch.zeros((), device=self.device)

    # ------------------------------------------------------------------
    # Monitoring / diagnostics
    # ------------------------------------------------------------------
    def balance_violation(self) -> torch.Tensor:
        """D - G - G_res - Q_n. Zero at a feasible point (0-dim tensor)."""
        return self.D - self.G - self.G_res - self.net_import()

    def box_violation(self) -> torch.Tensor:
        """Max violation of the D/G boxes and the q capacities (>= 0)."""
        v = [
            torch.relu(self.D_min - self.D),
            torch.relu(self.D - self.D_max),
            torch.relu(self.G_min - self.G),
            torch.relu(self.G - self.G_max),
        ]
        for m in self.neighbors:
            v.append(torch.relu(self.q[m] - self.k[m]))   # q <= kappa
        return torch.stack(v).max()

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------
    def clone(self) -> "Prosumer":
        """Deep-ish copy: independent state, shared read-only parameters."""
        new = copy.copy(self)
        new.D = self.D.clone()
        new.G = self.G.clone()
        new.q = {m: self.q[m].clone() for m in self.neighbors}
        new.q_received = {m: self.q_received[m].clone() for m in self.neighbors}
        return new

    def __repr__(self) -> str:
        return (
            f"Prosumer(id={self.node_id}, neighbors={self.neighbors}, "
            f"D={float(self.D):.3f}, G={float(self.G):.3f})"
        )