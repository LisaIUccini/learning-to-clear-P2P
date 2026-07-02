"""
prosumer.py
===========

Model of a single prosumer in a peer-to-peer (P2P) energy market.

This module contains **only the model** of one market participant:
its fixed physical parameters, its decision variables, its local cost
and gradient, and the minimal communication buffer needed to reason
about the reciprocity coupling with neighbours.

Everything algorithm-specific (Douglas-Rachford steps, GNN features,
learned perturbations) lives elsewhere and is added in later steps.

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