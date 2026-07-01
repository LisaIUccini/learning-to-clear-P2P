"""
energy_network.py
=================

The P2P energy market as a Generalized Nash Equilibrium Problem (GNEP).

`EnergyNetwork` owns the shared graph structure and all the *global*
views and game-level operators, while every purely local quantity is
delegated to the `Prosumer` objects it coordinates.

Responsibilities
----------------
    * graph bookkeeping (edge list, edge index, reverse-edge map)
    * global state vector   x = [D_0..D_{N-1}, G_0..G_{N-1}, Q_0..Q_{|E|-1}]
    * game pseudo-gradient  F(x) = col( grad_{x_n} J_n )_n
    * reciprocity projection  proj_{X^rec}  (the shadow / J_{gamma B} step)
    * one-hop communication round
    * monitoring (cost, balance, reciprocity, feasibility)

The Douglas-Rachford *algorithm* that uses these pieces lives in a
separate module (added in a later step).

Conventions
-----------
Edges are directed and ordered as `edge_list`: entry i is a pair (m, n)
meaning "prosumer n's trade decision toward neighbour m", i.e. q_{mn}.
The reverse edge (n, m) carries q_{nm}. Reciprocity couples the two:
    q_{mn} + q_{nm} <= 0.
The graph is assumed symmetric (m in Omega_n  <=>  n in Omega_m); this
is checked at construction.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from prosumer import Prosumer


class EnergyNetwork:
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        prosumers: Sequence[Prosumer],
        neighbors: Dict[int, List[int]],
        device: Optional[torch.device] = None,
    ):
        self.prosumers: List[Prosumer] = list(prosumers)
        self.neighbors: Dict[int, List[int]] = {
            n: sorted(neighbors[n]) for n in neighbors
        }
        self.N = len(self.prosumers)
        self.device = device if device is not None else torch.device("cpu")

        # --- graph structure ---
        self.edge_list: List[Tuple[int, int]] = self._build_edge_list()
        self.edge_index = self._build_edge_index()
        self.E = len(self.edge_list)

        self._idx_map: Dict[Tuple[int, int], int] = {
            e: i for i, e in enumerate(self.edge_list)
        }
        self._check_symmetry()

        # reverse-edge index: inv_idx[i] = position of (n, m) when edge_list[i]=(m, n)
        self.inv_idx = torch.tensor(
            [self._idx_map[(n, m)] for (m, n) in self.edge_list],
            dtype=torch.long, device=self.device,
        ) if self.E > 0 else torch.zeros(0, dtype=torch.long, device=self.device)

        # incoming_idx[n] = edge positions i with edge_list[i] = (m, n)  (n's own q's)
        self.incoming_idx: Dict[int, List[int]] = {
            n: [self._idx_map[(m, n)] for m in self.neighbors[n]]
            for n in range(self.N)
        }

        # unique undirected pairs (each edge counted once) for reciprocity checks
        self._unique_pairs = self._build_unique_pairs()

        self.dim = 2 * self.N + self.E

        # --- cached parameter vectors (for vectorised F, cost, projection) ---
        self._cache_parameters()

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def _build_edge_list(self) -> List[Tuple[int, int]]:
        """Directed edges (m, n): q_{mn} is n's decision. Sorted n, then m."""
        edges = []
        for n in sorted(self.neighbors.keys()):
            for m in self.neighbors[n]:
                edges.append((m, n))
        return edges

    def _build_edge_index(self) -> torch.Tensor:
        if not self.edge_list:
            return torch.zeros(2, 0, dtype=torch.long, device=self.device)
        src, dst = zip(*self.edge_list)     # (m -> n)
        return torch.tensor([src, dst], dtype=torch.long, device=self.device)

    def _check_symmetry(self) -> None:
        for (m, n) in self.edge_list:
            if (n, m) not in self._idx_map:
                raise ValueError(
                    f"Graph is not symmetric: edge ({m},{n}) has no reverse "
                    f"({n},{m}). Reciprocity requires an undirected graph."
                )

    def _build_unique_pairs(self) -> List[Tuple[int, int]]:
        pairs, seen = [], set()
        for (m, n) in self.edge_list:
            if (m, n) not in seen:
                seen.add((m, n))
                seen.add((n, m))
                pairs.append((m, n))
        return pairs

    def _cache_parameters(self) -> None:
        """Stack per-node and per-edge parameters into tensors for vectorised ops."""
        ps = self.prosumers
        stack = lambda attr: torch.stack([getattr(p, attr) for p in ps])
        self.a_tilde_vec = stack("a_tilde")           # (N,)
        self.a_vec = stack("a")                        # (N,)
        self.b_vec = stack("b")                        # (N,)
        self.D_star_vec = stack("D_star")              # (N,)
        self.G_res_vec = stack("G_res")                # (N,)
        self.D_min_vec = stack("D_min"); self.D_max_vec = stack("D_max")
        self.G_min_vec = stack("G_min"); self.G_max_vec = stack("G_max")

        if self.E > 0:
            self.c_edge = torch.stack(
                [ps[n].c[m] for (m, n) in self.edge_list])       # (E,)
            self.k_edge = torch.stack(
                [ps[n].k[m] for (m, n) in self.edge_list])       # (E,)
        else:
            self.c_edge = torch.zeros(0, device=self.device)
            self.k_edge = torch.zeros(0, device=self.device)

    # ------------------------------------------------------------------
    # Global state vector  x = [D, G, Q]
    # ------------------------------------------------------------------
    def flatten(self) -> torch.Tensor:
        D = torch.stack([p.D for p in self.prosumers])
        G = torch.stack([p.G for p in self.prosumers])
        Q = (torch.stack([self.prosumers[n].q[m] for (m, n) in self.edge_list])
             if self.E > 0 else torch.zeros(0, device=self.device))
        return torch.cat([D, G, Q])

    def unflatten(self, x: torch.Tensor) -> None:
        N = self.N
        for n in range(N):
            self.prosumers[n].D = x[n]
            self.prosumers[n].G = x[N + n]
        base = 2 * N
        for i, (m, n) in enumerate(self.edge_list):
            self.prosumers[n].q[m] = x[base + i]

    def get_DGQ(self, x: torch.Tensor):
        """Split a global vector into (D, G, Q) blocks."""
        N = self.N
        return x[:N], x[N:2 * N], x[2 * N:]

    # ------------------------------------------------------------------
    # Game pseudo-gradient  F(x)
    # ------------------------------------------------------------------
    def pseudo_gradient(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """F(x) aligned with `flatten`: shape (dim,).

        If `x` is None, evaluate at the current prosumer state.
        Vectorised using the cached parameter tensors.
        """
        if x is None:
            x = self.flatten()
        D, G, Q = self.get_DGQ(x)
        F_D = 2.0 * self.a_tilde_vec * (D - self.D_star_vec)
        F_G = self.a_vec * G + self.b_vec
        F_Q = self.c_edge                       # constant in q
        return torch.cat([F_D, F_G, F_Q])

    # ------------------------------------------------------------------
    # Reciprocity projection  proj_{X^rec}   (the shadow / J_{gamma B} step)
    # ------------------------------------------------------------------
    def project_reciprocity_Q(self, Q: torch.Tensor) -> torch.Tensor:
        """Euclidean projection of the trade block Q onto X^rec.

        For each directed edge i = (m, n) with reverse q_{nm} = Q[inv_idx[i]]:

            Qbar_i = Q_i - relu(Q_i + Q_{inv(i)}) / 2

        This projects each pair (q_{mn}, q_{nm}) onto {q_{mn} + q_{nm} <= 0}.
        Differentiable (relu). D, G are untouched by X^rec.
        """
        if self.E == 0:
            return Q
        return Q - torch.relu(Q + Q[self.inv_idx]) / 2.0

    def project_reciprocity(self, x: torch.Tensor) -> torch.Tensor:
        """proj_{X^rec}(x): leave (D, G) unchanged, project the Q block."""
        D, G, Q = self.get_DGQ(x)
        return torch.cat([D, G, self.project_reciprocity_Q(Q)])

    # ------------------------------------------------------------------
    # Local prox  J_{gamma A}   (resolvent of A = F + N_{X^loc})
    # ------------------------------------------------------------------
    def _node_sum(self, edge_vals: torch.Tensor) -> torch.Tensor:
        """Sum edge values into their owning node: out[n] = sum_{i: owner=n} v_i."""
        out = torch.zeros(self.N, device=self.device, dtype=edge_vals.dtype)
        if self.E > 0:
            out = out.index_add(0, self.edge_index[1], edge_vals)   # dst = owner
        return out

    def prox_local(self, gamma: float, r: torch.Tensor,
                   n_bisect: int = 60) -> torch.Tensor:
        """Exact, differentiable prox of gamma*J over X^loc, per node.

        Solves   min_{xi in X^loc_n}  gamma J_n(xi) + 1/2 ||xi - r_n||^2
        for every node in parallel, via dualisation of the single balance
        equality with a per-node multiplier nu, closed-form primal updates
        with box clamps, and a bisection on nu (the balance residual is
        monotone in nu). Gradients are obtained through a final implicit
        Newton step, so the whole map r -> prox is differentiable.

        Parameters
        ----------
        gamma : float                     resolvent step size (> 0)
        r     : (dim,) reflected point [r_D, r_G, r_Q]
        Returns
        -------
        (dim,) tensor [D*, G*, q*], the prox solution.
        """
        rD, rG, rQ = self.get_DGQ(r)
        owner = self.edge_index[1] if self.E > 0 else None

        at, a, b = self.a_tilde_vec, self.a_vec, self.b_vec
        Dstar, Gres = self.D_star_vec, self.G_res_vec
        Dmin, Dmax = self.D_min_vec, self.D_max_vec
        Gmin, Gmax = self.G_min_vec, self.G_max_vec
        c_edge, k_edge = self.c_edge, self.k_edge

        denomD = 2.0 * gamma * at + 1.0
        denomG = gamma * a + 1.0

        def D_of(nu):
            return torch.clamp((2.0 * gamma * at * Dstar + rD + nu) / denomD,
                               Dmin, Dmax)

        def G_of(nu):
            return torch.clamp((-gamma * b + rG - nu) / denomG, Gmin, Gmax)

        def q_of(nu):
            if self.E == 0:
                return rQ
            return torch.minimum(rQ - gamma * c_edge - nu[owner], k_edge)

        def balance_of(nu):
            return D_of(nu) - G_of(nu) - Gres - self._node_sum(q_of(nu))

        # --- bracket + bisection (no grad): find nu* with balance = 0 ---
        with torch.no_grad():
            lo = -torch.ones(self.N, device=self.device)
            hi = torch.ones(self.N, device=self.device)
            for _ in range(80):
                need_lo = balance_of(lo) > 0
                need_hi = balance_of(hi) < 0
                if not bool(need_lo.any()) and not bool(need_hi.any()):
                    break
                lo = torch.where(need_lo, lo * 2.0 - 1.0, lo)
                hi = torch.where(need_hi, hi * 2.0 + 1.0, hi)
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                pos = balance_of(mid) > 0
                hi = torch.where(pos, mid, hi)
                lo = torch.where(pos, lo, mid)
            nu = 0.5 * (lo + hi)

        # --- implicit Newton step for differentiability ---
        g_val = balance_of(nu)                       # ~0, carries grad in params/r
        D_raw = (2.0 * gamma * at * Dstar + rD + nu) / denomD
        G_raw = (-gamma * b + rG - nu) / denomG
        mD = ((D_raw > Dmin) & (D_raw < Dmax)).to(g_val.dtype)
        mG = ((G_raw > Gmin) & (G_raw < Gmax)).to(g_val.dtype)
        if self.E > 0:
            q_raw = rQ - gamma * c_edge - nu[owner]
            mq = (q_raw < k_edge).to(g_val.dtype)
            nq = self._node_sum(mq)
        else:
            nq = torch.zeros(self.N, device=self.device)
        g_prime = mD / denomD + mG / denomG + nq
        g_prime = torch.clamp(g_prime, min=1e-6)
        nu_star = nu - g_val / g_prime

        return torch.cat([D_of(nu_star), G_of(nu_star), q_of(nu_star)])

    # ------------------------------------------------------------------
    # Vector-based evaluations (do not touch prosumer state)
    # ------------------------------------------------------------------
    def cost_at(self, x: torch.Tensor) -> torch.Tensor:
        """Global cost sum_n J_n evaluated at a global vector x (0-dim)."""
        D, G, Q = self.get_DGQ(x)
        node = (self.a_tilde_vec * (D - self.D_star_vec) ** 2
                + 0.5 * self.a_vec * G ** 2 + self.b_vec * G).sum()
        trade = (self.c_edge * Q).sum() if self.E > 0 else torch.zeros((), device=self.device)
        return node + trade

    def balance_at(self, x: torch.Tensor) -> torch.Tensor:
        """Per-node balance residual D - G - G_res - Q_n at x, shape (N,)."""
        D, G, Q = self.get_DGQ(x)
        return D - G - self.G_res_vec - self._node_sum(Q)

    def reciprocity_max_at(self, x: torch.Tensor) -> torch.Tensor:
        """max_e relu(q_{mn}+q_{nm}) at x (0-dim). Zero if x in X^rec."""
        _, _, Q = self.get_DGQ(x)
        if self.E == 0:
            return torch.zeros((), device=self.device)
        return torch.relu(Q + Q[self.inv_idx]).max()

    # ------------------------------------------------------------------
    # Communication round
    # ------------------------------------------------------------------
    def communicate(self, training: bool = False) -> None:
        """Every prosumer sends its q toward each neighbour.

        After the call: prosumer[n].q_received[m] = prosumer[m].q[n],
        for every edge (m, n).
        """
        for (m, n) in self.edge_list:
            self.prosumers[m].send_q_to(self.prosumers[n], training=training)

    # ------------------------------------------------------------------
    # Monitoring / diagnostics
    # ------------------------------------------------------------------
    def cost(self) -> torch.Tensor:
        """Global cost = sum_n J_n (0-dim tensor)."""
        return torch.stack([p.local_cost() for p in self.prosumers]).sum()

    def net_imports(self) -> torch.Tensor:
        """Q_n for each node, shape (N,)."""
        return torch.stack([p.net_import() for p in self.prosumers])

    def balance_violation(self) -> torch.Tensor:
        """D_n - G_n - G_res_n - Q_n for each node, shape (N,)."""
        return torch.stack([p.balance_violation() for p in self.prosumers])

    def reciprocity_violation(self) -> torch.Tensor:
        """max(0, q_{mn} + q_{nm}) over unique edges, shape (n_pairs,)."""
        if not self._unique_pairs:
            return torch.zeros(0, device=self.device)
        vals = []
        for (m, n) in self._unique_pairs:
            q_mn = self.prosumers[n].q[m]
            q_nm = self.prosumers[m].q[n]
            vals.append(torch.relu(q_mn + q_nm))
        return torch.stack(vals)

    def max_box_violation(self) -> torch.Tensor:
        return torch.stack([p.box_violation() for p in self.prosumers]).max()

    def is_feasible(self, tol: float = 1e-4) -> bool:
        """True if box, capacity, balance and reciprocity all hold within tol."""
        if float(self.max_box_violation()) > tol:
            return False
        if float(self.balance_violation().abs().max()) > tol:
            return False
        if self._unique_pairs and float(self.reciprocity_violation().max()) > tol:
            return False
        return True

    def feasibility_report(self, tol: float = 1e-4) -> Dict[str, float]:
        """Numeric summary of the constraint violations (for debugging / plots)."""
        rec = (float(self.reciprocity_violation().max())
               if self._unique_pairs else 0.0)
        return {
            "box": float(self.max_box_violation()),
            "balance": float(self.balance_violation().abs().max()),
            "reciprocity": rec,
            "feasible": self.is_feasible(tol),
        }

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def init_state(self, generator: Optional[torch.Generator] = None) -> "EnergyNetwork":
        for p in self.prosumers:
            p.init_state(generator=generator)
        return self

    def clone(self) -> "EnergyNetwork":
        """New network with independently cloned prosumers; graph data shared."""
        new = copy.copy(self)
        new.prosumers = [p.clone() for p in self.prosumers]
        return new

    def __repr__(self) -> str:
        return f"EnergyNetwork(N={self.N}, |E|={self.E}, dim={self.dim})"