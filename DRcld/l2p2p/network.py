"""The P2P market as a distributed computation.

    Prosumer  - what one node owns: parameters, state (D, G, q) and the only
                buffer that receives outside information (`q_received`,
                filled once per iteration by `Network.communicate`). Runs its
                own share of Algorithm 1 (`step`).
    Network   - graph bookkeeping plus a vectorised mirror of the per-node
                computations, used only for centralised, differentiable
                training; every vectorised operation has a per-node
                counterpart on `Prosumer`.

Algorithm 1 at prosumer n (paper Sec. 3.1):

    line 5   send q_mn to (receive q_nm from) each m in Omega_n
    line 6   qtilde_mn = q_mn - 1/2 max(0, q_mn + q_nm)        (proj onto X^rec)
    line 8   r_mn = 2 qtilde_mn - q_mn                          (reflection)
    line 9   z_n = argmin_{xi in X^loc_n} gamma J_n(xi) + 1/2 ||xi - (D_n, G_n, r_n)||^2
    line 10  x_hat_n = x_n + lambda (z_n - xtilde_n)
    line 11  v_n = GNN(...)                                     (gnn.py)
    line 12  x_n <- x_hat_n + v_n

The projection and the reflection leave (D, G) unchanged: X^rec only
constrains trades.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

STALL_EMA_DECAY = 0.95   # ~20-iteration effective window for Prosumer.stall_ema


# ======================================================================
# Prosumer: local state + local computation
# ======================================================================
class Prosumer:
    def __init__(self, node_id: int, neighbors: Sequence[int], device,
                 D_min, D_max, G_min, G_max, G_res, a_tilde, a, b, D_star,
                 c: Dict[int, torch.Tensor], k: Dict[int, torch.Tensor]):
        self.node_id = int(node_id)
        self.neighbors: List[int] = sorted(int(m) for m in neighbors)
        self.device = device
        self.D_min, self.D_max = D_min, D_max
        self.G_min, self.G_max = G_min, G_max
        self.G_res = G_res
        self.a_tilde, self.a, self.b, self.D_star = a_tilde, a, b, D_star
        self.c, self.k = c, k

        z = torch.zeros((), device=device)
        self.D, self.G = z.clone(), z.clone()
        self.q: Dict[int, torch.Tensor] = {m: z.clone() for m in self.neighbors}
        self.q_received: Dict[int, torch.Tensor] = {m: z.clone() for m in self.neighbors}

        # residual bookkeeping (monitoring, and the GNN's temporal features)
        self.residual_sq = z.clone()          # ||x^{k+1}_n - x^k_n||^2
        self.base_residual_sq = z.clone()     # ||x_hat_n - x_n||^2: the base operator's step only
                                                # (excludes the GNN's v^k; == residual_sq
                                                # when no model is deployed) -- monitoring /
                                                # DRConfig.stop_on="unperturbed" only
        self.prev_residual = z.clone()        # ||x^k_n - x^{k-1}_n||, own move
                                                # on the PREVIOUS iteration (0 at
                                                # k=0) -- a GNN input feature, see
                                                # gnn.py; purely local bookkeeping
        self.stall_ema = z.clone()            # EMA of prev_residual (STALL_EMA_DECAY,
                                                # ~20-iteration effective window) -- a
                                                # SUSTAINED smallness signal for the GNN's
                                                # slow channel (gnn.py), also purely local
        self._round = 0                        # this node's own iteration count
        self._gnn_mem = None                   # private GRU state (gnn.py)
        self.gnn_agent = None                  # set by gnn.deploy()

    def random_init(self, rng: Optional[torch.Generator] = None) -> None:
        def u():
            return torch.rand((), device=self.device, generator=rng)
        self.D = self.D_min + (self.D_max - self.D_min) * u()
        self.G = self.G_min + (self.G_max - self.G_min) * u()
        self.q = {m: (2.0 * u() - 1.0) * self.k[m] for m in self.neighbors}
        self.residual_sq = torch.zeros((), device=self.device)
        self.prev_residual = torch.zeros((), device=self.device)
        self.stall_ema = torch.zeros((), device=self.device)
        self._round = 0
        self._gnn_mem = None

    # ------------------------------------------------------------------
    def flatten(self) -> torch.Tensor:
        q = (torch.stack([self.q[m] for m in self.neighbors])
             if self.neighbors else torch.zeros(0, device=self.device))
        return torch.cat([self.D.reshape(1), self.G.reshape(1), q])

    # ------------------------------------------------------------------
    # Algorithm 1, lines 4-9 (this node's own share)
    # ------------------------------------------------------------------
    def shadow_q(self) -> Dict[int, torch.Tensor]:
        """Line 4: qbar_mn = q_mn - 1/2 max(0, q_mn + q_nm). Needs a fresh
        `q_received` (filled by `Network.communicate`)."""
        return {m: self.q[m] - 0.5 * torch.relu(self.q[m] + self.q_received[m])
                for m in self.neighbors}

    def local_prox(self, gamma: float, rD: torch.Tensor, rG: torch.Tensor,
                    rq: Dict[int, torch.Tensor], n_bisect: int = 60
                    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """Line 9: argmin_{xi in X^loc_n} gamma J_n(xi) + 1/2 ||xi-r||^2.

        X^loc_n is a box (D, G, q<=kappa) plus one linear equality
        (D = G + G_res + sum q_m). Dualising the equality with a scalar
        multiplier nu gives closed-form primal updates and a balance
        residual that is monotone increasing in nu, so nu* is found by
        bisection -- purely local, no communication.
        """
        g = float(gamma)
        denomD = 2.0 * g * self.a_tilde + 1.0
        denomG = g * self.a + 1.0
        # D_of/G_of/q_sum are each evaluated ~30-40 times per local_prox call
        # (bracket expansion + bisection), but only the "+ nu"/"- nu" term
        # actually varies across those calls -- hoist the nu-INDEPENDENT part
        # of each numerator out of the loop. Same expression, same operation
        # order as before (nu is still the last term added/subtracted before
        # the division/minimum), so results are bit-identical, just not
        # recomputed 30-40x per call.
        numer_D_const = 2.0 * g * self.a_tilde * self.D_star + rD
        numer_G_const = rG - g * self.b
        # D_of/G_of are both a plain elementwise (add-or-subtract, divide,
        # clamp) with no reduction, so -- unlike batching phi_edge's matmuls
        # earlier -- stacking them into one call is exact, not an
        # approximation: each of the 2 output entries only ever depends on
        # its own input entry, so torch.clamp on a length-2 tensor computes
        # the identical two scalar clamps, just as one dispatch instead of
        # two. Halves the D_of/G_of call count in the hot bisection loop.
        DG_min = torch.stack([self.D_min, self.G_min])
        DG_max = torch.stack([self.D_max, self.G_max])
        DG_numer_const = torch.stack([numer_D_const, numer_G_const])
        DG_denom = torch.stack([denomD, denomG])
        DG_sign = torch.tensor([1.0, -1.0], device=self.device)

        def DG_of(nu):
            return torch.clamp((DG_numer_const + DG_sign * nu) / DG_denom, DG_min, DG_max)

        # The bisection loop only needs the SUM of q_m(nu) at every step, so
        # the per-neighbour dict (needed only once, at the end) is kept out
        # of the hot loop -- same values, far fewer python-level tensor ops.
        if self.neighbors:
            rq_vec = torch.stack([rq[m] for m in self.neighbors])
            c_vec = torch.stack([self.c[m] for m in self.neighbors])
            k_vec = torch.stack([self.k[m] for m in self.neighbors])
            q_const_vec = rq_vec - g * c_vec

            def q_sum(nu):
                return torch.minimum(q_const_vec - nu, k_vec).sum()
        else:
            def q_sum(nu):
                return torch.zeros((), device=self.device)

        def q_of(nu):
            return {m: torch.minimum(rq[m] - g * self.c[m] - nu, self.k[m])
                    for m in self.neighbors}

        def balance(nu):
            dg = DG_of(nu)
            return dg[0] - dg[1] - self.G_res - q_sum(nu)

        with torch.no_grad():
            lo = -torch.ones((), device=self.device)
            hi = torch.ones((), device=self.device)
            for _ in range(20):
                # balance(lo)/balance(hi) each used twice below -- compute
                # once and reuse, instead of recomputing (same values, same
                # comparisons, just not redone).
                b_lo, b_hi = balance(lo), balance(hi)
                if bool(b_lo <= 0) and bool(b_hi >= 0):
                    break
                if bool(b_lo > 0):
                    lo = lo * 2.0 - 1.0
                if bool(b_hi < 0):
                    hi = hi * 2.0 + 1.0
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                if bool(balance(mid) > 0):
                    hi = mid
                else:
                    lo = mid
            nu = 0.5 * (lo + hi)

        dg_final = DG_of(nu)
        return dg_final[0], dg_final[1], q_of(nu)

    # ------------------------------------------------------------------
    # One iteration of Algorithm 1 at this node
    # ------------------------------------------------------------------
    def step(self, operator) -> None:
        """Base update `operator.local_update` (lines 6-10, e.g. DR), then the
        learned correction if a model is deployed (`gnn.deploy`, lines 11-12),
        computed from this node's own state and the trades already received:
        no further communication. Needs a fresh `q_received`."""
        D_new, G_new, q_new = operator.local_update(self)
        base_parts = [(D_new - self.D) ** 2, (G_new - self.G) ** 2]
        base_parts += [(q_new[m] - self.q[m]) ** 2 for m in self.neighbors]
        self.base_residual_sq = torch.stack(base_parts).sum().detach()

        if self.gnn_agent is not None:
            state = self.gnn_agent.gnn.local_state(self)
            v_D, v_G, v_q, mem_new = self.gnn_agent.perturbation(state, self._round)
            D_new = D_new + v_D
            G_new = G_new + v_G
            q_new = {m: q_new[m] + v_q[m] for m in self.neighbors}
            self._gnn_mem = mem_new

        parts = [(D_new - self.D) ** 2, (G_new - self.G) ** 2]
        parts += [(q_new[m] - self.q[m]) ** 2 for m in self.neighbors]
        self.residual_sq = torch.stack(parts).sum()

        self.D, self.G, self.q = D_new, G_new, q_new
        self._round += 1
        self.prev_residual = torch.sqrt(self.residual_sq.detach() + 1e-12)
        self.stall_ema = STALL_EMA_DECAY * self.stall_ema + (1.0 - STALL_EMA_DECAY) * self.prev_residual

    def reset(self) -> None:
        self.residual_sq = torch.zeros((), device=self.device)
        self.base_residual_sq = torch.zeros((), device=self.device)
        self.prev_residual = torch.zeros((), device=self.device)
        self.stall_ema = torch.zeros((), device=self.device)
        self._round = 0
        self._gnn_mem = None

# ======================================================================
# Network: graph bookkeeping + vectorised (centralised) mirror
# ======================================================================
class Network:
    def __init__(self, prosumers: Sequence[Prosumer], neighbors: Dict[int, List[int]],
                 device=None):
        self.prosumers: List[Prosumer] = list(prosumers)
        self.neighbors = {n: sorted(neighbors[n]) for n in neighbors}
        self.N = len(self.prosumers)
        self.device = device or torch.device("cpu")

        # directed edge i = (m, n): q_mn is n's own decision
        self.edge_list: List[Tuple[int, int]] = [
            (m, n) for n in range(self.N) for m in self.neighbors[n]
        ]
        self.E = len(self.edge_list)
        self.idx_map = {e: i for i, e in enumerate(self.edge_list)}
        for (m, n) in self.edge_list:
            if (n, m) not in self.idx_map:
                raise ValueError(f"graph not symmetric at edge ({m},{n})")
        self.inv_idx = torch.tensor(
            [self.idx_map[(n, m)] for (m, n) in self.edge_list],
            dtype=torch.long, device=self.device
        ) if self.E else torch.zeros(0, dtype=torch.long, device=self.device)
        self.owner = torch.tensor([n for (_, n) in self.edge_list], dtype=torch.long,
                                   device=self.device) if self.E else torch.zeros(
            0, dtype=torch.long, device=self.device)

        self.dim = 2 * self.N + self.E
        self._cache_params()

    def _cache_params(self) -> None:
        ps = self.prosumers
        stack = lambda attr: torch.stack([getattr(p, attr) for p in ps])
        self.a_tilde_vec, self.a_vec, self.b_vec = stack("a_tilde"), stack("a"), stack("b")
        self.D_star_vec, self.G_res_vec = stack("D_star"), stack("G_res")
        self.D_min_vec, self.D_max_vec = stack("D_min"), stack("D_max")
        self.G_min_vec, self.G_max_vec = stack("G_min"), stack("G_max")
        if self.E:
            self.c_edge = torch.stack([ps[n].c[m] for (m, n) in self.edge_list])
            self.k_edge = torch.stack([ps[n].k[m] for (m, n) in self.edge_list])
        else:
            self.c_edge = torch.zeros(0, device=self.device)
            self.k_edge = torch.zeros(0, device=self.device)

    # ------------------------------------------------------------------
    def random_init(self, rng: Optional[torch.Generator] = None) -> None:
        for p in self.prosumers:
            p.random_init(rng)

    def reset(self) -> None:
        for p in self.prosumers:
            p.reset()

    # ------------------------------------------------------------------
    # Global vector view x = [D, G, Q]
    # ------------------------------------------------------------------
    def flatten(self) -> torch.Tensor:
        D = torch.stack([p.D for p in self.prosumers])
        G = torch.stack([p.G for p in self.prosumers])
        Q = (torch.stack([self.prosumers[n].q[m] for (m, n) in self.edge_list])
             if self.E else torch.zeros(0, device=self.device))
        return torch.cat([D, G, Q])

    def unflatten(self, x: torch.Tensor) -> None:
        N = self.N
        for n in range(N):
            self.prosumers[n].D = x[n]
            self.prosumers[n].G = x[N + n]
        for i, (m, n) in enumerate(self.edge_list):
            self.prosumers[n].q[m] = x[2 * N + i]

    def get_DGQ(self, x: torch.Tensor):
        N = self.N
        return x[:N], x[N:2 * N], x[2 * N:]

    # ------------------------------------------------------------------
    # Line 3: the ONLY place information crosses between nodes
    # ------------------------------------------------------------------
    def communicate(self) -> None:
        for (m, n) in self.edge_list:
            self.prosumers[n].q_received[m] = self.prosumers[m].q[n].detach().clone()

    # ------------------------------------------------------------------
    # Vectorised mirror of Prosumer.shadow_q / reflection (proj X^rec)
    # ------------------------------------------------------------------
    def project_reciprocity_Q(self, Q: torch.Tensor) -> torch.Tensor:
        if self.E == 0:
            return Q
        return Q - 0.5 * torch.relu(Q + Q[self.inv_idx])

    def project_reciprocity(self, x: torch.Tensor) -> torch.Tensor:
        D, G, Q = self.get_DGQ(x)
        return torch.cat([D, G, self.project_reciprocity_Q(Q)])

    # ------------------------------------------------------------------
    # Vectorised, DIFFERENTIABLE mirror of Prosumer.local_prox -- the
    # centralised twin used ONLY for training (autodiff needs the whole
    # state at once). One implicit Newton step after the (no-grad)
    # bisection makes the map r -> prox differentiable in r (and in the
    # network's own parameters): the balance residual is piecewise-linear
    # in nu, so a single correction from an accurate nu essentially lands
    # on the true root, and its derivative recovers the implicit-function
    # gradient dnu*/dr that the raw bisection result (a python float
    # comparison, no grad) would otherwise be blind to.
    # ------------------------------------------------------------------
    def prox_local(self, gamma, r: torch.Tensor, n_bisect: int = 60) -> torch.Tensor:
        """`gamma` is either the usual scalar (identical behaviour to
        before this parameter) or a per-node tensor of shape (N,) -- e.g.
        a GNN's own instance-adaptive gamma_i prediction (see
        `gamma_gnn.py`). Every node-indexed use of gamma (denomD/denomG/
        D_of/G_of, all already shaped (N,)) broadcasts a (N,) gamma
        correctly with NO change needed; only `q_of`'s per-EDGE use needs
        gamma indexed by each edge's OWNER node (`gamma[owner]`, the same
        "edge i=(m,n), n is the owner, n's own decision" convention
        `network.py`'s own module docstring already establishes elsewhere)
        -- for a plain scalar this reduces to the original expression
        exactly, so every existing call site (scalar gamma) is unaffected."""
        rD, rG, rQ = self.get_DGQ(r)
        at, a, b = self.a_tilde_vec, self.a_vec, self.b_vec
        Dstar, Gres = self.D_star_vec, self.G_res_vec
        Dmin, Dmax, Gmin, Gmax = self.D_min_vec, self.D_max_vec, self.G_min_vec, self.G_max_vec
        c_edge, k_edge = self.c_edge, self.k_edge
        owner = self.owner
        gamma_e = gamma[owner] if (torch.is_tensor(gamma) and gamma.dim() > 0) else gamma

        denomD = 2.0 * gamma * at + 1.0
        denomG = gamma * a + 1.0

        def D_of(nu):
            return torch.clamp((2.0 * gamma * at * Dstar + rD + nu) / denomD, Dmin, Dmax)

        def G_of(nu):
            return torch.clamp((rG - gamma * b - nu) / denomG, Gmin, Gmax)

        def q_of(nu):
            return rQ - gamma_e * c_edge - nu[owner] if self.E else rQ

        def node_sum(v):
            out = torch.zeros(self.N, device=self.device, dtype=v.dtype)
            return out.index_add(0, owner, v) if self.E else out

        def q_clamped(nu):
            return torch.minimum(q_of(nu), k_edge) if self.E else rQ

        def balance(nu):
            return D_of(nu) - G_of(nu) - Gres - node_sum(q_clamped(nu))

        with torch.no_grad():
            lo = -torch.ones(self.N, device=self.device)
            hi = torch.ones(self.N, device=self.device)
            for _ in range(20):
                need_lo, need_hi = balance(lo) > 0, balance(hi) < 0
                if not bool(need_lo.any()) and not bool(need_hi.any()):
                    break
                lo = torch.where(need_lo, lo * 2.0 - 1.0, lo)
                hi = torch.where(need_hi, hi * 2.0 + 1.0, hi)
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                pos = balance(mid) > 0
                hi = torch.where(pos, mid, hi)
                lo = torch.where(pos, lo, mid)
            nu = 0.5 * (lo + hi)

        # one differentiable Newton correction (see docstring)
        g_val = balance(nu)
        D_raw = (2.0 * gamma * at * Dstar + rD + nu) / denomD
        G_raw = (rG - gamma * b - nu) / denomG
        mD = ((D_raw > Dmin) & (D_raw < Dmax)).to(g_val.dtype)
        mG = ((G_raw > Gmin) & (G_raw < Gmax)).to(g_val.dtype)
        if self.E:
            q_raw = q_of(nu)
            mq = (q_raw < k_edge).to(g_val.dtype)
            n_unclamped = node_sum(mq)
        else:
            n_unclamped = torch.zeros(self.N, device=self.device)
        g_prime = torch.clamp(mD / denomD + mG / denomG + n_unclamped, min=1e-6)
        nu_star = nu - g_val / g_prime

        return torch.cat([D_of(nu_star), G_of(nu_star), q_clamped(nu_star)])

    # ------------------------------------------------------------------
    # Monitoring / evaluation (global reductions -- not part of the
    # distributed algorithm, used only to report progress or train)
    # ------------------------------------------------------------------
    def cost_at(self, x: torch.Tensor) -> torch.Tensor:
        D, G, Q = self.get_DGQ(x)
        node = (self.a_tilde_vec * (D - self.D_star_vec) ** 2
                + 0.5 * self.a_vec * G ** 2 + self.b_vec * G).sum()
        trade = (self.c_edge * Q).sum() if self.E else torch.zeros((), device=self.device)
        return node + trade

    def dist_to_Xloc(self, x: torch.Tensor) -> torch.Tensor:
        """Euclidean distance from `x` ([D,G,Q]) to the LOCAL constraint
        set X^loc = {D in [D_min,D_max], G in [G_min,G_max], Q <= k_edge}
        -- the per-prosumer box `local_prox`/`prox_local` projects onto
        exactly (the OTHER constraint set, X^rec = balance + reciprocity,
        is what `project_reciprocity` handles). X^loc is separable per
        coordinate, so its projection is closed-form clipping and the
        distance is just the L2 norm of each coordinate's own clip
        residual. Used to report dist(xbar^k, X_loc) -- how far the
        SHADOW iterate (always X^rec-feasible by construction) still sits
        from the box constraints at iteration k, a genuine convergence
        diagnostic distinct from the residual."""
        D, G, Q = self.get_DGQ(x)
        dD = torch.clamp(self.D_min_vec - D, min=0.0) + torch.clamp(D - self.D_max_vec, min=0.0)
        dG = torch.clamp(self.G_min_vec - G, min=0.0) + torch.clamp(G - self.G_max_vec, min=0.0)
        dQ = torch.clamp(Q - self.k_edge, min=0.0) if self.E else torch.zeros(0, device=self.device)
        return torch.sqrt((dD ** 2).sum() + (dG ** 2).sum() + (dQ ** 2).sum())

    def balance_at(self, x: torch.Tensor) -> torch.Tensor:
        D, G, Q = self.get_DGQ(x)
        out = torch.zeros(self.N, device=self.device)
        if self.E:
            out = out.index_add(0, self.owner, Q)
        return D - G - self.G_res_vec - out

    def physical_scale(self) -> torch.Tensor:
        """Per-coordinate physical reference [half D-box, half G-box, kappa],
        matching `flatten()`'s [D, G, Q] layout."""
        half_D = (self.D_max_vec - self.D_min_vec) / 2.0
        half_G = (self.G_max_vec - self.G_min_vec) / 2.0
        return torch.cat([half_D, half_G, self.k_edge])

    def per_node_move_sq(self, x_new: torch.Tensor, x_old: torch.Tensor) -> torch.Tensor:
        """Per-node squared move ||[D,G,q]_n^new - [D,G,q]_n^old||^2, shape
        (N,) -- the vectorised mirror of `Prosumer.residual_sq`, used by
        `training.rollout` to maintain the same per-node `prev_residual`
        GNN feature the distributed path keeps (`Prosumer.prev_residual`)."""
        D_new, G_new, Q_new = self.get_DGQ(x_new)
        D_old, G_old, Q_old = self.get_DGQ(x_old)
        out = (D_new - D_old) ** 2 + (G_new - G_old) ** 2
        if self.E:
            out = out + torch.zeros(self.N, device=self.device, dtype=x_new.dtype).index_add(
                0, self.owner, (Q_new - Q_old) ** 2)
        return out

    def aux_residual(self) -> float:
        """||x^{k+1}-x^k||: sum of the prosumers' own local contributions."""
        return float(sum(float(p.residual_sq) for p in self.prosumers)) ** 0.5

    def base_residual(self) -> float:
        """||x_hat^k - x^k||: the base operator's part of the last step,
        without the learned correction (== aux_residual() without a model)."""
        return float(sum(float(p.base_residual_sq) for p in self.prosumers)) ** 0.5
