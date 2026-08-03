"""
gnn.py
======

The learned perturbation map Phi_theta (Algorithm 1, line 8).

Role in the algorithm
----------------------
After the plain DR update (line 7) each prosumer adds a small, LEARNED,
SUMMABLE correction:

    x^{k+1}_n  <-  x^{k+1}_n + v^k_n ,      v^k_n = scale * rho^k * d_n(x^k)

where d_n is bounded (component n of a tanh-squashed, physically-rescaled
network output). This is a perturbed Krasnoselskii-Mann iteration: the
unperturbed DR recursion x^{k+1} = x^k + lambda*(J_{gammaA}(2 J_{gammaB}
x^k - x^k) - J_{gammaB} x^k) is (lambda/2)-averaged nonexpansive for
lambda in (0,2) (Lions & Mercier 1979), and a classical result on
perturbed averaged/KM iterations (e.g. Combettes 2001) states that adding
ANY sequence e_k with sum_k ||e_k|| < infinity preserves convergence to
the same fixed-point set. We enforce that BY CONSTRUCTION, not by hoping
training keeps it small:

    * both output heads end in tanh          => |Phi_theta| <= 1 componentwise
    * scaled by a node/edge-local PHYSICAL reference (half the D/G box
      width, the trade capacity kappa)        => v^k_n bounded by a fixed
                                                  constant of the instance
    * multiplied by the decaying amplitude rho^k, rho in (0,1) fixed
                                              => geometric decay

    sum_k ||v^k|| <= scale * ||physical_scale|| / (1 - rho) < infinity,

for ANY theta, trained or not, in- or out-of-distribution. The learned
part can only affect SPEED, never correctness: it cannot break the
convergence guarantee that lines 4-7 alone already have.

Two execution paths, ONE set of weights
----------------------------------------
Training is necessarily CENTRALISED (back-propagation through the
unrolled trajectory needs the whole state at once, see training.py), while
deployment must be DISTRIBUTED (each prosumer queries its own copy of the
model using only local data). Both are provided here and compute the SAME
function:

    direction(net, x)      -- centralised, vectorised over the whole graph.
                              Used for training and for fast simulation.

    local_forward(state)   -- what a single prosumer runs after the
                              communication step, using ONLY its own
                              variables/parameters and the q values
                              received from its direct neighbours (1 hop,
                              reuses the communication DR already does --
                              no extra round). Used for real distributed
                              inference.

Because the weights theta are shared by every node/edge, and every MLP
size depends only on feature dimensions and d_h (never on N, |E| or
|Omega_n|), a trained model can be copied to every node and run on
networks of any size and topology.

Features (strictly local, computed after the line-3 communication round)
--------------------------------------------------------------------------
Node n:
    h_D      position of D_n in its box, in [-1, 1]
    h_G      position of G_n in its box, in [-1, 1]
    h_Dstar  position of the TARGET D*_n in the same box, in [-1, 1]
    h_bal    local balance residual (D_n - G_n - DeltaG_n - sum_m q_mn),
             normalised by the D-box width -- how far n is from feasible
    h_atilde, h_a, h_b   n's own cost-curvature coefficients, min-max
             normalised against `GNNConfig`'s reference ranges (NOT the
             running network's statistics -- see note below)
    h_prev_res   how much n's OWN (D,G,q) moved on the PREVIOUS iteration
             (the local part of ||x^k-x^{k-1}||, normalised by the D-box
             width) -- a direct, self-supervised progress signal: large
             and not shrinking says "I am not converging here yet",
             small says "this is settling, ease off"

Directed edge (m -> n), i.e. n's link toward neighbour m:
    e_qown   q_{mn} / kappa_{mn}   (n's own decision toward m)
    e_qrecv  q_{nm} / kappa_{mn}   (received from m during communicate())
    e_c      n's own trading cost c_{nm}, normalised against the reference
             range

Why fixed reference ranges, not batch statistics
--------------------------------------------------
A single prosumer cannot compute a mean/std over the whole network (that
needs a global reduction, exactly what `communicate()` is built to avoid).
Normalisation therefore uses FIXED constants declared in `GNNConfig`
(`a_tilde_range`, `a_range`, `b_range`, `c_range`), matching the sampling
ranges of the `MarketConfig` the training data was drawn from --
`GNNConfig.from_market_config` builds them automatically so the two never
drift apart. This is a compile-time constant every node already knows
(it is part of the deployed model), not a quantity derived at runtime from
other nodes' data, so it costs no communication and stays local.

Temporal memory (across DR iterations, NOT graph hops)
----------------------------------------------------------
The node-update step is a GRUCell, not a stateless MLP: mem_n^{k+1} =
GRUCell([h_n^k, agg_n^k], mem_n^k). mem_n is PRIVATE, PER-NODE state that
persists across `dr_step` calls on the SAME prosumer -- it costs no
communication whatsoever (nothing about it is sent or received, exactly
like `Prosumer._round`), so the "one communication round per iteration"
invariant is unaffected. It lets the network build up a notion of "have
my recent corrections here actually been helping" over several iterations,
instead of reacting to the current instant alone -- the earlier, memoryless
version could not distinguish "just started" from "been pushing here for
50 iterations with no visible progress".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from energy_network import EnergyNetwork

Range = Tuple[float, float]


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class GNNConfig:
    """Hyper-parameters of the perturbation map.

    `hidden_dim` (d_h) MUST be a fixed constant, independent of N, |E| and
    the neighbourhood sizes: that is what makes the weights reusable
    across topologies. `rho` and `scale` are NOT learned parameters (kept
    fixed so the summability bound stays a clean, checkable constant, see
    `PerturbationGNN.summability_bound`).
    """
    hidden_dim: int = 32              # d_h -- fixed, independent of N and |E|
    rho: float = 0.9                  # perturbation decay in (0,1) -> summability
    scale: float = 1.0                # amplitude, as a FRACTION of the local
                                      # physical range (box width / capacity)
    activation: str = "tanh"          # hidden activation: tanh | relu

    # reference ranges for feature normalisation -- MUST match (or bound)
    # the MarketConfig the training data is drawn from. Defaults mirror
    # MarketConfig's own defaults; use `from_market_config` to stay in sync.
    a_tilde_range: Range = (0.5, 1.5)
    a_range: Range = (0.5, 2.0)
    b_range: Range = (0.2, 1.0)
    c_range: Range = (0.05, 0.30)

    def __post_init__(self):
        if not (0.0 < self.rho < 1.0):
            raise ValueError("rho must lie in (0,1) to ensure summability")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not (self.scale > 0):
            raise ValueError("scale must be > 0")

    @classmethod
    def from_market_config(cls, market_cfg, **kwargs) -> "GNNConfig":
        """Build reference ranges consistent with the dataset's MarketConfig.

        Keeps feature normalisation from silently drifting out of sync
        with whatever ranges actually generated the training instances.
        """
        return cls(
            a_tilde_range=tuple(market_cfg.a_tilde_range),
            a_range=tuple(market_cfg.a_range),
            b_range=tuple(market_cfg.b_range),
            c_range=tuple(market_cfg.c_range),
            **kwargs,
        )


def _mlp(d_in: int, d_out: int, d_hidden: int, activation: str) -> nn.Sequential:
    """Two-layer MLP. Sizes depend only on features and d_h, never on N."""
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        act(),
        nn.Linear(d_hidden, d_out),
    )


def _norm(v, rng: Range):
    """Min-max normalise `v` against a FIXED reference range, into [-1, 1].

    Clamped, so values slightly outside the nominal range (a different
    instance distribution, an ablation) saturate instead of exploding.
    Works for both batched tensors and 0-dim scalars via broadcasting.
    """
    lo, hi = rng
    return 2.0 * torch.clamp((v - lo) / (hi - lo), 0.0, 1.0) - 1.0


# ======================================================================
# What a single prosumer knows after the communication step
# ======================================================================
@dataclass
class LocalNodeState:
    """Strictly local view of prosumer n (no global information).

    q_own[m]  = q_{mn}  : n's own trade decision toward neighbour m
    q_recv[m] = q_{nm}  : the value RECEIVED from neighbour m (one-hop)
    The remaining fields are n's own fixed parameters.
    """
    node_id: int
    neighbors: List[int]
    D: torch.Tensor
    G: torch.Tensor
    D_star: torch.Tensor
    G_res: torch.Tensor
    D_min: torch.Tensor
    D_max: torch.Tensor
    G_min: torch.Tensor
    G_max: torch.Tensor
    a_tilde: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    q_own: Dict[int, torch.Tensor]
    q_recv: Dict[int, torch.Tensor]
    c: Dict[int, torch.Tensor]
    kappa: Dict[int, torch.Tensor]
    mem: torch.Tensor            # (hidden_dim,) recurrent state, own private history
    prev_residual: torch.Tensor  # 0-dim, local move on the PREVIOUS iteration


def extract_local_state(net: EnergyNetwork, x: torch.Tensor, n: int,
                        mem: torch.Tensor, prev_residual: torch.Tensor
                        ) -> LocalNodeState:
    """Build node n's local view from a global iterate.

    Used for testing/cross-checking `direction` against `local_forward`: in
    a real deployment each prosumer assembles this from its own memory plus
    the messages it received (`Prosumer.local_gnn_state`), never from x.
    """
    D, G, Q = net.get_DGQ(x)
    nb = net.neighbors[n]
    idx = {m: net._idx_map[(m, n)] for m in nb}          # edge (m -> n)
    p = net.prosumers[n]
    return LocalNodeState(
        node_id=n, neighbors=list(nb),
        D=D[n], G=G[n], D_star=p.D_star, G_res=p.G_res,
        D_min=p.D_min, D_max=p.D_max, G_min=p.G_min, G_max=p.G_max,
        a_tilde=p.a_tilde, a=p.a, b=p.b,
        q_own={m: Q[idx[m]] for m in nb},                 # q_{mn}
        q_recv={m: Q[net.inv_idx[idx[m]]] for m in nb},   # q_{nm}
        c={m: p.c[m] for m in nb}, kappa={m: p.k[m] for m in nb},
        mem=mem, prev_residual=prev_residual,
    )


# ======================================================================
# The perturbation network
# ======================================================================
class PerturbationGNN(nn.Module):
    """Phi_theta: one 1-hop message-passing layer + two bounded output heads.

    A third, per-node GATE head (sigmoid, in [0,1]) scales BOTH output
    heads uniformly for that node: v_n = scale * rho^k * gate_n *
    tanh(...). This is a purely additive safety margin -- gate_n = 0
    recovers plain DR exactly for that node -- it can only ever SHRINK the
    already-bounded output, so it does not affect the summability bound
    (`summability_bound`), only gives the network the structural capacity
    to switch itself off when uncertain instead of always committing to a
    full-strength correction.

    The node-update step is a GRUCell carrying a PER-NODE recurrent state
    `mem` across DR iterations (see the module docstring's "Temporal
    memory" section) -- purely a private history, no communication.
    """

    D_NODE = 8   # h_D, h_G, h_Dstar, h_bal, h_atilde, h_a, h_b, h_prev_res
    D_EDGE = 3   # e_qown, e_qrecv, e_c

    def __init__(self, config: Optional[GNNConfig] = None):
        super().__init__()
        self.cfg = config if config is not None else GNNConfig()
        d_h = self.cfg.hidden_dim
        act = self.cfg.activation

        self.phi_msg = _mlp(self.D_EDGE, d_h, d_h, act)                # message
        self.phi_upd = nn.GRUCell(self.D_NODE + d_h, d_h)              # node update + memory
        self.phi_node = _mlp(d_h, 2, d_h, act)                         # -> (D, G)
        self.phi_edge = _mlp(d_h + self.D_EDGE, 1, d_h, act)           # -> q_mn
        self.phi_gate = _mlp(d_h, 1, d_h, act)                         # -> confidence gate

    def init_mem(self, net: EnergyNetwork) -> torch.Tensor:
        """Zero memory for every node, shape (N, hidden_dim) -- call once
        at the start of a rollout/run, then thread the returned value
        through successive `forward` calls."""
        return torch.zeros(net.N, self.cfg.hidden_dim, device=net.device)

    def init_local_mem(self, device=None) -> torch.Tensor:
        """Zero memory for a single node, shape (hidden_dim,)."""
        return torch.zeros(self.cfg.hidden_dim, device=device)

    # ==================================================================
    # CENTRALISED PATH  (training / fast simulation)
    # ==================================================================
    def build_features(self, net: EnergyNetwork, x: torch.Tensor,
                       prev_residual: torch.Tensor):
        """Node and edge features from the global iterate x.

        Every entry is computed from quantities the corresponding node
        owns or received one-hop, so this is only a vectorised rewriting
        of the distributed computation `local_forward` performs.
        `prev_residual` is (N,), the local move on the PREVIOUS iteration
        (zeros on the very first call, see `init_mem`'s counterpart at the
        call site).
        """
        cfg = self.cfg
        D, G, Q = net.get_DGQ(x)
        dD = torch.clamp(net.D_max_vec - net.D_min_vec, min=1e-6)
        dG = torch.clamp(net.G_max_vec - net.G_min_vec, min=1e-6)

        h_D = 2.0 * (D - net.D_min_vec) / dD - 1.0
        h_G = 2.0 * (G - net.G_min_vec) / dG - 1.0
        h_Dstar = 2.0 * (net.D_star_vec - net.D_min_vec) / dD - 1.0
        h_bal = net.balance_at(x) / dD
        h_prev = prev_residual / dD
        h_at = _norm(net.a_tilde_vec, cfg.a_tilde_range)
        h_a = _norm(net.a_vec, cfg.a_range)
        h_b = _norm(net.b_vec, cfg.b_range)
        h = torch.stack([h_D, h_G, h_Dstar, h_bal, h_prev, h_at, h_a, h_b],
                        dim=1)                                       # (N, D_NODE)

        if net.E == 0:
            return h, torch.zeros(0, self.D_EDGE, device=net.device)

        kappa = torch.clamp(net.k_edge, min=1e-6)
        q_own = Q / kappa
        q_recv = Q[net.inv_idx] / kappa
        h_c = _norm(net.c_edge, cfg.c_range)
        e = torch.stack([q_own, q_recv, h_c], dim=1)                    # (E, D_EDGE)
        return h, e

    def forward(self, net: EnergyNetwork, x: torch.Tensor, mem: torch.Tensor,
               prev_residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bounded direction Phi_theta(x) for the whole network, shape (dim,).

        Components lie in [-1, 1] (normalised units); `direction` rescales
        them to physical units and the decaying amplitude rho^k is applied
        outside (see `perturbation_vector`). Returns (direction, mem_new):
        `mem_new` must be fed back in as `mem` on the NEXT call for the
        same rollout (see `init_mem`).
        """
        h, e = self.build_features(net, x, prev_residual)
        N, d_h = net.N, self.cfg.hidden_dim

        if net.E > 0:
            owner = net.edge_index[1]          # dst = n, owner of q_{mn}
            msg = self.phi_msg(e)                                      # message
            agg = torch.zeros(N, d_h, device=net.device, dtype=msg.dtype)
            agg = agg.index_add(0, owner, msg)                         # aggregate
            deg = torch.zeros(N, device=net.device, dtype=msg.dtype)
            deg = deg.index_add(0, owner,
                                torch.ones(net.E, device=net.device,
                                           dtype=msg.dtype))
            agg = agg / torch.clamp(deg, min=1.0).sqrt().unsqueeze(1)
        else:
            agg = torch.zeros(N, d_h, device=net.device, dtype=h.dtype)

        mem_new = self.phi_upd(torch.cat([h, agg], dim=1), mem)         # node update + memory
        gate = torch.sigmoid(self.phi_gate(mem_new)).squeeze(1)         # (N,) confidence
        delta_node = torch.tanh(self.phi_node(mem_new)) * gate.unsqueeze(1)  # -> (D, G)

        if net.E > 0:
            owner = net.edge_index[1]
            h_owner = mem_new[owner]
            delta_edge = torch.tanh(
                self.phi_edge(torch.cat([h_owner, e], dim=1))).squeeze(1) * gate[owner]
        else:
            delta_edge = torch.zeros(0, device=net.device, dtype=h.dtype)

        # layout [D, G, Q], matching EnergyNetwork.flatten()
        direction = torch.cat([delta_node[:, 0], delta_node[:, 1], delta_edge])
        return direction, mem_new

    def physical_scale(self, net: EnergyNetwork) -> torch.Tensor:
        """Per-coordinate physical reference used to rescale [-1,1] outputs.

        Half the box width for D and G (so a full-scale, undecayed
        perturbation could move the variable across half its own box), the
        trade capacity kappa for q. Shape (dim,), matches `flatten()`.
        """
        half_D = (net.D_max_vec - net.D_min_vec) / 2.0
        half_G = (net.G_max_vec - net.G_min_vec) / 2.0
        return torch.cat([half_D, half_G, net.k_edge])

    def direction(self, net: EnergyNetwork, x: torch.Tensor, mem: torch.Tensor,
                 prev_residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bounded perturbation direction in PHYSICAL units, shape (dim,).

        |direction_i| <= physical_scale_i componentwise: this is the
        quantity `perturbation_vector` multiplies by scale * rho^k.
        Returns (direction, mem_new).
        """
        d, mem_new = self.forward(net, x, mem, prev_residual)
        return d * self.physical_scale(net), mem_new

    # ==================================================================
    # DISTRIBUTED PATH  (inference at a single prosumer)
    # ==================================================================
    def local_forward(self, s: LocalNodeState):
        """Perturbation direction computed by prosumer n on its own.

        Uses ONLY s: n's own state/parameters, its own recurrent memory
        and previous-residual signal, plus the q values received from its
        direct neighbours. Mathematically identical to the corresponding
        slice of `direction` (already in PHYSICAL units).

        Returns
        -------
        delta_D, delta_G : 0-dim tensors, perturbation of (D_n, G_n)
        delta_q          : {m: 0-dim tensor}, perturbation of q_{mn}
        mem_new          : (hidden_dim,), feed back in as `s.mem` next call
        """
        cfg = self.cfg
        dD = torch.clamp(s.D_max - s.D_min, min=1e-6)
        dG = torch.clamp(s.G_max - s.G_min, min=1e-6)
        h_D = 2.0 * (s.D - s.D_min) / dD - 1.0
        h_G = 2.0 * (s.G - s.G_min) / dG - 1.0
        h_Dstar = 2.0 * (s.D_star - s.D_min) / dD - 1.0
        q_own_sum = (torch.stack([s.q_own[m] for m in s.neighbors]).sum()
                    if s.neighbors else torch.zeros((), dtype=s.D.dtype))
        h_bal = (s.D - s.G - s.G_res - q_own_sum) / dD
        h_prev = s.prev_residual / dD
        h_at = _norm(s.a_tilde, cfg.a_tilde_range)
        h_a = _norm(s.a, cfg.a_range)
        h_b = _norm(s.b, cfg.b_range)
        h = torch.stack([h_D.reshape(()), h_G.reshape(()), h_Dstar.reshape(()),
                         h_bal.reshape(()), h_prev.reshape(()), h_at.reshape(()),
                         h_a.reshape(()), h_b.reshape(())])              # (D_NODE,)

        d_h = self.cfg.hidden_dim
        k_nb = len(s.neighbors)
        mem = s.mem.reshape(1, d_h)

        if k_nb == 0:
            agg = torch.zeros(1, d_h, dtype=h.dtype)
            mem_new = self.phi_upd(torch.cat([h, agg.squeeze(0)]).unsqueeze(0), mem).squeeze(0)
            gate = torch.sigmoid(self.phi_gate(mem_new)).reshape(())
            delta_node = torch.tanh(self.phi_node(mem_new)) * gate
            return (delta_node[0] * (dD / 2.0), delta_node[1] * (dG / 2.0), {}, mem_new)

        rows = []
        for m in s.neighbors:
            kap = torch.clamp(s.kappa[m], min=1e-6)
            q_own = s.q_own[m] / kap
            q_recv = s.q_recv[m] / kap
            h_c = _norm(s.c[m], cfg.c_range)
            rows.append(torch.stack([q_own.reshape(()), q_recv.reshape(()),
                                     h_c.reshape(())]))
        e = torch.stack(rows)                                          # (k, D_EDGE)

        msg = self.phi_msg(e)                                          # message
        agg = msg.sum(dim=0) / float(k_nb) ** 0.5                      # aggregate
        mem_new = self.phi_upd(torch.cat([h, agg]).unsqueeze(0), mem).squeeze(0)  # update + memory
        gate = torch.sigmoid(self.phi_gate(mem_new)).reshape(())       # confidence
        delta_node = torch.tanh(self.phi_node(mem_new)) * gate         # -> (D, G)

        delta_D = delta_node[0] * (dD / 2.0)
        delta_G = delta_node[1] * (dG / 2.0)

        delta_q = {}
        for i, m in enumerate(s.neighbors):
            out = torch.tanh(self.phi_edge(torch.cat([mem_new, e[i]]))) * gate
            delta_q[m] = out.reshape(()) * torch.clamp(s.kappa[m], min=1e-6)

        return delta_D, delta_G, delta_q, mem_new

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def n_parameters(self) -> int:
        """Number of trainable weights (independent of N and |E|)."""
        return sum(p.numel() for p in self.parameters())

    def summability_bound(self, net: EnergyNetwork) -> float:
        """Upper bound on sum_k ||v^k|| for this instance.

        scale * ||physical_scale(net)||_2 / (1 - rho): finite for any
        trained (or untrained) weights, since the bound never uses theta.
        """
        with torch.no_grad():
            norm = torch.linalg.vector_norm(self.physical_scale(net))
        return float(self.cfg.scale * norm / (1.0 - self.cfg.rho))


# ======================================================================
# Wiring into the DR solver (centralised simulation / training)
# ======================================================================
def perturbation_vector(gnn: PerturbationGNN, net: EnergyNetwork, x: torch.Tensor,
                        k: int, mem: torch.Tensor, prev_residual: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """v^k = scale * rho^k * direction(x): the centralised mirror of what
    `LocalAgent.perturbation` computes distributedly, used by the
    differentiable training rollout in training.py. Returns (v, mem_new)."""
    amp = gnn.cfg.scale * (gnn.cfg.rho ** k)
    d, mem_new = gnn.direction(net, x, mem, prev_residual)
    return amp * d, mem_new


# ======================================================================
# Deployment: hand the trained weights to every prosumer
# ======================================================================
class LocalAgent:
    """A prosumer's on-board copy of the trained perturbation model.

    In a real deployment each device holds its own instance (same
    weights). `perturbation` is what line 8 of Algorithm 1 executes
    locally, right after the communication round.
    """

    def __init__(self, gnn: PerturbationGNN, node_id: int,
                 rho: Optional[float] = None, scale: Optional[float] = None):
        self.gnn = gnn
        self.node_id = int(node_id)
        self.rho = gnn.cfg.rho if rho is None else rho
        self.scale = gnn.cfg.scale if scale is None else scale

    @torch.no_grad()
    def perturbation(self, state: LocalNodeState, k: int):
        """v^k_n for this prosumer: (delta_D, delta_G, {m: delta_q}, mem_new).

        `mem_new` must be stored by the caller (`Prosumer.dr_step`) and
        passed back in as `state.mem` on the next call -- it is this
        node's own recurrent history, untouched by the amplitude schedule
        (only the direction is scaled, not the memory).
        """
        amp = self.scale * (self.rho ** k)
        dD, dG, dq, mem_new = self.gnn.local_forward(state)
        return amp * dD, amp * dG, {m: amp * v for m, v in dq.items()}, mem_new


def deploy(gnn: PerturbationGNN, net: EnergyNetwork, share_weights: bool = True) -> None:
    """Install the trained model on every prosumer of `net`.

    After this call each prosumer runs line 8 on its own during `dr_step`.
    `share_weights=True` keeps a single copy of the tensors in memory
    (fine for simulation); set it to False to give every node a
    physically independent copy, as when shipping the weights to
    real devices.
    """
    import copy as _copy
    gnn.eval()
    for p in net.prosumers:
        model = gnn if share_weights else _copy.deepcopy(gnn)
        model.eval()
        p.gnn_agent = LocalAgent(model, p.node_id)


def undeploy(net: EnergyNetwork) -> None:
    """Remove the learned perturbation: the prosumers fall back to plain DR."""
    for p in net.prosumers:
        p.gnn_agent = None
