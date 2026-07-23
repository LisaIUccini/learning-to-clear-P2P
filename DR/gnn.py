"""
gnn.py
======

The learned perturbation map Phi_theta of L2Play (Algorithm 1, line 8).

Two execution paths, ONE set of weights
---------------------------------------
Training is necessarily CENTRALISED (back-propagation through the unrolled
iteration needs the whole trajectory), while deployment must be
DISTRIBUTED (each prosumer queries its own copy of the model). Both are
provided here and are numerically identical:

    forward(net, x)        -- centralised, vectorised over the whole graph.
                              Used for training and for fast simulation.

    local_forward(state)   -- what a single prosumer runs after the
                              communication step, using ONLY its own
                              variables and the q values received from its
                              direct neighbours. Used for real distributed
                              inference.

Because the weights theta are shared by every node and every edge, and all
MLP sizes depend only on the feature dimensions and d_h (never on N, |E| or
|Omega_n|), the trained model can simply be copied to every node and run on
networks of any size and topology (zero-shot generalisation).

Deployment
----------
    gnn = PerturbationGNN(cfg)            # train centrally ...
    torch.save(gnn.state_dict(), "phi.pt")

    agents = deploy(gnn, net)             # ... then hand a copy to each node
    dD, dG, dq = agents[n].perturbation(local_state, k)

Role in the algorithm
---------------------
After the plain DR update (line 7) each prosumer adds

    x^{k+1}_n  <-  x^{k+1}_n + v^k_n ,      v^k = scale * rho^k * Phi_theta

Theorem 1 guarantees convergence to a v-GNE for ANY learned map, provided
the perturbation sequence is summable (v in ell_1). We enforce that BY
CONSTRUCTION:

    * both output heads end with tanh  =>  |Phi_theta| <= 1 componentwise,
      hence ||Phi_theta||_2 <= sqrt(dim);
    * the output is scaled by the decaying amplitude rho^k, rho in (0,1).

    sum_k ||v^k|| <= scale * sqrt(dim) / (1 - rho) < infinity,

so summability holds whatever the network outputs -- even untrained or
out-of-distribution. The learned part can only affect speed, never
correctness.

Architecture (single message-passing layer, eqs. 6-12 of the paper)
-------------------------------------------------------------------
For the directed edge (m -> n), q_{mn} is n's own decision toward m and
q_{nm} is what n RECEIVES from m during the communication step.

    node features   h_n      = (D_n, G_n)                   in R^2   (eq. 6)
    edge features   e_mn     = (q_nm)                       in R^1   (eq. 7)
    message         msg_m->n = phi_msg(e_mn)                in R^dh  (eq. 8)
    aggregation     h_agg_n  = 1/sqrt(|Omega_n|) sum_m msg  in R^dh  (eq. 9)
    node update     h_new_n  = phi_upd(h_n, h_agg_n)        in R^dh  (eq. 10)
    node head       delta_n  = phi_node(h_new_n)            in R^2   (eq. 11)
    edge head       delta_mn = phi_edge(h_new_n, e_mn)      in R^1   (eq. 12)

delta_n perturbs (D_n, G_n); delta_mn perturbs q_{mn}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from energy_network import EnergyNetwork


# ======================================================================
# Configuration
# ======================================================================
@dataclass
class GNNConfig:
    """Hyper-parameters of the perturbation map.

    hidden_dim (d_h) MUST be a fixed constant, independent of N, |E| and the
    neighbourhood sizes: that is what makes the weights reusable across
    topologies.
    """
    hidden_dim: int = 32          # d_h -- fixed, independent of N and |E|
    rho: float = 0.9              # perturbation decay in (0,1) -> summability
    scale: float = 1.0            # global amplitude of the perturbation
    activation: str = "tanh"      # hidden activation: tanh | relu
    normalize: bool = True        # rescale features with LOCAL parameters only
    rich_edge_features: bool = False  # False: e_mn = (q_nm)            [paper]
                                      # True : e_mn = (q_mn, q_nm, sum) [ablation]

    def __post_init__(self):
        if not (0.0 < self.rho < 1.0):
            raise ValueError("rho must lie in (0,1) to ensure summability")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")


def _mlp(d_in: int, d_out: int, d_hidden: int, activation: str) -> nn.Sequential:
    """Two-layer MLP. Sizes depend only on features and d_h, never on N."""
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        act(),
        nn.Linear(d_hidden, d_out),
    )


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
    q_own: Dict[int, torch.Tensor]
    q_recv: Dict[int, torch.Tensor]
    D_min: torch.Tensor
    D_max: torch.Tensor
    G_min: torch.Tensor
    G_max: torch.Tensor
    kappa: Dict[int, torch.Tensor]


def extract_local_state(net: EnergyNetwork, x: torch.Tensor,
                        n: int) -> LocalNodeState:
    """Build node n's local view from a global iterate.

    Used for testing and simulation: in a real deployment each prosumer
    assembles this from its own memory plus the messages it received.
    """
    D, G, Q = net.get_DGQ(x)
    nb = net.neighbors[n]
    idx = {m: net._idx_map[(m, n)] for m in nb}          # edge (m -> n)
    p = net.prosumers[n]
    return LocalNodeState(
        node_id=n, neighbors=list(nb),
        D=D[n], G=G[n],
        q_own={m: Q[idx[m]] for m in nb},                 # q_{mn}
        q_recv={m: Q[net.inv_idx[idx[m]]] for m in nb},   # q_{nm}
        D_min=p.D_min, D_max=p.D_max, G_min=p.G_min, G_max=p.G_max,
        kappa={m: p.k[m] for m in nb},
    )


# ======================================================================
# The perturbation network
# ======================================================================
class PerturbationGNN(nn.Module):
    """Phi_theta: one message-passing layer + two bounded output heads."""

    def __init__(self, config: Optional[GNNConfig] = None):
        super().__init__()
        self.cfg = config if config is not None else GNNConfig()
        d_h = self.cfg.hidden_dim

        self.d_node = 2                                        # (D, G)
        self.d_edge = 3 if self.cfg.rich_edge_features else 1  # (q_nm) [paper]

        act = self.cfg.activation
        self.phi_msg = _mlp(self.d_edge, d_h, d_h, act)                # eq. 8
        self.phi_upd = _mlp(self.d_node + d_h, d_h, d_h, act)          # eq. 10
        self.phi_node = _mlp(d_h, 2, d_h, act)                         # eq. 11
        self.phi_edge = _mlp(d_h + self.d_edge, 1, d_h, act)           # eq. 12

    # ==================================================================
    # CENTRALISED PATH  (training / fast simulation)
    # ==================================================================
    def build_features(self, net: EnergyNetwork, x: torch.Tensor):
        """Node and edge features from the global iterate x.

        Every entry is computed from quantities the corresponding node
        owns or received one-hop, so this is only a vectorised rewriting
        of the distributed computation.
        """
        D, G, Q = net.get_DGQ(x)

        if self.cfg.normalize:
            dD = torch.clamp(net.D_max_vec - net.D_min_vec, min=1e-6)
            dG = torch.clamp(net.G_max_vec - net.G_min_vec, min=1e-6)
            h_D = 2.0 * (D - net.D_min_vec) / dD - 1.0
            h_G = 2.0 * (G - net.G_min_vec) / dG - 1.0
            kappa = torch.clamp(net.k_edge, min=1e-6)
        else:
            h_D, h_G = D, G
            kappa = torch.ones_like(net.k_edge)

        h = torch.stack([h_D, h_G], dim=1)                    # (N, 2)   eq. 6

        if net.E == 0:
            return h, torch.zeros(0, self.d_edge, device=net.device)

        q_own = Q / kappa                     # q_{mn}: n's own decision
        q_recv = Q[net.inv_idx] / kappa       # q_{nm}: received from m  eq. 7

        if self.cfg.rich_edge_features:
            e = torch.stack([q_own, q_recv, q_own + q_recv], dim=1)   # (E,3)
        else:
            e = q_recv.unsqueeze(1)                                    # (E,1)
        return h, e

    def forward(self, net: EnergyNetwork, x: torch.Tensor) -> torch.Tensor:
        """Bounded direction Phi_theta(x) for the whole network, shape (dim,).

        Components lie in [-1, 1]; the decay rho^k is applied outside
        (see `make_perturbation_fn`), keeping this module stateless with
        respect to the iteration counter.
        """
        h, e = self.build_features(net, x)
        N, d_h = net.N, self.cfg.hidden_dim

        if net.E > 0:
            owner = net.edge_index[1]          # dst = n, owner of q_{mn}

            msg = self.phi_msg(e)                                   # eq. 8
            agg = torch.zeros(N, d_h, device=net.device, dtype=msg.dtype)
            agg = agg.index_add(0, owner, msg)                      # eq. 9
            deg = torch.zeros(N, device=net.device, dtype=msg.dtype)
            deg = deg.index_add(0, owner,
                                torch.ones(net.E, device=net.device,
                                           dtype=msg.dtype))
            agg = agg / torch.clamp(deg, min=1.0).sqrt().unsqueeze(1)
        else:
            agg = torch.zeros(N, d_h, device=net.device, dtype=h.dtype)

        h_new = self.phi_upd(torch.cat([h, agg], dim=1))            # eq. 10
        delta_node = torch.tanh(self.phi_node(h_new))               # eq. 11

        if net.E > 0:
            h_owner = h_new[net.edge_index[1]]
            delta_edge = torch.tanh(
                self.phi_edge(torch.cat([h_owner, e], dim=1))).squeeze(1)  # eq.12
        else:
            delta_edge = torch.zeros(0, device=net.device, dtype=h.dtype)

        # layout [D, G, Q], matching EnergyNetwork.flatten()
        return torch.cat([delta_node[:, 0], delta_node[:, 1], delta_edge])

    # ==================================================================
    # DISTRIBUTED PATH  (inference at a single prosumer)
    # ==================================================================
    def local_forward(self, s: LocalNodeState):
        """Perturbation direction computed by prosumer n on its own.

        Uses ONLY s: n's own state and parameters plus the q values received
        from its direct neighbours. Mathematically identical to the
        corresponding slice of `forward`.

        Returns
        -------
        delta_D, delta_G : 0-dim tensors, perturbation of (D_n, G_n)
        delta_q          : {m: 0-dim tensor}, perturbation of q_{mn}
        """
        # --- local features (eq. 6) ---
        if self.cfg.normalize:
            dD = torch.clamp(s.D_max - s.D_min, min=1e-6)
            dG = torch.clamp(s.G_max - s.G_min, min=1e-6)
            h_D = 2.0 * (s.D - s.D_min) / dD - 1.0
            h_G = 2.0 * (s.G - s.G_min) / dG - 1.0
        else:
            h_D, h_G = s.D, s.G
        h = torch.stack([h_D.reshape(()), h_G.reshape(())])          # (2,)

        d_h = self.cfg.hidden_dim
        k_nb = len(s.neighbors)

        if k_nb == 0:
            agg = torch.zeros(d_h, dtype=h.dtype)
            h_new = self.phi_upd(torch.cat([h, agg]))
            delta_node = torch.tanh(self.phi_node(h_new))
            return delta_node[0], delta_node[1], {}

        # --- edge features (eq. 7), one row per neighbour ---
        rows = []
        for m in s.neighbors:
            kap = (torch.clamp(s.kappa[m], min=1e-6) if self.cfg.normalize
                   else torch.ones_like(s.kappa[m]))
            q_recv = s.q_recv[m] / kap
            if self.cfg.rich_edge_features:
                q_own = s.q_own[m] / kap
                rows.append(torch.stack([q_own.reshape(()), q_recv.reshape(()),
                                         (q_own + q_recv).reshape(())]))
            else:
                rows.append(q_recv.reshape(1))
        e = torch.stack(rows)                                        # (k, d_edge)

        msg = self.phi_msg(e)                                        # eq. 8
        agg = msg.sum(dim=0) / float(k_nb) ** 0.5                    # eq. 9
        h_new = self.phi_upd(torch.cat([h, agg]))                    # eq. 10
        delta_node = torch.tanh(self.phi_node(h_new))                # eq. 11

        delta_q = {}
        for i, m in enumerate(s.neighbors):                          # eq. 12
            out = self.phi_edge(torch.cat([h_new, e[i]]))
            delta_q[m] = torch.tanh(out).reshape(())

        return delta_node[0], delta_node[1], delta_q

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def n_parameters(self) -> int:
        """Number of trainable weights (independent of N and |E|)."""
        return sum(p.numel() for p in self.parameters())

    def summability_bound(self, dim: int) -> float:
        """Upper bound on sum_k ||v^k||: scale * sqrt(dim) / (1 - rho)."""
        return self.cfg.scale * (dim ** 0.5) / (1.0 - self.cfg.rho)


# ======================================================================
# Wiring into the DR solver (centralised simulation)
# ======================================================================
def make_perturbation_fn(gnn: PerturbationGNN,
                         rho: Optional[float] = None,
                         scale: Optional[float] = None):
    """Build the `perturbation(net, x, k) -> v` hook used by DRSolver.

    Applies the decaying amplitude:  v^k = scale * rho^k * Phi_theta(x).
    Since Phi_theta is bounded by construction, the sequence is summable
    and Theorem 1 applies.
    """
    rho = gnn.cfg.rho if rho is None else rho
    scale = gnn.cfg.scale if scale is None else scale

    def perturbation(net: EnergyNetwork, x: torch.Tensor, k: int) -> torch.Tensor:
        return (scale * (rho ** k)) * gnn(net, x)

    return perturbation


# ======================================================================
# Deployment: hand the trained weights to every prosumer
# ======================================================================
class LocalAgent:
    """A prosumer's on-board copy of the trained perturbation model.

    In a real deployment each device holds its own instance (same weights).
    `perturbation` is what line 8 of Algorithm 1 executes locally, right
    after the communication round.
    """

    def __init__(self, gnn: PerturbationGNN, node_id: int,
                 rho: Optional[float] = None, scale: Optional[float] = None):
        self.gnn = gnn
        self.node_id = int(node_id)
        self.rho = gnn.cfg.rho if rho is None else rho
        self.scale = gnn.cfg.scale if scale is None else scale

    @torch.no_grad()
    def perturbation(self, state: LocalNodeState, k: int):
        """v^k_n for this prosumer: (delta_D, delta_G, {m: delta_q})."""
        amp = self.scale * (self.rho ** k)
        dD, dG, dq = self.gnn.local_forward(state)
        return amp * dD, amp * dG, {m: amp * v for m, v in dq.items()}


def deploy(gnn: PerturbationGNN, net, share_weights: bool = True) -> None:
    """Install the trained model on every prosumer of `net`.

    After this call each prosumer runs line 8 on its own during `dr_step`.
    `share_weights=True` keeps a single copy of the tensors in memory (fine
    for simulation); set it to False to give every node a physically
    independent copy, as when shipping the weights to real devices.
    """
    import copy as _copy
    gnn.eval()
    for p in net.prosumers:
        model = gnn if share_weights else _copy.deepcopy(gnn)
        model.eval()
        p.gnn_agent = LocalAgent(model, p.node_id)


def undeploy(net) -> None:
    """Remove the learned perturbation: the prosumers fall back to plain DR."""
    for p in net.prosumers:
        p.gnn_agent = None