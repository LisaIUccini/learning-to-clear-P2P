"""
Distributed L2WS: a GNN-based warm-start predictor.

centralized_model.py's `WarmStartMLP` reads a flat `theta` vector built from EVERY
node's and EVERY edge's private data (`build_theta`) into one monolithic
MLP -- fully centralised, flagged as this benchmark's main unresolved
limitation. This module replaces it with a
message-passing GNN so h_w becomes a genuinely distributed, one-hop-at-a-
time protocol, architecturally mirroring this project's own `gnn.py`
(`PerturbationGNN`) -- same scatter/index_add message-passing pattern, same
"deployable on any N" property (every MLP's size depends only on feature
counts, never N or |E|) -- but predicting a ONE-SHOT absolute warm start
z^0 from STATIC market parameters, not a per-iteration correction from
LIVE DR state.

    node n's raw features (static, always locally known):
        D_min, D_max, D_star, G_min, G_max, G_res, a_tilde, a, b   (9)
    edge (m,n)'s raw features (shared, known to both endpoints):
        c_mn, kappa_mn                                              (2)

    h_n^0        = phi_embed(normalise(raw_node_n))
    for L rounds (shared weights across rounds -- literally "the same
    local protocol every node runs, L times", not L distinct layers):
        msg_(m->n) = phi_msg([h_m^l ; e_mn])
        agg_n^l    = (1/sqrt(|Omega_n|)) sum_m msg_(m->n)
        h_n^{l+1}  = GRUCell(agg_n^l, h_n^l)
    D_n = D_min_n + (tanh(phi_D(h_n^L))+1)/2 * (D_max_n - D_min_n)   (box
    G_n = G_min_n + (tanh(phi_G(h_n^L))+1)/2 * (G_max_n - G_min_n)    midpoint
    q_mn = tanh(phi_Q([h_n^L ; e_mn])) * kappa_mn                     at init,
                                                                       zero-init
                                                                       last layer
                                                                       -- same
                                                                       trick as
                                                                       WarmStartMLP)

`source = net.owner[net.inv_idx]` recovers each edge's SENDER m without a
new cached field: edge i=(m,n) has owner[i]=n (network.py's own
convention); inv_idx[i] is the index of the REVERSE entry (n,m), whose
owner is m -- `owner[inv_idx[i]] == m`. Exactly the trick gnn.py's own
`Q[net.inv_idx]` ("received" value) already relies on.

Communication-cost model (for the paper's round-count figures, same
ANALYTICAL convention as A2DR's "1+2*diameter" and plain centralised
L2WS's "2*diameter" -- none of these are measured from literally running
parallel OS processes, see time_parallel_equivalent.py's own docstring for
why not): L one-time rounds (one per message-passing layer), each a
genuine one-hop exchange of hidden embeddings -- vs. the centralised
version's 2*diameter (gather everything to one point, broadcast the
prediction back). For L < diameter (the common case, L is a small fixed
hyperparameter while diameter grows with network size/sparsity), this is
strictly cheaper, not just "fair".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .. import market

D_NODE_STATIC = 9   # D_min, D_max, D_star, G_min, G_max, G_res, a_tilde, a, b
D_EDGE_STATIC = 2   # c, kappa


def _mlp(d_in: int, d_out: int, d_hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_out))


@dataclass
class DistributedWarmStartConfig:
    hidden_dim: int = 32
    n_rounds: int = 2   # L -- message-passing rounds (one-time communication cost, see module docstring)
    shared_weights: bool = True   # True: same phi_msg/phi_upd applied L times (one local protocol,
                                   # run L times). False: L independent layers (own phi_msg_l/phi_upd_l
                                   # each round) -- more parameters, no longer "the same code on every
                                   # node for L rounds", but potentially more expressive per round.
    update_type: str = "gru"      # "gru": phi_upd = GRUCell(agg, h) (gnn.py's own choice, designed for
                                   # dozens of DR iterations, not necessarily ideal for L=1-5 structural
                                   # rounds). "residual": h_new = h + mlp([h; agg]), a plain residual
                                   # update -- avoids GRU's gating overhead at small L and the
                                   # oversmoothing GRUs can cause when stacked (node embeddings
                                   # converging toward each other) by keeping an explicit skip connection.

    def __post_init__(self):
        if self.update_type not in ("gru", "residual"):
            raise ValueError(f"update_type must be 'gru' or 'residual', got {self.update_type!r}")


class NodeEdgeNormalizer:
    """z-score normalisation for the GNN's raw per-node/per-edge features,
    fit ONCE on the training set's instances (stacked across every node /
    every edge of every training instance -- NOT per-instance, unlike
    ThetaNormalizer's flat theta vector, since here the model must stay
    deployable on any N: there is no fixed per-node "slot" to normalise
    independently, only a shared distribution over what any one node's
    features look like)."""

    def __init__(self, specs: List["market.MarketSpec"], device):
        node_rows, edge_rows = [], []
        for spec in specs:
            net = market.build_network(spec, device=device)
            node_rows.append(torch.stack([
                net.D_min_vec, net.D_max_vec, net.D_star_vec,
                net.G_min_vec, net.G_max_vec, net.G_res_vec,
                net.a_tilde_vec, net.a_vec, net.b_vec,
            ], dim=1))
            if net.E:
                edge_rows.append(torch.stack([net.c_edge, net.k_edge], dim=1))
        node_all = torch.cat(node_rows, dim=0)
        self.node_mean = node_all.mean(dim=0)
        self.node_std = node_all.std(dim=0).clamp(min=1e-6)
        if edge_rows:
            edge_all = torch.cat(edge_rows, dim=0)
            self.edge_mean = edge_all.mean(dim=0)
            self.edge_std = edge_all.std(dim=0).clamp(min=1e-6)
        else:
            self.edge_mean = torch.zeros(D_EDGE_STATIC, device=device)
            self.edge_std = torch.ones(D_EDGE_STATIC, device=device)


def _make_update(update_type: str, d_h: int) -> nn.Module:
    if update_type == "gru":
        return nn.GRUCell(d_h, d_h)
    return _mlp(2 * d_h, d_h, d_h)   # "residual": consumed as h + f([h; agg])


class DistributedWarmStartGNN(nn.Module):
    def __init__(self, cfg: Optional[DistributedWarmStartConfig] = None):
        super().__init__()
        self.cfg = cfg or DistributedWarmStartConfig()
        d_h = self.cfg.hidden_dim
        L = self.cfg.n_rounds
        self.phi_embed = _mlp(D_NODE_STATIC, d_h, d_h)
        if self.cfg.shared_weights:
            self.phi_msg = _mlp(d_h + D_EDGE_STATIC, d_h, d_h)
            self.phi_upd = _make_update(self.cfg.update_type, d_h)
        else:
            self.phi_msg = nn.ModuleList([_mlp(d_h + D_EDGE_STATIC, d_h, d_h) for _ in range(L)])
            self.phi_upd = nn.ModuleList([_make_update(self.cfg.update_type, d_h) for _ in range(L)])
        self.phi_D = _mlp(d_h, 1, d_h)
        self.phi_G = _mlp(d_h, 1, d_h)
        self.phi_Q = _mlp(d_h + D_EDGE_STATIC, 1, d_h)
        # zero-init last layer of every output head -- tanh(0)=0, so D/G
        # start EXACTLY at their own box midpoint and q starts at 0,
        # matching WarmStartMLP's own init convention (box mid / Q=0)
        # without needing a per-node bias lookup.
        for head in (self.phi_D, self.phi_G, self.phi_Q):
            last_linear = [m for m in head if isinstance(m, nn.Linear)][-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
        # residual update: zero-init its last layer too, so at init h^L = h^0
        # (0 rounds' worth of change) for every round -- same "start neutral"
        # convention as the output heads above. Not applicable to GRUCell
        # (no equivalent no-op initialisation; left at its PyTorch default).
        if self.cfg.update_type == "residual":
            upd_modules = [self.phi_upd] if self.cfg.shared_weights else list(self.phi_upd)
            for upd in upd_modules:
                last_linear = [m for m in upd if isinstance(m, nn.Linear)][-1]
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_centralised(self, net, normalizer: NodeEdgeNormalizer) -> torch.Tensor:
        """Vectorised over the whole graph (same reason gnn.py's own
        forward_centralised exists: training backpropagates through the
        unrolled DR trajectory and needs the whole state at once) -- but
        every aggregation is a LOCAL one-hop sum (scatter/index_add over
        `net.owner`), so this computes EXACTLY what L independent
        per-node agents running L rounds of message passing would, just
        batched for speed. See module docstring for the source-index trick."""
        d_h = self.cfg.hidden_dim
        N, E = net.N, net.E
        raw_node = torch.stack([
            net.D_min_vec, net.D_max_vec, net.D_star_vec,
            net.G_min_vec, net.G_max_vec, net.G_res_vec,
            net.a_tilde_vec, net.a_vec, net.b_vec,
        ], dim=1)
        node_n = (raw_node - normalizer.node_mean) / normalizer.node_std
        h = self.phi_embed(node_n)   # (N, d_h)

        if E > 0:
            raw_edge = torch.stack([net.c_edge, net.k_edge], dim=1)
            e = (raw_edge - normalizer.edge_mean) / normalizer.edge_std
            source = net.owner[net.inv_idx]   # sender m for each edge entry (m,n) -- see docstring
            deg = torch.zeros(N, device=net.device, dtype=h.dtype)
            deg = deg.index_add(0, net.owner, torch.ones(E, device=net.device, dtype=h.dtype))
            inv_sqrt_deg = torch.clamp(deg, min=1.0).sqrt().unsqueeze(1)

            for l in range(self.cfg.n_rounds):
                phi_msg_l = self.phi_msg if self.cfg.shared_weights else self.phi_msg[l]
                phi_upd_l = self.phi_upd if self.cfg.shared_weights else self.phi_upd[l]
                msg = phi_msg_l(torch.cat([h[source], e], dim=1))        # (E, d_h)
                agg = torch.zeros(N, d_h, device=net.device, dtype=h.dtype)
                agg = agg.index_add(0, net.owner, msg) / inv_sqrt_deg
                if self.cfg.update_type == "gru":
                    h = phi_upd_l(agg, h)
                else:
                    h = h + phi_upd_l(torch.cat([h, agg], dim=1))
        # else: no edges, h stays at its embedded value (0 rounds possible)

        D_pos = torch.tanh(self.phi_D(h)).squeeze(-1)
        G_pos = torch.tanh(self.phi_G(h)).squeeze(-1)
        D_out = net.D_min_vec + (D_pos + 1.0) / 2.0 * (net.D_max_vec - net.D_min_vec)
        G_out = net.G_min_vec + (G_pos + 1.0) / 2.0 * (net.G_max_vec - net.G_min_vec)

        if E > 0:
            Q_pos = torch.tanh(self.phi_Q(torch.cat([h[net.owner], e], dim=1))).squeeze(-1)
            Q_out = Q_pos * net.k_edge
        else:
            Q_out = torch.zeros(0, device=net.device, dtype=h.dtype)

        return torch.cat([D_out, G_out, Q_out])


class DistributedPredictor:
    """Same `.predict(spec) -> warm_start tensor` contract as
    `model.L2WSPredictor`, so it's a drop-in replacement wherever a
    predictor object (not the raw module) is needed (evaluate_with_curves,
    the L2WS `run_pair`)."""

    def __init__(self, gnn: DistributedWarmStartGNN, normalizer: NodeEdgeNormalizer, device=None):
        self.gnn = gnn
        self.normalizer = normalizer
        self.device = device or torch.device("cpu")

    def save(self, path) -> None:
        n = self.normalizer
        torch.save({"state_dict": self.gnn.state_dict(), "cfg": self.gnn.cfg.__dict__,
                    "node_mean": n.node_mean, "node_std": n.node_std,
                    "edge_mean": n.edge_mean, "edge_std": n.edge_std}, path)

    @classmethod
    def load(cls, path, device=None) -> "DistributedPredictor":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        gnn = DistributedWarmStartGNN(DistributedWarmStartConfig(**blob["cfg"]))
        gnn.load_state_dict(blob["state_dict"])
        gnn.eval()
        normalizer = NodeEdgeNormalizer.__new__(NodeEdgeNormalizer)
        for key in ("node_mean", "node_std", "edge_mean", "edge_std"):
            setattr(normalizer, key, blob[key])
        return cls(gnn, normalizer, device=device)

    def warm_start(self, net, spec: "market.MarketSpec", differentiable: bool) -> torch.Tensor:
        """Warm start during training, computed on the already built `net`."""
        warm_start = self.gnn.forward_centralised(net, self.normalizer)
        return warm_start if differentiable else warm_start.detach()

    def predict(self, spec: "market.MarketSpec", differentiable: bool = False) -> torch.Tensor:
        net = market.build_network(spec, device=self.device)
        if differentiable:
            return self.gnn.forward_centralised(net, self.normalizer)
        with torch.no_grad():
            return self.gnn.forward_centralised(net, self.normalizer)

    def parameters(self):
        return self.gnn.parameters()

    def train(self):
        self.gnn.train()

    def eval(self):
        self.gnn.eval()
