"""
gamma_gnn.py
============

Learns per-node, INSTANCE-ADAPTIVE DR prox step sizes gamma_i, inspired by
Doerks, Haeusner, Hernandez Escobar & Sjoelund, "Learning to accelerate
distributed ADMM using graph neural networks" (arXiv:2509.05288, 2025) --
their "Local alpha_i" task (Sec. 4.1), ported to this project's own DR map.

Unlike gnn.py's PerturbationGNN (Phi_w), which ADDS a learned correction
on top of plain DR at every iteration, this predicts a MULTIPLICATIVE
hyperparameter -- the prox step size gamma used INSIDE the existing,
unmodified DR update (network.py's own prox_local/local_prox) -- the DR
algorithm itself is untouched, only WHAT VALUE of gamma each node uses
for its own local prox. No per-iteration correction, no extra channel.

Convergence caveat (be upfront about this, do not overclaim): the ADMM
paper's own result (Sec. 3.2) states the ergodic average converges for
ANY global step size alpha>0 OR any per-node alpha_i>0, citing established
references for both. For THIS project's own DR map, there is no
equivalent per-node-varying-gamma theorem on record -- the existing
project convergence theorem assumes one shared scalar gamma. Empirically,
this project has observed DR converge (at different speeds, never
diverge) across a wide scalar-gamma range; per-node gamma_i is a
genuinely new, UNVERIFIED case -- validated here only empirically (smoke
tests + held-out test-set convergence), not by a proof. Treat results as
suggestive, not as a settled theoretical guarantee, until/unless checked
more carefully.

Predicted ONCE from the instance's STATIC market data (NOT live DR
state) before the DR loop starts -- mirrors the distributed L2WS warm-
start's own "theta -> one prediction" design (one-time communication),
not Phi_w's "live state, every iteration" design (which needs its own
one-hop round every single iteration). Architecture reuses the validated-
best distributed L2WS config (L message-passing rounds, shared weights,
residual update) -- same features/source-index trick as
l2ws_benchmark/distributed_model.py -- only the output head differs: ONE
scalar gamma_i per node instead of a warm-start z^0. Zero-init last layer
so training starts EXACTLY at the market's own already-validated scalar
gamma_base (the safe, known-convergent default), departing from it only
as training finds an improvement -- same "start neutral" convention used
throughout this project (WarmStartMLP's box-midpoint init,
DistributedWarmStartGNN's zero-init heads).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

import market

D_NODE_STATIC = 9   # D_min, D_max, D_star, G_min, G_max, G_res, a_tilde, a, b
D_EDGE_STATIC = 2   # c, kappa


def _mlp(d_in: int, d_out: int, d_hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_out))


@dataclass
class GammaGNNConfig:
    hidden_dim: int = 32
    n_rounds: int = 3           # L -- validated winner from the L2WS distributed ablation
    gamma_base: float = 4.0     # the market's own already-validated SCALAR gamma (safe default,
                                 # exact output at init since the last layer is zero-init)
    log_range: float = 1.0986   # ln(3.0) -- gamma_i in [gamma_base/3, gamma_base*3] at full
                                 # saturation (tanh->+-1); keeps predictions in a plausible
                                 # neighbourhood of the known-good scalar rather than unbounded


class NodeEdgeNormalizer:
    """Same z-score normalisation as l2ws_benchmark/distributed_model.py's
    own class (duplicated, not imported -- this module lives at the
    project root like gnn.py, with no dependency on the experiments/
    subtree), fit ONCE on the training set's instances, stacked across
    every node / every edge of every training instance."""

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


class GammaGNN(nn.Module):
    def __init__(self, cfg: Optional[GammaGNNConfig] = None):
        super().__init__()
        self.cfg = cfg or GammaGNNConfig()
        d_h = self.cfg.hidden_dim
        self.phi_embed = _mlp(D_NODE_STATIC, d_h, d_h)
        self.phi_msg = _mlp(d_h + D_EDGE_STATIC, d_h, d_h)
        self.phi_upd = _mlp(2 * d_h, d_h, d_h)   # residual update: h + f([h; agg])
        self.phi_gamma = _mlp(d_h, 1, d_h)
        last_linear = [m for m in self.phi_upd if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)
        last_linear = [m for m in self.phi_gamma if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_centralised(self, net, normalizer: NodeEdgeNormalizer) -> torch.Tensor:
        """Returns gamma_i, shape (N,), strictly positive, equal to
        cfg.gamma_base at initialisation (zero-init phi_gamma -> raw=0 ->
        tanh(0)=0 -> exp(0)=1)."""
        d_h = self.cfg.hidden_dim
        N, E = net.N, net.E
        raw_node = torch.stack([
            net.D_min_vec, net.D_max_vec, net.D_star_vec,
            net.G_min_vec, net.G_max_vec, net.G_res_vec,
            net.a_tilde_vec, net.a_vec, net.b_vec,
        ], dim=1)
        node_n = (raw_node - normalizer.node_mean) / normalizer.node_std
        h = self.phi_embed(node_n)

        if E > 0:
            raw_edge = torch.stack([net.c_edge, net.k_edge], dim=1)
            e = (raw_edge - normalizer.edge_mean) / normalizer.edge_std
            source = net.owner[net.inv_idx]
            deg = torch.zeros(N, device=net.device, dtype=h.dtype)
            deg = deg.index_add(0, net.owner, torch.ones(E, device=net.device, dtype=h.dtype))
            inv_sqrt_deg = torch.clamp(deg, min=1.0).sqrt().unsqueeze(1)

            for _ in range(self.cfg.n_rounds):
                msg = self.phi_msg(torch.cat([h[source], e], dim=1))
                agg = torch.zeros(N, d_h, device=net.device, dtype=h.dtype)
                agg = agg.index_add(0, net.owner, msg) / inv_sqrt_deg
                h = h + self.phi_upd(torch.cat([h, agg], dim=1))

        raw_gamma = torch.tanh(self.phi_gamma(h)).squeeze(-1)   # (N,), in (-1, 1), =0 at init
        gamma_i = self.cfg.gamma_base * torch.exp(raw_gamma * self.cfg.log_range)
        return gamma_i


def rollout_gamma(net, gamma_vec: torch.Tensor, cfg, x0: Optional[torch.Tensor] = None,
                   differentiable: bool = True):
    """Same unrolled-DR loop as train.rollout, but with NO GNN correction
    and a PER-NODE gamma vector (shape (N,)) substituted for the usual
    scalar cfg.gamma at every call to prox_local -- the DR update itself
    is completely unmodified, see module docstring. `cfg` needs only
    .lam/.bisect/.unroll (a train.TrainConfig or any object with those
    three fields works)."""
    x = (net.flatten() if x0 is None else x0)
    if not differentiable:
        x = x.detach()
    xbar = net.project_reciprocity(x)
    residuals: List[torch.Tensor] = []
    ctx = torch.enable_grad() if differentiable else torch.no_grad()
    with ctx:
        for k in range(cfg.unroll):
            refl = 2.0 * xbar - x
            z = net.prox_local(gamma_vec, refl, n_bisect=cfg.bisect)
            x_new = x + cfg.lam * (z - xbar)
            residuals.append(torch.linalg.vector_norm(x_new - x))
            xbar = net.project_reciprocity(x_new)
            x = x_new
    return x, residuals


class GammaPredictor:
    """`.predict(spec) -> gamma_i tensor (N,)`, same contract shape as
    L2WSPredictor/DistributedPredictor's `.predict`."""

    def __init__(self, gnn: GammaGNN, normalizer: NodeEdgeNormalizer, device=None):
        self.gnn = gnn
        self.normalizer = normalizer
        self.device = device or torch.device("cpu")

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
