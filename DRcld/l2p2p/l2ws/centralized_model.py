"""
Centralized L2WS benchmark: warm-start MLP + market-parameter feature builder.

Implements the architecture of Sambharya, Hall, Amos & Stellato (JMLR 2024),
"Learning to Warm-Start Fixed-Point Optimization Algorithms": a feedforward
ReLU network h_w(theta) mapping a problem's parameters theta to a predicted
warm start z^0 = h_w(theta) for the SAME Douglas-Rachford fixed-point
operator used throughout (fixed_point.run_dr / training.rollout). At inference, z^0 is fed into `cfg.unroll` (training) or
until convergence (test) un-perturbed DR steps -- NO per-iteration
correction, unlike this project's own Phi_w (gnn.py), which alters every
step. Training differentiates through those steps (training.rollout(net,
None, cfg, x0=h_w(theta))) against the k-step
fixed-point residual (the paper's `\\ell^{fp}_\\theta`, Eq. 5) -- the paper's
own loss choice, since it needs no precomputed ground-truth fixed point.

Because h_w only ever touches the STARTING point, the DR operator that
follows is completely unmodified and keeps its own convergence guarantees
intact -- exactly what makes this a meaningful, independent benchmark
against Phi_w (which trades that property for a stronger per-step
correction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .. import market


def build_theta(spec: "market.MarketSpec") -> torch.Tensor:
    """Fixed-size feature vector theta for one market instance: every
    per-node primitive array concatenated, then per-edge c/kappa in a
    FIXED, SORTED edge-key order. Valid as a fixed-size MLP input only
    across instances sharing the same topology -- true for one base
    market's level+width-perturbed dataset (`scenarios.sample_level_width_dataset`,
    which changes values only, never the graph), the setting of this benchmark."""
    node_feats = np.concatenate([
        spec.D_min, spec.D_max, spec.D_star,
        spec.G_min, spec.G_max, spec.G_res,
        spec.a_tilde, spec.a, spec.b,
    ]).astype(np.float32)
    edge_keys = sorted(spec.c.keys())
    if edge_keys:
        c_feats = np.array([spec.c[k] for k in edge_keys], dtype=np.float32)
        kappa_feats = np.array([spec.kappa[k] for k in edge_keys], dtype=np.float32)
        edge_feats = np.concatenate([c_feats, kappa_feats])
    else:
        edge_feats = np.zeros(0, dtype=np.float32)
    return torch.from_numpy(np.concatenate([node_feats, edge_feats]))


def theta_dim(spec: "market.MarketSpec") -> int:
    return int(build_theta(spec).shape[0])


@dataclass
class WarmStartConfig:
    hidden_dim: int = 64
    n_hidden_layers: int = 1   # paper sweeps 0, 1, 2 (500 neurons, in its
                                # much larger OSQP/SCS problems); this market
                                # has theta_dim in the low dozens and
                                # out_dim = 2*N+E ~ 30-65 for N=10


class WarmStartMLP(nn.Module):
    """h_w(theta) = W_L phi(W_{L-1} phi(... phi(W_1 theta))), ReLU phi, as
    in the paper (Eq. 4) -- plain feedforward, no graph structure, matching
    the paper's own architecture choice (no GNN encoder here)."""

    def __init__(self, theta_dim_: int, out_dim: int, cfg: Optional[WarmStartConfig] = None,
                 init_point: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg or WarmStartConfig()
        dims = [theta_dim_] + [self.cfg.hidden_dim] * self.cfg.n_hidden_layers + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        # Zero-init the LAST layer's weight matrix (so at init the output is
        # just its bias) and seed that bias at a reasonable midpoint of the
        # feasible box (D mid, G mid, Q=0) instead of the origin -- purely
        # an initialisation convenience (the paper's own Figure 1: warm-start
        # LOCATION matters a lot for convergence speed even before any
        # training), not part of the paper's method itself.
        last_linear = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last_linear.weight)
        if init_point is not None:
            with torch.no_grad():
                last_linear.bias.copy_(init_point)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)


class ThetaNormalizer:
    """z-score normalisation fit once on the TRAINING set's theta vectors
    (mean/std per feature, never refit on val/test -- standard
    train-only-fit convention). Static per-instance market parameters have
    no natural shared scale across a_tilde/a/b/D/G/c/kappa, unlike the
    GNN's own STATE features (gnn.py's build_features), which are already
    instance-relative via each instance's own box bounds."""

    def __init__(self, thetas: List[torch.Tensor], eps: float = 1e-6):
        stacked = torch.stack(thetas)
        self.mean = stacked.mean(dim=0)
        self.std = stacked.std(dim=0).clamp(min=eps)

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.mean) / self.std


class L2WSPredictor:
    """Bundles the trained MLP + its fitted normalizer + the box-midpoint
    fallback into one `predict(spec) -> warm_start tensor` call, so this
    object alone can serve as the `model` argument to
    `l2ws.training.run_pair`."""

    def __init__(self, mlp: WarmStartMLP, normalizer: ThetaNormalizer, device=None):
        self.mlp = mlp
        self.normalizer = normalizer
        self.device = device or torch.device("cpu")

    def save(self, path) -> None:
        torch.save({"state_dict": self.mlp.state_dict(), "cfg": self.mlp.cfg.__dict__,
                    "normalizer_mean": self.normalizer.mean, "normalizer_std": self.normalizer.std}, path)

    @classmethod
    def load(cls, path, base_spec: "market.MarketSpec", device=None) -> "L2WSPredictor":
        """`base_spec` fixes the input/output sizes (same graph as training)."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        out_dim = market.build_network(base_spec).dim
        mlp = WarmStartMLP(theta_dim(base_spec), out_dim, WarmStartConfig(**blob["cfg"]))
        mlp.load_state_dict(blob["state_dict"])
        mlp.eval()
        normalizer = ThetaNormalizer.__new__(ThetaNormalizer)
        normalizer.mean, normalizer.std = blob["normalizer_mean"], blob["normalizer_std"]
        return cls(mlp, normalizer, device=device)

    def warm_start(self, net, spec: "market.MarketSpec", differentiable: bool) -> torch.Tensor:
        """Warm start during training (`net` is unused: theta comes from `spec`)."""
        warm_start = self.mlp(self.normalizer(build_theta(spec).to(self.device)))
        return warm_start if differentiable else warm_start.detach()

    def predict(self, spec: "market.MarketSpec", differentiable: bool = False) -> torch.Tensor:
        theta = build_theta(spec).to(self.device)
        theta_n = self.normalizer(theta)
        if differentiable:
            return self.mlp(theta_n)
        with torch.no_grad():
            return self.mlp(theta_n)

    def parameters(self):
        return self.mlp.parameters()

    def train(self):
        self.mlp.train()

    def eval(self):
        self.mlp.eval()
