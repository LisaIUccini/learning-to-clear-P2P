"""
gnn.py
======

The learned perturbation Phi_w (paper Sec. III-B, Eq. 10-14), added to the
plain DR update at line 12 of Algorithm 1:

    v^k_n = rho^k * Phi_w( h^k_n, {e^k_mn}_{m in Omega_n}, mem^{k-1}_n )

Two execution paths compute the SAME function of the SAME weights:

    forward_centralised  - vectorised over the whole graph via cached edge
                            tensors (scatter aggregation). Needed only
                            because training back-propagates through an
                            unrolled trajectory and autodiff needs the
                            whole state at once (Sec. III-C).

    forward_local         - exactly what one prosumer runs after
                            `Network.communicate()` (line 3), using ONLY
                            its own state/parameters, its own recurrent
                            memory, and the q it received from its direct
                            neighbours -- the SAME one-hop exchange
                            Algorithm 1 already performs for the DR step.
                            The GNN issues NO message of its own: the
                            edge feature e_mn is built purely from q_mn
                            (already local) and q_nm (received at line 3),
                            so message-passing here is a LOCAL computation
                            on already-available data, not a new
                            communication round.

Features (all strictly local; paper: "constructed from bilateral
variables already exchanged during Algorithm 1")
--------------------------------------------------------------------------
node n:  h_D, h_G      position of D_n, G_n in their own box, in [-1,1]
         h_Dstar       position of the target D*_n in the D-box
         h_bal         local balance residual D_n-G_n-DeltaG_n-Q_n,
                        normalised by the D-box width
         h_prev        this node's OWN move on the PREVIOUS iteration,
                        ||[D,G,q]^{k-1 -> k}||, normalised by the D-box
                        width -- a direct, zero-communication progress
                        signal ("am I still moving a lot, or have I
                        basically settled?"); zero on the very first
                        iteration.
         h_stall       exponential moving average of h_prev over recent
                        iterations (decay 0.95, effective window ~20
                        iterations), same normalisation -- a SUSTAINED
                        smallness signal, distinguishing "just took one
                        small step" from "have been stuck near-flat for a
                        while" (a plateau). Both h_prev and h_stall are
                        purely local bookkeeping (mirror the iteration
                        counter every node already keeps), not messages.
         h_atilde, h_a, h_b   n's own cost coefficients, normalised
                        against FIXED reference ranges (a node cannot
                        compute batch statistics without a global
                        reduction, so normalisation uses compile-time
                        constants matching the training distribution --
                        see GNNConfig)

edge (m -> n):  e_qown  = q_mn / kappa_mn   (n's own decision toward m)
                e_qrecv = q_nm / kappa_mn   (received at line 3)
                e_c     = c_nm, normalised

Message passing -> recurrent update -> TWO gated, bounded output channels
--------------------------------------------------------------------------
    msg_(m->n)  = phi_msg(e_mn)
    agg_n       = (1/sqrt(max(1,|Omega_n|))) sum_m msg_(m->n)
    mem_n^k     = GRUCell([h_n^k ; agg_n^k], mem_n^{k-1})

    -- fast channel (rho, scale -- short decay, k99 ~ 90-150) --
    gate_n        = sigmoid(phi_gate(mem_n^k))
    Delta_node    = gate_n * tanh(phi_node(mem_n^k))
    Delta_mn      = gate_n * tanh(phi_edge([mem_n^k ; e_mn]))

    -- slow channel (rho_slow, scale_slow -- long decay, k99 ~ 300-450) --
    gate_n_slow   = sigmoid(phi_gate_slow(mem_n^k))     ("off" by default,
                                                          see bias init below)
    Delta_node_slow = gate_n_slow * tanh(phi_node_slow(mem_n^k))
    Delta_mn_slow   = gate_n_slow * tanh(phi_edge_slow([mem_n^k ; e_mn]))

    v^k_n = scale*rho^k*Delta_node + scale_slow*rho_slow^k*Delta_node_slow
            (and likewise for the edge/trade components)

WHY TWO CHANNELS (see train.py for the measured motivation): a single
decay schedule that is short enough to protect timing-sensitive late-stage
plateaus (measured: essential) is, BY THE SAME TOKEN, too short to ever
meaningfully help genuinely slow instances (1000+ iterations) -- it is
gone long before their outcome is decided. Splitting the correction into
a fast, low-risk channel (protects win-rate, unchanged from before) and a
separate slow channel that starts almost entirely OFF (phi_gate_slow's
bias is initialised strongly negative) and must be talked into activating
by sustained evidence of genuine slow progress (h_stall) lets training
pursue large wins on hard instances without the two objectives fighting
over the SAME schedule.

`gate_n` / `gate_n_slow` are per-node CONFIDENCE gates, not in the paper's
Eq. 10-14 but a safe, well-motivated addition: purely multiplicative on an
already tanh-bounded quantity, so each can only ever SHRINK its channel's
correction toward "do nothing", never amplify it.

Every MLP's size depends only on the feature counts and the hidden width
d_h, never on N or |Omega_n|: one trained model is architecturally
deployable on any network size or topology (paper's third contribution).

Summability by construction (Theorem 1's hypothesis)
--------------------------------------------------------------------------
Each channel's raw correction is bounded componentwise by 1 (tanh, then a
gate in [0,1]) for ANY weights w. Rescaling by a fixed PHYSICAL reference
and multiplying by its own geometrically decaying rho^k (rho in (0,1))
bounds each channel's own sum; the total perturbation is the SUM of the
two channels, and a sum of two summable sequences is summable (triangle
inequality), so

    sum_k ||v^k|| <= sum_k ||v^k_fast|| + sum_k ||v^k_slow||
                  <= (scale/(1-rho) + scale_slow/(1-rho_slow)) * ||physical_scale||
                  < infinity

for any w, trained or not -- exactly the hypothesis Theorem 1 needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

Range = Tuple[float, float]

D_NODE = 9   # h_D, h_G, h_Dstar, h_bal, h_prev, h_stall, h_atilde, h_a, h_b
D_EDGE = 3   # e_qown, e_qrecv, e_c


@dataclass
class GNNConfig:
    hidden_dim: int = 32       # d_h -- fixed, independent of N and |E|

    # fast channel: short decay, protects timing-sensitive plateaus (see
    # train.py's measured comparison -- this pairing is the validated one)
    rho: float = 0.95          # in (0,1) for summability
    scale: float = 0.4         # fraction of the physical reference

    # slow channel: long decay, starts almost OFF (phi_gate_slow's bias),
    # meant to pursue large wins on genuinely slow instances (1000+
    # iterations) without disturbing the fast channel's protected regime
    rho_slow: float = 0.99     # in (0,1) for summability; k99 ~ 458
    scale_slow: float = 0.4    # fraction of the physical reference

    a_tilde_range: Range = (0.5, 1.5)
    a_range: Range = (0.5, 2.0)
    b_range: Range = (0.2, 1.0)
    c_range: Range = (0.05, 0.30)

    def __post_init__(self):
        if not (0.0 < self.rho < 1.0):
            raise ValueError("rho must lie in (0,1) for summability")
        if not (0.0 < self.rho_slow < 1.0):
            raise ValueError("rho_slow must lie in (0,1) for summability")

    @classmethod
    def from_market_config(cls, mc, **kw) -> "GNNConfig":
        return cls(a_tilde_range=tuple(mc.a_tilde_range), a_range=tuple(mc.a_range),
                   b_range=tuple(mc.b_range), c_range=tuple(mc.c_range), **kw)


def _mlp(d_in: int, d_out: int, d_hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.Tanh(), nn.Linear(d_hidden, d_out))


def _norm(v, rng: Range):
    lo, hi = rng
    return 2.0 * torch.clamp((v - lo) / (hi - lo), 0.0, 1.0) - 1.0


@dataclass
class LocalNodeState:
    """Everything a single prosumer needs to compute its own v^k_n -- built
    entirely from `Prosumer`'s own fields (see `PerturbationGNN.local_state`),
    nothing belonging to a non-neighbour."""
    neighbors: List[int]
    D: torch.Tensor; G: torch.Tensor; D_star: torch.Tensor; G_res: torch.Tensor
    D_min: torch.Tensor; D_max: torch.Tensor; G_min: torch.Tensor; G_max: torch.Tensor
    a_tilde: torch.Tensor; a: torch.Tensor; b: torch.Tensor
    q_own: Dict[int, torch.Tensor]; q_recv: Dict[int, torch.Tensor]
    c: Dict[int, torch.Tensor]; kappa: Dict[int, torch.Tensor]
    mem: torch.Tensor
    prev_residual: torch.Tensor
    stall: torch.Tensor


class PerturbationGNN(nn.Module):
    def __init__(self, cfg: Optional[GNNConfig] = None):
        super().__init__()
        self.cfg = cfg or GNNConfig()
        d_h = self.cfg.hidden_dim
        self.phi_msg = _mlp(D_EDGE, d_h, d_h)
        self.phi_upd = nn.GRUCell(D_NODE + d_h, d_h)

        # fast channel
        self.phi_node = _mlp(d_h, 2, d_h)
        self.phi_edge = _mlp(d_h + D_EDGE, 1, d_h)
        self.phi_gate = _mlp(d_h, 1, d_h)
        # slow channel
        self.phi_node_slow = _mlp(d_h, 2, d_h)
        self.phi_edge_slow = _mlp(d_h + D_EDGE, 1, d_h)
        self.phi_gate_slow = _mlp(d_h, 1, d_h)

        # Bias both gates to start CONSERVATIVE: "do nothing" is the
        # default a node has to be talked out of. The slow channel starts
        # even MORE conservative (sigmoid(-4) ~= 0.018 vs sigmoid(-2) ~=
        # 0.12) since it has a much longer reach and therefore much more
        # opportunity to do harm before training has taught it when that
        # reach is actually warranted.
        with torch.no_grad():
            self.phi_gate[-1].bias.fill_(-2.0)
            self.phi_gate_slow[-1].bias.fill_(-4.0)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_mem(self, N: int, device) -> torch.Tensor:
        return torch.zeros(N, self.cfg.hidden_dim, device=device)

    def decay(self, k: int) -> float:
        return self.cfg.rho ** k

    def decay_slow(self, k: int) -> float:
        return self.cfg.rho_slow ** k

    # ==================================================================
    # CENTRALISED PATH (training: needs the whole graph state at once)
    # ==================================================================
    def physical_scale(self, net) -> torch.Tensor:
        """Per-coordinate physical reference [half D-box, half G-box, kappa],
        matching `flatten()`'s [D, G, Q] layout."""
        half_D = (net.D_max_vec - net.D_min_vec) / 2.0
        half_G = (net.G_max_vec - net.G_min_vec) / 2.0
        return torch.cat([half_D, half_G, net.k_edge])

    def build_features(self, net, x: torch.Tensor, prev_residual: torch.Tensor,
                        stall: torch.Tensor):
        cfg = self.cfg
        D, G, Q = net.get_DGQ(x)
        dD = torch.clamp(net.D_max_vec - net.D_min_vec, min=1e-6)
        dG = torch.clamp(net.G_max_vec - net.G_min_vec, min=1e-6)
        h = torch.stack([
            2.0 * (D - net.D_min_vec) / dD - 1.0,
            2.0 * (G - net.G_min_vec) / dG - 1.0,
            2.0 * (net.D_star_vec - net.D_min_vec) / dD - 1.0,
            net.balance_at(x) / dD,
            prev_residual / dD,
            stall / dD,
            _norm(net.a_tilde_vec, cfg.a_tilde_range),
            _norm(net.a_vec, cfg.a_range),
            _norm(net.b_vec, cfg.b_range),
        ], dim=1)                                                  # (N, D_NODE)

        if net.E == 0:
            return h, torch.zeros(0, D_EDGE, device=net.device)
        kappa = torch.clamp(net.k_edge, min=1e-6)
        e = torch.stack([Q / kappa, Q[net.inv_idx] / kappa,
                          _norm(net.c_edge, cfg.c_range)], dim=1)   # (E, D_EDGE)
        return h, e

    def forward_centralised(self, net, x: torch.Tensor, mem: torch.Tensor,
                             prev_residual: torch.Tensor, stall: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bounded directions in PHYSICAL units, shape (dim,) = [D,G,Q]
        each. Returns (direction_fast, direction_slow, mem_new); the
        caller applies each channel's own amplitude schedule (see
        `train.rollout` / `LocalAgent.perturbation`)."""
        h, e = self.build_features(net, x, prev_residual, stall)
        N, d_h = net.N, self.cfg.hidden_dim

        if net.E > 0:
            msg = self.phi_msg(e)
            agg = torch.zeros(N, d_h, device=net.device, dtype=msg.dtype)
            agg = agg.index_add(0, net.owner, msg)
            deg = torch.zeros(N, device=net.device, dtype=msg.dtype)
            deg = deg.index_add(0, net.owner, torch.ones(net.E, device=net.device, dtype=msg.dtype))
            agg = agg / torch.clamp(deg, min=1.0).sqrt().unsqueeze(1)
        else:
            agg = torch.zeros(N, d_h, device=net.device, dtype=h.dtype)

        mem_new = self.phi_upd(torch.cat([h, agg], dim=1), mem)
        scale_vec = self.physical_scale(net)

        def _channel(phi_node, phi_edge, phi_gate):
            gate = torch.sigmoid(phi_gate(mem_new))                    # (N, 1)
            node_out = torch.tanh(phi_node(mem_new)) * gate            # (N, 2)
            if net.E > 0:
                edge_out = (torch.tanh(phi_edge(torch.cat([mem_new[net.owner], e], dim=1)))
                            * gate[net.owner]).squeeze(1)
            else:
                edge_out = torch.zeros(0, device=net.device, dtype=h.dtype)
            direction = torch.cat([node_out[:, 0], node_out[:, 1], edge_out])
            return direction * scale_vec

        direction_fast = _channel(self.phi_node, self.phi_edge, self.phi_gate)
        # scale_slow == 0 makes the caller's amp_slow * direction_slow exactly
        # zero regardless of direction_slow's value (LocalAgent.perturbation /
        # train.rollout) -- skip the slow channel's MLPs entirely rather than
        # compute and then discard them; identical result, ~2x fewer linear/
        # tanh evaluations per step when the slow channel is inactive (the
        # validated recipe throughout this project).
        if self.cfg.scale_slow == 0.0:
            direction_slow = torch.zeros_like(direction_fast)
        else:
            direction_slow = _channel(self.phi_node_slow, self.phi_edge_slow, self.phi_gate_slow)
        return direction_fast, direction_slow, mem_new

    # ==================================================================
    # DISTRIBUTED PATH (single prosumer, after `Network.communicate()`)
    # ==================================================================
    def local_state(self, prosumer) -> LocalNodeState:
        if prosumer._gnn_mem is None:
            prosumer._gnn_mem = torch.zeros(self.cfg.hidden_dim, device=prosumer.device)
        return LocalNodeState(
            neighbors=list(prosumer.neighbors), D=prosumer.D, G=prosumer.G,
            D_star=prosumer.D_star, G_res=prosumer.G_res,
            D_min=prosumer.D_min, D_max=prosumer.D_max,
            G_min=prosumer.G_min, G_max=prosumer.G_max,
            a_tilde=prosumer.a_tilde, a=prosumer.a, b=prosumer.b,
            q_own=dict(prosumer.q), q_recv=dict(prosumer.q_received),
            c=prosumer.c, kappa=prosumer.k, mem=prosumer._gnn_mem,
            prev_residual=prosumer.prev_residual, stall=prosumer.stall_ema,
        )

    def forward_local(self, s: LocalNodeState):
        """Same function as `forward_centralised`, evaluated by ONE node
        using only `s`. Returns ((dD_fast,dG_fast,dq_fast),
        (dD_slow,dG_slow,dq_slow), mem_new), already in PHYSICAL units
        (not yet scaled by either channel's decay -- see
        `LocalAgent.perturbation`)."""
        cfg = self.cfg
        dD = torch.clamp(s.D_max - s.D_min, min=1e-6)
        dG = torch.clamp(s.G_max - s.G_min, min=1e-6)
        h = torch.stack([
            (2.0 * (s.D - s.D_min) / dD - 1.0).reshape(()),
            (2.0 * (s.G - s.G_min) / dG - 1.0).reshape(()),
            (2.0 * (s.D_star - s.D_min) / dD - 1.0).reshape(()),
            ((s.D - s.G - s.G_res - self._q_sum(s)) / dD).reshape(()),
            (s.prev_residual / dD).reshape(()),
            (s.stall / dD).reshape(()),
            _norm(s.a_tilde, cfg.a_tilde_range).reshape(()),
            _norm(s.a, cfg.a_range).reshape(()),
            _norm(s.b, cfg.b_range).reshape(()),
        ])                                                          # (D_NODE,)

        d_h = cfg.hidden_dim
        k_nb = len(s.neighbors)
        mem = s.mem.reshape(1, d_h)

        def _channel_empty(phi_node, phi_gate, mem_new):
            gate = torch.sigmoid(phi_gate(mem_new)).reshape(())
            node_out = torch.tanh(phi_node(mem_new)) * gate
            return node_out[0] * (dD / 2.0), node_out[1] * (dG / 2.0), {}

        if k_nb == 0:
            agg = torch.zeros(1, d_h, dtype=h.dtype)
            mem_new = self.phi_upd(torch.cat([h, agg.squeeze(0)]).unsqueeze(0), mem).squeeze(0)
            fast = _channel_empty(self.phi_node, self.phi_gate, mem_new)
            if cfg.scale_slow == 0.0:
                slow = (torch.zeros((), dtype=h.dtype), torch.zeros((), dtype=h.dtype), {})
            else:
                slow = _channel_empty(self.phi_node_slow, self.phi_gate_slow, mem_new)
            return fast, slow, mem_new

        kappa_vec = torch.clamp(torch.stack([s.kappa[m] for m in s.neighbors]), min=1e-6)
        q_own = torch.stack([s.q_own[m] for m in s.neighbors]) / kappa_vec
        q_recv = torch.stack([s.q_recv[m] for m in s.neighbors]) / kappa_vec
        c_n = _norm(torch.stack([s.c[m] for m in s.neighbors]), cfg.c_range)
        e = torch.stack([q_own, q_recv, c_n], dim=1)                # (k, D_EDGE)

        msg = self.phi_msg(e)
        agg = msg.sum(dim=0) / float(k_nb) ** 0.5
        mem_new = self.phi_upd(torch.cat([h, agg]).unsqueeze(0), mem).squeeze(0)

        def _channel(phi_node, phi_edge, phi_gate):
            gate = torch.sigmoid(phi_gate(mem_new)).reshape(())
            node_out = torch.tanh(phi_node(mem_new)) * gate
            delta_D = node_out[0] * (dD / 2.0)
            delta_G = node_out[1] * (dG / 2.0)
            delta_q = {}
            for i, m in enumerate(s.neighbors):
                out = torch.tanh(phi_edge(torch.cat([mem_new, e[i]]))) * gate
                delta_q[m] = out.reshape(()) * kappa_vec[i]
            return delta_D, delta_G, delta_q

        fast = _channel(self.phi_node, self.phi_edge, self.phi_gate)
        # see forward_centralised's matching comment: scale_slow == 0 makes
        # amp_slow * slow's-outputs exactly zero at the caller regardless of
        # their value, so skip the slow channel's MLPs entirely in that case.
        if cfg.scale_slow == 0.0:
            zero = torch.zeros((), dtype=h.dtype)
            slow = (zero, zero, {m: zero for m in s.neighbors})
        else:
            slow = _channel(self.phi_node_slow, self.phi_edge_slow, self.phi_gate_slow)
        return fast, slow, mem_new

    @staticmethod
    def _q_sum(s: LocalNodeState) -> torch.Tensor:
        if not s.neighbors:
            return torch.zeros((), dtype=s.D.dtype)
        return torch.stack([s.q_own[m] for m in s.neighbors]).sum()


# ======================================================================
# Deployment: one LocalAgent per prosumer, all sharing the trained weights
# ======================================================================
class LocalAgent:
    def __init__(self, gnn: PerturbationGNN):
        self.gnn = gnn

    @torch.no_grad()
    def perturbation(self, state: LocalNodeState, k: int):
        amp_fast = self.gnn.cfg.scale * self.gnn.decay(k)
        amp_slow = self.gnn.cfg.scale_slow * self.gnn.decay_slow(k)
        (dD_f, dG_f, dq_f), (dD_s, dG_s, dq_s), mem_new = self.gnn.forward_local(state)
        dD = amp_fast * dD_f + amp_slow * dD_s
        dG = amp_fast * dG_f + amp_slow * dG_s
        dq = {m: amp_fast * dq_f[m] + amp_slow * dq_s[m] for m in dq_f}
        return dD, dG, dq, mem_new


def deploy(gnn: PerturbationGNN, net) -> None:
    """Install the trained model on every prosumer -- after this call each
    node runs lines 11-12 of Algorithm 1 on its own during `dr_step`."""
    gnn.eval()
    agent = LocalAgent(gnn)
    for p in net.prosumers:
        p.gnn_agent = agent


def undeploy(net) -> None:
    for p in net.prosumers:
        p.gnn_agent = None
