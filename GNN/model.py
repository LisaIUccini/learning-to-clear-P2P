import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Small MLP helper
# ------------------------------------------------------------------

class MLP(nn.Module):
    """
    Multi-layer perceptron with ReLU activations.
    Architecture: Linear -> ReLU -> ... -> Linear (no activation on output).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# GNN Message Passing layer (manual, no PyG MessagePassing base)
# ------------------------------------------------------------------

class GNNLayer(nn.Module):
    """
    One message-passing layer implementing:

        msg_{m->n}  = phi_msg(edge_attr_{m->n})           shape (d_h,)
        h_agg_n     = sum_{m in Omega_n} msg_{m->n}       shape (d_h,)
        h_new_n     = phi_upd([x_n, h_agg_n])             shape (d_h,)

    where:
        edge_attr_{m->n} = q_{nm}  (trade of m communicated to n)
        x_n              = [D_n, G_n]

    Implemented manually with scatter_add — no dependency on PyG
    MessagePassing, which has shape issues with local subgraphs.
    Weight sharing: phi_msg and phi_upd are shared across all edges/nodes.
    """

    def __init__(self, node_feat_size: int, edge_feat_size: int, d_h: int):
        super().__init__()

        self.d_h = d_h

        # phi_msg: edge feature -> message vector
        self.phi_msg = MLP(
            input_size=edge_feat_size,
            hidden_size=d_h,
            output_size=d_h,
        )

        # phi_upd: [node features, aggregated message] -> new hidden state
        self.phi_upd = MLP(
            input_size=node_feat_size + d_h,
            hidden_size=d_h,
            output_size=d_h,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : node features (num_nodes, node_feat_size)
        edge_index : (2, num_edges)  — edge_index[0]=src, edge_index[1]=dst
        edge_attr  : edge features (num_edges, edge_feat_size)

        Returns
        -------
        h_new : (num_nodes, d_h)
        """
        num_nodes = x.size(0)

        # Step 1 — Message: apply phi_msg to each edge's feature
        # edge_attr may be (num_edges, 1) — squeeze to (num_edges,) then
        # unsqueeze back if needed, or just ensure 2D input to phi_msg.
        # edge_attr shape must be (num_edges, edge_feat_size) — no extra dims.
        if edge_attr.dim() == 3:
            edge_attr = edge_attr.squeeze(1)  # (num_edges, 1, feat) -> (num_edges, feat)
        # msg shape: (num_edges, d_h)
        msg = self.phi_msg(edge_attr)

        # Step 2 — Aggregation: sum / sqrt(degree) at each destination node
        # sqrt(degree) normalization balances between sum (too large for
        # high-degree nodes) and mean (too small, loses signal).
        # With N=3 (degree~2): sum/sqrt(2) ≈ sum*0.71 — almost unchanged.
        # With N=10 (degree~9): sum/sqrt(9) = sum/3 — moderate reduction.
        dst = edge_index[1]  # (num_edges,)

        # Compute sum of messages
        h_agg = torch.zeros(num_nodes, self.d_h, device=x.device)
        h_agg.scatter_add_(
            0,
            dst.unsqueeze(1).expand_as(msg),  # (num_edges, d_h)
            msg
        )

        # Count degree per node
        degree = torch.zeros(num_nodes, 1, device=x.device)
        degree.scatter_add_(
            0,
            dst.unsqueeze(1),
            torch.ones(dst.size(0), 1, device=x.device)
        )
        degree = degree.clamp(min=1)

        # Normalize by sqrt(degree)
        h_agg = h_agg / torch.sqrt(degree)

        # Step 3 — Update: apply phi_upd to [node_features, aggregated_messages]
        # h_new shape: (num_nodes, d_h)
        h_new = self.phi_upd(torch.cat([x, h_agg], dim=-1))

        return h_new


# ------------------------------------------------------------------
# Full GNN perturbation model
# ------------------------------------------------------------------

class GNNUpdate(nn.Module):
    """
    GNN-based learned perturbation for the Forward-Backward algorithm.

    At each FB iteration t, given a PyG Data object (global or local subgraph),
    produces summable perturbations:

        v_nodes[n] = m_t * tanh(phi_node(h_new[n]))             shape (N, 2)
        v_edges[i] = m_t * tanh(phi_edge(h_new[n], edge_attr[i]))  shape (|E|, 1)
                     for edge i = (m->n), owned by node n

    where m_t = phi_m * rho^t ensures summability (v in ell_1).

    Distributed execution: each node n independently queries the GNN
    with its local subgraph (ego node + neighbors), using only:
        - its own state (D_n, G_n)
        - messages q_{nm} from neighbors (communicated by them)
    No global information required.

    Weight sharing: phi_msg, phi_upd, phi_node, phi_edge have fixed
    input/output sizes independent of N and graph topology.
    The same weights work for any graph size without retraining.

    Architecture:
        GNNLayer  : phi_msg (edge_feat -> d_h) + phi_upd ([node_feat, d_h] -> d_h)
        phi_node  : MLP (d_h -> 2)
        phi_edge  : MLP (d_h + edge_feat -> 3)
    """

    def __init__(
        self,
        d_h: int = 32,
        rho: float = 0.9,
        node_feat_size: int = 2,
        edge_feat_size: int = 1,
    ):
        super().__init__()

        self.rho = rho

        # Message passing layer
        self.gnn_layer = GNNLayer(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            d_h=d_h,
        )

        # Node-level output: hidden state -> (v_D, v_G)
        self.phi_node = MLP(
            input_size=d_h,
            hidden_size=d_h,
            output_size=2,
        )

        # Edge-level output: (h_new[n], edge_attr) -> (v_q, v_z, v_u)
        # Only uses h_new[n] — the node that owns q_{mn} — which already
        # aggregates all neighbor messages. Fully local, no need for h_new[m].
        self.phi_edge = MLP(
            input_size=d_h + edge_feat_size,
            hidden_size=d_h,
            output_size=3,
        )

        # Learnable amplitude, kept positive via softplus
        self.phi_m_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, data, t: int):
        """
        Compute perturbations for iteration t.

        Works on both the global graph (all N nodes) and local subgraphs
        (ego node + neighbors) — same weights, different graph sizes.

        Parameters
        ----------
        data : torch_geometric.data.Data
            FB:
                .x         : (num_nodes, 2)  — [D, G]
                .edge_attr : (num_edges, 1)  — [q]
            ADMM:
                .x         : (num_nodes, 3)  — [D, G, u_avg]
                .edge_attr : (num_edges, 2)  — [q, z]
            In both cases z and u are used as context for the GNN
            but are NOT perturbed — only D, G, q are safe to perturb.
        t    : current iteration index (int >= 0)

        Returns
        -------
        v_nodes : (num_nodes, 2)  — perturbations for [D, G]
        v_edges : (num_edges, 3)  — perturbations for [q, z, u]
        """
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr

        h_new = self.gnn_layer(x, edge_index, edge_attr)  # (num_nodes, d_h)

        delta_nodes = self.phi_node(h_new)  # (num_nodes, 2)

        dst = edge_index[1]
        edge_input  = torch.cat([h_new[dst], edge_attr], dim=-1)
        delta_edges = self.phi_edge(edge_input)  # (num_edges, 1)

        phi_m = F.softplus(self.phi_m_raw)
        m_t   = phi_m * (self.rho ** t)

        v_nodes = m_t * torch.tanh(delta_nodes)  # (num_nodes, 2)
        v_edges = m_t * torch.tanh(delta_edges)  # (num_edges, 3)

        return v_nodes, v_edges