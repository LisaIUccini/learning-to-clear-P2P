import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ------------------------------------------------------------------
# Small MLP helper
# ------------------------------------------------------------------

class MLP(nn.Module):
    """
    Simple multi-layer perceptron with ReLU activations.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear
    The last layer has no activation (raw output, scaled by caller).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, n_layers: int = 2):
        """
        Parameters
        ----------
        input_size  : dimension of input vector
        hidden_size : dimension of hidden layers
        output_size : dimension of output vector
        n_layers    : number of hidden layers (default 2)
        """
        super().__init__()

        layers = []
        in_dim = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))  # no activation on output

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# GNN Message Passing layer
# ------------------------------------------------------------------

class GNNLayer(MessagePassing):
    """
    One message-passing layer implementing:

        msg_{m->n}  = phi_msg(edge_attr_{m->n})           shape (d_h,)
        h_agg_n     = sum_{m in Omega_n} msg_{m->n}       shape (d_h,)
        h_new_n     = phi_upd([x_n, h_agg_n])             shape (d_h,)

    where:
        edge_attr_{m->n} = q_{nm}  (trade decision of m, communicated to n)
        x_n              = [D_n, G_n]

    Parameters are shared across all nodes and all edges (weight sharing).
    Aggregation: sum (permutation invariant).
    """

    def __init__(self, node_feat_size: int, edge_feat_size: int, d_h: int):
        """
        Parameters
        ----------
        node_feat_size : dimension of node input features (2: D and G)
        edge_feat_size : dimension of edge input features (1: q_{nm})
        d_h            : hidden dimension
        """
        super().__init__(aggr="add")  # sum aggregation

        # phi_msg: edge features -> message vector
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
        x          : node features (N, node_feat_size)
        edge_index : (2, |E|)
        edge_attr  : edge features (|E|, edge_feat_size)

        Returns
        -------
        h_new : updated node hidden states (N, d_h)
        """
        # propagate calls message() for each edge, aggregates by sum,
        # then calls update() for each node
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute message for each edge.
        Called by propagate() for every edge (m->n).

        Parameters
        ----------
        edge_attr : (|E|, edge_feat_size)  — features of each edge

        Returns
        -------
        msg : (|E|, d_h)
        """
        return self.phi_msg(edge_attr)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Update each node's hidden state after aggregation.
        Called by propagate() for every node n.

        Parameters
        ----------
        aggr_out : (N, d_h)  — aggregated messages sum_{m in Omega_n} msg_{m->n}
        x        : (N, node_feat_size)  — original node features [D_n, G_n]

        Returns
        -------
        h_new : (N, d_h)
        """
        return self.phi_upd(torch.cat([x, aggr_out], dim=-1))


# ------------------------------------------------------------------
# Full GNN perturbation model
# ------------------------------------------------------------------

class GNNUpdate(nn.Module):
    """
    GNN-based learned perturbation for the Forward-Backward algorithm.

    At each FB iteration t, given the current network state as a PyG Data object,
    produces summable perturbations:

        v_nodes[n]   = m_t * tanh(phi_node(h_new[n]))       shape (N, 2)
        v_edges[i]   = m_t * tanh(phi_edge(h_new[m], h_new[n], edge_attr[i]))
                                                             shape (|E|, 1)

    where m_t = phi_m * rho^t ensures summability (v in ell_1).

    By Proposition 3 of Martin & Belgioioso (2026), injecting a summable
    perturbation into a KM iteration preserves convergence guarantees.

    Architecture (weight sharing across all nodes and edges):
        GNNLayer  : phi_msg (edge -> d_h) + phi_upd ([node, agg] -> d_h)
        phi_node  : MLP (d_h -> 2)        node-level output
        phi_edge  : MLP (2*d_h + 1 -> 1)  edge-level output
    """

    def __init__(
        self,
        d_h: int = 32,
        rho: float = 0.9,
        node_feat_size: int = 2,
        edge_feat_size: int = 1,
    ):
        """
        Parameters
        ----------
        d_h            : hidden dimension (default 32)
        rho            : exponential decay factor, in (0,1) (default 0.9)
        node_feat_size : node input features — 2 (D and G)
        edge_feat_size : edge input features — 1 (q_{nm})
        """
        super().__init__()

        self.rho = rho

        # Message passing layer
        self.gnn_layer = GNNLayer(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            d_h=d_h,
        )

        # Node-level output head: hidden state -> (v_D, v_G)
        self.phi_node = MLP(
            input_size=d_h,
            hidden_size=d_h,
            output_size=2,
        )

        # Edge-level output head: (h_src, h_dst, edge_attr) -> v_q
        # input: d_h (source hidden) + d_h (dest hidden) + edge_feat_size
        self.phi_edge = MLP(
            input_size=2 * d_h + edge_feat_size,
            hidden_size=d_h,
            output_size=1,
        )

        # Learnable scalar amplitude for the perturbation (kept positive via softplus)
        self.phi_m_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, data, t: int):
        """
        Compute perturbations for FB iteration t.

        Parameters
        ----------
        data : torch_geometric.data.Data
            .x          : (N, 2)    node features [D_n, G_n]
            .edge_index : (2, |E|)
            .edge_attr  : (|E|, 1)  edge features [q_{nm}]
        t    : current FB iteration index (int >= 0)

        Returns
        -------
        v_nodes : torch.Tensor (N, 2)   perturbations for D and G
        v_edges : torch.Tensor (|E|, 1) perturbations for Q
        """
        x          = data.x           # (N, 2)
        edge_index = data.edge_index   # (2, |E|)
        edge_attr  = data.edge_attr    # (|E|, 1)

        # --- Message passing: compute updated node hidden states ---
        # h_new[n] aggregates information from neighbors via q_{nm}
        h_new = self.gnn_layer(x, edge_index, edge_attr)  # (N, d_h)

        # --- Node-level output ---
        # delta_nodes[n] = direction of perturbation for (D_n, G_n)
        delta_nodes = self.phi_node(h_new)  # (N, 2)

        # --- Edge-level output ---
        # For each edge i = (m, n): use hidden states of both endpoints
        # and the edge feature q_{nm}
        src = edge_index[0]  # (|E|,) source indices m
        dst = edge_index[1]  # (|E|,) destination indices n

        edge_input = torch.cat(
            [h_new[src], h_new[dst], edge_attr], dim=-1
        )  # (|E|, 2*d_h + edge_feat_size)

        delta_edges = self.phi_edge(edge_input)  # (|E|, 1)

        # --- Summable scaling: m_t = phi_m * rho^t ---
        # phi_m is kept positive via softplus
        phi_m = torch.nn.functional.softplus(self.phi_m_raw)
        m_t   = phi_m * (self.rho ** t)

        # --- Final perturbation: m_t * tanh(delta) ---
        # tanh bounds the direction, m_t ensures summability
        v_nodes = m_t * torch.tanh(delta_nodes)  # (N, 2)
        v_edges = m_t * torch.tanh(delta_edges)  # (|E|, 1)

        return v_nodes, v_edges