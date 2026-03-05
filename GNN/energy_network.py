import copy
import torch
import numpy as np
from scipy.optimize import minimize
from torch_geometric.data import Data

def compute_gne(
    net: EnergyNetwork,
    gamma: float,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the Generalized Nash Equilibrium (GNE) of a given EnergyNetwork
    instance by running the pure Forward-Backward algorithm to convergence.

    Uses the exact (non-differentiable) projection — only called offline
    during dataset generation, never during training.

    Parameters
    ----------
    net      : EnergyNetwork instance with a fixed initial state
    gamma    : FB step size
    max_iter : maximum number of FB iterations
    tol      : convergence tolerance on ||x_{t+1} - x_t||^2

    Returns
    -------
    x_star : torch.Tensor (dim,)  —  flattened GNE state vector
    """
    n = net.clone()

    for it in range(max_iter):
        x_old = n.flatten().detach().clone()

        # Forward step
        n_new = n.forward_step(gamma)

        # Backward step (exact projection)
        n_new.project_exact()

        # Check convergence
        x_new = n_new.flatten()
        residual = torch.sum((x_new - x_old) ** 2).item()

        n = n_new

        if residual < tol:
            print(f"    GNE converged at iteration {it+1}, residual={residual:.2e}")
            break
    else:
        print(f"    GNE did not converge after {max_iter} iterations, "
              f"final residual={residual:.2e}")

    return n


class EnergyNetwork:
    """
    Peer-to-peer energy market modeled as a Generalized Nash Equilibrium Problem.

    All state and parameters are represented as tensors — no per-agent objects.

    State variables (decision variables):
        D : torch.Tensor (N,)   — energy demand of each prosumer
        G : torch.Tensor (N,)   — controllable flexibility of each prosumer
        Q : torch.Tensor (|E|,) — bilateral trades, Q[i] = q_{mn} for edge i=(m,n)
                                   q_{mn} is the decision of node n:
                                   how much n buys (>0) or sells (<0) to m

    Fixed parameters (node-level):
        D_min, D_max : (N,)  demand bounds
        G_min, G_max : (N,)  flexibility bounds
        G_res        : (N,)  renewable generation (given, not optimized)
        a_tilde      : (N,)  usage benefit coefficient
        a, b         : (N,)  flexibility cost coefficients
        D_star       : (N,)  target demand

    Fixed parameters (edge-level, ordered as edge_list):
        c_edge : (|E|,)  c_edge[i] = c_{nm} for edge i=(m,n)
                         cost that node n assigns to trading with m
        k_edge : (|E|,)  k_edge[i] = k_{mn} capacity on link (m,n)

    Graph structure:
        edge_list    : list of tuples (m, n), fixed ordering
        edge_index   : torch.LongTensor (2, |E|)
        inverse_edge : dict {(m,n): idx_of_(n,m)} for fast reverse-edge lookup
    """

    def __init__(
        self,
        N: int,
        neighbors: dict,
        D_min: torch.Tensor,
        D_max: torch.Tensor,
        G_min: torch.Tensor,
        G_max: torch.Tensor,
        G_res: torch.Tensor,
        a_tilde: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        D_star: torch.Tensor,
        c: dict,
        k: dict,
        device: torch.device,
    ):
        self.N         = N
        self.neighbors = neighbors
        self.device    = device

        # Node-level fixed parameters
        self.D_min   = D_min.to(device)
        self.D_max   = D_max.to(device)
        self.G_min   = G_min.to(device)
        self.G_max   = G_max.to(device)
        self.G_res   = G_res.to(device)
        self.a_tilde = a_tilde.to(device)
        self.a       = a.to(device)
        self.b       = b.to(device)
        self.D_star  = D_star.to(device)

        # Graph structure (built once, never changes)
        self.edge_list    = self._build_edge_list()
        self.edge_index   = self._build_edge_index()
        self.inverse_edge = self._build_inverse_edge()

        # Edge-level fixed parameters (ordered as edge_list)
        self.c_edge = torch.tensor(
            [c[(m, n)].item() for (m, n) in self.edge_list],
            dtype=torch.float32, device=device
        )
        self.k_edge = torch.tensor(
            [k[(m, n)].item() for (m, n) in self.edge_list],
            dtype=torch.float32, device=device
        )

        # Precompute incoming edge indices for each node n:
        # incoming_idx[n] = [i : edge_list[i] = (m, n)]
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}
        self.incoming_idx = {
            n: [idx_map[(m, n)] for m in self.neighbors[n] if (m, n) in idx_map]
            for n in range(self.N)
        }

        self.dim = 2 * N + len(self.edge_list)
        self.D, self.G, self.Q = self._init_state()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_edge_list(self):
        """
        Ordered list of directed edges (m, n) where m in neighbors[n].
        Ordering: outer loop on n (sorted), inner loop on m (sorted).
        Edge (m, n) carries q_{mn} = decision of node n.
        """
        edge_list = []
        for n in sorted(self.neighbors.keys()):
            for m in sorted(self.neighbors[n]):
                edge_list.append((m, n))
        return edge_list

    def _build_edge_index(self):
        """
        PyG edge_index tensor of shape (2, |E|).
        edge_index[0, i] = source m, edge_index[1, i] = destination n.
        """
        src = [m for (m, n) in self.edge_list]
        dst = [n for (m, n) in self.edge_list]
        return torch.tensor([src, dst], dtype=torch.long, device=self.device)

    def _build_inverse_edge(self):
        """
        inverse_edge[(m, n)] = index of (n, m) in edge_list.
        Used in get_pyg_data() to fetch q_{nm} as the feature of edge (m->n).
        """
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}
        return {(m, n): idx_map[(n, m)] for (m, n) in self.edge_list}

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def _init_state(self):
        D = self.D_min + (self.D_max - self.D_min) * torch.rand(self.N, device=self.device)
        G = self.G_min + (self.G_max - self.G_min) * torch.rand(self.N, device=self.device)
        Q = torch.rand(len(self.edge_list), device=self.device) * self.k_edge
        return D, G, Q

    # ------------------------------------------------------------------
    # Flatten / unflatten
    # ------------------------------------------------------------------

    def flatten(self):
        """Returns x = [D, G, Q] of shape (dim,)."""
        return torch.cat([self.D, self.G, self.Q])

    def unflatten(self, x: torch.Tensor):
        """Sets D, G, Q from x of shape (dim,)."""
        self.D = x[:self.N]
        self.G = x[self.N: 2 * self.N]
        self.Q = x[2 * self.N:]

    # ------------------------------------------------------------------
    # Global pseudo-gradient F(x)
    # ------------------------------------------------------------------

    def F(self):
        """
        grad_D[n] = 2 * a_tilde[n] * (D[n] - D_star[n])
        grad_G[n] = a[n] * G[n] + b[n]
        grad_Q[i] = c_edge[i]   (constant w.r.t. Q)

        Returns (N,), (N,), (|E|,)
        """
        grad_D = 2.0 * self.a_tilde * (self.D - self.D_star)
        grad_G = self.a * self.G + self.b
        grad_Q = self.c_edge
        return grad_D, grad_G, grad_Q

    # ------------------------------------------------------------------
    # Forward step
    # ------------------------------------------------------------------

    def forward_step(self, gamma: float):
        """
        x_new = x - gamma * F(x).
        Does not modify self. Returns a new EnergyNetwork.
        """
        grad_D, grad_G, grad_Q = self.F()
        new_net   = self.clone()
        new_net.D = self.D - gamma * grad_D
        new_net.G = self.G - gamma * grad_G
        new_net.Q = self.Q - gamma * grad_Q
        return new_net

    # ------------------------------------------------------------------
    # Projection onto the feasible set
    # ------------------------------------------------------------------

    def project_exact(self):
        """
        Exact projection via scipy SLSQP. Not differentiable.
        Used during inference. Updates state in place.
        """
        x0      = self.flatten()
        x_orig  = x0.detach().clone()
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}

        def objective(x):
            x_t = torch.tensor(x, dtype=torch.float32)
            return (0.5 * torch.sum((x_t - x_orig) ** 2)).item()

        def grad_objective(x):
            x_t = torch.tensor(x, dtype=torch.float32)
            return (x_t - x_orig).numpy().astype(float)

        eq_constraints = []
        for n in range(self.N):
            q_idxs = [2 * self.N + idx_map[(m, n)] for m in self.neighbors[n]]
            def make_eq(n_, q_idxs_):
                def eq_fn(x):
                    return (
                        float(x[n_])
                        - float(x[self.N + n_])
                        - float(self.G_res[n_])
                        - sum(float(x[i]) for i in q_idxs_)
                    )
                return eq_fn
            eq_constraints.append({"type": "eq", "fun": make_eq(n, q_idxs)})

        ineq_constraints = []
        visited = set()
        for (m, n) in self.edge_list:
            if (m, n) in visited:
                continue
            visited.add((m, n))
            visited.add((n, m))
            i_mn = 2 * self.N + idx_map[(m, n)]
            i_nm = 2 * self.N + idx_map[(n, m)]
            def make_ineq(a_, b_):
                def ineq_fn(x):
                    return -(float(x[a_]) + float(x[b_]))
                return ineq_fn
            ineq_constraints.append({"type": "ineq", "fun": make_ineq(i_mn, i_nm)})

        bounds = (
            [(float(self.D_min[n]), float(self.D_max[n])) for n in range(self.N)]
            + [(float(self.G_min[n]), float(self.G_max[n])) for n in range(self.N)]
            + [(float("-inf"), float(self.k_edge[i])) for i in range(len(self.edge_list))]
        )

        result = minimize(
            objective,
            x0.detach().cpu().numpy(),
            jac=grad_objective,
            method="SLSQP",
            bounds=bounds,
            constraints=eq_constraints + ineq_constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )

        if not result.success:
            raise ValueError(f"Exact projection failed: {result.message}")

        x_proj = torch.tensor(result.x, dtype=torch.float32, device=self.device)
        self.unflatten(x_proj)

    def project_approx(self, n_iter: int = 30, lr: float = 0.1):
        """
        Approximate differentiable projection. Used during training.
        Updates state in place.
        """
        self.D = torch.clamp(self.D, self.D_min, self.D_max)
        self.G = torch.clamp(self.G, self.G_min, self.G_max)

        Q = self.Q.detach().clone()
        Q.requires_grad_(True)

        for _ in range(n_iter):
            eq_loss = torch.tensor(0.0, device=self.device)
            for n in range(self.N):
                if self.incoming_idx[n]:
                    q_sum = Q[self.incoming_idx[n]].sum()
                else:
                    q_sum = torch.tensor(0.0, device=self.device)
                eq_loss = eq_loss + (
                    self.D[n] - self.G[n] - self.G_res[n] - q_sum
                ) ** 2

            grads = torch.autograd.grad(eq_loss, Q, create_graph=True)[0]
            Q = Q - lr * grads
            Q = torch.clamp(Q, -self.k_edge, self.k_edge)

        self.Q = Q

    # ------------------------------------------------------------------
    # PyG Data object for the GNN
    # ------------------------------------------------------------------

    def get_pyg_data(self):
        """
        Build a PyG Data object with the current state.

        Node features x : (N, 2)  —  x[n] = [D_n, G_n]

        Edge features edge_attr : (|E|, 1)
            For edge i = (m->n): feature is Q[inverse_edge[(m,n)]] = q_{nm}
            i.e. the trade decision of node m w.r.t. n, communicated by m to n.
        """
        x_node = torch.stack([self.D, self.G], dim=1)  # (N, 2)

        inv_idx = torch.tensor(
            [self.inverse_edge[(m, n)] for (m, n) in self.edge_list],
            dtype=torch.long, device=self.device
        )
        edge_attr = self.Q[inv_idx].unsqueeze(1)  # (|E|, 1)

        return Data(
            x=x_node,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
        )

    # ------------------------------------------------------------------
    # Apply GNN perturbation
    # ------------------------------------------------------------------

    def apply_perturbation(self, v_nodes: torch.Tensor, v_edges: torch.Tensor):
        """
        Add GNN perturbations to the current state.

        v_nodes : (N, 2)   — v_nodes[n] = [v_D_n, v_G_n]
        v_edges : (|E|, 1) — v_edges[i] = perturbation for Q[i] = q_{mn}
                             perturbation decided by node n for trade with m
        """
        self.D = self.D + v_nodes[:, 0]
        self.G = self.G + v_nodes[:, 1]
        self.Q = self.Q + v_edges.squeeze(1)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def cost(self):
        """Global cost = sum of all local costs J_n(x_n)."""
        node_cost = (
            self.a_tilde * (self.D - self.D_star) ** 2
            + 0.5 * self.a * self.G ** 2
            + self.b * self.G
        ).sum()
        edge_cost = (self.c_edge * self.Q).sum()
        return node_cost + edge_cost

    def net_imports(self):
        """Q_n = sum_{m in Omega_n} q_{mn} for each node. Returns (N,)."""
        net_Q = torch.zeros(self.N, device=self.device)
        for n in range(self.N):
            if self.incoming_idx[n]:
                idxs = torch.tensor(self.incoming_idx[n], device=self.device)
                net_Q[n] = self.Q[idxs].sum()
        return net_Q

    def balance_violation(self):
        """D[n] - G[n] - G_res[n] - net_imports[n]. Should be 0 at feasibility."""
        return self.D - self.G - self.G_res - self.net_imports()

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self):
        """
        Copy with independent state (D, G, Q).
        Fixed parameters and graph structure are shared (read-only).
        """
        new_net   = copy.copy(self)
        new_net.D = self.D.clone()
        new_net.G = self.G.clone()
        new_net.Q = self.Q.clone()
        return new_net

    def __repr__(self):
        return (
            f"EnergyNetwork(N={self.N}, "
            f"|E|={len(self.edge_list)}, "
            f"dim={self.dim})"
        )