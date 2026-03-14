import copy
import torch
import numpy as np
from scipy.optimize import minimize
from torch_geometric.data import Data

from prosumer import Prosumer


class EnergyNetwork:
    """
    Peer-to-peer energy market as a Generalized Nash Equilibrium Problem.

    Coordinates a list of Prosumer objects. Holds the shared graph structure
    and provides global operations: flatten/unflatten, projection, pseudo-gradient,
    PyG data construction, and monitoring.

    All per-node logic (local cost, gradient, communication) lives in Prosumer.

    Attributes
    ----------
    prosumers   : list[Prosumer]   — one per node, indexed by node_id
    N           : int              — number of prosumers
    neighbors   : dict             — {n: list[int]}  graph adjacency
    edge_list   : list[(m,n)]      — ordered directed edges
    edge_index  : LongTensor (2,E) — PyG format
    inverse_edge: dict             — {(m,n): idx of (n,m) in edge_list}
    dim         : int              — total state dimension = 2N + |E|
    """

    def __init__(self, prosumers: list, neighbors: dict, device: torch.device):
        """
        Parameters
        ----------
        prosumers : list[Prosumer]  — already constructed, one per node
        neighbors : dict {n: list[int]}
        device    : torch.device
        """
        self.prosumers = prosumers
        self.neighbors = neighbors
        self.N         = len(prosumers)
        self.device    = device

        # Graph structure (built once)
        self.edge_list    = self._build_edge_list()
        self.edge_index   = self._build_edge_index()
        self.inverse_edge = self._build_inverse_edge()

        # edge_list index for each incoming edge of node n
        # incoming_idx[n] = list of edge_list indices i such that edge_list[i] = (m, n)
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}
        self.incoming_idx = {
            n: [idx_map[(m, n)] for m in self.neighbors[n] if (m, n) in idx_map]
            for n in range(self.N)
        }

        # Precompute edge-level parameter tensors (for projection and monitoring)
        self.c_edge = torch.tensor(
            [prosumers[n].c[m].item() for (m, n) in self.edge_list],
            dtype=torch.float32, device=device
        )
        self.k_edge = torch.tensor(
            [prosumers[n].k[m].item() for (m, n) in self.edge_list],
            dtype=torch.float32, device=device
        )

        self.dim = 2 * self.N + len(self.edge_list)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_edge_list(self):
        """
        Ordered directed edge list: (m, n) means q_{mn} is n's decision.
        Outer loop on n (sorted), inner loop on m (sorted).
        """
        edge_list = []
        for n in sorted(self.neighbors.keys()):
            for m in sorted(self.neighbors[n]):
                edge_list.append((m, n))
        return edge_list

    def _build_edge_index(self):
        src = [m for (m, n) in self.edge_list]
        dst = [n for (m, n) in self.edge_list]
        return torch.tensor([src, dst], dtype=torch.long, device=self.device)

    def _build_inverse_edge(self):
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}
        return {(m, n): idx_map[(n, m)] for (m, n) in self.edge_list}

    # ------------------------------------------------------------------
    # Flatten / unflatten  (global state vector x = [D, G, Q])
    # ------------------------------------------------------------------

    def flatten(self) -> torch.Tensor:
        """
        Returns x = [D_0,...,D_{N-1}, G_0,...,G_{N-1}, Q_0,...,Q_{|E|-1}]
        shape (dim,).

        Q is ordered as edge_list: Q[i] = q_{mn} for edge_list[i] = (m, n),
        which is prosumer n's trade decision w.r.t. neighbor m.
        """
        D = torch.stack([p.D for p in self.prosumers])
        G = torch.stack([p.G for p in self.prosumers])
        Q = torch.stack([self.prosumers[n].q[m] for (m, n) in self.edge_list])
        return torch.cat([D, G, Q])

    def unflatten(self, x: torch.Tensor):
        """
        Writes D, G, q back to each Prosumer from global vector x of shape (dim,).
        """
        for n in range(self.N):
            self.prosumers[n].D = x[n]
            self.prosumers[n].G = x[self.N + n]

        for i, (m, n) in enumerate(self.edge_list):
            self.prosumers[n].q[m] = x[2 * self.N + i]

    # ------------------------------------------------------------------
    # Communication round
    # ------------------------------------------------------------------

    def communicate(self):
        """
        Each prosumer sends its q decisions to all neighbors.
        After this call, prosumer[n].q_received[m] = prosumer[m].q[n]
        for all (m, n) edges.

        This is the distributed communication step that precedes each
        FB iteration.
        """
        for (m, n) in self.edge_list:
            # prosumer m sends q[n] to prosumer n's receive buffer
            self.prosumers[m].send_q_to(self.prosumers[n])

    # ------------------------------------------------------------------
    # Forward step  (y = x - gamma * F(x))
    # ------------------------------------------------------------------

    def forward_step(self, gamma: float) -> "EnergyNetwork":
        """
        Applies one gradient step on every prosumer without modifying self.
        Returns a new EnergyNetwork with updated state.
        """
        new_net = self.clone()
        for n in range(self.N):
            new_net.prosumers[n] = self.prosumers[n].forward_step(gamma)
        return new_net

    # ------------------------------------------------------------------
    # Projection onto the feasible set
    # ------------------------------------------------------------------

    def project_exact(self):
        """
        Exact projection via scipy SLSQP (not differentiable).
        Used during inference. Returns (x_proj, success) without modifying self.
        """
        x0     = self.flatten()
        x_orig = x0.detach().clone()
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}

        def objective(x):
            x_t = torch.tensor(x, dtype=torch.float32)
            return (0.5 * torch.sum((x_t - x_orig) ** 2)).item()

        def grad_objective(x):
            x_t = torch.tensor(x, dtype=torch.float32)
            return (x_t - x_orig).numpy().astype(float)

        # Balance equality constraints: D[n] = G[n] + G_res[n] + sum_m q_{mn}
        eq_constraints = []
        for n in range(self.N):
            q_idxs = [2 * self.N + idx_map[(m, n)] for m in self.neighbors[n]]
            def make_eq(n_, q_idxs_):
                def eq_fn(x):
                    return (
                        float(x[n_])
                        - float(x[self.N + n_])
                        - float(self.prosumers[n_].G_res.item())
                        - sum(float(x[i]) for i in q_idxs_)
                    )
                return eq_fn
            eq_constraints.append({"type": "eq", "fun": make_eq(n, q_idxs)})

        # Reciprocity inequality constraints: q_{mn} + q_{nm} <= 0
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
            [(float(p.D_min.item()), float(p.D_max.item())) for p in self.prosumers]
            + [(float(p.G_min.item()), float(p.G_max.item())) for p in self.prosumers]
            + [(float("-inf"), float(self.k_edge[i].item())) for i in range(len(self.edge_list))]
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
            print(f"Warning: Exact projection failed — {result.message}")

        x_proj = torch.tensor(result.x, dtype=torch.float32, device=self.device)
        return x_proj, result.success

    def project_approx(self, n_iter: int = 30, lr: float = 0.1,
                       create_graph: bool = False):
        """
        Differentiable approximate projection. Updates prosumers in-place.
        Used during training (create_graph=True enables backprop).

        Clamps D and G to their box constraints, then iteratively corrects
        Q to satisfy the balance equality.
        """
        # Clamp D and G locally for each prosumer
        for p in self.prosumers:
            p.D = torch.clamp(p.D, p.D_min, p.D_max)
            p.G = torch.clamp(p.G, p.G_min, p.G_max)

        # Collect Q as a single tensor for differentiable correction
        Q = torch.stack([self.prosumers[n].q[m] for (m, n) in self.edge_list])
        Q = Q.detach().clone()
        Q.requires_grad_(True)

        D = torch.stack([p.D.detach() for p in self.prosumers])
        G = torch.stack([p.G.detach() for p in self.prosumers])
        G_res = torch.stack([p.G_res for p in self.prosumers])

        for _ in range(n_iter):
            eq_loss = torch.tensor(0.0, device=self.device)
            for n in range(self.N):
                if self.incoming_idx[n]:
                    q_sum = Q[self.incoming_idx[n]].sum()
                else:
                    q_sum = torch.tensor(0.0, device=self.device)
                eq_loss = eq_loss + (D[n] - G[n] - G_res[n] - q_sum) ** 2

            if create_graph:
                grads = torch.autograd.grad(eq_loss, Q)[0]
                Q = Q - lr * grads
                Q = torch.clamp(Q, -self.k_edge, self.k_edge)

        # Write corrected Q back into each prosumer
        for i, (m, n) in enumerate(self.edge_list):
            self.prosumers[n].q[m] = Q[i]

    # ------------------------------------------------------------------
    # Global PyG Data (for centralised training)
    # ------------------------------------------------------------------

    def get_pyg_data(self) -> Data:
        """
        Build a global PyG Data object with the full network state.
        Used during centralised training where all node/edge data is available.

        Node features x : (N, 2)   — x[n] = [D_n, G_n]
        Edge features   : (|E|, 1) — for edge (m->n): q_{nm} (m's decision sent to n)
                                      = prosumers[m].q[n]
        """
        x_node = torch.stack(
            [torch.stack([p.D, p.G]) for p in self.prosumers]
        )  # (N, 2)

        # For edge (m->n): feature is q_{nm} = prosumer m's trade w.r.t. n
        edge_attr = torch.stack(
            [self.prosumers[m].q[n].reshape(1) for (m, n) in self.edge_list]
        )  # (|E|, 1)

        return Data(
            x=x_node,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
        )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def cost(self) -> torch.Tensor:
        """Global cost = sum of all local costs J_n."""
        return sum(p.local_cost() for p in self.prosumers)

    def net_imports(self) -> torch.Tensor:
        """Q_n = sum_{m in Omega_n} q_{mn} for each node. Returns (N,)."""
        return torch.stack([p.net_import() for p in self.prosumers])

    def balance_violation(self) -> torch.Tensor:
        """D[n] - G[n] - G_res[n] - net_import[n]. Should be 0 at feasibility."""
        return torch.stack([p.balance_violation() for p in self.prosumers])

    def constraint_violation(self, tol: float = 1e-4) -> bool:
        """
        Returns True if any constraint is violated beyond tol.

        Checks: D/G bounds, Q capacity, balance equality, reciprocity.
        """
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}
        Q = self.flatten()[2 * self.N:]

        for p in self.prosumers:
            if p.D < p.D_min - tol or p.D > p.D_max + tol:
                return True
            if p.G < p.G_min - tol or p.G > p.G_max + tol:
                return True

        if (Q > self.k_edge + tol).any():
            return True

        if (self.balance_violation().abs() > tol).any():
            return True

        visited = set()
        for (m, n) in self.edge_list:
            if (m, n) in visited or (n, m) not in idx_map:
                continue
            visited.add((m, n))
            visited.add((n, m))
            q_mn = float(self.prosumers[n].q[m])
            q_nm = float(self.prosumers[m].q[n])
            if q_mn + q_nm > tol:
                return True

        return False

    def _init_state(self):
        """Re-randomize state of all prosumers."""
        for p in self.prosumers:
            p._init_state()

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self) -> "EnergyNetwork":
        """
        Returns a new EnergyNetwork with independently cloned prosumers.
        Graph structure and edge tensors are shared (read-only).
        """
        new_net           = copy.copy(self)
        new_net.prosumers = [p.clone() for p in self.prosumers]
        return new_net

    def __repr__(self):
        return (
            f"EnergyNetwork(N={self.N}, "
            f"|E|={len(self.edge_list)}, "
            f"dim={self.dim})"
        )