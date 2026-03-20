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

        # Single idx_map built once and reused everywhere
        self._idx_map = {edge: i for i, edge in enumerate(self.edge_list)}
        self.inverse_edge = {(m, n): self._idx_map[(n, m)] for (m, n) in self.edge_list}

        # edge_list index for each incoming edge of node n
        # incoming_idx[n] = list of edge_list indices i such that edge_list[i] = (m, n)
        self.incoming_idx = {
            n: [self._idx_map[(m, n)] for m in self.neighbors[n] if (m, n) in self._idx_map]
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

        # Pre-cached bounds for project_exact (avoid repeated .item() calls)
        self._bounds_D = [(float(p.D_min.item()), float(p.D_max.item())) for p in self.prosumers]
        self._bounds_G = [(float(p.G_min.item()), float(p.G_max.item())) for p in self.prosumers]
        self._bounds_Q = [(float("-inf"), float(self.k_edge[i].item())) for i in range(len(self.edge_list))]
        self._bounds = self._bounds_D + self._bounds_G + self._bounds_Q

        # Pre-cached G_res float values for project_exact
        self._G_res_vals = [float(p.G_res.item()) for p in self.prosumers]

        # Pre-cached G_res tensor for project_approx
        self._G_res_tensor = torch.stack([p.G_res for p in self.prosumers])

        # Pre-cached unique edge pairs for reciprocity checks
        self._unique_edge_pairs = []
        visited = set()
        for (m, n) in self.edge_list:
            if (m, n) not in visited and (n, m) in self._idx_map:
                visited.add((m, n))
                visited.add((n, m))
                self._unique_edge_pairs.append((m, n))

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
        if not self.edge_list:
            return torch.zeros(2, 0, dtype=torch.long, device=self.device)
        src, dst = zip(*self.edge_list)
        return torch.tensor([src, dst], dtype=torch.long, device=self.device)

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
        N = self.N
        for n in range(N):
            self.prosumers[n].D = x[n]
            self.prosumers[n].G = x[N + n]

        base = 2 * N
        for i, (m, n) in enumerate(self.edge_list):
            self.prosumers[n].q[m] = x[base + i]

    def flatten_admm(self) -> torch.Tensor:
        """
        Returns full ADMM state sigma = [x, z, u] as a single vector.
 
        sigma = [D_0,...,D_{N-1}, G_0,...,G_{N-1}, Q_0,...,Q_{|E|-1},
                 Z_0,...,Z_{|E|-1}, U_0,...,U_{|E|-1}]
 
        Z and U are ordered as edge_list: Z[i] = z_{mn} for edge_list[i] = (m,n).
        """
        prosumers = self.prosumers
        x = self.flatten()
        Z = torch.stack([prosumers[n].z[m] for (m, n) in self.edge_list])
        U = torch.stack([prosumers[n].u[m] for (m, n) in self.edge_list])
        return torch.cat([x, Z, U])

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

    def admm_reset(self):
        """
        Reset z, u, z_received to zero for all prosumers.
        Must be called before starting a new ADMM run.
        """
        for p in self.prosumers:
            p.admm_reset()

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
        N      = self.N
        idx_map = self._idx_map

        def objective(x):
            x_t = torch.tensor(x, dtype=torch.float32)
            return (0.5 * torch.sum((x_t - x_orig) ** 2)).item()

        def grad_objective(x):
            x_t = torch.tensor(x, dtype=torch.float32)
            return (x_t - x_orig).numpy().astype(float)

        # Balance equality constraints: D[n] = G[n] + G_res[n] + sum_m q_{mn}
        eq_constraints = []
        G_res_vals = self._G_res_vals
        for n in range(N):
            q_idxs = [2 * N + idx_map[(m, n)] for m in self.neighbors[n]]
            def make_eq(n_, q_idxs_, g_res_):
                def eq_fn(x):
                    return (
                        float(x[n_])
                        - float(x[N + n_])
                        - g_res_
                        - sum(float(x[i]) for i in q_idxs_)
                    )
                return eq_fn
            eq_constraints.append({"type": "eq", "fun": make_eq(n, q_idxs, G_res_vals[n])})

        # Reciprocity inequality constraints: q_{mn} + q_{nm} <= 0
        ineq_constraints = []
        base = 2 * N
        for (m, n) in self._unique_edge_pairs:
            i_mn = base + idx_map[(m, n)]
            i_nm = base + idx_map[(n, m)]
            def make_ineq(a_, b_):
                def ineq_fn(x):
                    return -(float(x[a_]) + float(x[b_]))
                return ineq_fn
            ineq_constraints.append({"type": "ineq", "fun": make_ineq(i_mn, i_nm)})

        result = minimize(
            objective,
            x0.detach().cpu().numpy(),
            jac=grad_objective,
            method="SLSQP",
            bounds=self._bounds,
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
        G_res = self._G_res_tensor

        # Precompute per-node q_sum using incoming_idx for vectorized balance
        for _ in range(n_iter):
            eq_loss = torch.tensor(0.0, device=self.device)
            for n in range(self.N):
                idx = self.incoming_idx[n]
                q_sum = Q[idx].sum() if idx else torch.tensor(0.0, device=self.device)
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

    def get_pyg_data(self, algorithm: str = "fb") -> Data:
        """
        Build a global PyG Data object with the full network state.
        Used during centralised training where all node/edge data is available.

        algorithm="fb"  (default, unchanged):
            Node features x : (N, 2)   — [D_n, G_n]
            Edge features   : (|E|, 1) — q_{nm}

        algorithm="admm":
            Node features x : (N, 3)   — [D_n, G_n, u_avg_n]
                              u_avg_n = mean(u[m]) over neighbors m
            Edge features   : (|E|, 3) — [q_{nm}, z_{nm}, u_{nm}]
                              z_{nm} = prosumer m's consensus copy of q_{nm}
                              u_{nm} = prosumer m's dual variable for q_{nm}
        """
        if algorithm == "fb":
            x_node = torch.stack(
                [torch.stack([p.D, p.G]) for p in self.prosumers]
            )  # (N, 2)
            edge_attr = torch.stack(
                [self.prosumers[m].q[n].reshape(1) for (m, n) in self.edge_list]
            )  # (|E|, 1)

        else:  # admm
            _zero = torch.tensor(0.0, device=self.device)
            prosumers = self.prosumers

            def u_avg(p):
                if not p.neighbors:
                    return _zero
                return torch.stack([p.u[m] for m in p.neighbors]).mean()

            x_node = torch.stack(
                [torch.stack([p.D, p.G, u_avg(p)]) for p in prosumers]
            )  # (N, 3)

            # Build edge_attr as (|E|, 3) without per-edge torch.stack
            q_vals = [prosumers[m].q[n] for (m, n) in self.edge_list]
            z_vals = [prosumers[m].z[n] for (m, n) in self.edge_list]
            u_vals = [prosumers[m].u[n] for (m, n) in self.edge_list]
            edge_attr = torch.stack([
                torch.stack(q_vals),
                torch.stack(z_vals),
                torch.stack(u_vals),
            ], dim=1)  # (|E|, 3)

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

    def constraint_violation(self, tol: float = 1e-3) -> bool:
        """
        Returns True if any constraint is violated beyond tol.

        Checks: D/G bounds, Q capacity, balance equality, reciprocity.
        """
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

        for (m, n) in self._unique_edge_pairs:
            q_mn = float(self.prosumers[n].q[m])
            q_nm = float(self.prosumers[m].q[n])
            if q_mn + q_nm > tol:
                print("violation coupling constraint")
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