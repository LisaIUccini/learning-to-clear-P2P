import copy
import torch
import numpy as np
from scipy.optimize import minimize
from torch_geometric.data import Data

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
    def project_differentiable(self):
        """
        Differentiable projection via qpth (OptNet).
        Identical to project_exact in the forward pass:
            min  0.5 * ||x - x0||^2
            s.t. D_min <= D <= D_max
                G_min <= G <= G_max
                Q[i]  <= k_edge[i]              (no lower bound, as in project_exact)
                D[n] = G[n] + G_res[n] + sum_m q_{mn}   (balance equality)
                q_{mn} + q_{nm} <= 0            (reciprocity)

        Forward  : QPFunction solves the QP exactly
        Backward : OptNet implicit differentiation (KKT)
        """
        from qpth.qp import QPFunction

        N       = self.N
        E       = len(self.edge_list)
        dim     = 2 * N + E
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}
        dev     = self.device

        x0 = self.flatten()   # (dim,) — preserves grad_fn

        # ----------------------------------------------------------------
        # Objective: min 0.5 ||x - x0||^2  ≡  min 0.5 x^T I x - x0^T x
        # ----------------------------------------------------------------
        Q_mat = torch.eye(dim, device=dev)          # (dim, dim)
        p     = -x0.unsqueeze(0)                    # (1, dim)

        # ----------------------------------------------------------------
        # Equality constraints: A_eq @ x = b_eq
        # Identical to project_exact:
        #   D[n] - G[n] - sum_m q_{mn} = G_res[n]
        # ----------------------------------------------------------------
        A_eq = torch.zeros(N, dim, device=dev)
        b_eq = self.G_res.clone()

        for n in range(N):
            A_eq[n, n]     =  1.0           # D[n]
            A_eq[n, N + n] = -1.0           # -G[n]
            for m in self.neighbors[n]:
                A_eq[n, 2*N + idx_map[(m, n)]] = -1.0  # -q_{mn}

        A_eq = A_eq.unsqueeze(0)            # (1, N, dim)
        b_eq = b_eq.unsqueeze(0)            # (1, N)

        # ----------------------------------------------------------------
        # Inequality constraints: G_ineq @ x <= h
        # Identical to project_exact bounds + reciprocity:
        #
        #   -D[n] <= -D_min[n]       (D lower bound)
        #    D[n] <=  D_max[n]       (D upper bound)
        #   -G[n] <= -G_min[n]       (G lower bound)
        #    G[n] <=  G_max[n]       (G upper bound)
        #    Q[i] <=  k_edge[i]      (capacity, NO lower bound as in project_exact)
        #    q_{mn} + q_{nm} <= 0    (reciprocity)
        # ----------------------------------------------------------------
        visited     = set()
        recip_pairs = []
        for (m, n) in self.edge_list:
            if (m, n) not in visited:
                visited.add((m, n))
                visited.add((n, m))
                recip_pairs.append((idx_map[(m, n)], idx_map[(n, m)]))

        n_ineq = 4*N + E + len(recip_pairs)
        G_mat  = torch.zeros(n_ineq, dim, device=dev)
        h_vec  = torch.zeros(n_ineq,      device=dev)

        row = 0
        # D lower bound: -D[n] <= -D_min[n]
        for n in range(N):
            G_mat[row, n] = -1.0
            h_vec[row]    = -self.D_min[n]
            row += 1
        # D upper bound: D[n] <= D_max[n]
        for n in range(N):
            G_mat[row, n] =  1.0
            h_vec[row]    =  self.D_max[n]
            row += 1
        # G lower bound: -G[n] <= -G_min[n]
        for n in range(N):
            G_mat[row, N+n] = -1.0
            h_vec[row]      = -self.G_min[n]
            row += 1
        # G upper bound: G[n] <= G_max[n]
        for n in range(N):
            G_mat[row, N+n] =  1.0
            h_vec[row]      =  self.G_max[n]
            row += 1
        # Capacity: Q[i] <= k_edge[i]  — NO lower bound, identical to project_exact
        for i in range(E):
            G_mat[row, 2*N+i] =  1.0
            h_vec[row]        =  self.k_edge[i]
            row += 1
        # Reciprocity: q_{mn} + q_{nm} <= 0
        for (i_mn, i_nm) in recip_pairs:
            G_mat[row, 2*N+i_mn] = 1.0
            G_mat[row, 2*N+i_nm] = 1.0
            h_vec[row]            = 0.0
            row += 1

        G_mat = G_mat.unsqueeze(0)          # (1, n_ineq, dim)
        h_vec = h_vec.unsqueeze(0)          # (1, n_ineq)

        # ----------------------------------------------------------------
        # Solve — QPFunction handles both forward (QP solve) and
        # backward (implicit differentiation via KKT conditions)
        # ----------------------------------------------------------------
        qp     = QPFunction(verbose=False, eps=1e-12, maxIter=10000)
        x_proj = qp(Q_mat, p, G_mat, h_vec, A_eq, b_eq).squeeze(0)  # (dim,)

        self.D = x_proj[:N]
        self.G = x_proj[N:2*N]
        self.Q = x_proj[2*N:]

    def project_exact(self):
        """
        osqp, clarabel, prova altri solver
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
            #raise ValueError(f"Exact projection failed: {result.message}")
            print(f"Warning: Exact projection failed")

        x_proj = torch.tensor(result.x, dtype=torch.float32, device=self.device)
        #self.unflatten(x_proj)
        return x_proj, result.success
    

    def project_approx(self, n_iter: int = 30, lr: float = 0.1,
                       create_graph: bool = False):
        """
        Approximate differentiable projection. Updates state in place.

        Parameters
        ----------
        n_iter       : number of gradient correction steps
        lr           : step size for the correction
        create_graph : keep computational graph for backprop through projection.
                       Set True during training, False during inference.
        """
        self.D = torch.clamp(self.D, self.D_min, self.D_max)
        self.G = torch.clamp(self.G, self.G_min, self.G_max)

        Q = self.Q.detach().clone()
        Q.requires_grad_(True)

        # D and G are detached — they are fixed during Q correction
        D = self.D.detach()
        G = self.G.detach()

        for _ in range(n_iter):
            eq_loss = torch.tensor(0.0, device=self.device)
            for n in range(self.N):
                if self.incoming_idx[n]:
                    q_sum = Q[self.incoming_idx[n]].sum()
                else:
                    q_sum = torch.tensor(0.0, device=self.device)
                eq_loss = eq_loss + (
                    D[n] - G[n] - self.G_res[n] - q_sum
                ) ** 2

            if create_graph:
                grads = torch.autograd.grad(eq_loss, Q)[0]
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

    def get_local_pyg_data(self, n: int):
        """
        Build a PyG Data object for node n and its local neighborhood only.

        Used for distributed execution: node n uses only its own state
        and the messages from neighbors to compute its perturbation.

        Local graph re-indexing:
            node 0       : prosumer n  (ego node)
            node 1..k    : neighbors m1, m2, ... (sorted)

        Node features x : (1+k, 2)
            x[0]   = [D_n, G_n]
            x[1+i] = [D_mi, G_mi]

        Edge features edge_attr : (k, 1)
            edge_attr[i] = q_{nm_i} = Q[inverse_edge[(m_i, n)]]
            i.e. trade decision of neighbor m_i communicated to n.

        Edge index: all edges point toward ego node 0.
            edge_index = [[1, 2, ..., k],
                          [0, 0, ..., 0]]

        Returns
        -------
        data                  : torch_geometric.data.Data
        local_edge_global_idx : list[int]  global edge_list indices of incoming
                                edges (m->n), same order as edge_attr rows.
                                Used to write back v_edges into the global Q.
        """
        neighbors_n = sorted(self.neighbors[n])
        k = len(neighbors_n)

        # Node features: ego + neighbors
        ego_feat = torch.stack([self.D[n], self.G[n]]).unsqueeze(0)  # (1, 2)
        if k > 0:
            nbr_feat = torch.stack(
                [torch.stack([self.D[m], self.G[m]]) for m in neighbors_n]
            )  # (k, 2)
            x_local = torch.cat([ego_feat, nbr_feat], dim=0)  # (1+k, 2)
        else:
            x_local = ego_feat

        # Edge index: neighbors -> ego
        if k > 0:
            src_local = torch.arange(1, k + 1, dtype=torch.long, device=self.device)
            dst_local = torch.zeros(k, dtype=torch.long, device=self.device)
            edge_index_local = torch.stack([src_local, dst_local], dim=0)  # (2, k)
        else:
            edge_index_local = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        # Edge features: q_{nm} for each neighbor m
        # local_edge_global_idx[i] = index of (n, m_i) in edge_list (inverse edge)
        # The feature of edge (m_i -> n) is Q[inverse_edge[(m_i, n)]] = q_{n,m_i}
        local_edge_global_idx = [
            self.inverse_edge[(m, n)] for m in neighbors_n
        ]

        if k > 0:
            # Q[inv_i] is a scalar tensor — reshape to (1,) then stack to (k, 1)
            edge_attr_local = torch.stack(
                [self.Q[inv_i].reshape(1) for inv_i in local_edge_global_idx]
            )  # (k, 1)
        else:
            edge_attr_local = torch.zeros(0, 1, device=self.device)

        data = Data(
            x=x_local,
            edge_index=edge_index_local,
            edge_attr=edge_attr_local,
            num_nodes=1 + k,
        )
        return data, local_edge_global_idx

    # ------------------------------------------------------------------
    # Apply GNN perturbation
    # ------------------------------------------------------------------

    def apply_perturbation_node(
        self,
        n: int,
        v_node_n: torch.Tensor,
        v_edge_n: torch.Tensor,
    ):
        """
        Apply perturbations produced by node n to its own local state.

        Called inside the distributed loop — each node n updates only
        its own D[n], G[n], and Q[incoming edges of n].

        Parameters
        ----------
        n        : node index
        v_node_n : torch.Tensor (2,)  — [v_D_n, v_G_n]
        v_edge_n : torch.Tensor (k,)  — perturbations for Q[incoming_idx[n]],
                                        i.e. q_{mn} for each m in Omega_n (sorted)
        """

        D_new = self.D.clone()
        G_new = self.G.clone()
        Q_new = self.Q.clone()

        D_new[n] = self.D[n] + v_node_n[0]
        G_new[n] = self.G[n] + v_node_n[1]

        for local_i, global_i in enumerate(self.incoming_idx[n]):
            Q_new[global_i] = self.Q[global_i] + v_edge_n[local_i]

        self.D = D_new
        self.G = G_new
        self.Q = Q_new

        """
        self.D[n] = self.D[n] + v_node_n[0]
        self.G[n] = self.G[n] + v_node_n[1]

        for local_i, global_i in enumerate(self.incoming_idx[n]):
            self.Q[global_i] = self.Q[global_i] + v_edge_n[local_i]
        """

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
    
    def constraint_violation(self, tol: float = 1e-4) -> bool:
        """
        Returns True if any constraint is violated beyond tol, False otherwise.

        Constraints checked:
            1. D bounds     : D_min[n] <= D[n] <= D_max[n]
            2. G bounds     : G_min[n] <= G[n] <= G_max[n]
            3. Q capacity   : Q[i] <= k_edge[i]
            4. Balance      : D[n] = G[n] + G_res[n] + sum_{m} q_{mn}
            5. Reciprocity  : Q[idx(m,n)] + Q[idx(n,m)] <= 0
        """
        idx_map = {edge: i for i, edge in enumerate(self.edge_list)}

        # D and G bounds
        if (self.D < self.D_min - tol).any(): return True
        if (self.D > self.D_max + tol).any(): return True
        if (self.G < self.G_min - tol).any(): return True
        if (self.G > self.G_max + tol).any(): return True

        # Q capacity
        if (self.Q > self.k_edge + tol).any(): return True

        # Balance equality
        if (self.balance_violation().abs() > tol).any(): return True

        # Reciprocity
        visited = set()
        for (m, n) in self.edge_list:
            if (m, n) in visited or (n, m) not in idx_map:
                continue
            visited.add((m, n))
            visited.add((n, m))
            if float(self.Q[idx_map[(m,n)]]) + float(self.Q[idx_map[(n,m)]]) > tol:
                return True

        return False

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