import copy
import torch
import cvxpy as cp


class Prosumer:
    """
    Single prosumer in a peer-to-peer energy market.

    Each prosumer owns:
        - its fixed physical parameters (immutable after construction)
        - its decision variables: D, G, and q (one scalar per neighbor)
        - a receive buffer: q_received[m] = q_{nm} communicated by neighbor m
          i.e. how much m wants to trade with n (m's decision, sent to n)

    Decision variables (local indexing by neighbor):
        D           : scalar tensor  — energy demand
        G           : scalar tensor  — controllable flexibility activation
        q[m]        : scalar tensor  — trade decision of this node w.r.t. neighbor m
                      q[m] > 0: buy from m; q[m] < 0: sell to m

    Received buffer (filled by neighbors before each FB step):
        q_received[m] : scalar tensor  — q_{nm} as decided by m and communicated to n

    Parameters (fixed, set at construction):
        node_id     : int            — global index of this prosumer
        neighbors   : list[int]      — sorted list of neighbor indices
        D_min, D_max: scalar tensors — demand bounds
        G_min, G_max: scalar tensors — flexibility bounds
        G_res       : scalar tensor  — renewable generation (not optimized)
        a_tilde     : scalar tensor  — demand utility curvature
        a, b        : scalar tensors — flexibility cost coefficients
        D_star      : scalar tensor  — target demand
        c[m]        : dict scalar    — per-unit trading cost w.r.t. neighbor m
        k[m]        : dict scalar    — capacity on link to neighbor m
    """

    def __init__(
        self,
        node_id: int,
        neighbors: list,
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
        self.node_id   = node_id
        self.neighbors = sorted(neighbors)
        self.device    = device

        # Fixed parameters — stored as scalar tensors on device
        self.D_min   = D_min.to(device)
        self.D_max   = D_max.to(device)
        self.G_min   = G_min.to(device)
        self.G_max   = G_max.to(device)
        self.G_res   = G_res.to(device)
        self.a_tilde = a_tilde.to(device)
        self.a       = a.to(device)
        self.b       = b.to(device)
        self.D_star  = D_star.to(device)

        # c[m] and k[m]: per-neighbor dicts of scalar tensors
        self.c = {m: c[m].to(device) for m in self.neighbors}
        self.k = {m: k[m].to(device) for m in self.neighbors}

        # Decision variables — initialized randomly in _init_state
        self.D = None     # scalar tensor
        self.G = None     # scalar tensor
        self.q = {}       # dict {m: scalar tensor}, one per neighbor

        # Receive buffer — filled by neighbors before each step
        # q_received[m] = q_{nm}, the trade decision of neighbor m w.r.t. this node
        self.q_received = {m: torch.zeros(1, device=device) for m in self.neighbors}

        self._init_state()

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def _init_state(self):
        """Randomize D, G, q within their feasible bounds."""
        self.D = self.D_min + (self.D_max - self.D_min) * torch.rand(1, device=self.device).squeeze()
        self.G = self.G_min + (self.G_max - self.G_min) * torch.rand(1, device=self.device).squeeze()
        self.q = {
            m: torch.rand(1, device=self.device).squeeze() * self.k[m]
            for m in self.neighbors
        }

    # ------------------------------------------------------------------
    # Local net import
    # ------------------------------------------------------------------

    def net_import(self) -> torch.Tensor:
        """
        Q_n = sum_{m in Omega_n} q[m]

        Returns scalar tensor.
        """
        if not self.neighbors:
            return torch.zeros(1, device=self.device).squeeze()
        return torch.stack([self.q[m] for m in self.neighbors]).sum()

    # ------------------------------------------------------------------
    # Local pseudo-gradient (F_n components)
    # ------------------------------------------------------------------

    def grad_D(self) -> torch.Tensor:
        """grad_{D_n} J_n = 2 * a_tilde * (D - D_star)"""
        return 2.0 * self.a_tilde * (self.D - self.D_star)

    def grad_G(self) -> torch.Tensor:
        """grad_{G_n} J_n = a * G + b"""
        return self.a * self.G + self.b

    def grad_q(self, m: int) -> torch.Tensor:
        """grad_{q_{mn}} J_n = c[m]  (constant w.r.t. q)"""
        return self.c[m]

    # ------------------------------------------------------------------
    # Forward step (gradient descent on local cost)
    # ------------------------------------------------------------------

    def forward_step(self, gamma: float) -> "Prosumer":
        """
        Compute y_n = x_n - gamma * grad J_n  without modifying self.

        Returns a new Prosumer with updated D, G, q (no projection yet).
        Parameters and q_received are copied by reference (read-only).
        """
        new_p = self._shallow_copy()
        new_p.D = self.D - gamma * self.grad_D()
        new_p.G = self.G - gamma * self.grad_G()
        new_p.q = {m: self.q[m] - gamma * self.grad_q(m) for m in self.neighbors}
        return new_p

    # ------------------------------------------------------------------
    # Local projection via CvxpyLayer
    # ------------------------------------------------------------------

    def project_local(self) -> "Prosumer":
        """
        Project the current state onto the local feasible set by solving
        the local QP via cvxpy.

        Uses only information available locally:
            - own state (D, G, q)
            - q_received from neighbors (reciprocity RHS)

        Not differentiable — use only during inference.
        For training, use project_approx() on EnergyNetwork instead.

        Problem (dim = 2 + k variables, k = |Omega_n|):

            min   (1/2) || x - x_hat ||^2
            s.t.  D_min <= D <= D_max               (box on D)
                  G_min <= G <= G_max               (box on G)
                  q_i   <= k_i  for all i           (capacity)
                  D - G - G_res - sum_i q_i = 0     (balance)
                  q_i   <= -q_received_i  for all i (reciprocity)

        Modifies D, G, q in-place and returns self for chaining.

        Returns
        -------
        self (in-place update)
        """
        k   = len(self.neighbors)
        dim = 2 + k

        # Current state as numpy array (cvxpy works with numpy)
        x_hat = self.flatten().detach().cpu().numpy()

        # Reciprocity RHS: q_received from each neighbor
        if k > 0:
            q_recv = [
                float(self.q_received[m].item()) for m in self.neighbors
            ]

        # Fixed bounds
        D_min_val = float(self.D_min.item())
        D_max_val = float(self.D_max.item())
        G_min_val = float(self.G_min.item())
        G_max_val = float(self.G_max.item())
        G_res_val = float(self.G_res.item())
        k_caps    = [float(self.k[m].item()) for m in self.neighbors]

        # CVXPY problem
        x = cp.Variable(dim)

        objective = cp.Minimize(0.5 * cp.sum_squares(x - x_hat))

        constraints = []

        # 1. Box on D
        constraints += [x[0] >= D_min_val, x[0] <= D_max_val]

        # 2. Box on G
        constraints += [x[1] >= G_min_val, x[1] <= G_max_val]

        # 3. Capacity on each q
        for i in range(k):
            constraints.append(x[2 + i] <= k_caps[i])

        # 4. Balance equality
        if k > 0:
            constraints.append(
                x[0] - x[1] - G_res_val - cp.sum(x[2:]) == 0
            )
        else:
            constraints.append(x[0] - x[1] == G_res_val)

        # 5. Reciprocity
        for i in range(k):
            constraints.append(x[2 + i] <= -q_recv[i])

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            print(f"Warning: Prosumer {self.node_id} projection status: "
                  f"{problem.status}")
            return self

        # Write projected values back as tensors
        x_proj = torch.tensor(x.value, dtype=torch.float32, device=self.device)
        self.unflatten(x_proj)
        return self

    def send_q_to(self, other: "Prosumer"):
        """
        Send this node's trade decision q[other.node_id] to other's receive buffer.

        Called during the communication round before each FB step.
        Equivalent to: other.q_received[self.node_id] = self.q[other.node_id]
        """
        n = self.node_id
        m = other.node_id
        other.q_received[n] = self.q[m].detach().clone()

    # ------------------------------------------------------------------
    # Local PyG data for distributed GNN inference
    # ------------------------------------------------------------------

    def get_local_pyg_data(self):
        """
        Build a PyG-compatible dict with the local subgraph of this node.

        Node re-indexing:
            local 0      : this prosumer (ego)
            local 1..k   : neighbors (sorted order)

        Node features x : (1+k, 2)
            x[0] = [D_n, G_n]
            x[i] = [D_mi, G_mi]  — filled from q_received (D/G of neighbors
                                    are not available locally; we use zeros)

        Edge features edge_attr : (k, 1)
            edge_attr[i] = q_received[m_i]  — trade decision communicated by m_i

        Edge index: all edges point to ego node (local index 0).
            edge_index[0] = [1, 2, ..., k]
            edge_index[1] = [0, 0, ..., 0]

        Note: neighbor node features (D, G) are not available locally in a
        truly distributed setting. We use zeros as placeholders — the GNN
        only uses them to fill h_new, which is dominated by the edge messages.

        Returns
        -------
        data : dict with keys 'x', 'edge_index', 'edge_attr'  (all tensors)
               Compatible with GNNLayer / GNNUpdate.
        """
        from torch_geometric.data import Data

        k = len(self.neighbors)

        # Ego node features
        ego_feat = torch.stack([self.D, self.G]).unsqueeze(0)  # (1, 2)

        if k > 0:
            # Neighbor features: zeros (not available locally)
            nbr_feat = torch.zeros(k, 2, device=self.device)
            x_local  = torch.cat([ego_feat, nbr_feat], dim=0)  # (1+k, 2)

            # Edge index: neighbors -> ego
            src = torch.arange(1, k + 1, dtype=torch.long, device=self.device)
            dst = torch.zeros(k,         dtype=torch.long, device=self.device)
            edge_index = torch.stack([src, dst], dim=0)  # (2, k)

            # Edge features: q received from each neighbor (sorted order)
            edge_attr = torch.stack(
                [self.q_received[m].reshape(1) for m in self.neighbors]
            )  # (k, 1)
        else:
            x_local    = ego_feat
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            edge_attr  = torch.zeros(0, 1, device=self.device)

        return Data(
            x=x_local,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=1 + k,
        )

    # ------------------------------------------------------------------
    # Apply GNN perturbation (distributed, local write)
    # ------------------------------------------------------------------

    def apply_perturbation(self, v_D: torch.Tensor, v_G: torch.Tensor, v_q: dict):
        """
        Add perturbations to local decision variables (in-place).

        Parameters
        ----------
        v_D  : scalar tensor  — perturbation for D
        v_G  : scalar tensor  — perturbation for G
        v_q  : dict {m: scalar tensor}  — perturbations for q[m], one per neighbor
        """
        self.D = self.D + v_D
        self.G = self.G + v_G
        for m in self.neighbors:
            self.q[m] = self.q[m] + v_q[m]

    # ------------------------------------------------------------------
    # Flatten / unflatten (local state vector)
    # ------------------------------------------------------------------

    def flatten(self) -> torch.Tensor:
        """
        Returns local state vector [D, G, q[m0], q[m1], ...] shape (2+k,).
        Neighbor order: sorted(neighbors).
        """
        parts = [self.D.unsqueeze(0), self.G.unsqueeze(0)]
        parts += [self.q[m].unsqueeze(0) for m in self.neighbors]
        return torch.cat(parts)

    def unflatten(self, x: torch.Tensor):
        """
        Sets D, G, q from local state vector of shape (2+k,).
        Neighbor order must match sorted(neighbors).
        """
        self.D = x[0]
        self.G = x[1]
        for i, m in enumerate(self.neighbors):
            self.q[m] = x[2 + i]

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self) -> "Prosumer":
        """
        Returns a copy with independent D, G, q, q_received.
        Fixed parameters (c, k, bounds) are shared (read-only).
        """
        new_p = self._shallow_copy()
        new_p.D          = self.D.clone()
        new_p.G          = self.G.clone()
        new_p.q          = {m: self.q[m].clone()          for m in self.neighbors}
        new_p.q_received = {m: self.q_received[m].clone() for m in self.neighbors}
        return new_p

    def _shallow_copy(self) -> "Prosumer":
        """
        Shallow copy: shares parameters, creates new state containers.
        Not meant to be used directly — use clone() for independent state.
        """
        new_p = copy.copy(self)
        return new_p

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def balance_violation(self) -> torch.Tensor:
        """D - G - G_res - sum_m q[m]. Should be 0 at feasibility."""
        return self.D - self.G - self.G_res - self.net_import()

    def local_cost(self) -> torch.Tensor:
        """J_n = a_tilde*(D-D*)^2 + 0.5*a*G^2 + b*G + sum_m c[m]*q[m]"""
        node_cost = (
            self.a_tilde * (self.D - self.D_star) ** 2
            + 0.5 * self.a * self.G ** 2
            + self.b * self.G
        )
        trade_cost = sum(self.c[m] * self.q[m] for m in self.neighbors) if self.neighbors else torch.zeros(1, device=self.device).squeeze()
        return node_cost + trade_cost

    def __repr__(self):
        return (
            f"Prosumer(id={self.node_id}, "
            f"neighbors={self.neighbors}, "
            f"D={self.D:.3f}, G={self.G:.3f})"
        )