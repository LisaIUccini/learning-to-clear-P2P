import copy
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer



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

        # ADMM auxiliary variables (reset before each run via admm_reset())
        # z[m] : consensus copy of q[m]           — updated via z_update()
        # u[m] : scaled dual variable (lambda/rho) — updated via u_update()
        self.z = {m: torch.zeros(1, device=device).squeeze() for m in self.neighbors}
        self.u = {m: torch.zeros(1, device=device).squeeze() for m in self.neighbors}

        # ADMM receive buffers — filled by neighbors during admm_communicate()
        # z_received[m] : z_{nm} held by neighbor m (needed for z_update)
        # q_received is reused for q_{nm} received from m
        self.z_received = {m: torch.zeros(1, device=device).squeeze() for m in self.neighbors}
        self.u_received = {m: torch.zeros(1, device=device).squeeze() for m in self.neighbors}

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

    # ------------------------------------------------------------------
    # ADMM methods
    # ------------------------------------------------------------------

    def admm_reset(self):
        """
        Reset z, u, u_received, z_received to zero before a new ADMM run.
        Must be called on all prosumers before starting.
        """
        for m in self.neighbors:
            self.z[m]          = torch.zeros(1, device=self.device).squeeze()
            self.u[m]          = torch.zeros(1, device=self.device).squeeze()
            self.z_received[m] = torch.zeros(1, device=self.device).squeeze()
            self.u_received[m] = torch.zeros(1, device=self.device).squeeze()

    def admm_x_update(self, rho: float) -> "Prosumer":
        """
        ADMM x-update (exact, via cvxpy): solve the local augmented Lagrangian QP.

        Problem (scaled ADMM form):

            min   J_n(D, G, q) + (rho/2) * sum_m || q[m] - z[m] + u[m] ||^2
            s.t.  D_min <= D <= D_max
                  G_min <= G <= G_max
                  q[m]  <= k[m]   for all m
                  D - G - G_res - sum_m q[m] = 0   (balance)

        Note: z[m] is this node's own consensus copy, NOT z_received.
        At convergence: q[m] = z[m], and z[m] + z_received[m] <= 0
        (reciprocity satisfied globally).

        Not differentiable — use only during inference.
        For training use admm_x_update_approx().
        """
        k   = len(self.neighbors)
        dim = 2 + k

        D_min_val = float(self.D_min.item())
        D_max_val = float(self.D_max.item())
        G_min_val = float(self.G_min.item())
        G_max_val = float(self.G_max.item())
        G_res_val = float(self.G_res.item())
        k_caps    = [float(self.k[m].item()) for m in self.neighbors]

        # z[m] - u[m] : the ADMM target for q[m]
        z_minus_u = [
            float(self.z[m].item()) - float(self.u[m].item())
            for m in self.neighbors
        ]

        a_tilde_val = float(self.a_tilde.item())
        a_val       = float(self.a.item())
        b_val       = float(self.b.item())
        D_star_val  = float(self.D_star.item())
        c_vals      = [float(self.c[m].item()) for m in self.neighbors]

        x = cp.Variable(dim)
        D = x[0]
        G = x[1]

        # Local cost J_n
        node_cost  = a_tilde_val * cp.square(D - D_star_val) + \
                     0.5 * a_val * cp.square(G) + b_val * G
        trade_cost = cp.sum([c_vals[i] * x[2 + i] for i in range(k)]) \
                     if k > 0 else 0
        J_n = node_cost + trade_cost

        # ADMM penalty: (rho/2) * sum_m (q[m] - z[m] + u[m])^2
        #             = (rho/2) * sum_m (q[m] - (z[m] - u[m]))^2
        if k > 0:
            penalty = (rho / 2.0) * cp.sum(
                [cp.square(x[2 + i] - z_minus_u[i]) for i in range(k)]
            )
        else:
            penalty = 0

        objective = cp.Minimize(J_n + penalty)

        constraints = []
        constraints += [D >= D_min_val, D <= D_max_val]
        constraints += [G >= G_min_val, G <= G_max_val]
        for i in range(k):
            constraints.append(x[2 + i] <= k_caps[i])
        if k > 0:
            constraints.append(D - G - G_res_val - cp.sum(x[2:]) == 0)
        else:
            constraints.append(D - G == G_res_val)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            print(f"Warning: Prosumer {self.node_id} x-update status: "
                  f"{problem.status}")
            return self

        x_sol = torch.tensor(x.value, dtype=torch.float32, device=self.device)
        self.unflatten_local(x_sol)
        return self

    def admm_x_update_approx(self, rho: float) -> "Prosumer":
        """
        ADMM x-update (differentiable, for training): gradient step on J_n + penalty.

        Gradient of (rho/2)||q[m] - z[m] + u[m]||^2 w.r.t. q[m]
            = rho * (q[m] - z[m] + u[m])

        z[m] and u[m] are detached (they are parameters, not optimisation variables).
        """
        grad_D = self.grad_D()
        grad_G = self.grad_G()

        grad_q = {
            m: self.grad_q(m) + rho * (
                self.q[m] - self.z[m].detach() + self.u[m].detach()
            )
            for m in self.neighbors
        }

        # Step size: 1/(2*a_tilde + rho) for D, 1/(a + rho) for G and q
        step_D = 1.0 / (2.0 * float(self.a_tilde.item()) + rho)
        step_G = 1.0 / (float(self.a.item()) + rho)
        step_q = step_G

        self.D = torch.clamp(self.D - step_D * grad_D, self.D_min, self.D_max)
        self.G = torch.clamp(self.G - step_G * grad_G, self.G_min, self.G_max)
        for m in self.neighbors:
            self.q[m] = torch.clamp(
                self.q[m] - step_q * grad_q[m],
                max= self.k[m]
            )
        return self

    # ------------------------------------------------------------------
    # ADMM x-update (differentiable, exact via cvxpylayers)
    # ------------------------------------------------------------------
    
    def _build_cvxpy_layer(self, rho_val: float):
        """
        Build a CvxpyLayer for the ADMM x-update QP.
        Called once and cached — the problem structure is fixed,
        only the parameter values change between calls.

        The QP solves:
            min  J_n(D,G,q) + (rho/2) sum_m ||q[m] - target[m]||^2
            s.t. D_min <= D <= D_max
                 G_min <= G <= G_max
                 q[m] <= k[m]  for all m
                 D - G - G_res - sum(q) = 0

        Parameters (differentiable):
            target : (k,) — z[m] - u[m] for each neighbor m

        The solution x* is differentiable w.r.t. target via implicit
        differentiation of the KKT conditions (handled by cvxpylayers).
        """
        k = len(self.neighbors)
        dim = 2 + k

        # Fixed scalar values
        D_min_val   = float(self.D_min.item())
        D_max_val   = float(self.D_max.item())
        G_min_val   = float(self.G_min.item())
        G_max_val   = float(self.G_max.item())
        G_res_val   = float(self.G_res.item())
        a_tilde_val = float(self.a_tilde.item())
        a_val       = float(self.a.item())
        b_val       = float(self.b.item())
        D_star_val  = float(self.D_star.item())
        c_vals      = [float(self.c[m].item()) for m in self.neighbors]
        k_caps      = [float(self.k[m].item()) for m in self.neighbors]

        # cvxpy variable and parameter
        x = cp.Variable(dim)
        target = cp.Parameter(k)  # z[m] - u[m], differentiable

        D = x[0]
        G = x[1]

        # Objective: J_n + ADMM penalty
        node_cost  = a_tilde_val * cp.square(D - D_star_val) + \
                     0.5 * a_val * cp.square(G) + b_val * G
        trade_cost = cp.sum([c_vals[i] * x[2 + i] for i in range(k)]) \
                     if k > 0 else 0
        J_n = node_cost + trade_cost

        if k > 0:
            penalty = (rho_val / 2.0) * cp.sum(
                [cp.square(x[2 + i] - target[i]) for i in range(k)]
            )
        else:
            penalty = 0

        objective = cp.Minimize(J_n + penalty)

        # Constraints
        constraints = []
        constraints += [D >= D_min_val, D <= D_max_val]
        constraints += [G >= G_min_val, G <= G_max_val]
        for i in range(k):
            constraints.append(x[2 + i] <= k_caps[i])
        if k > 0:
            constraints.append(D - G - G_res_val - cp.sum(x[2:]) == 0)
        else:
            constraints.append(D - G == G_res_val)

        problem = cp.Problem(objective, constraints)

        # Build CvxpyLayer: target is the differentiable parameter
        self._cvxpy_layer = CvxpyLayer(
            problem, parameters=[target], variables=[x]
        )
        self._cvxpy_layer_rho = rho_val

    def admm_x_update_diff(self, rho: float) -> "Prosumer":
        """
        ADMM x-update: exact QP solution, differentiable via cvxpylayers.

        Solves the same QP as admm_x_update but gradients flow through
        the solution w.r.t. the ADMM target (z[m] - u[m]).

        The CvxpyLayer is built once and cached. If rho changes,
        the layer is rebuilt.
        """
        k = len(self.neighbors)

        # Build/rebuild layer if needed
        if not hasattr(self, '_cvxpy_layer') or self._cvxpy_layer_rho != rho:
            self._build_cvxpy_layer(rho)

        # Construct the differentiable parameter: target[i] = z[m_i] - u[m_i]
        target_vals = torch.stack([
            self.z[m] - self.u[m] for m in self.neighbors
        ]).double() if k > 0 else torch.zeros(0, dtype=torch.float64, device=self.device)

        # Solve — output is differentiable w.r.t. target_vals
        sol, = self._cvxpy_layer(target_vals, solver_args={'solve_method': 'SCS'})

        # cvxpylayers returns float64, convert back to float32
        sol = sol.float()

        # Unpack solution back into prosumer state
        self.D = sol[0]
        self.G = sol[1]
        for i, m in enumerate(self.neighbors):
            self.q[m] = sol[2 + i]

        return self

    def admm_z_update(self):
        """
        ADMM z-update: closed-form projection for each edge pair (n, m).

        For edge (n, m), n holds:
            a = q[m] + u[m]           (own augmented variable)
            b = q_received[m] + u_received[m]  (neighbor m's augmented variable)

        We project (a, b) onto the halfspace {a + b <= 0}:
            z[m] = a - clamp(a+b, min=0) / 2

        This is equivalent to:
            if a + b <= 0:  z[m] = a             (already feasible)
            else:           z[m] = a - (a+b)/2   (project onto a+b=0 hyperplane)

        but is differentiable (torch.clamp has well-defined gradients),
        allowing gradients to flow through z during training.

        At convergence: z[m] + z_received[m] <= 0  (reciprocity)
        and q[m] = z[m] for all m (consensus).

        Requires u_received[m] to be up to date — call admm_communicate() first.
        """
        for m in self.neighbors:
            a = self.q[m]          + self.u[m]
            b = self.q_received[m] + self.u_received[m]
            s = a + b
            self.z[m] = a - torch.clamp(s, min=0.0) / 2.0

    def admm_u_update(self):
        """
        ADMM u-update (scaled dual): u[m] += q[m] - z[m].
        Primal residual: q[m] - z[m] -> 0 at convergence.
        """
        for m in self.neighbors:
            self.u[m] = self.u[m] + self.q[m] - self.z[m]

    def send_z_to(self, other: "Prosumer", training: bool = False):
        """Send z[other.node_id] to other's z_received buffer."""
        if training:
            other.z_received[self.node_id] = self.z[other.node_id]
        else:
            other.z_received[self.node_id] = self.z[other.node_id].detach().clone()

    def send_u_to(self, other: "Prosumer", training: bool = False):
        """Send u[other.node_id] to other's u_received buffer."""
        if training:
            other.u_received[self.node_id] = self.u[other.node_id]
        else:
            other.u_received[self.node_id] = self.u[other.node_id].detach().clone()

    def unflatten_local(self, x: torch.Tensor):
        """
        Write [D, G, q_m0, q_m1, ...] back from local vector (2+k,).
        Used by admm_x_update after solving the local QP.
        """
        self.D = x[0]
        self.G = x[1]
        for i, m in enumerate(self.neighbors):
            self.q[m] = x[2 + i]

    def send_q_to(self, other: "Prosumer", training: bool = False):
        """
        Send this node's trade decision q[other.node_id] to other's receive buffer.

        Called during the communication round before each FB step.
        Equivalent to: other.q_received[self.node_id] = self.q[other.node_id]

        If training=True, preserves gradient flow between nodes (centralised).
        If training=False (default), detaches for distributed execution.
        """
        n = self.node_id
        m = other.node_id
        if training:
            other.q_received[n] = self.q[m]
        else:
            other.q_received[n] = self.q[m].detach().clone()

    # ------------------------------------------------------------------
    # Local PyG data for distributed GNN inference
    # ------------------------------------------------------------------

    def get_local_pyg_data(self, algorithm: str = "fb"):
        """
        Build a PyG-compatible Data object with the local subgraph of this node.

        Node re-indexing:
            local 0    : this prosumer (ego)
            local 1..k : neighbors (sorted order)

        algorithm="fb"  (default, unchanged):
            Node features x : (1+k, 2) — [D, G] for ego; zeros for neighbors
            Edge features   : (k,   1) — q_received[m] for each neighbor m

        algorithm="admm":
            Node features x : (1+k, 3) — [D, G, u_avg] for ego; zeros for neighbors
                              u_avg = mean(u[m]) over neighbors (own dual state)
                              Neighbor node features: zeros (not available locally)
            Edge features   : (k,   2) — [q_received[m], z_received[m]]
                              z_received[m] = neighbor m's consensus copy (communicated)

        Edge index: all edges point to ego node (local index 0).
        """
        from torch_geometric.data import Data

        k = len(self.neighbors)

        if algorithm == "fb":
            feat_size = 2
            ego_feat  = torch.stack([self.D, self.G]).unsqueeze(0)  # (1, 2)
        else:  # admm
            feat_size = 3
            if k > 0:
                u_avg = torch.stack([self.u[m] for m in self.neighbors]).mean()
            else:
                u_avg = torch.zeros(1, device=self.device).squeeze()
            ego_feat = torch.stack([self.D, self.G, u_avg]).unsqueeze(0)  # (1, 3)

        if k > 0:
            nbr_feat   = torch.zeros(k, feat_size, device=self.device)
            x_local    = torch.cat([ego_feat, nbr_feat], dim=0)  # (1+k, feat_size)

            src        = torch.arange(1, k + 1, dtype=torch.long, device=self.device)
            dst        = torch.zeros(k,         dtype=torch.long, device=self.device)
            edge_index = torch.stack([src, dst], dim=0)  # (2, k)

            if algorithm == "fb":
                edge_attr = torch.stack(
                    [self.q_received[m].reshape(1) for m in self.neighbors]
                )  # (k, 1)
            else:  # admm
                edge_attr = torch.stack([
                    torch.stack([
                        self.q_received[m].reshape(1),
                        self.z_received[m].reshape(1),
                    ], dim=-1).squeeze(0)
                    for m in self.neighbors
                ])  # (k, 2)
        else:
            x_local    = ego_feat
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            edge_attr  = torch.zeros(0, 1 if algorithm == "fb" else 2,
                                     device=self.device)

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

        Only D, G, q are perturbed — z and u are managed by ADMM and must
        not be perturbed directly to preserve constraint feasibility.

        Parameters
        ----------
        v_D  : scalar tensor  — perturbation for D
        v_G  : scalar tensor  — perturbation for G
        v_q  : dict {m: scalar tensor}  — perturbations for q[m]
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
        Returns a copy with independent D, G, q, q_received, z, u, z_received, u_received.
        Fixed parameters (c, k, bounds) are shared (read-only).
        """
        new_p = self._shallow_copy()
        new_p.D          = self.D.clone()
        new_p.G          = self.G.clone()
        new_p.q          = {m: self.q[m].clone()          for m in self.neighbors}
        new_p.q_received = {m: self.q_received[m].clone() for m in self.neighbors}
        new_p.z          = {m: self.z[m].clone()          for m in self.neighbors}
        new_p.u          = {m: self.u[m].clone()          for m in self.neighbors}
        new_p.z_received = {m: self.z_received[m].clone() for m in self.neighbors}
        new_p.u_received = {m: self.u_received[m].clone() for m in self.neighbors}
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