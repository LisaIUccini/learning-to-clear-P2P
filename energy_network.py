import torch
import numpy as np
from scipy.optimize import minimize

class EnergyNetwork:
    def __init__(self,
                 N,
                 neighbors,
                 D_min, D_max,
                 G_min, G_max,
                 G_res,
                 a_tilde, a, b,
                 D_star,
                 c, k,
                 device):

        self.N = N
        self.neighbors = neighbors
        self.device = device

        # bounds
        self.D_min = D_min.to(device)
        self.D_max = D_max.to(device)
        self.G_min = G_min.to(device)
        self.G_max = G_max.to(device)
        self.G_res = G_res.to(device)

        # cost parameters
        self.a_tilde = a_tilde.to(device)
        self.a = a.to(device)
        self.b = b.to(device)
        self.D_star = D_star.to(device)

        self.c = c
        self.k = k

        self.edge_list = self.build_edge_list()
        self.dim = 2 * N + len(self.edge_list)

        self.D, self.G, self.q = self.init_state()
        self.x = self.flatten(self.D, self.G, self.q)

        
    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def init_state(self):
        D = torch.zeros(self.N, device=self.device)
        G = torch.zeros(self.N, device=self.device)
        q = {}
        for n in self.neighbors:
            for m in self.neighbors[n]:
                q[(m, n)] = torch.tensor(0.0, device=self.device)
        return D, G, q
    

    def build_edge_list(self):
        """
        Build a fixed, ordered list of directed edges (m, n)
        such that m in neighbors[n].
        """
        edge_list = []
        for n in sorted(self.neighbors.keys()):
            for m in sorted(self.neighbors[n]):
                edge_list.append((m, n))
        return edge_list
    
    # --------------------------------------------------
    # Flatten / unflatten
    # --------------------------------------------------
    def flatten(self, D, G, q):
        q_vec = torch.tensor([q[e].item() if isinstance(q[e], torch.Tensor) else q[e] for e in self.edge_list])
        return torch.cat([D, G, q_vec])

    def unflatten(self, x):
        D = x[:self.N]
        G = x[self.N:2*self.N]
        q_vec = x[2*self.N:]
        q = {self.edge_list[i]: q_vec[i] for i in range(len(self.edge_list))}
        return D, G, q
    
    # --------------------------------------------------
    # Cost
    # --------------------------------------------------
    def cost(self):
        D, G, q = self.D, self.G, self.q
        cost = 0.0
        for n in range(self.N):
            cost += (
                self.a_tilde[n] * (D[n] - self.D_star[n])**2
                + 0.5 * self.a[n] * G[n]**2
                + self.b[n] * G[n]
            )
            for m in self.neighbors[n]:
                cost += self.c[(n, m)] * q[(m, n)]
        return cost

    # --------------------------------------------------
    # Gradient operator F
    # --------------------------------------------------
    def F(self):
        grad_D = torch.zeros(self.N, device=self.device)
        grad_G = torch.zeros(self.N, device=self.device)
        grad_q = {e: torch.tensor(0.0, device=self.device) for e in self.q}

        for n in range(self.N):
            # derivative w.r.t D_n
            grad_D[n] = 2 * self.a_tilde[n] * (self.D[n] - self.D_star[n])
            
            # derivative w.r.t G_n
            grad_G[n] = self.a[n] * self.G[n] + self.b[n]

            # derivative w.r.t q_{mn}
            for m in self.neighbors[n]:
                grad_q[(m, n)] = self.c[(m, n)]
        return grad_D, grad_G, grad_q
    
    # --------------------------------------------------
    # Projection onto constraints
    # --------------------------------------------------
    def project_all_agents_global_torch(self):
        q_keys = list(self.q.keys())
        key_to_idx = {key: i for i, key in enumerate(q_keys)}

        # Convert q dict into tensor vector
        q_vals = torch.stack([self.q[key].to(self.device) if isinstance(self.q[key], torch.Tensor) else torch.tensor(self.q[key], device=self.device, dtype=torch.float32)
                            for key in q_keys]).flatten()
        
        # Concatenate into a single tensor
        x0 = torch.cat([self.D, self.G, q_vals])
        x_orig = x0.clone()

        # Objective function (torch tensor, ma scipy richiede numpy)
        def objective(x):
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            val = 0.5 * torch.sum((x_t - x_orig)**2)
            return val.item()  # scipy richiede float

        def grad_objective(x):
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            grad = (x_t - x_orig).detach().cpu().numpy().astype(float)  # scipy richiede numpy
            return grad

        # Equality constraints: D[n] = G[n] + G_res[n] + sum(q_mn)
        eq_constraints = []
        for n in range(self.N):
            q_idxs = [2*self.N + key_to_idx[(m, n)] for m in self.neighbors[n] if (m, n) in key_to_idx]

            def make_eq(n_, q_idxs_):
                def eq_fn(x):
                    return float(x[n_]) - float(x[self.N + n_]) - float(self.G_res[n_]) - float(sum(x[q_idxs_]))
                return eq_fn

            eq_constraints.append({'type': 'eq', 'fun': make_eq(n, q_idxs)})

        # Inequality constraints: q[(m,n)] + q[(n,m)] <= 0
        ineq_constraints = []
        visited = set()
        for n in self.neighbors:
            for m in self.neighbors[n]:
                if (m, n) in visited:
                    continue
                visited.add((n, m))
                if (m, n) not in key_to_idx or (n, m) not in key_to_idx:
                    continue
                idx_mn = 2*self.N + key_to_idx[(m, n)]
                idx_nm = 2*self.N + key_to_idx[(n, m)]

                def make_ineq(i_mn, i_nm):
                    def ineq_fn(x):
                        return -(float(x[i_mn]) + float(x[i_nm]))
                    return ineq_fn

                ineq_constraints.append({'type': 'ineq', 'fun': make_ineq(idx_mn, idx_nm)})

        # Bounds
        bounds = (
            [(float(self.D_min[n]), float(self.D_max[n])) for n in range(self.N)]
        + [(float(self.G_min[n]), float(self.G_max[n])) for n in range(self.N)]
        + [(float('-inf'), float(self.k[key])) for key in q_keys]
        )

        # Minimization
        result = minimize(
            objective, x0.detach().cpu().numpy(), jac=grad_objective, method='SLSQP', bounds=bounds,
            constraints=eq_constraints + ineq_constraints,
            options={'ftol': 1e-12, 'maxiter': 2000}
        )

        if not result.success:
            raise ValueError(f"Proiezione globale non riuscita: {result.message}")

        x_proj = torch.tensor(result.x, dtype=torch.float32, device=self.device)
        D_proj = x_proj[:self.N]
        G_proj = x_proj[self.N:2*self.N]
        q_proj = {key: x_proj[2*self.N + i] for i, key in enumerate(q_keys)}
        
        self.D, self.G, self.q = D_proj, G_proj, q_proj
        self.x = self.flatten(self.D, self.G, self.q)

    def project_all_agents_global_approx(self, n_iter=30, lr=0.1):
        """
        Differentiable projection on bounds and approximate equality constraints.
        Pure torch, compatible with backward().
        
        Parameters
        ----------
        D, G: torch.Tensor shape (N,)
        q: dict of torch.Tensor scalars
        N: int
        neighbors: dict of lists
        G_res, D_min, D_max, G_min, G_max: torch.Tensor shape (N,)
        k: dict of upper bounds for q
        device: torch.device
        n_iter: number of iterative corrections for equality constraint
        
        Returns
        -------
        D_proj, G_proj: torch.Tensor shape (N,)
        q_proj: dict of torch.Tensor
        """
        
        # --- Clamp D e G ---
        D_proj = torch.clamp(self.D, self.D_min, self.D_max)
        G_proj = torch.clamp(self.G, self.G_min, self.G_max)

        # --- Prepara q come tensor leaf ---
        q_keys = list(self.q.keys())
        q_vec = torch.stack([self.q[key] for key in q_keys]).to(self.device).detach()
        q_vec.requires_grad_(True)

        # Limiti superiori q
        k_vec = torch.tensor([self.k[key] for key in q_keys], dtype=torch.float32, device=self.device)

        # --- Map dai key agli indici ---
        key_to_idx = {key: i for i, key in enumerate(q_keys)}

        # --- Loop iterativo per approssimare vincoli ---
        for _ in range(n_iter):
            # Calcola violazioni vincoli di uguaglianza
            eq_loss = 0.0
            for n in range(self.N):
                q_sum = sum(q_vec[key_to_idx[(m, n)]] for m in self.neighbors[n] if (m, n) in key_to_idx)
                eq_loss += (D_proj[n] - (G_proj[n] + self.G_res[n] + q_sum))**2

            # Gradient step
            grads = torch.autograd.grad(eq_loss, q_vec, create_graph=True)[0]
            q_vec = q_vec - lr * grads

            # Clamp differenziabile su bounds
            q_vec = torch.clamp(q_vec, -k_vec, k_vec)#torch.minimum(q_vec, k_vec)

        # --- Ricostruisci dizionario q ---
        q_proj = {key: q_vec[key_to_idx[key]] for key in q_keys}

        self.D, self.G, self.q = D_proj, G_proj, q_proj
        self.x = self.flatten(self.D, self.G, self.q)

    def __getstate__(self):
        return self.D, self.G,  self.q
    def __getflattenstate__(self):
        return self.x
    def __setstate__(self, D, G, q):
        self.D = D
        self.G = G
        self.q = q
        self.x = self.flatten(self.D, self.G, self.q)
    def __setflattenstate__(self, x):
        self.D, self.G, self.q = self.unflatten(x)
        self.x = x
    def getQ(self):
        Q=np.zeros(self.N)
        for n in range(self.N):
            Qn = sum(self.q[(m, n)].detach() for m in self.neighbors[n])
            Q[n]=Qn
        return Q

    def clone(self):
        new_net = EnergyNetwork(
            self.N, self.neighbors,
            self.D_min, self.D_max,
            self.G_min, self.G_max,
            self.G_res,
            self.a_tilde, self.a, self.b,
            self.D_star,
            self.c, self.k,
            self.device
        )
        new_net.__setstate__(*self.__getstate__())
        return new_net
