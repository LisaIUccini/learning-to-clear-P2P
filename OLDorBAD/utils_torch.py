import numpy as np
from scipy.optimize import minimize
import cvxpy as cp
#from cvxpylayers.torch import CvxpyLayer
import torch

def initialize_DGq_torch(D_min, D_max, G_min, G_max, neighbors, k, N):
    D = torch.zeros(N, dtype=torch.float32) #D_min + (D_max - D_min) * torch.rand(N)
    G = torch.zeros(N, dtype=torch.float32) #G_min + (G_max - G_min) * torch.rand(N)

    # Bilateral trades: q[(m,n)] = quantity exchanged between node n and its neighbor m
    # q[(m,n)] >= 0 node n buys from m
    # q[(m,n)] <= 0 node n sells -q[(m,n)] to m
    q = {}
    for n in neighbors:
        for m in neighbors[n]:
            q[(m, n)] = 0.0 #torch.rand(1) * k[(m, n)]  # shape (1,) tensor

    return D, G, q

def cost(N, D, G, q, a_tilde, a, b, D_star, c, neighbors):
    """
    Compute the cost function F(D,G,q) for all agents.

    Parameters:
    - D, G: torch tensors, shape (N,)
    - q: dict with keys (m,n), values torch tensors
    - a_tilde, a, b: torch tensors, shape (N,)
    - c: dict of torch tensors for edges
    - D_star: torch tensor, shape (N,)
    - neighbors: dict of neighbor lists

    Returns:
    - cost: torch tensor scalar
    """
    cost = 0.0

    for n in range(N):
        # Cost for D and G
        cost = cost + a_tilde[n] * (D[n] - D_star[n])**2 + 0.5 * a[n] * G[n]**2 + b[n] * G[n]

        # Cost for q
        for m in neighbors[n]:
            if (m, n) in q:
                cost = cost + c[(n, m)] * q[(m, n)]

    return cost

def F_torch(D, G, N, a_tilde, a, b, D_star, c, neighbors, device):
    """
    Compute gradients w.r.t. D, G, q using PyTorch tensors.

    Parameters:
    - D, G: torch tensors, shape (N,)
    - neighbors: dict of neighbor lists
    - N: int, number of nodes
    - a_tilde, a, b: torch tensors, shape (N,)
    - c: dict of torch tensors for edges
    - D_star: torch tensor, shape (N,)
    - device: optional, torch.device

    Returns:
    - grad_D, grad_G: torch tensors, shape (N,)
    - grad_q: dict with keys (m,n), values torch tensors
    """
    grad_D = torch.zeros(N, dtype=torch.float32, device=device)
    grad_G = torch.zeros(N, dtype=torch.float32, device=device)
    grad_q = {}

    for n in range(N):
        # derivative w.r.t D_n
        grad_D[n] = 2 * a_tilde[n] * (D[n] - D_star[n])
        
        # derivative w.r.t G_n
        grad_G[n] = a[n] * G[n] + b[n]

        # derivative w.r.t q_{mn}
        for m in neighbors[n]:
            grad_q[(m, n)] = c[(m, n)]

    return grad_D, grad_G, grad_q


def projection_loss_torch(
    x, x_orig,
    N, neighbors,
    G_res,
    D_min, D_max,
    G_min, G_max,
    k,
    r=1000.0
):
    """
    Loss differenziabile per la proiezione globale.
    Tutto in PyTorch, compatibile con backward().
    """

    # =========================
    # Base objective
    # =========================
    loss = 100000.0 *torch.sum((x - x_orig) ** 2)

    # =========================
    # Unflatten
    # =========================
    D, G, q = unflatten_state(x, N, neighbors)

    # =========================
    # Equality constraints
    # D[n] = G[n] + G_res[n] + sum_m q[(m,n)]
    # =========================
    for n in range(N):
        flow_sum = torch.zeros((), device=x.device)
        for m in neighbors[n]:
            if (m, n) in q:
                flow_sum = flow_sum + q[(m, n)]

        eq_violation = D[n] - G[n] - G_res[n] - flow_sum
        loss = loss +  eq_violation ** 2

    # =========================
    # Inequality coupling
    # q[(m,n)] + q[(n,m)] <= 0
    # =========================
    visited = set()
    for n in neighbors:
        for m in neighbors[n]:
            if (m, n) in visited:
                continue
            visited.add((n, m))

            if (m, n) in q and (n, m) in q:
                viol = torch.relu(q[(m, n)] + q[(n, m)])
                loss = loss +   viol ** 2

    # =========================
    # (Optional) Soft box penalties
    # =========================
    # Se vuoi rendere i box constraints soft (opzionale)
    loss += torch.sum(torch.relu(D_min - D) ** 2)
    loss += torch.sum(torch.relu(D - D_max) ** 2)
    loss += torch.sum(torch.relu(G_min - G) ** 2)
    loss += torch.sum(torch.relu(G - G_max) ** 2)
    for key in q:
        loss += 10.0 * torch.sum(torch.relu(q[key] - k[key]) ** 2)

    return loss

def build_edge_list(neighbors):
        """
        Build a fixed, ordered list of directed edges (m, n)
        such that m in neighbors[n].
        """
        edge_list = []
        for n in sorted(neighbors.keys()):
            for m in sorted(neighbors[n]):
                edge_list.append((m, n))
        return edge_list
    
def flatten_state(D, G, q, neighbors):
    """
    Flatten (D, G, q) into a single 1D tensor.
    Works both if D,G are tensor or np.array, q contains floats or tensors.
    """

    # Converti D e G in tensori se non lo sono
    if not isinstance(D, torch.Tensor):
        D = torch.tensor(D, dtype=torch.float32)
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G, dtype=torch.float32)

    edge_list = build_edge_list(neighbors)

    # Assicurati che ogni q[e] sia uno scalare tensor
    q_vec = torch.tensor([q[e].item() if isinstance(q[e], torch.Tensor) else q[e] for e in edge_list],
                         dtype=torch.float32)

    # Concatenate in 1D
    x = torch.cat([D.flatten(), G.flatten(), q_vec.flatten()])

    return x

def unflatten_state(x, N, neighbors):
    """
    Unflatten vector x into (D, G, q).

    Parameters
    ----------
    x : np.ndarray, shape (2N + E,)
    N : int
    edge_list : list of tuples (m,n)

    Returns
    -------
    D : np.ndarray, shape (N,)
    G : np.ndarray, shape (N,)
    q : dict with keys (m,n)
    """
    edge_list = build_edge_list(neighbors)

    D = x[:N]
    G = x[N:2*N]

    q_vec = x[2*N:]
    q = {edge_list[i]: q_vec[i] for i in range(len(edge_list))}

    return D, G, q


def project_all_agents_global_torch(D, G, q, N, neighbors, G_res, D_min, D_max, G_min, G_max, k, device):
    q_keys = list(q.keys())
    key_to_idx = {key: i for i, key in enumerate(q_keys)}

    # Convert q dict into tensor vector
    q_vals = torch.stack([q[key].to(device) if isinstance(q[key], torch.Tensor) else torch.tensor(q[key], device=device, dtype=torch.float32)
                          for key in q_keys]).flatten()
    
    # Concatenate into a single tensor
    x0 = torch.cat([D, G, q_vals])
    x_orig = x0.clone()

    # Objective function (torch tensor, ma scipy richiede numpy)
    def objective(x):
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        val = 0.5 * torch.sum((x_t - x_orig)**2)
        return val.item()  # scipy richiede float

    def grad_objective(x):
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        grad = (x_t - x_orig).detach().cpu().numpy().astype(float)  # scipy richiede numpy
        return grad

    # Equality constraints: D[n] = G[n] + G_res[n] + sum(q_mn)
    eq_constraints = []
    for n in range(N):
        q_idxs = [2*N + key_to_idx[(m, n)] for m in neighbors[n] if (m, n) in key_to_idx]

        def make_eq(n_, q_idxs_):
            def eq_fn(x):
                return float(x[n_]) - float(x[N + n_]) - float(G_res[n_]) - float(sum(x[q_idxs_]))
            return eq_fn

        eq_constraints.append({'type': 'eq', 'fun': make_eq(n, q_idxs)})

    # Inequality constraints: q[(m,n)] + q[(n,m)] <= 0
    ineq_constraints = []
    visited = set()
    for n in neighbors:
        for m in neighbors[n]:
            if (m, n) in visited:
                continue
            visited.add((n, m))
            if (m, n) not in key_to_idx or (n, m) not in key_to_idx:
                continue
            idx_mn = 2*N + key_to_idx[(m, n)]
            idx_nm = 2*N + key_to_idx[(n, m)]

            def make_ineq(i_mn, i_nm):
                def ineq_fn(x):
                    return -(float(x[i_mn]) + float(x[i_nm]))
                return ineq_fn

            ineq_constraints.append({'type': 'ineq', 'fun': make_ineq(idx_mn, idx_nm)})

    # Bounds
    bounds = (
        [(float(D_min[n]), float(D_max[n])) for n in range(N)]
      + [(float(G_min[n]), float(G_max[n])) for n in range(N)]
      + [(float('-inf'), float(k[key])) for key in q_keys]
    )

    # Minimization
    result = minimize(
        objective, x0.detach().cpu().numpy(), jac=grad_objective, method='SLSQP', bounds=bounds,
        constraints=eq_constraints + ineq_constraints,
        options={'ftol': 1e-12, 'maxiter': 2000}
    )

    if not result.success:
        raise ValueError(f"Proiezione globale non riuscita: {result.message}")

    x_proj = torch.tensor(result.x, dtype=torch.float32, device=device)
    D_proj = x_proj[:N]
    G_proj = x_proj[N:2*N]
    q_proj = {key: x_proj[2*N + i] for i, key in enumerate(q_keys)}

    return D_proj, G_proj, q_proj

def project_all_agents_global_approx(D, G, q, N, neighbors, G_res, D_min, D_max, G_min, G_max, k, device, n_iter=30, lr=0.1):
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
    # --- Assicurati che i bounds siano tensor su device ---
    D_min = D_min.to(device) if isinstance(D_min, torch.Tensor) else torch.tensor(D_min, dtype=torch.float32, device=device)
    D_max = D_max.to(device) if isinstance(D_max, torch.Tensor) else torch.tensor(D_max, dtype=torch.float32, device=device)
    G_min = G_min.to(device) if isinstance(G_min, torch.Tensor) else torch.tensor(G_min, dtype=torch.float32, device=device)
    G_max = G_max.to(device) if isinstance(G_max, torch.Tensor) else torch.tensor(G_max, dtype=torch.float32, device=device)
    G_res  = G_res.to(device)  if isinstance(G_res,  torch.Tensor) else torch.tensor(G_res, dtype=torch.float32, device=device)

    # --- Clamp D e G ---
    D_proj = torch.clamp(D, D_min, D_max)
    G_proj = torch.clamp(G, G_min, G_max)

    # --- Prepara q come tensor leaf ---
    q_keys = list(q.keys())
    q_vec = torch.stack([q[key] for key in q_keys]).to(device).detach()
    q_vec.requires_grad_(True)

    # Limiti superiori q
    k_vec = torch.tensor([k[key] for key in q_keys], dtype=torch.float32, device=device)

    # --- Map dai key agli indici ---
    key_to_idx = {key: i for i, key in enumerate(q_keys)}

    # --- Loop iterativo per approssimare vincoli ---
    for _ in range(n_iter):
        # Calcola violazioni vincoli di uguaglianza
        eq_loss = 0.0
        for n in range(N):
            q_sum = sum(q_vec[key_to_idx[(m, n)]] for m in neighbors[n] if (m, n) in key_to_idx)
            eq_loss += (D_proj[n] - (G_proj[n] + G_res[n] + q_sum))**2

        # Gradient step
        grads = torch.autograd.grad(eq_loss, q_vec, create_graph=True)[0]
        q_vec = q_vec - lr * grads

        # Clamp differenziabile su bounds
        q_vec = torch.clamp(q_vec, -k_vec, k_vec)#torch.minimum(q_vec, k_vec)

    # --- Ricostruisci dizionario q ---
    q_proj = {key: q_vec[key_to_idx[key]] for key in q_keys}

    return D_proj, G_proj, q_proj
