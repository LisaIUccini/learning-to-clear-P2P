import numpy as np
from scipy.optimize import minimize

"""Return F(x) = grad J"""
def F(D, G, N, a_tilde, a, b, D_star, c, neighbors):
    
    grad_D = np.zeros(N)
    grad_G = np.zeros(N)
    grad_q = {}

    for n in range(N):
        # derivative with respect to D_n
        grad_D[n] = 2 * a_tilde[n] * (D[n] - D_star[n])

        # derivative with respect to G_n
        grad_G[n] = a[n] * G[n] + b[n]

        # derivative with respect to q_{mn}
        for m in neighbors[n]:
            grad_q[(m, n)] = c[(m, n)]

    return grad_D, grad_G, grad_q


"""
Project D, G, q on:
- bounds constraint
- balancing constraint D_n = G_n + G_res_n + Qn
- constraint q_{mn} <= k_{mn}
- coupling constraint q[(m,n)] + q[(n,m)] <= 0
Formulated as a QP
"""
def project_all_agents_global(D, G, q, N, neighbors, G_res, D_min, D_max, G_min, G_max, k):

    q_keys = list(q.keys())
    key_to_idx = {key: i for i, key in enumerate(q_keys)}

    q_vals = np.array([float(q[key]) for key in q_keys], dtype=float)
    x0 = np.concatenate([D, G, q_vals])
    x_orig = x0.copy()

    # Objective: quadratic distance from the original point
    def objective(x):
        return 0.5 * float(np.dot(x - x_orig, x - x_orig))

    def grad_objective(x):
        return (x - x_orig).astype(float)

    # Equality constraint: for each n, D[n] = G[n] + G_res + sum(q[(m,n)])
    eq_constraints = []
    for n in range(N):
        q_idxs = [2*N + key_to_idx[(m, n)] for m in neighbors[n] if (m, n) in key_to_idx]

        def make_eq(n_, q_idxs_):
            def eq_fn(x):
                return float(x[n_]) - float(x[N + n_]) - float(G_res[n_]) - float(np.sum(x[q_idxs_]))
            return eq_fn

        eq_constraints.append({'type': 'eq', 'fun': make_eq(n, q_idxs)})

    # Inequality Coupling constraint: q[(m,n)] + q[(n,m)] <= 0 
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
                    # scipy wants >= 0, so we write -(q[(m,n)] + q[(n,m)]) >= 0
                    return -(float(x[i_mn]) + float(x[i_nm]))
                return ineq_fn

            ineq_constraints.append({'type': 'ineq', 'fun': make_ineq(idx_mn, idx_nm)})

    # Bounds constraint
    bounds = (
        [(float(D_min[n]), float(D_max[n])) for n in range(N)]
      + [(float(G_min[n]), float(G_max[n])) for n in range(N)]
      + [(float(-np.inf), float(k[key])) for key in q_keys]
    )

    # Optimization
    result = minimize(
        objective, x0, jac=grad_objective, method='SLSQP', bounds=bounds,
        constraints=eq_constraints + ineq_constraints,
        options={'ftol': 1e-12, 'maxiter': 2000}
    )

    if not result.success:
        raise ValueError(f"Proiezione globale non riuscita: {result.message}")

    x_proj = result.x
    D_proj = x_proj[:N]
    G_proj = x_proj[N:2*N]
    q_proj = {key: float(x_proj[2*N + i]) for i, key in enumerate(q_keys)}

    return D_proj, G_proj, q_proj