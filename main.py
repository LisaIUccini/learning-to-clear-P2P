import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from plot_results import plot_fig
from plot_results import plot_convergence

N = 3 #numero di nodi
gamma = 0.1 #passo Forward-Backward
max_iter = 100

neighbors = {
    0:[1,2],
    1:[0,2],
    2:[0,1]
}

#Vincoli su D e G
D_min = np.array([0.0, 0.0, 0.0])
D_max = np.array([10.0, 10.0, 10])
G_min = np.zeros(N)
G_max = np.array([10.0, 0.0, 0.0])

#generazioni fonti rinnovabili nota
G_res = np.array([0.0, 3.0, 5.0])

#utilità: U_n(D) = -a_tilde_n (D - D_star)^2
#coefficiente D
a_tilde = np.array([5.0, 15.0, 10.0])
#domanda ottima
D_star = np.array([6.0, 3.0, 3.0])

# costo generazione: C_n(G) = 0.5 a_n G^2 + b_n G
a = np.array([4.0, 4.0, 4.0])
b = np.array([0.0, 0.0, 0.0])

# costo di scambio c_{mn}
c = {}
# capacità k_{mn}
k = {}

for n in neighbors:
    for m in neighbors[n]:
        c[(m, n)] = 0.1   # esempio
        if m==0:
            k[(m, n)] = 10.0      # capacità fisica
        else:
            k[(m, n)] = 5.0      # capacità fisica

# Inizializzazione
D = np.random.rand(N)
G = np.random.rand(N)

# q[(m,n)] = flusso da m a n
q = {}
for n in neighbors:
    for m in neighbors[n]:
        q[(m, n)] = 0.0



def F(D, G, q):
    """Calcola F(x) = grad J"""
    grad_D = np.zeros(N)
    grad_G = np.zeros(N)
    grad_q = {}

    for n in range(N):
        # derivata rispetto a D_n
        grad_D[n] = 2 * a_tilde[n] * (D[n] - D_star[n])

        # derivata rispetto a G_n
        grad_G[n] = a[n] * G[n] + b[n]

        # derivata rispetto a q_{mn}
        for m in neighbors[n]:
            grad_q[(m, n)] = c[(m, n)]

    return grad_D, grad_G, grad_q

def project_all_agents_global(D, G, q):
    """
    Proiezione locale sul nodo n:
    - vincoli box
    - vincolo di bilancio
    - vincoli q_{mn} <= k_{mn}
    - q[(m,n)] + q[(n,m)] <= 0
    """

    q_keys = list(q.keys())
    key_to_idx = {key: i for i, key in enumerate(q_keys)}
    Q_len = len(q_keys)

    q_vals = np.array([float(q[key]) for key in q_keys], dtype=float)
    x0 = np.concatenate([D, G, q_vals])
    x_orig = x0.copy()

    # Obiettivo: distanza quadratica dal punto originale
    def objective(x):
        return 0.5 * float(np.dot(x - x_orig, x - x_orig))

    def grad_objective(x):
        return (x - x_orig).astype(float)

    # ── Vincoli di uguaglianza: per ogni n, D[n] = G[n] + G_res + sum(q[(m,n)]) ──
    eq_constraints = []
    for n in range(N):
        q_idxs = [2*N + key_to_idx[(m, n)] for m in neighbors[n] if (m, n) in key_to_idx]

        def make_eq(n_, q_idxs_):
            def eq_fn(x):
                return float(x[n_]) - float(x[N + n_]) - float(G_res[n_]) - float(np.sum(x[q_idxs_]))
            return eq_fn

        eq_constraints.append({'type': 'eq', 'fun': make_eq(n, q_idxs)})

    # ── Vincoli di disuguaglianza: q[(m,n)] + q[(n,m)] <= 0 ─────────────────
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
                    # scipy vuole >= 0, quindi scriviamo -(q[(m,n)] + q[(n,m)]) >= 0
                    return -(float(x[i_mn]) + float(x[i_nm]))
                return ineq_fn

            ineq_constraints.append({'type': 'ineq', 'fun': make_ineq(idx_mn, idx_nm)})

    # ── Bounds ────────────────────────────────────────────────────────────────
    bounds = (
        [(float(D_min[n]), float(D_max[n])) for n in range(N)]
      + [(float(G_min[n]), float(G_max[n])) for n in range(N)]
      + [(float(-np.inf), float(k[key])) for key in q_keys]
    )

    # ── Ottimizzazione ────────────────────────────────────────────────────────
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

def forward_backward(D, G, q, gamma=0.1, max_iter=100, tol=1e-6):
    residuals = []
    for it in range(max_iter):
        # --- FORWARD ---
        F_D, F_G, F_q = F(D, G, q)
        D_tmp = D - gamma * F_D
        G_tmp = G - gamma * F_G
        q_tmp = {key: q[key] - gamma * F_q[key] for key in q}


        # --- PROIEZIONE  ---
        D_new = D.copy()
        G_new = G.copy()
        q_new = q_tmp.copy()

        D_new, G_new, q_new = project_all_agents_global(D_tmp,G_tmp,q_tmp)

        # criterio di arresto
        x_current = np.concatenate([D, G, np.array([q[k] for k in sorted(q.keys())])])
        x_proj = np.concatenate([D_new, G_new, np.array([q_new[k] for k in sorted(q_new.keys())])])
        err = np.linalg.norm(x_proj - x_current)
        residuals.append(err)
        if err < tol:
            print(f"Convergenza raggiunta a iterazione {it}")
            break


        D, G, q = D_new, G_new, q_new

    return D_new, G_new, q_new, residuals


D, G, q, residuals = forward_backward(D, G, q, gamma, max_iter)

# -----------------------------
# Risultati finali
# -----------------------------
Q = np.zeros(N)
print("Risultati finali:")
print("D =", D)
print("G =", G)
print("q =\n", q)
for n in range(N):
    Qn = sum(q[(m, n)] for m in neighbors[n])
    Q[n]=Qn
    print("Q",n,"=",Qn)
    print("balance ",n,"=",G[n]+G_res[n]+Qn-D[n])

plot_fig(N,D,G,q,Q, G_res)
plot_convergence(residuals)
