import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

N = 3 #numero di nodi
gamma = 0.1 #passo Forward-Backward
max_iter = 1000

neighbors = {
    0:[1,2],
    1:[0,2],
    2:[0,1]
}

#Vincoli su D e G
D_min = np.array([0.0, 0.0, 0.0])
D_max = np.array([10.0, 10.0, 10])
G_min = np.zeros(N)
G_max = np.array([0.0, 0.0, 0.0])

#generazioni fonti rinnovabili nota
G_res = np.array([0.0, 3.0, 5.0])

#utilità: U_n(D) = -a_tilde_n (D - D_star)^2
#coefficiente D
a_tilde = np.array([5.0, 15.0, 10.0])
#domanda ottima
D_star = np.array([6.0, 3.0, 3.0])

# costo generazione: C_n(G) = 0.5 a_n G^2 + b_n G
a = np.array([4.0, 1.0, 1.0])
b = np.array([0.0, 0.0, 0.0])

# costo di scambio c_{mn}
c = {}
# capacità k_{mn}
k = {}

for n in neighbors:
    for m in neighbors[n]:
        c[(m, n)] = 0.1*(-G_res[m])      # esempio
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

def project_local_node(n, D, G, q):
    """
    Proiezione locale sul nodo n:
    - vincoli box
    - vincolo di bilancio
    - vincoli q_{mn} <= k_{mn}
    """
    """
    metodo semplice che fa caare perchè si annullano tra di loro, che lessi
    
    # proiezione box su G
    G_n = np.clip(G[n], G_min[n], G_max[n])
    
    # proiezione su q_{mn} <= k_{mn}
    q_n = {}
    for m in neighbors[n]:
        q_n[(m, n)] = min(q[(m, n)], k[(m, n)])

    # ricostruzione D dal bilancio
    Q_n = sum(q_n[(m, n)] for m in neighbors[n])
    D_n = G_n + G_res[n] + Q_n

    # proiezione box su D
    D_n = np.clip(D_n, D_min[n], D_max[n])

    # ricalibrazione G per rispettare il bilancio
    G_n = D_n - G_res[n] - Q_n
    #G_n = np.clip(G_n, G_min[n], G_max[n])
    """

    neigh_n = neighbors[n]

    # Vettore variabili: x = [D[n], G[n], q[(m0,n)], q[(m1,n)], ...]
    q_orig = np.array([float(q[(m, n)]) for m in neigh_n], dtype=float)
    x0 = np.array([float(D[n]), float(G[n])] + q_orig.tolist(), dtype=float)
    x_orig = x0.copy()

    # Obiettivo: distanza quadratica dal punto originale
    def objective(x):
        return 0.5 * float(np.dot(x - x_orig, x - x_orig))

    def grad_objective(x):
        return (x - x_orig).astype(float)

    # Vincolo di uguaglianza: D[n] - G[n] - G_res - sum(q[(m,n)]) = 0
    def eq_constraint(x):
        return float(x[0]) - float(x[1]) - float(G_res[n]) - float(np.sum(x[2:]))
    
    # Bounds
    bounds = [(float(D_min[n]), float(D_max[n])), (float(G_min[n]), float(G_max[n]))] + \
             [(float(-np.inf), float(k[(m, n)])) for m in neigh_n]

    result = minimize(
        objective,
        x0,
        jac=grad_objective,
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': eq_constraint}],
        options={'ftol': 1e-12, 'maxiter': 1000}
    )

    if not result.success:
        raise ValueError(f"Proiezione non riuscita per n={n}: {result.message}")

    x_proj = result.x
    D_proj_n = float(x_proj[0])
    G_proj_n = float(x_proj[1])
    q_proj_n = {(m, n): float(x_proj[2 + i]) for i, m in enumerate(neigh_n)}

    return D_proj_n, G_proj_n, q_proj_n


def project_reciprocity(q):
    """
    Proiezione su q_{mn} + q_{nm} <= 0
    applicata edge-wise
    """
    q_proj = q.copy()
    visited = set()

    for (m, n) in q:
        if (n, m) in q and (n, m) not in visited:
            s = q[(m, n)] + q[(n, m)]
            if s > 0:
                correction = s / 2
                q_proj[(m, n)] -= correction
                q_proj[(n, m)] -= correction
            visited.add((m, n))
            visited.add((n, m))

    return q_proj


def forward_backward(D, G, q, gamma=0.1, max_iter=100, tol=1e-6):
    residuals = []
    for it in range(max_iter):
        print("\n---------ITER ",it,"---------")
        # --- FORWARD ---
        F_D, F_G, F_q = F(D, G, q)
        D_tmp = D - gamma * F_D
        G_tmp = G - gamma * F_G
        q_tmp = {key: q[key] - gamma * F_q[key] for key in q}

        # --- PROIEZIONE GLOBALE ---
        q_tmp = project_reciprocity(q_tmp)

        # --- PROIEZIONE LOCALE ---
        D_new = D.copy()
        G_new = G.copy()
        q_new = q_tmp.copy()

        
        for n in range(N):
            D_n, G_n, q_n = project_local_node(n, D_tmp, G_tmp, q_tmp)
            D_new[n] = D_n
            G_new[n] = G_n
            for m in neighbors[n]:
                q_new[(m, n)] = q_n[(m, n)]
        
        # criterio di arresto
        """
        err = np.linalg.norm(D_new - D) + np.linalg.norm(G_new - G)
        if err < tol:
            print(f"Convergenza raggiunta a iterazione {it}")
            break
        """
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
print("Risultati finali:")
print("D =", D)
print("G =", G)
print("q =\n", q)
for n in range(N):
    Qn = sum(q[(m, n)] for m in neighbors[n])
    print("Q",n,"=",Qn)
    print("balance ",n,"=",G[n]+G_res[n]+Qn-D[n])

plt.figure(figsize=(6,4))
plt.semilogy(residuals, marker='o')
plt.xlabel('Iterazione')
plt.ylabel('Residuo VI ||x - Pi_X(x - γF(x))||')
plt.title('Convergenza Forward-Backward')
plt.grid(True)
plt.show()