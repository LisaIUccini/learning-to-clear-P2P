import numpy as np

from plot_results import plot_fig
from plot_results import plot_convergence

from utils import F
from utils import project_all_agents_global

"""""""""""""""""""""""""""""
Initialize problem parameters
"""""""""""""""""""""""""""""

N = 3 # number of prosumers
gamma = 0.1 # step Forward-Backward
max_iter = 100 

# set of neighbors of each prosumer
neighbors = {
    0:[1,2],
    1:[0,2],
    2:[0,1]
}

# Box Constraints on D and G
D_min = np.array([0.0, 0.0, 0.0])
D_max = np.array([10.0, 10.0, 10])
G_min = np.zeros(N)
G_max = np.array([10.0, 0.0, 0.0])

# generations of renewable sources note
G_res = np.array([0.0, 3.0, 5.0])

# coefficiets of Usage Benefit: U_n(D) = -a_tilde_n (D - D_star)^2
a_tilde = np.array([5.0, 15.0, 10.0])
D_star = np.array([6.0, 3.0, 3.0])

# coefficiets of Flexibility Activation Cost: C_n(G) = 0.5 a_n G^2 + b_n G
a = np.array([4.0, 4.0, 4.0])
b = np.array([0.0, 0.0, 0.0])

# price preferences c_{mn}
c = {}
# interconnection capacity constraint  k_{mn}
k = {}
for n in neighbors:
    for m in neighbors[n]:
        c[(m, n)] = 0.1   # esempio
        if m==0:
            k[(m, n)] = 10.0      # capacità fisica
        else:
            k[(m, n)] = 5.0      # capacità fisica


D = D_min + (D_max - D_min) * np.random.rand(N) # Demand: quantity of energy consumed at node n
G = G_min + (G_max - G_min) * np.random.rand(N) # Flexibility activation: quantity of energy produced from node n

# Bilateral trades: q[(m,n)] = quantity exchanged between node n and its neighbor m
# q[(m,n)] >= 0 node n buys from m
# q[(m,n)] <= 0 node n sells -q[(m,n)] to m
q = {}
for n in neighbors:
    for m in neighbors[n]:
        q[(m, n)] = 0 + (k[(m, n)] - 0)*np.random.rand()


"""""""""""""""""""""""""""""
Iterative function solves the fixed point problem with FBS (Forward-Backward Splitting) method
"""""""""""""""""""""""""""""
def forward_backward(D, G, q, gamma=0.1, max_iter=100, tol=1e-6):
    residuals = []
    for it in range(max_iter):
        
        # --- FORWARD ---
        F_D, F_G, F_q = F(D, G, N, a_tilde, a, b, D_star, c, neighbors)
        D_tmp = D - gamma * F_D
        G_tmp = G - gamma * F_G
        q_tmp = {key: q[key] - gamma * F_q[key] for key in q}


        # --- PROJECTION  ---
        D_new = D.copy()
        G_new = G.copy()
        q_new = q_tmp.copy()
        D_new, G_new, q_new = project_all_agents_global(D_tmp,G_tmp,q_tmp, N, neighbors, G_res, D_min, D_max, G_min, G_max, k)

        # Convergence Condition
        x_current = np.concatenate([D, G, np.array([q[k] for k in sorted(q.keys())])])
        x_proj = np.concatenate([D_new, G_new, np.array([q_new[k] for k in sorted(q_new.keys())])])
        err = np.linalg.norm(x_proj - x_current)
        residuals.append(err)
        if err < tol:
            print(f"Convergenza raggiunta a iterazione {it}")
            break

        D, G, q = D_new, G_new, q_new

    return D_new, G_new, q_new, residuals


"""""""""""""""""""""""""""""
MAIN
"""""""""""""""""""""""""""""

D, G, q, residuals = forward_backward(D, G, q, gamma, max_iter)


# final results plot
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

plot_fig(N,D,G,q,Q, G_res, k)
plot_convergence(residuals)
