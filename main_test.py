import torch
import numpy as np
from energy_network import EnergyNetwork
import solveVI_dynamic_DR as solveVI
import matplotlib.pyplot as plt
from typing import Optional

from plot_results import plot_fig, plot_convergence, plot_convergenceDR

# ─────────────────────────────────────────────────────────────────────────────
# Eliminazione vincoli di uguaglianza  (sostituzione di D_n)
# ─────────────────────────────────────────────────────────────────────────────

def eliminate_equalities(M:     torch.Tensor,
                          q0:    torch.Tensor,
                          D_mat: torch.Tensor,
                          d_vec: torch.Tensor,
                          E_mat: torch.Tensor,
                          e_vec: torch.Tensor):
    """
    Elimina i vincoli di uguaglianza  E_mat @ u = e_vec  dal problema AVI,
    sostituendo le variabili dipendenti (D_n) con la loro espressione esplicita.

    Il vincolo di bilanciamento di ogni agente n è:
        D_n = ΔG_n + G_n + Σ_{m∈Ω_n} q_{mn}   →   D_n è dipendente

    Le variabili LIBERE sono: G_n e q_{mn} per ogni n, m.

    Costruisce la sostituzione affine:
        u = S @ u_free + s

    e restituisce il problema AVI ridotto alle sole variabili libere:
        AVI(C_r, M_r, q0_r)   con   C_r = { u_free : D_r @ u_free <= d_r }

    Parametri
    ----------
    M, q0        : matrici AVI originali
    D_mat, d_vec : vincoli disuguaglianza originali (reciprocità, capacità, box)
    E_mat, e_vec : vincoli uguaglianza (bilanciamento)

    Ritorna
    -------
    M_r, q0_r    : matrici AVI ridotte
    D_r, d_r     : vincoli disuguaglianza ridotti
    free_idx     : indici delle variabili libere in u
    dep_idx      : indici delle variabili dipendenti (D_n) in u
    S, s         : sostituzione  u = S @ u_free + s
    """
    n_tot  = M.shape[0]
    dtype  = M.dtype
    device = M.device

    # Ogni riga di E_mat ha esattamente un +1 — è la colonna di D_n
    dep_idx = []
    for i in range(E_mat.shape[0]):
        col = torch.where(E_mat[i] == 1.0)[0]
        dep_idx.append(col[0].item())

    free_idx = [j for j in range(n_tot) if j not in dep_idx]
    n_free   = len(free_idx)

    # Dalla condizione E_mat u = e_vec:
    #   E_dep @ u_dep + E_free @ u_free = e_vec
    #   u_dep = E_dep^{-1} (e_vec - E_free @ u_free)
    #         = P @ u_free + p
    E_dep_inv = torch.linalg.inv(E_mat[:, dep_idx])
    P = -E_dep_inv @ E_mat[:, free_idx]   # (N, n_free)
    p =  E_dep_inv @ e_vec                # (N,)

    # Matrice di sostituzione  u = S @ u_free + s
    S = torch.zeros(n_tot, n_free, dtype=dtype, device=device)
    s = torch.zeros(n_tot,         dtype=dtype, device=device)
    for k, j in enumerate(free_idx):
        S[j, k] = 1.0          # variabili libere: copiate direttamente
    for k, j in enumerate(dep_idx):
        S[j, :] = P[k, :]      # variabili dipendenti: espressione lineare
        s[j]    = p[k]

    # Problema ridotto (sostituendo u = S u_free + s ovunque)
    M_r  = S.T @ M @ S
    q0_r = S.T @ (M @ s + q0)
    D_r  = D_mat @ S
    d_r  = d_vec - D_mat @ s

    return M_r, q0_r, D_r, d_r, free_idx, dep_idx, S, s




def solve_energy_game(net:        EnergyNetwork,
                      lam:        float = 0.5,
                      H:          Optional[torch.Tensor] = None,
                      max_iter:   int   = 200,
                      tol:        float = 1e-6,
                      verbose:    bool  = True) -> solveVI.DRResult:
    """
    Risolve il Market Energy Game come AVI con Douglas-Rachford splitting.

    Parametri
    ----------
    net         : EnergyNetwork
    lam         : float   rilassamento λ ∈ (0,1]  (paper: 0.5)
    H           : torch.Tensor  precondizionatore (default: identità)
    max_iter    : int     iterazioni massime DR
    tol         : float   soglia su ‖r(u^k)‖
    warm_start  : bool    True = usa stato corrente di net come u^0
    verbose     : bool

    Ritorna
    -------
    result : DRResult
        .u_star    : torch.Tensor  soluzione u* completa (D_n inclusi)
        .residuals : list          storia ‖r(u^k)‖ (sul problema ridotto)
        .n_iter    : int
        .converged : bool
    """

    # ── Step 1: M, q0 ─────────────────────────────────────────────────────
    M, q0 = net.M, net.q0

    # ── Step 2: D_mat, d_vec (disuguaglianze) ─────────────────────────────
    D_mat, d_vec = net.D_mat, net.d_vec

    # ── Step 3: E_mat, e_vec (bilanciamento) ──────────────────────────────
    E_mat, e_vec = net.E_mat, net.e_vec

    # ── Step 4: elimina D_n → problema ridotto ────────────────────────────
    M_r, q0_r, D_r, d_r, free_idx, dep_idx, S, s = eliminate_equalities(M, q0, D_mat, d_vec, E_mat, e_vec)

    # ── Step 5: π_C sul poliedro ridotto ──────────────────────────────────
    project_C = net.make_projection1()

    # ── Step 6: punto iniziale u^0 ────────────────────────────────────────
    free_idx = [1, 2, 3, 5, 6, 7, 9, 10, 11]   #← tutto tranne i D_n (0, 4, 8)
    dep_idx  = [0, 4, 8]                        # ← solo i D_n
    u_full_init, offset, dim_n = net.get_u()
    u_init = u_full_init[free_idx]

    # ── Step 7: Douglas-Rachford (eq. 33 del paper) ───────────────────────
    result = solveVI.douglas_rachford_avi(
        M         = M_r,
        q         = q0_r,
        project_C = project_C,
        H         = H,
        lam       = lam,
        max_iter  = max_iter,
        tol       = tol,
        u_init    = u_init,
        verbose   = verbose,
        log_every = 100,
    )

    # ── Step 8: ricostruisce u* completo  u = S @ u_free* + s ─────────────
    u_full_star   = S @ result.u_star + s
    result.u_star = u_full_star

    # ── Step 9: scrive la soluzione nella rete ────────────────────────────
    net.set_from_u(result.u_star, offset, dim_n)

    return result


# ---------------------------------------------------------
#  Device & seeds
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

# =============== Initialize problem parameters ===============
N = 3 # number of prosumers
gamma = 0.1 # step Forward-Backward
max_iter = 200
tol=1e-6

# set of neighbors of each prosumer
neighbors = {
    0:[1,2],
    1:[0,2],
    2:[0,1]
}

# Box Constraints on D and G
D_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
D_max = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
G_min = torch.zeros(N, dtype=torch.float32)
G_max = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32)

# generations of renewable sources note
G_res = torch.tensor([0.0, 3.0, 5.0], dtype=torch.float32)

# coefficiets of Usage Benefit: U_n(D) = -a_tilde_n (D - D_star)^2
a_tilde = torch.tensor([5.0, 15.0, 10.0], dtype=torch.float32)
D_star  = torch.tensor([6.0, 3.0, 3.0], dtype=torch.float32)

# coefficiets of Flexibility Activation Cost: C_n(G) = 0.5 a_n G^2 + b_n G
a = torch.tensor([4.0, 4.0, 4.0], dtype=torch.float32)
b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

# price preferences c_{mn}
c = {}
# interconnection capacity constraint  k_{mn}
k = {}
for n in neighbors:
    for m in neighbors[n]:
        c[(m, n)] = torch.tensor(0.1, dtype=torch.float32)
        if m == 0:
            k[(m, n)] = torch.tensor(10.0, dtype=torch.float32)
        else:
            k[(m, n)] = torch.tensor(5.0, dtype=torch.float32)

n = EnergyNetwork(N, neighbors, D_min, D_max, G_min, G_max, G_res, a_tilde, a, b, D_star, c, k, device)


result = solve_energy_game(
        net        = n,
        lam        = 0.5,
        max_iter   = 1000,
        tol        = 1e-6,
        verbose    = True,
    )

fig = plot_convergenceDR(result, title="3-node network")
fig.savefig("solveVIDR.png",
            dpi=150, bbox_inches='tight')
plt.close(fig)
Dl,Gl,ql = n.__getstate__()
plot_fig(N,Dl,Gl,ql,n.getQ(), G_res, k, "graph_visualization_VIDR.png")
print("\n  Plot salvato.")
