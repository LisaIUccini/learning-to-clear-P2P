import torch
import numpy as np
from energy_network import EnergyNetwork
import solveVI_dynamic_DR as solveVI
import matplotlib.pyplot as plt
from typing import Optional
import osqp
import scipy.sparse as sp

from plot_results import plot_fig, plot_convergence, plot_convergenceDR, check_and_report_v2


def make_osqp_solver(D_mat: torch.Tensor,
                     d_vec: torch.Tensor,
                     E_mat: torch.Tensor,
                     e_vec: torch.Tensor,
                     osqp_settings: Optional[dict] = None):
    """
    Costruisce e restituisce un qp_solver basato su OSQP da passare
    direttamente a douglas_rachford_avi(qp_solver=...).

    Il solver risolve ad ogni iterazione k del DR:

        min   0.5 y^T (H+M1) y  +  q_k^T y
        s.t.  D_mat y  ≤  d_vec
              E_mat y  =  e_vec

    Convertito nel formato OSQP:

        min   0.5 y^T P y  +  q^T y
        s.t.  l  ≤  A y  ≤  u

    con:
        P = H + M1
        A = vstack([D_mat, E_mat])
        l = [-inf, ..., -inf, e_vec]
        u = [d_vec,           e_vec]

    Parametri
    ----------
    D_mat, d_vec : vincoli disuguaglianza  D y ≤ d
    E_mat, e_vec : vincoli uguaglianza     E y = e
    osqp_settings : dict di settings OSQP (opzionale)
                    Default: eps_abs=1e-8, eps_rel=1e-8, warm_starting=True,
                             verbose=False, max_iter=4000

    Ritorna
    -------
    qp_solver : Callable
        Signature: qp_solver(H_plus_M1, q_k, project_C, u_init) -> y
        (project_C è ignorato — OSQP gestisce i vincoli direttamente)
    """

    # ── Converti in numpy/scipy sparse una volta sola ─────────────────
    D_np = D_mat.detach().cpu().numpy().astype(np.float64)
    d_np = d_vec.detach().cpu().numpy().astype(np.float64)
    E_np = E_mat.detach().cpu().numpy().astype(np.float64)
    e_np = e_vec.detach().cpu().numpy().astype(np.float64)

    n_ineq = D_np.shape[0]
    n_eq   = E_np.shape[0]

    # A = [D_mat; E_mat]  (stack verticale)
    A_np = np.vstack([D_np, E_np])
    A_sp = sp.csc_matrix(A_np)

    # Bounds:
    #   disuguaglianze:  -inf ≤ D y ≤ d
    #   uguaglianze:     e    ≤ E y ≤ e
    l_np = np.concatenate([-np.inf * np.ones(n_ineq), e_np])
    u_np = np.concatenate([d_np, e_np])

    # Settings OSQP di default
    default_settings = dict(
        eps_abs       = 1e-8,
        eps_rel       = 1e-8,
        warm_starting = True,
        verbose       = False,
        max_iter      = 4000,
        polish        = True,    # post-processing per soluzione più precisa
    )
    if osqp_settings:
        default_settings.update(osqp_settings)

    # Istanza OSQP — viene riusata tra le iterazioni DR (warm start)
    prob = osqp.OSQP()
    _initialized = [False]  # flag per primo setup

    def qp_solver(H_plus_M1:  torch.Tensor,
                  q_k:         torch.Tensor,
                  u_init:      Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Risolve il QP:
            min  0.5 y^T P y + q_k^T y
            s.t. D y ≤ d,  E y = e

        P = H_plus_M1 (cambia ad ogni chiamata se H cambia,
                       ma in DR è costante → aggiorniamo solo q_k).

        u_init : warm start (soluzione dell'iterazione precedente)
        """
        device = q_k.device
        dtype  = q_k.dtype

        P_np  = H_plus_M1.detach().cpu().numpy().astype(np.float64)
        q_np  = q_k.detach().cpu().numpy().astype(np.float64)
        P_sp  = sp.csc_matrix(np.triu(P_np))   # OSQP vuole triangolo superiore

        if not _initialized[0]:
            # Prima chiamata: setup completo
            prob.setup(P=P_sp, q=q_np,
                       A=A_sp, l=l_np, u=u_np,
                       **default_settings)
            _initialized[0] = True
        else:
            # Chiamate successive: aggiorna solo q_k (P è costante nel DR)
            prob.update(q=q_np)

        # Warm start con la soluzione dell'iterazione precedente
        if u_init is not None:
            x0 = u_init.detach().cpu().numpy().astype(np.float64)
            prob.warm_start(x=x0)

        # Risolvi
        sol = prob.solve()

        if sol.info.status not in ('solved', 'solved_inaccurate'):
            # fallback: restituisce warm start o zero
            if u_init is not None:
                return u_init.clone()
            return torch.zeros_like(q_k)

        y = torch.tensor(sol.x, dtype=dtype, device=device)
        return y

    return qp_solver




def solve_energy_game(net:        EnergyNetwork,
                      lam:        float = 0.5,
                      H:          Optional[torch.Tensor] = None,
                      max_iter:   int   = 200,
                      tol:        float = 1e-5,
                      osqp_settings: Optional[dict] = None,
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

    # ── Step 1: M, q0, D_mat, d_vec (disuguaglianze), E_mat, e_vec (bilanciamento) ─────────────────────────────────────────────────────
    M, q0, D_mat, d_vec, E_mat, e_vec = net.getDynamicParameters()

    # ── Step 4: elimina D_n → problema ridotto ────────────────────────────
    #M_r, q0_r, D_r, d_r, free_idx, dep_idx, S, s = eliminate_equalities(M, q0, D_mat, d_vec, E_mat, e_vec)

    # ── Step 5: π_C sul poliedro ridotto ──────────────────────────────────
    qp_solver = make_osqp_solver(D_mat, d_vec, E_mat, e_vec, osqp_settings)
    project_C = net.make_projection_full()

    # ── Step 6: punto iniziale u^0 ────────────────────────────────────────
    #free_idx = [1, 2, 3, 5, 6, 7, 9, 10, 11]   #← tutto tranne i D_n (0, 4, 8)
    #dep_idx  = [0, 4, 8]                        # ← solo i D_n
    u_full_init, offset, dim_n = net.get_u()
    #u_init = u_full_init[free_idx]

    # ── Step 7: Douglas-Rachford (eq. 33 del paper) ───────────────────────
    result = solveVI.douglas_rachford_avi(
        M         = M,
        q         = q0,
        project_C = project_C,
        H         = H,
        lam       = lam,
        max_iter  = max_iter,
        tol       = tol,
        u_init    = u_full_init,
        qp_solver  = qp_solver,   
        verbose   = verbose,
        log_every = 100,
    )

    # ── Step 8: ricostruisce u* completo  u = S @ u_free* + s ─────────────
    #u_full_star   = S @ result.u_star + s
    #result.u_star = u_full_star

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
        lam        = 0.75,
        max_iter   = 1000,
        tol        = 1e-5,
        verbose    = True,
    )

check_and_report_v2(n, result)

fig = plot_convergenceDR(result, title="3-node network")
fig.savefig("solveVIDR.png",
            dpi=150, bbox_inches='tight')
plt.close(fig)
Dl,Gl,ql = n.__getstate__()
plot_fig(N,Dl,Gl,ql,n.getQ(), G_res, k, "graph_visualization_VIDR.png")
print("\n  Plot salvato.")
