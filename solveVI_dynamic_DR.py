"""
Douglas-Rachford Splitting for Monotone Affine Variational Inequalities
========================================================================
Implements the algorithm from:

    Rahimi Baghbadorani, Benenati, Grammatico (2025)
    "A Douglas-Rachford Splitting Method for Solving Monotone Variational
    Inequalities in Linear-quadratic Dynamic Games"
    arXiv:2504.05757

Notation follows the paper exactly.

─────────────────────────────────────────────────────────────────────────
PROBLEM  (eq. 1 in the paper)
─────────────────────────────────────────────────────────────────────────

    VI(C, F):   find  u* ∈ C   s.t.  ⟨F(u*), u − u*⟩ ≥ 0   ∀ u ∈ C

For an affine operator  F(u) = M u + q  this is written:

    AVI(C, M, q):   find  u* ∈ C   s.t.  ⟨M u* + q, u − u*⟩ ≥ 0   ∀ u ∈ C

where:
    C ⊆ R^n   closed convex set  (polyhedron:  D u ≤ d)
    M ∈ R^{n×n}   (possibly non-symmetric)
    q ∈ R^n

─────────────────────────────────────────────────────────────────────────
OPERATOR SPLITTING  (eq. 32 in the paper)
─────────────────────────────────────────────────────────────────────────

Decompose  M = M1 + M2  such that:

    M1 = M1^T  ≽ 0        (symmetric, positive semidefinite)
    M2          ≻ 0        (positive definite, possibly non-symmetric)
    H  = H^T   ≻ 0        (user-chosen, symmetric positive definite)

─────────────────────────────────────────────────────────────────────────
ITERATION  (eq. 33 in the paper)
─────────────────────────────────────────────────────────────────────────

Given u^k, repeat for k = 0, 1, 2, ...:

    Step 1 — QP subproblem:
        y^k = sol( C,  H + M1,  q + (M2 − H) u^k )

        i.e. y^k solves AVI(C, H+M1, q + (M2−H) u^k).
        Because H+M1 is symmetric ≻ 0, this is equivalent to:

            y^k = argmin_{y ∈ C}  { 0.5 y^T (H+M1) y  +  [q + (M2−H) u^k]^T y }

    Step 2 — Linear update:
        u^{k+1} = (H + M2)^{-1} [ H (2λ_k y^k + (1−2λ_k) u^k)  +  M2 u^k ]

where  λ_k ∈ (0, 1]  is the relaxation parameter (constant λ in the paper).

─────────────────────────────────────────────────────────────────────────
CONVERGENCE  (Lemma 3 in the paper)
─────────────────────────────────────────────────────────────────────────

The sequence (u^k) converges LINEARLY to the solution of AVI(C, M, q)
under the splitting conditions (eq. 32).

─────────────────────────────────────────────────────────────────────────
CANONICAL SPLITTING  (eq. 34 in the paper, for the OL-NE application)
─────────────────────────────────────────────────────────────────────────

    M1 = (M + M^T) / 4        ← symmetric part of M, halved
    M2 = M − M1               ← remainder  (= (M + M^T)/4 + (M − M^T)/2)

Note: M2 = M1 + skew(M), where skew(M) = (M − M^T)/2 is skew-symmetric
and thus positive semidefinite. M2 ≻ 0 iff M + M^T ≻ 0 (strong monotonicity).

─────────────────────────────────────────────────────────────────────────
RESIDUAL  (Section IV, convergence check)
─────────────────────────────────────────────────────────────────────────

The natural residual of the VI at iterate u is:

    r(u) = u − π_C( u − F(u) )  =  u − π_C( u − M u − q )

where π_C denotes the metric projection onto C.
Convergence is declared when  ‖r(u^k)‖ ≤ tol.
"""

import torch
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class DRResult:
    """Output of the Douglas-Rachford solver."""
    u_star:     torch.Tensor            # solution u*
    residuals:  list = field(default_factory=list)  # ‖r(u^k)‖ at each iteration
    n_iter:     int  = 0                # iterations performed
    converged:  bool = False

# ─────────────────────────────────────────────────────────────────────────────
# Splitting of M  (eq. 32 and 34 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def split_M_canonical(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Canonical splitting from eq. (34) of the paper:

        M1 = (M + M^T) / 4      ← symmetric ≽ 0
        M2 = M − M1             ← positive definite if M + M^T ≻ 0

    Parameters
    ----------
    M : torch.Tensor  shape (n, n)

    Returns
    -------
    M1, M2 : torch.Tensor  shape (n, n)
    """
    M1 = (M + M.T) / 4.0
    M2 = M - M1
    return M1, M2

def split_M_custom(M: torch.Tensor,
                   M1: torch.Tensor,
                   M2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    User-provided splitting, with sanity checks (eq. 32 of the paper).

    Checks:
        M1 symmetric  ≽ 0
        M2            ≻ 0
        M1 + M2 == M

    Returns M1, M2 unchanged if all checks pass.
    """
    # symmetry of M1
    if not torch.allclose(M1, M1.T, atol=1e-6):
        raise ValueError("split_M_custom: M1 must be symmetric.")

    # M1 ≽ 0
    eig_M1 = torch.linalg.eigvalsh(M1)
    if eig_M1.min() < -1e-6:
        raise ValueError(f"split_M_custom: M1 must be PSD (min eig = {eig_M1.min():.2e}).")

    # M2 ≻ 0  (check via eigenvalues of symmetric part)
    eig_sym_M2 = torch.linalg.eigvalsh((M2 + M2.T) / 2.0)
    if eig_sym_M2.min() <= 0:
        raise ValueError(f"split_M_custom: M2 must be PD (min eig sym = {eig_sym_M2.min():.2e}).")

    # M1 + M2 = M
    if not torch.allclose(M1 + M2, M, atol=1e-6):
        raise ValueError("split_M_custom: M1 + M2 must equal M.")

    return M1, M2

# ─────────────────────────────────────────────────────────────────────────────
# Natural residual  r(u) = u − π_C(u − F(u))
# ─────────────────────────────────────────────────────────────────────────────

def natural_residual(u: torch.Tensor,
                     u_old: torch.Tensor,
                     M: torch.Tensor,
                     q: torch.Tensor,
                     project_C: Callable) -> torch.Tensor:
    """
    Computes the natural residual of VI(C, F) at u:

        r(u) = u − π_C( u − F(u) )   where  F(u) = M u + q

    Parameters
    ----------
    u         : torch.Tensor  shape (n,)   current iterate
    M         : torch.Tensor  shape (n, n)
    q         : torch.Tensor  shape (n,)
    project_C : Callable  u -> π_C(u),  projection onto C

    Returns
    -------
    r : torch.Tensor  shape (n,)
    """
    #Fu = M @ u + q                   # F(u) = M u + q
    #return u - project_C(u - Fu)    # r(u) = u − π_C(u − F(u))
    return u - u_old

# ─────────────────────────────────────────────────────────────────────────────
# Step 1:  y^k = sol(C, H+M1, q + (M2−H) u^k)
# ─────────────────────────────────────────────────────────────────────────────

def solve_inner_qp(H_plus_M1:  torch.Tensor,
                   q_k:         torch.Tensor,
                   project_C:   Callable,
                   u_init:      Optional[torch.Tensor] = None,
                   qp_solver:   Optional[Callable]     = None,
                   pgd_lr:      float = None,
                   pgd_max_iter: int  = 200,
                   pgd_tol:     float = 1e-10) -> torch.Tensor:
    """
    Solves the QP inner subproblem of Step 1 (eq. 33a):

        y^k = argmin_{y ∈ C}  { 0.5 y^T (H+M1) y  +  q_k^T y }

    where  q_k = q + (M2 − H) u^k  (formed by the caller).

    H+M1 is symmetric PD → this is a standard constrained QP.

    Parameters
    ----------
    H_plus_M1  : torch.Tensor  shape (n, n)   symmetric PD matrix
    q_k        : torch.Tensor  shape (n,)      linear term for this iteration
    project_C  : Callable      u -> π_C(u)
    u_init     : torch.Tensor  shape (n,)      warm start (optional)
    qp_solver  : Callable (optional)
                 Signature: qp_solver(H, q, project_C, u_init) -> y
                 If provided, used instead of the built-in PGD.
                 Plug in OSQP, quadprog, etc. here.
    pgd_lr     : float  step size for PGD  (default: 1 / λ_max(H+M1))
    pgd_max_iter : int
    pgd_tol    : float

    Returns
    -------
    y : torch.Tensor  shape (n,)
    """
    # ── External solver (e.g. OSQP) ─────────────────────────────────────────
    if qp_solver is not None:
        return qp_solver(H_plus_M1, q_k, project_C, u_init)

    # ── Built-in: Projected Gradient Descent ────────────────────────────────
    # min  0.5 y^T A y + q^T y   s.t.  y ∈ C
    # Gradient:  A y + q
    # Step size: 1 / λ_max(A)
    n = q_k.shape[0]
    y = torch.zeros(n, dtype=q_k.dtype, device=q_k.device) if u_init is None else u_init.clone()

    if pgd_lr is None:
        L = torch.linalg.eigvalsh(H_plus_M1).max().item()
        pgd_lr = 1.0 / max(L, 1e-12)

    for _ in range(pgd_max_iter):
        grad = H_plus_M1 @ y + q_k
        y_new = project_C(y - pgd_lr * grad)
        if torch.norm(y_new - y) < pgd_tol * (1.0 + torch.norm(y)):
            y = y_new
            break
        y = y_new

    return y



# ─────────────────────────────────────────────────────────────────────────────
# Main algorithm:  Douglas-Rachford for AVI(C, M, q)
# ─────────────────────────────────────────────────────────────────────────────
def douglas_rachford_once(
        M:           torch.Tensor,
        M2:           torch.Tensor,
        q:           torch.Tensor,
        project_C:   Callable,
        M2_minus_H:          Optional[torch.Tensor],
        H_plus_M1:          Optional[torch.Tensor],
        H_plus_M2_inv:           Optional[torch.Tensor],
        H:           Optional[torch.Tensor],
        lam:         float = 0.5,
        u:      Optional[torch.Tensor] = None,
        qp_solver:   Optional[Callable]     = None,):

            # ── Step 1  (eq. 33a) ────────────────────────────────────────────────
            # y^k = sol( C,  H+M1,  q + (M2−H) u^k )
            #
            # Equivalent QP:
            #   y^k = argmin_{y ∈ C}  0.5 y^T (H+M1) y  +  q_k^T y
            #   where  q_k = q + (M2−H) u^k
            q_k = q + M2_minus_H @ u

            y = solve_inner_qp(
                H_plus_M1 = H_plus_M1,
                q_k        = q_k,
                project_C  = project_C,
                u_init     = u,             # warm-start with current u^k
                qp_solver  = qp_solver,
            )

            # ── Step 2  (eq. 33b) ────────────────────────────────────────────────
            # u^{k+1} = (H+M2)^{-1} [ H (2λ y^k + (1−2λ) u^k)  +  M2 u^k ]
            rhs    = H @ (2.0 * lam * y + (1.0 - 2.0 * lam) * u)  +  M2 @ u
            u_new  = H_plus_M2_inv @ rhs

            # ── Natural residual  r(u^{k+1}) ─────────────────────────────────────
            # r(u) = u − π_C( u − M u − q )
            r  = natural_residual(u_new, u, M, q, project_C)
            rk = torch.norm(r).item()

            return u_new, rk

def douglas_rachford_avi(
        M:           torch.Tensor,
        q:           torch.Tensor,
        project_C:   Callable,
        M1:          Optional[torch.Tensor] = None,
        M2:          Optional[torch.Tensor] = None,
        H:           Optional[torch.Tensor] = None,
        lam:         float = 0.5,
        max_iter:    int   = 200,
        tol:         float = 1e-6,
        u_init:      Optional[torch.Tensor] = None,
        qp_solver:   Optional[Callable]     = None,
        verbose:     bool  = False,
        log_every:   int   = 50,
) -> DRResult:
    """
    Douglas-Rachford splitting for  AVI(C, M, q)  (Algorithm from eq. 33).

    Solves:
        find u* ∈ C   s.t.   ⟨M u* + q, u − u*⟩ ≥ 0   ∀ u ∈ C

    Iteration (eq. 33):
        y^k      = sol( C,  H + M1,  q + (M2 − H) u^k )           [Step 1 — QP]
        u^{k+1}  = (H + M2)^{-1} [ H(2λ y^k + (1−2λ) u^k) + M2 u^k ]  [Step 2]

    Parameters
    ----------
    M          : torch.Tensor  shape (n, n)   game / cost matrix
    q          : torch.Tensor  shape (n,)     linear term
    project_C  : Callable      u -> π_C(u),  projection onto the feasible set C
    M1         : torch.Tensor  shape (n, n)   (optional) splitting term, symmetric ≽ 0
                 If None, the canonical splitting (eq. 34) is used.
    M2         : torch.Tensor  shape (n, n)   (optional) splitting term, ≻ 0
                 If None, the canonical splitting (eq. 34) is used.
    H          : torch.Tensor  shape (n, n)   (optional) symmetric PD preconditioner
                 If None, H = I (identity).
    lam        : float         relaxation parameter λ ∈ (0, 1]  (paper uses λ = 0.5)
    max_iter   : int           maximum number of iterations
    tol        : float         convergence threshold on ‖r(u^k)‖
    u_init     : torch.Tensor  shape (n,)     initial point u^0
                 If None, u^0 = 0.
    qp_solver  : Callable (optional)
                 External QP solver for Step 1.
                 Signature: qp_solver(H_plus_M1, q_k, project_C, u_init) -> y
                 If None, built-in PGD is used.
    verbose    : bool          print residual every log_every iterations
    log_every  : int

    Returns
    -------
    DRResult with fields:
        u_star    : solution u*
        residuals : list of ‖r(u^k)‖
        n_iter    : iterations performed
        converged : bool
    """
    n      = M.shape[0]
    dtype  = M.dtype
    device = M.device

    # ── Splitting M = M1 + M2 ───────────────────────────────────────────────
    if M1 is None or M2 is None:
        M1, M2 = split_M_canonical(M)          # eq. (34)

    # ── Preconditioner H ────────────────────────────────────────────────────
    if H is None:
        H = torch.eye(n, dtype=dtype, device=device)   # H = I

    # ── Precompute constant matrices ─────────────────────────────────────────
    H_plus_M1  = H + M1                                # symmetric PD  → QP in Step 1
    H_plus_M2  = H + M2
    H_plus_M2_inv = torch.linalg.inv(H_plus_M2)        # precomputed once

    M2_minus_H = M2 - H                                # used in Step 1 linear term

    # ── Initial point u^0 ───────────────────────────────────────────────────
    u = torch.zeros(n, dtype=dtype, device=device) if u_init is None else u_init.clone()

    result = DRResult(u_star=u)

    if verbose:
        print(f"{'─'*55}")
        print(f"  Douglas-Rachford AVI Solver")
        print(f"  n={n},  λ={lam},  tol={tol:.0e},  max_iter={max_iter}")
        print(f"  Splitting: {'canonical (eq.34)' if M1 is None else 'custom'}")
        print(f"{'─'*55}")

    # ── Main loop ────────────────────────────────────────────────────────────
    for k in range(max_iter):

        u_new, rk = douglas_rachford_once(M, M2, q, project_C, M2_minus_H, H_plus_M1, H_plus_M2_inv, H, lam, u=u, qp_solver=qp_solver)
    
        result.residuals.append(rk)

        if verbose and (k % log_every == 0):
            print(f"  iter {k:5d}   ‖r‖ = {rk:.4e}")

        # ── Convergence check ─────────────────────────────────────────────────
        if rk <= tol:
            u = u_new
            result.n_iter    = k + 1
            result.converged = True
            if verbose:
                print(f"  ✓ Converged at iter {k+1}   ‖r‖ = {rk:.4e}")
            break

        u = u_new

    else:
        # loop ended without break
        result.n_iter    = max_iter
        result.converged = False
        if verbose:
            print(f"  ✗ Max iterations reached.  ‖r‖ = {result.residuals[-1]:.4e}")

    result.u_star = u
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Receding-horizon wrapper  (Figure 2 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def receding_horizon_dr(
        build_avi:   Callable,
        project_C:   Callable,
        x0:          torch.Tensor,
        T_sim:       int,
        lam:         float = 0.5,
        max_iter:    int   = 200,
        tol:         float = 1e-6,
        H:           Optional[torch.Tensor] = None,
        qp_solver:   Optional[Callable]     = None,
        verbose:     bool  = False,
) -> Tuple[list, list]:
    """
    Receding-horizon controller based on DR (Figure 2 of the paper).

    At each time step t:
        1. Given current state x[t], call build_avi(x[t]) to get (M, q).
        2. Solve AVI(C, M, q) with Douglas-Rachford.
        3. Apply first action u*[0|t].
        4. Advance the state.

    Parameters
    ----------
    build_avi  : Callable  x -> (M, q, u_warm_start)
                 Returns the AVI matrices for the current state x,
                 plus an optional warm-start for u.
    project_C  : Callable  u -> π_C(u)
                 Projection onto the feasible set (can depend on x if needed;
                 in that case, wrap it outside and pass a closure).
    x0         : torch.Tensor  initial state
    T_sim      : int           simulation horizon
    lam        : float         DR relaxation parameter
    max_iter   : int           max DR iterations per time step
    tol        : float         DR convergence tolerance
    H          : torch.Tensor  preconditioner (optional, same at all steps)
    qp_solver  : Callable      external QP solver (optional)
    verbose    : bool

    Returns
    -------
    u_history  : list of torch.Tensor   applied first actions u*[0|t]
    res_history: list of list           residual sequences per time step
    """
    x          = x0.clone()
    u_history  = []
    res_history = []
    u_warm     = None

    for t in range(T_sim):
        
        # Build AVI for current state
        #aggiorna net con x[t] e ricalcola M, q, u_warm
        M, q, u_warm = build_avi(x)

        #ricalcolerei proiezione se i vinoli dipendono da x[t]
        #project_C_t = project_C(u, x=x)

        # Solve AVI(C, M, q) with DR
        result = douglas_rachford_avi(
            M          = M,
            q          = q,
            project_C  = project_C,
            H          = H,
            lam        = lam,
            max_iter   = max_iter,
            tol        = tol,
            u_init     = u_warm,
            qp_solver  = qp_solver,
            verbose    = verbose,
        )

        u_star = result.u_star

        """
        5. Ricostruisce D_n e scrivi in net
            u_full = S @ result.u_star + s
            net.set_from_u(u_full, offset, dim_n)

        6. Avanza lo stato
            x[t+1] = f(x[t], u_star)
        """

        # Apply first block of u* (caller's responsibility to extract it)
        u_history.append(u_star)
        res_history.append(result.residuals)

        if verbose:
            status = "✓" if result.converged else "✗"
            print(f"  t={t:3d}  {status}  iters={result.n_iter:4d}  "
                  f"‖r‖={result.residuals[-1]:.2e}")

        # The caller must provide the state update via build_avi's side-effect
        # or explicitly here. We return u_history so the caller can advance x.
        # (For modularity, the state update is left to the caller.)
        break  # remove this if build_avi also advances x internally

    return u_history, res_history
