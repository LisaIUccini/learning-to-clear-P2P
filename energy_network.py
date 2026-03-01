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

        # stato iniziale
        self.D, self.G, self.q = self.init_state()

        # stato globale flattenato (D_0..D_{N-1}, G_0..G_{N-1}, q_edges...)
        self.x = self.flatten(self.D, self.G, self.q)

        # AVI reformulation
        self.u, offset, dim_m = self.get_u()
        self.M, self.q0 = self.build_M_and_q0()
        self.D_mat, self.d_vec = self.build_constraint_matrices()
        self.E_mat, self.e_vec = self.build_equality_matrix()

    
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

        
    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def init_state(self):
        D = torch.zeros(self.N, device=self.device)#self.D_min + (self.D_max - self.D_min) * torch.rand(self.N, device=self.device)
        G = torch.zeros(self.N, device=self.device)#self.G_min + (self.G_max - self.G_min) * torch.rand(self.N, device=self.device)
        q = {}
        for n in self.neighbors:
            for m in self.neighbors[n]:
                q[(m, n)] = torch.tensor(0.0, device=self.device)#0 + (self.k[(m,n)] - 0) * torch.rand(1, device=self.device).item(
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


    # --------------------------------------------------
    # AVI Reformulation - j_n, u, M, q00, D_mat, d_vac
    # --------------------------------------------------
    #
    # Per ogni agente n definiamo:
    #
    #   u_n = [ D_n, G_n, q_{m1,n}, q_{m2,n}, ... ]   ∈ R^{2 + |Ω_n|}
    #
    # dove m1, m2, ... sono i vicini di n in ordine (sorted).
    # Il vettore globale è:
    #
    #   u = col(u_0, u_1, ..., u_{N-1})   ∈ R^{sum_n (2 + |Ω_n|)}
    #
    # ATTENZIONE: questo ordine è DIVERSO da self.x (che usa
    # [D_0..D_{N-1}, G_0..G_{N-1}, q_edges...]).
    # I due encoding coesistono: self.x per il resto del codice,
    # u per il solver DR.
    # ------------------------------------------------------------------

    def build_agent_index(self):
        """
        Costruisce l'indice globale per ogni agente n nel vettore u.

        Ritorna
        -------
        offset : dict  n -> indice globale della prima componente di u_n in u
        dim_n  : dict  n -> dimensione del blocco u_n  (= 2 + |Ω_n|)
        dim_u  : int   dimensione totale di u
        """
        offset = {}
        dim_n  = {}
        off = 0
        for n in range(self.N):
            neighbors_n = sorted(self.neighbors.get(n, []))
            d = 2 + len(neighbors_n)
            offset[n] = off
            dim_n[n]  = d
            off += d
        return offset, dim_n, off   # off = dim_u


    def get_u_n(self, n, offset=None):
        """
        Estrae il blocco u_n = [D_n, G_n, q_{m1,n}, q_{m2,n}, ...]
        dallo stato corrente (self.D, self.G, self.q).

        Parametri
        ---------
        n      : indice agente
        offset : se None viene ricalcolato (puoi passarlo per efficienza)

        Ritorna
        -------
        u_n : torch.Tensor shape (2 + |Ω_n|,)
        """
        neighbors_n = sorted(self.neighbors.get(n, []))

        D_n = self.D[n].unsqueeze(0)                          # shape (1,)
        G_n = self.G[n].unsqueeze(0)                          # shape (1,)
        q_n = torch.stack([
            self.q[(m, n)] if isinstance(self.q[(m, n)], torch.Tensor)
            else torch.tensor(self.q[(m, n)], dtype=torch.float32, device=self.device)
            for m in neighbors_n
        ])                                                      # shape (|Ω_n|,)

        return torch.cat([D_n, G_n, q_n])                     # shape (2 + |Ω_n|,)
    
    def get_u(self):
        """
        Costruisce il vettore globale u = col(u_0, u_1, ..., u_{N-1}).

        Ritorna
        -------
        u      : torch.Tensor shape (dim_u,)
        offset : dict  n -> offset del blocco n in u
        dim_n  : dict  n -> dim del blocco n
        """
        offset, dim_n, dim_u = self.build_agent_index()
        blocks = [self.get_u_n(n) for n in range(self.N)]
        u = torch.cat(blocks)
        return u, offset, dim_n
    
    def set_from_u(self, u, offset, dim_n):
        """
        Scrive il vettore u nella rete (aggiorna self.D, self.G, self.q).

        Parametri
        ---------
        u      : torch.Tensor shape (dim_u,)
        offset : dict da build_agent_index()
        dim_n  : dict da build_agent_index()
        """
        for n in range(self.N):
            off = offset[n]
            neighbors_n = sorted(self.neighbors.get(n, []))

            self.D[n] = u[off]
            self.G[n] = u[off + 1]
            for k, m in enumerate(neighbors_n):
                self.q[(m, n)] = u[off + 2 + k]

        self.x = self.flatten(self.D, self.G, self.q)

    # ------------------------------------------------------------------
    # Matrici AVI:  M u + q0  =  gradiente del costo aggregato
    # ------------------------------------------------------------------
    def build_M_and_q0(self):
        """
        Costruisce la matrice di gioco M e il vettore lineare q0 per l'AVI:

            AVI(U, M, q0):  trovare u* ∈ U  t.c.  <M u* + q0, u - u*> >= 0

        Struttura di M per l'agente n (blocco diagonale R_n):

            R_n = diag( 2*ã_n,  a_n,  0, 0, ... )
                         ^^^^   ^^^   ^^^^^^^^^
                          D_n   G_n   q_{mn} : costo lineare → 0 nella parte quadratica

        Vettore r_n (parte lineare del costo di n):

            r_n = ( -2*ã_n*D*_n,  b_n,  c_{n,m1}, c_{n,m2}, ... )

        M = blkdiag(R_0, R_1, ..., R_{N-1})   (matrice a blocchi diagonali)
        q0 = col(r_0, r_1, ..., r_{N-1})

        Ritorna
        -------
        M  : torch.Tensor shape (dim_u, dim_u)
        q0 : torch.Tensor shape (dim_u,)
        """
        offset, dim_n, dim_u = self.build_agent_index()

        M  = torch.zeros(dim_u, dim_u, dtype=torch.float32, device=self.device)
        q0 = torch.zeros(dim_u,        dtype=torch.float32, device=self.device)

        for n in range(self.N):
            off        = offset[n]
            neighbors_n = sorted(self.neighbors.get(n, []))

            # --- blocco diagonale R_n ---
            M[off,     off    ] = 2.0 * self.a_tilde[n]   # coeff quadratico D_n
            M[off + 1, off + 1] = self.a[n]               # coeff quadratico G_n
            # le righe/colonne dei q rimangono 0 (costo lineare)

            # --- vettore lineare r_n ---
            q0[off    ] = -2.0 * self.a_tilde[n] * self.D_star[n]  # grad D_n a D=0
            q0[off + 1] = self.b[n]                                 # grad G_n a G=0
            for k, m in enumerate(neighbors_n):
                # c[(n,m)] è il costo che n paga per il trade q[(m,n)]
                q0[off + 2 + k] = self.c[(n, m)]

        return M, q0
    
    # ------------------------------------------------------------------
    # Matrici dei vincoli:  D_mat u <= d_vec
    # ------------------------------------------------------------------

    def build_constraint_matrices(self):
        """
        Costruisce D_mat e d_vec tali che:

            D_mat @ u <= d_vec

        raccoglie TRE tipi di vincoli (nell'ordine):

        Blocco 1 — Reciprocità  (|E| righe, una per arco non orientato)
            q_{mn} + q_{nm} <= 0

        Blocco 2 — Capacità  (2|E| righe, una per arco orientato)
            q_{mn} <= κ_{mn}

        Blocco 3 — Box D_n e G_n  (4N righe)
            D_n <= D_max_n,  -D_n <= -D_min_n
            G_n <= G_max_n,  -G_n <= -G_min_n

        Ritorna
        -------
        D_mat : torch.Tensor shape (n_ineq, dim_u)
        d_vec : torch.Tensor shape (n_ineq,)
        """
        offset, dim_n, dim_u = self.build_agent_index()

        # helper: indice globale di q_{mn} in u (si trova nel blocco di n)
        def global_idx_q(m, n):
            neighbors_n = sorted(self.neighbors.get(n, []))
            k = neighbors_n.index(m)
            return offset[n] + 2 + k

        # helper: indice globale di D_n in u
        def global_idx_D(n):
            return offset[n]

        # helper: indice globale di G_n in u
        def global_idx_G(n):
            return offset[n] + 1

        # --- raccoglie archi non orientati (m < n) ---
        undirected_edges = []
        seen = set()
        for n in range(self.N):
            for m in sorted(self.neighbors.get(n, [])):
                e = (min(m, n), max(m, n))
                if e not in seen:
                    seen.add(e)
                    undirected_edges.append(e)

        n_rec  = len(undirected_edges)                         # vincoli reciprocità
        n_cap  = sum(len(self.neighbors.get(n,[])) for n in range(self.N))  # vincoli capacità
        n_box  = 4 * self.N                                    # vincoli box
        n_ineq = n_rec + n_cap + n_box

        D_mat = torch.zeros(n_ineq, dim_u, dtype=torch.float32, device=self.device)
        d_vec = torch.zeros(n_ineq,         dtype=torch.float32, device=self.device)
        row = 0

        # ------ Blocco 1: Reciprocità ------
        # Per ogni arco non orientato (m, n) con m < n:
        #   q_{nm} (in u_m)  +  q_{mn} (in u_n)  <= 0
        for (m, n) in undirected_edges:
            # q_{nm}: flusso da n a m, memorizzato nel blocco di m
            D_mat[row, global_idx_q(n, m)] = 1.0
            # q_{mn}: flusso da m a n, memorizzato nel blocco di n
            D_mat[row, global_idx_q(m, n)] = 1.0
            d_vec[row] = 0.0
            row += 1

        # ------ Blocco 2: Capacità ------
        # Per ogni arco orientato (m,n) nell'edge_list: q_{mn} <= κ_{mn}
        for n in range(self.N):
            for m in sorted(self.neighbors.get(n, [])):
                D_mat[row, global_idx_q(m, n)] = 1.0
                d_vec[row] = float(self.k[(m, n)])
                row += 1

        # ------ Blocco 3: Box su D_n e G_n ------
        for n in range(self.N):
            gi_D = global_idx_D(n)
            gi_G = global_idx_G(n)

            D_mat[row,   gi_D] =  1.0;  d_vec[row]   = float(self.D_max[n]);  row += 1
            D_mat[row,   gi_D] = -1.0;  d_vec[row]   = float(-self.D_min[n]); row += 1
            D_mat[row,   gi_G] =  1.0;  d_vec[row]   = float(self.G_max[n]);  row += 1
            D_mat[row,   gi_G] = -1.0;  d_vec[row]   = float(-self.G_min[n]); row += 1

        return D_mat, d_vec
    
    def build_equality_matrix(self):
        """
        Costruisce E_mat e e_vec per i vincoli di bilanciamento:

            E_mat @ u = e_vec

        dove per ogni agente n:

            D_n - G_n - sum_{m in Ω_n} q_{mn}  =  ΔG_n   (= G_res[n])

        Ritorna
        -------
        E_mat : torch.Tensor shape (N, dim_u)
        e_vec : torch.Tensor shape (N,)
        """
        offset, dim_n, dim_u = self.build_agent_index()

        E_mat = torch.zeros(self.N, dim_u, dtype=torch.float32, device=self.device)
        e_vec = torch.zeros(self.N,        dtype=torch.float32, device=self.device)

        for n in range(self.N):
            off         = offset[n]
            neighbors_n = sorted(self.neighbors.get(n, []))

            E_mat[n, off    ] =  1.0   # +D_n
            E_mat[n, off + 1] = -1.0   # -G_n
            for k, m in enumerate(neighbors_n):
                E_mat[n, off + 2 + k] = -1.0   # -q_{mn}

            e_vec[n] = float(self.G_res[n])     # = ΔG_n

        return E_mat, e_vec
    
    def make_projection(self,
                        D_mat:    torch.Tensor,
                        d_vec:    torch.Tensor,
                        max_iter: int   = 200,
                        tol:      float = 1e-10):
        """
        Costruisce e restituisce la funzione  π_C(u)  che proietta
        un vettore u sul poliedro:

            C = { u : D_mat @ u  <=  d_vec }

        ─────────────────────────────────────────────────────────────
        PERCHÉ È UN METODO CHE RESTITUISCE UNA FUNZIONE?
        ─────────────────────────────────────────────────────────────
        Il solver Douglas-Rachford chiama la proiezione MOLTE volte
        (una per iterazione), passando ogni volta un vettore u diverso.
        Ma D_mat e d_vec non cambiano mai: vengono costruite una sola
        volta dalla rete.

        Soluzione: questo metodo "intrappola" D_mat e d_vec dentro
        la funzione project_C e la restituisce pronta all'uso.

        Utilizzo:
            project_C = net.make_projection(D_mat, d_vec)
            # ora project_C è una normale funzione  u -> π_C(u)
            u_proiettato = project_C(u)       # chiamata singola
            # oppure passala al solver:
            result = douglas_rachford_avi(M, q, project_C, ...)

        ─────────────────────────────────────────────────────────────
        ALGORITMO INTERNO: Dykstra
        ─────────────────────────────────────────────────────────────
        Proietta su un poliedro come intersezione di semispazi:

            C = H_0 ∩ H_1 ∩ ... ∩ H_{m-1}

        dove  H_i = { u : a_i · u ≤ d_i }  (i-esima riga di D_mat).

        Ad ogni iterazione cicla su tutti i semispazi. Se il punto
        corrente z viola il semispazio i (cioè a_i · z > d_i),
        lo proietta su H_i:

            z  ←  z  −  ((a_i · z − d_i) / ‖a_i‖²)  *  a_i

        Converge quando z non si muove più di tol tra due iterazioni.

        Parametri
        ----------
        D_mat    : torch.Tensor  shape (n_ineq, dim_u_ridotto)
                   Matrice dei vincoli (già ridotta dopo eliminate_equalities)
        d_vec    : torch.Tensor  shape (n_ineq,)
                   Termine noto dei vincoli
        max_iter : int    iterazioni massime di Dykstra
        tol      : float  soglia di convergenza

        Ritorna
        -------
        project_C : funzione  u (torch.Tensor) -> π_C(u) (torch.Tensor)
        """

        # Converti in numpy una volta sola — le matrici non cambiano mai
        # tra le iterazioni del DR, quindi questa conversione è fatta
        # una sola volta qui, non ad ogni chiamata di project_C
        D_np         = D_mat.detach().cpu().numpy().astype(np.float64)
        d_np         = d_vec.detach().cpu().numpy().astype(np.float64)
        n_ineq       = D_np.shape[0]

        # ‖a_i‖² per ogni riga — precompute una volta sola
        row_norms_sq = np.sum(D_np ** 2, axis=1)   # shape (n_ineq,)

        # ── La funzione che viene restituita ─────────────────────────
        # Questa è project_C: l'unico argomento è u (il vettore da
        # proiettare). D_np, d_np, row_norms_sq sono "intrappolati"
        # nel contesto della funzione (closure).
        def project_C(u: torch.Tensor) -> torch.Tensor:
            device = u.device
            dtype  = u.dtype
            z = u.detach().cpu().numpy().astype(np.float64)

            # Incrementi di Dykstra (uno per vincolo, inizializzati a 0)
            p = np.zeros((n_ineq, len(z)))

            for _ in range(max_iter):
                z_old = z.copy()

                for i in range(n_ineq):
                    if row_norms_sq[i] < 1e-14:
                        continue                        # riga nulla, salta

                    y_i   = z + p[i]                   # punto corretto
                    slack = D_np[i] @ y_i - d_np[i]   # violazione del vincolo i

                    if slack > 0:
                        # z viola H_i → proietta su H_i
                        proj_i = y_i - (slack / row_norms_sq[i]) * D_np[i]
                    else:
                        # z è già dentro H_i → nessuna modifica
                        proj_i = y_i

                    p[i] = y_i - proj_i   # aggiorna incremento Dykstra
                    z    = proj_i

                # Criterio di stop: z non si è mosso
                if np.linalg.norm(z - z_old) < tol:
                    break

            return torch.tensor(z, dtype=dtype, device=device)

        return project_C
    
    # ==================================================================
    # PROIEZIONE ESATTA  π_C(u_free)  sul poliedro dei vincoli
    # ==================================================================

    def make_projection1(self):
        """
        Costruisce e restituisce la funzione  π_C(u_free)  che proietta
        esattamente il vettore ridotto u_free (senza D_n) sul suo insieme
        ammissibile, sfruttando la struttura analitica di ogni vincolo.

        Dopo eliminate_equalities le variabili libere sono (G_n, q_{mn}),
        nell'ordine:
            u_free = [G_0, q_{·,0}, G_1, q_{·,1}, ..., G_{N-1}, q_{·,N-1}]

        cioè per ogni agente n il blocco è:
            [G_n, q_{m1,n}, q_{m2,n}, ...]   (senza D_n)

        I vincoli su queste variabili hanno tutti proiezione esatta:

            Vincolo          Forma               Proiezione
            ──────────────────────────────────────────────────────────
            Box G_n          G_n ∈ [G_min, G_max]  clamp scalare
            Capacità         q_{mn} ≤ κ_{mn}        clamp scalare
            Reciprocità      q_{mn}+q_{nm} ≤ 0      semispazio 2D esatto

        Proiezione sul semispazio {(a,b) : a+b ≤ 0}:
            se a+b > 0:  a' = a-(a+b)/2,  b' = b-(a+b)/2

        NON dipende da S o s — costruisce gli indici direttamente
        dalla struttura della rete (offset e neighbors), che è la
        fonte di verità per l'ordine delle variabili in u_free.

        Ritorna
        -------
        project_C : funzione  u_free (Tensor) -> π_C(u_free) (Tensor)
        """
        offset, dim_n, dim_u = self.build_agent_index()

        # ── Indici di G_n e q_{mn} in u_free ─────────────────────────
        # u_free è u completo senza le D_n.
        # Ogni D_n occupa offset[n]+0 in u → viene saltata.
        # Quindi u_free[k] corrisponde a u[free_idx[k]] dove
        # free_idx = lista crescente di tutti gli indici != offset[n].
        #
        # Costruiamo direttamente la lista free_idx dalla topologia:
        dep_set = {offset[n] for n in range(self.N)}   # indici di D_n in u
        free_idx_list = [j for j in range(dim_u) if j not in dep_set]
        # free_idx_list[k] = indice globale in u della k-esima var libera

        # Mappa inversa: indice globale in u → indice in u_free
        global_to_free = {j: k for k, j in enumerate(free_idx_list)}

        # g_idx[n]    = indice di G_n    in u_free
        # q_idx[(m,n)]= indice di q_{mn} in u_free
        g_idx = {}
        q_idx = {}
        for n in range(self.N):
            g_idx[n] = global_to_free[offset[n] + 1]
            for pos, m in enumerate(sorted(self.neighbors.get(n, []))):
                q_idx[(m, n)] = global_to_free[offset[n] + 2 + pos]

        # ── Archi non orientati per reciprocità ───────────────────────
        undirected_edges = []
        seen = set()
        for n in range(self.N):
            for m in sorted(self.neighbors.get(n, [])):
                e = (min(m, n), max(m, n))
                if e not in seen:
                    seen.add(e)
                    undirected_edges.append(e)

        # ── Bounds precomputati ───────────────────────────────────────
        g_lo = {n: float(self.G_min[n]) for n in range(self.N)}
        g_hi = {n: float(self.G_max[n]) for n in range(self.N)}
        k_hi = {(m, n): float(self.k[(m, n)])
                for n in range(self.N)
                for m in sorted(self.neighbors.get(n, []))}

        # ── Chiusura: cattura g_idx, q_idx, bounds ────────────────────
        def project_C(u_free: torch.Tensor) -> torch.Tensor:
            device = u_free.device
            dtype  = u_free.dtype
            z = u_free.detach().cpu().clone()

            # 1. Box su G_n  (clamp scalare esatto)
            for n in range(self.N):
                k = g_idx[n]
                z[k] = z[k].clamp(g_lo[n], g_hi[n])

            # 2. Capacità su q_{mn}  (clamp scalare esatto)
            for (m, n), k in q_idx.items():
                z[k] = z[k].clamp(max=k_hi[(m, n)])

            # 3. Reciprocità  q_{mn}+q_{nm} ≤ 0  (analitica esatta)
            for (m, n) in undirected_edges:
                k_mn = q_idx[(m, n)]
                k_nm = q_idx[(n, m)]
                s_val = z[k_mn] + z[k_nm]
                if s_val > 0:
                    half = s_val / 2.0
                    z[k_mn] = z[k_mn] - half
                    z[k_nm] = z[k_nm] - half

            return z.to(device=device, dtype=dtype)

        return project_C