import copy
import torch


class Prosumer:
    """
    Represents a single agent (prosumer) in the peer-to-peer energy market.

    Each prosumer controls three local decision variables:
        - D  : energy demand (scalar)
        - G  : controllable flexibility activation (scalar)
        - q  : dict {m: tensor} of bilateral trades with neighbors m in Omega_n

    Local parameters (fixed, known only to this prosumer):
        - a_tilde, D_star        : usage benefit coefficients
        - a, b                   : flexibility activation cost coefficients
        - D_min, D_max           : demand bounds
        - G_min, G_max           : flexibility bounds
        - G_res                  : renewable energy source generation (given, not optimized)
        - c  : dict {m: tensor}  : per-unit trading cost with neighbor m
        - k  : dict {m: tensor}  : max interconnection capacity with neighbor m

    The prosumer does NOT know about the global feasible set or the reciprocity
    constraint — those are handled by EnergyNetwork.
    """

    def __init__(
        self,
        idx: int,
        neighbors: list,
        D_min: torch.Tensor,
        D_max: torch.Tensor,
        G_min: torch.Tensor,
        G_max: torch.Tensor,
        G_res: torch.Tensor,
        a_tilde: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        D_star: torch.Tensor,
        c: dict,
        k: dict,
        device: torch.device,
    ):
        """
        Parameters
        ----------
        idx       : index of this prosumer in the network (integer)
        neighbors : list of neighbor indices [m1, m2, ...]
        D_min, D_max, G_min, G_max, G_res : scalar tensors (0-dim or shape ())
        a_tilde, a, b, D_star             : scalar tensors
        c : dict {m: tensor}  trading cost with neighbor m
        k : dict {m: tensor}  capacity with neighbor m
        device    : torch.device
        """
        self.idx = idx
        self.neighbors = neighbors  # list of neighbor indices
        self.device = device

        # --- Fixed local parameters ---
        self.D_min = D_min.to(device)
        self.D_max = D_max.to(device)
        self.G_min = G_min.to(device)
        self.G_max = G_max.to(device)
        self.G_res = G_res.to(device)

        self.a_tilde = a_tilde.to(device)
        self.a = a.to(device)
        self.b = b.to(device)
        self.D_star = D_star.to(device)

        # c[(m)] and k[(m)]: per-neighbor dicts, already indexed locally
        self.c = {m: c[m].to(device) for m in neighbors}
        self.k = {m: k[m].to(device) for m in neighbors}

        # --- Local state (initialized randomly within bounds) ---
        self.D, self.G, self.q = self._init_state()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_state(self):
        """
        Initialize D and G uniformly within their bounds.
        Initialize each q[m] uniformly in [0, k[m]].
        """
        D = self.D_min + (self.D_max - self.D_min) * torch.rand(
            1, device=self.device
        ).squeeze()
        G = self.G_min + (self.G_max - self.G_min) * torch.rand(
            1, device=self.device
        ).squeeze()
        q = {
            m: torch.rand(1, device=self.device).squeeze() * self.k[m]
            for m in self.neighbors
        }
        return D, G, q

    # ------------------------------------------------------------------
    # State access and mutation
    # ------------------------------------------------------------------

    def get_state(self):
        """
        Returns the current local state as (D, G, q).

        Returns
        -------
        D : scalar tensor
        G : scalar tensor
        q : dict {m: scalar tensor}
        """
        return self.D, self.G, self.q

    def set_state(self, D: torch.Tensor, G: torch.Tensor, q: dict):
        """
        Set the local state.

        Parameters
        ----------
        D : scalar tensor
        G : scalar tensor
        q : dict {m: scalar tensor}
        """
        self.D = D.to(self.device)
        self.G = G.to(self.device)
        self.q = {m: q[m].to(self.device) for m in self.neighbors}

    # ------------------------------------------------------------------
    # Local pseudo-gradient components
    # ------------------------------------------------------------------

    def grad_D(self) -> torch.Tensor:
        """
        Gradient of J_n w.r.t. D_n:
            dJ/dD_n = 2 * a_tilde_n * (D_n - D_star_n)
        """
        return 2.0 * self.a_tilde * (self.D - self.D_star)

    def grad_G(self) -> torch.Tensor:
        """
        Gradient of J_n w.r.t. G_n:
            dJ/dG_n = a_n * G_n + b_n
        """
        return self.a * self.G + self.b

    def grad_q(self, m: int) -> torch.Tensor:
        """
        Gradient of J_n w.r.t. q_{mn} (trade with neighbor m):
            dJ/dq_{mn} = c_{nm}   (constant, independent of q)

        Parameters
        ----------
        m : neighbor index
        """
        return self.c[m]

    # ------------------------------------------------------------------
    # Local cost (for monitoring only, not used in the FB loop)
    # ------------------------------------------------------------------

    def local_cost(self) -> torch.Tensor:
        """
        J_n(x_n) = 0.5*a_n*G_n^2 + b_n*G_n
                 + sum_{m in Omega_n} c_{nm}*q_{mn}
                 + a_tilde_n * (D_n - D_star_n)^2
        """
        cost = (
            0.5 * self.a * self.G ** 2
            + self.b * self.G
            + self.a_tilde * (self.D - self.D_star) ** 2
        )
        for m in self.neighbors:
            cost = cost + self.c[m] * self.q[m]
        return cost

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    def clone(self):
        """
        Returns a deep copy of this Prosumer with fully independent tensors.
        Parameters are shared (read-only), state is copied.
        """
        new = copy.copy(self)           # shallow copy of the object
        new.D = self.D.clone()
        new.G = self.G.clone()
        new.q = {m: self.q[m].clone() for m in self.neighbors}
        return new

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"Prosumer(idx={self.idx}, "
            f"D={self.D.item():.3f}, "
            f"G={self.G.item():.3f}, "
            f"neighbors={self.neighbors})"
        )