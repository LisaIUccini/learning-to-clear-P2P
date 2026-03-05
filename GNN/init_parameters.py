import torch


def _build_c_k(neighbors, c_fn, k_fn):
    """
    Helper to build c and k dicts given per-edge functions.

    Parameters
    ----------
    neighbors : dict {n: [m1, m2, ...]}
    c_fn : callable (m, n) -> float   trading cost c_{nm}
    k_fn : callable (m, n) -> float   capacity k_{mn}

    Returns
    -------
    c : dict {(m, n): tensor}
    k : dict {(m, n): tensor}
    """
    c = {}
    k = {}
    for n in neighbors:
        for m in neighbors[n]:
            c[(m, n)] = torch.tensor(c_fn(m, n), dtype=torch.float32)
            k[(m, n)] = torch.tensor(k_fn(m, n), dtype=torch.float32)
    return c, k


def initN3():
    """
    Small test network with N=3 prosumers, fully connected.

    Returns
    -------
    N, gamma, max_iter, tol,
    neighbors, D_min, D_max, G_min, G_max, G_res,
    D_star, a_tilde, a, b, c, k
    """
    N        = 3
    gamma    = 0.1
    max_iter = 100
    tol      = 1e-6

    neighbors = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }

    D_min   = torch.tensor([0.0,  0.0,  0.0],  dtype=torch.float32)
    D_max   = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor([10.0, 0.0, 0.0],   dtype=torch.float32)
    G_res   = torch.tensor([0.0,  3.0, 5.0],   dtype=torch.float32)

    a_tilde = torch.tensor([5.0,  15.0, 10.0], dtype=torch.float32)
    D_star  = torch.tensor([6.0,  3.0,  3.0],  dtype=torch.float32)
    a       = torch.tensor([4.0,  4.0,  4.0],  dtype=torch.float32)
    b       = torch.zeros(N, dtype=torch.float32)

    def c_fn(m, n):
        return 0.1

    def k_fn(m, n):
        return 10.0 if m == 0 else 5.0

    c, k = _build_c_k(neighbors, c_fn, k_fn)

    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, \
           G_res, D_star, a_tilde, a, b, c, k


def initN10():
    """
    Medium network with N=10 prosumers.
    Node 0 is the root (grid interface), connected to all others.

    Returns
    -------
    N, gamma, max_iter, tol,
    neighbors, D_min, D_max, G_min, G_max, G_res,
    D_star, a_tilde, a, b, c, k
    """
    N        = 10
    gamma    = 0.1
    max_iter = 250
    tol      = 1e-6

    neighbors = {
        0: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        1: [0, 2, 3],
        2: [0, 1, 4],
        3: [0, 1, 5],
        4: [0, 2, 6],
        5: [0, 3, 7],
        6: [0, 4, 8],
        7: [0, 5, 9],
        8: [0, 6, 9],
        9: [0, 7, 8],
    }

    D_min   = torch.zeros(N, dtype=torch.float32)
    D_max   = torch.full((N,), 10.0, dtype=torch.float32)
    G_min   = torch.zeros(N, dtype=torch.float32)
    G_max   = torch.tensor(
        [10.0, 0.0, 6.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        dtype=torch.float32
    )
    G_res   = torch.tensor(
        [0.0, 2.0, 3.0, 1.0, 0.5, 2.5, 1.5, 0.0, 3.0, 1.0],
        dtype=torch.float32
    )
    D_star  = torch.tensor(
        [6.0, 4.0, 5.0, 6.0, 5.0, 4.0, 5.0, 6.0, 4.0, 5.0],
        dtype=torch.float32
    )
    a_tilde = torch.tensor(
        [10.0, 12.0, 8.0, 10.0, 9.0, 11.0, 10.0, 12.0, 9.0, 10.0],
        dtype=torch.float32
    )
    a       = torch.tensor(
        [3.0, 2.0, 4.0, 3.0, 2.5, 3.5, 4.0, 3.0, 2.0, 3.0],
        dtype=torch.float32
    )
    b       = torch.zeros(N, dtype=torch.float32)

    def c_fn(m, n):
        return 0.1

    def k_fn(m, n):
        # Root node links have higher capacity
        return 10.0 if (m == 0 or n == 0) else 5.0

    c, k = _build_c_k(neighbors, c_fn, k_fn)

    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, \
           G_res, D_star, a_tilde, a, b, c, k


def initN30():
    """
    Large radial tree network with N=30 prosumers.
    Node 0 is the root (main grid interface).

    Returns
    -------
    N, gamma, max_iter, tol,
    neighbors, D_min, D_max, G_min, G_max, G_res,
    D_star, a_tilde, a, b, c, k
    """
    N        = 30
    gamma    = 0.1
    max_iter = 300
    tol      = 1e-6

    neighbors = {
        0:  [1, 2],
        1:  [0, 3, 4],
        2:  [0, 5, 6],
        3:  [1, 7, 8],
        4:  [1, 9],
        5:  [2, 10],
        6:  [2, 11, 12],
        7:  [3, 13, 14],
        8:  [3, 15],
        9:  [4, 16],
        10: [5, 16],
        11: [6, 17],
        12: [6],
        13: [7, 18],
        14: [7, 19],
        15: [8],
        16: [9, 10, 20],
        17: [11, 21],
        18: [13, 22],
        19: [14],
        20: [16, 23, 24],
        21: [17, 25],
        22: [18],
        23: [20, 26],
        24: [20],
        25: [21, 27],
        26: [23],
        27: [25, 28],
        28: [27, 29],
        29: [28],
    }

    D_min = torch.zeros(N, dtype=torch.float32)
    D_max = torch.tensor([
         0.0, 20.0, 20.0, 15.0, 10.0, 10.0, 15.0,  8.0,  8.0,  6.0,
         6.0,  8.0,  5.0,  6.0,  6.0,  5.0,  7.0,  6.0,  5.0,  5.0,
         7.0,  5.0,  4.0,  5.0,  4.0,  4.0,  4.0,  3.0,  3.0,  3.0,
    ], dtype=torch.float32)

    G_min = torch.zeros(N, dtype=torch.float32)
    G_max = torch.tensor([
        50.0, 8.0, 8.0, 5.0, 0.0, 0.0, 5.0, 3.0, 0.0, 3.0,
         0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0,
         2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=torch.float32)

    G_res = torch.tensor([
        0.0, 2.0, 1.5, 3.0, 4.0, 3.5, 2.0, 2.5, 4.0, 3.0,
        3.5, 2.0, 4.5, 3.0, 3.5, 4.0, 2.5, 3.0, 2.0, 3.5,
        2.5, 3.0, 4.0, 2.0, 4.0, 3.0, 4.0, 3.0, 3.5, 4.0,
    ], dtype=torch.float32)

    a_tilde = torch.tensor([
         0.0, 10.0, 10.0,  8.0, 12.0, 12.0,  8.0, 10.0, 10.0, 12.0,
        12.0, 10.0, 15.0, 12.0, 12.0, 15.0, 10.0, 12.0, 15.0, 15.0,
        10.0, 12.0, 15.0, 12.0, 15.0, 12.0, 15.0, 12.0, 12.0, 15.0,
    ], dtype=torch.float32)

    D_star = torch.tensor([
         0.0, 14.0, 14.0, 10.0,  7.0,  7.0, 10.0,  5.5,  5.5,  4.0,
         4.0,  5.5,  3.5,  4.0,  4.0,  3.5,  5.0,  4.0,  3.5,  3.5,
         5.0,  3.5,  3.0,  3.5,  3.0,  3.0,  3.0,  2.0,  2.0,  2.0,
    ], dtype=torch.float32)

    a = torch.tensor([
        1.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 6.0, 4.0, 5.0,
        4.0, 6.0, 4.0, 6.0, 4.0, 4.0, 5.0, 4.0, 6.0, 4.0,
        5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
    ], dtype=torch.float32)

    b = torch.zeros(N, dtype=torch.float32)

    def c_fn(m, n):
        if m == 0 or n == 0:
            return 0.5   # grid-connected link: higher tariff
        elif m <= 2 or n <= 2:
            return 0.2   # links adjacent to main hubs
        else:
            return 0.1   # peer-to-peer deeper links

    def k_fn(m, n):
        if m == 0 or n == 0:
            return 20.0
        elif m <= 2 or n <= 2:
            return 10.0
        else:
            return 5.0

    c, k = _build_c_k(neighbors, c_fn, k_fn)

    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, \
           G_res, D_star, a_tilde, a, b, c, k