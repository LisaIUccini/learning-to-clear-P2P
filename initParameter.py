
import torch

def initN3():
    N = 3 # number of prosumers
    gamma = 0.1 # step Forward-Backward
    max_iter = 100
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
    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, G_res, D_star, a_tilde, a, b, c, k

def initN10():
    N = 10 # number of prosumers
    gamma = 0.1 # step Forward-Backward
    max_iter = 250
    tol=1e-6

    neighbors = {
        0: [1,2,3,4,5,6,7,8,9],
        1: [0,2,3],
        2: [0,1,4],
        3: [0,1,5],
        4: [0,2,6],
        5: [0,3,7],
        6: [0,4,8],
        7: [0,5,9],
        8: [0,6,9],
        9: [0,7,8]
    }
    print("Neighbors:", neighbors)

    # Box Constraints on D and G
    D_min = torch.zeros(N, dtype=torch.float32)
    D_max = torch.full((N,), 10.0, dtype=torch.float32)
    G_min = torch.zeros(N, dtype=torch.float32)
    G_max = torch.tensor([10.0, 0.0,6.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0], dtype=torch.float32)


    # --- Generazione rinnovabile ---
    G_res = torch.tensor([0.0, 2.0, 3.0, 1.0, 0.5, 2.5, 1.5, 0.0, 3.0, 1.0], dtype=torch.float32)

    # --- Parametri funzione di beneficio ---
    D_star = torch.tensor([6.0, 4.0, 5.0, 6.0, 5.0, 4.0, 5.0, 6.0, 4.0, 5.0], dtype=torch.float32)
    a_tilde = torch.tensor([10.0, 12.0, 8.0, 10.0, 9.0, 11.0, 10.0, 12.0, 9.0, 10.0], dtype=torch.float32)

    # --- Parametri costo flessibilità ---
    a = torch.tensor([3.0, 2.0, 4.0, 3.0, 2.5, 3.5, 4.0, 3.0, 2.0, 3.0], dtype=torch.float32)
    b = torch.zeros(N, dtype=torch.float32)

    # Price preferences c_{mn} and interconnection capacities k_{mn}
    c = {}
    k = {}
    for n in neighbors:
        for m in neighbors[n]:
            c[(m, n)] = torch.tensor(0.1, dtype=torch.float32)
            if m == 0 or n == 0:
                k[(m, n)] = torch.tensor(10.0, dtype=torch.float32)  # root node can exchange di più
            else:
                k[(m, n)] = torch.tensor(5.0, dtype=torch.float32)

    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, G_res, D_star, a_tilde, a, b, c, k


def initN30():
    
    N = 30 # number of prosumers
    gamma = 0.1 # step Forward-Backward
    max_iter = 30
    tol=1e-6

    # set of neighbors of each prosumer
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
        10: [5, 16],        # node 16 is shared between subtrees 9 and 10
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
        0.0,  # 0  - root (grid interface, no demand)
        20.0,  # 1  - industrial/commercial hub
        20.0,  # 2  - industrial/commercial hub
        15.0,  # 3
        10.0,  # 4
        10.0,  # 5
        15.0,  # 6
        8.0,  # 7
        8.0,  # 8
        6.0,  # 9
        6.0,  # 10
        8.0,  # 11
        5.0,  # 12
        6.0,  # 13
        6.0,  # 14
        5.0,  # 15
        7.0,  # 16
        6.0,  # 17
        5.0,  # 18
        5.0,  # 19
        7.0,  # 20
        5.0,  # 21
        4.0,  # 22
        5.0,  # 23
        4.0,  # 24
        4.0,  # 25
        4.0,  # 26
        3.0,  # 27
        3.0,  # 28
        3.0,  # 29
    ], dtype=torch.float32)

    # ----------------------------------------------------------------
    # FLEXIBILITY (CONTROLLABLE GENERATION) BOUNDS [G_min, G_max] (kWh)
    # Node 0 = main grid: large generation capacity.
    # Some nodes have CHP or storage (non-zero G_max).
    # Pure consumers have G_max = 0.
    # ----------------------------------------------------------------
    G_min = torch.zeros(N, dtype=torch.float32)
    G_max = torch.tensor([
        50.0,  # 0  - main grid (unlimited supply modeled as large cap)
        8.0,  # 1  - CHP unit
        8.0,  # 2  - CHP unit
        5.0,  # 3  - storage/CHP
        0.0,  # 4  - pure consumer
        0.0,  # 5  - pure consumer
        5.0,  # 6  - storage
        3.0,  # 7  - small CHP
        0.0,  # 8  - pure consumer
        3.0,  # 9  - battery storage
        0.0,  # 10 - pure consumer
        3.0,  # 11 - small CHP
        0.0,  # 12 - pure consumer
        2.0,  # 13 - small storage
        0.0,  # 14 - pure consumer
        0.0,  # 15 - pure consumer
        2.0,  # 16 - small storage
        0.0,  # 17 - pure consumer
        2.0,  # 18 - small storage
        0.0,  # 19 - pure consumer
        2.0,  # 20 - small CHP
        0.0,  # 21 - pure consumer
        0.0,  # 22 - pure consumer
        2.0,  # 23 - small storage
        0.0,  # 24 - pure consumer
        0.0,  # 25 - pure consumer
        0.0,  # 26 - pure consumer
        0.0,  # 27 - pure consumer
        0.0,  # 28 - pure consumer
        0.0,  # 29 - pure consumer
    ], dtype=torch.float32)

    # ----------------------------------------------------------------
    # RENEWABLE ENERGY SOURCES ΔG_n (kWh) — photovoltaic generation
    # Nodes with solar panels; leaf/residential nodes more likely to have PV.
    # ----------------------------------------------------------------
    G_res = torch.tensor([
        0.0,  # 0  - grid node, no PV
        2.0,  # 1
        1.5,  # 2
        3.0,  # 3  - good solar exposure
        4.0,  # 4  - residential PV
        3.5,  # 5
        2.0,  # 6
        2.5,  # 7
        4.0,  # 8  - residential PV
        3.0,  # 9
        3.5,  # 10
        2.0,  # 11
        4.5,  # 12 - leaf node, high PV
        3.0,  # 13
        3.5,  # 14
        4.0,  # 15 - leaf node
        2.5,  # 16
        3.0,  # 17
        2.0,  # 18
        3.5,  # 19 - leaf node
        2.5,  # 20
        3.0,  # 21
        4.0,  # 22 - leaf node
        2.0,  # 23
        4.0,  # 24 - leaf node
        3.0,  # 25
        4.0,  # 26 - leaf node
        3.0,  # 27
        3.5,  # 28
        4.0,  # 29 - leaf node
    ], dtype=torch.float32)

    # ----------------------------------------------------------------
    # USAGE BENEFIT COEFFICIENTS: U_n(D) = -ã_n (D - D*_n)^2 + b̃_n
    # Higher ã_n → prosumer is more sensitive to deviations from target.
    # D_star: target demand (comfort point), set near D_max * 0.7
    # ----------------------------------------------------------------
    a_tilde = torch.tensor([
        0.0,  # 0  - root, no usage benefit
        10.0,  # 1
        10.0,  # 2
        8.0,  # 3
        12.0,  # 4
        12.0,  # 5
        8.0,  # 6
        10.0,  # 7
        10.0,  # 8
        12.0,  # 9
        12.0,  # 10
        10.0,  # 11
        15.0,  # 12
        12.0,  # 13
        12.0,  # 14
        15.0,  # 15
        10.0,  # 16
        12.0,  # 17
        15.0,  # 18
        15.0,  # 19
        10.0,  # 20
        12.0,  # 21
        15.0,  # 22
        12.0,  # 23
        15.0,  # 24
        12.0,  # 25
        15.0,  # 26
        12.0,  # 27
        12.0,  # 28
        15.0,  # 29
    ], dtype=torch.float32)

    D_star = torch.tensor([
        0.0,   # 0
        14.0,   # 1
        14.0,   # 2
        10.0,   # 3
        7.0,   # 4
        7.0,   # 5
        10.0,   # 6
        5.5,   # 7
        5.5,   # 8
        4.0,   # 9
        4.0,   # 10
        5.5,   # 11
        3.5,   # 12
        4.0,   # 13
        4.0,   # 14
        3.5,   # 15
        5.0,   # 16
        4.0,   # 17
        3.5,   # 18
        3.5,   # 19
        5.0,   # 20
        3.5,   # 21
        3.0,   # 22
        3.5,   # 23
        3.0,   # 24
        3.0,   # 25
        3.0,   # 26
        2.0,   # 27
        2.0,   # 28
        2.0,   # 29
    ], dtype=torch.float32)

    # ----------------------------------------------------------------
    # FLEXIBILITY ACTIVATION COST: C_n(G) = 0.5 * a_n * G^2 + b_n * G
    # Constraint from paper: -b_n/a_n >= G_min_n  →  b_n <= -a_n * G_min = 0
    # Node 0 (grid) has low marginal cost; CHPs slightly higher.
    # ----------------------------------------------------------------
    a = torch.tensor([
        1.0,  # 0  - grid: low cost
        4.0,  # 1
        4.0,  # 2
        5.0,  # 3
        4.0,  # 4
        4.0,  # 5
        5.0,  # 6
        6.0,  # 7
        4.0,  # 8
        5.0,  # 9
        4.0,  # 10
        6.0,  # 11
        4.0,  # 12
        6.0,  # 13
        4.0,  # 14
        4.0,  # 15
        5.0,  # 16
        4.0,  # 17
        6.0,  # 18
        4.0,  # 19
        5.0,  # 20
        4.0,  # 21
        4.0,  # 22
        5.0,  # 23
        4.0,  # 24
        4.0,  # 25
        4.0,  # 26
        4.0,  # 27
        4.0,  # 28
        4.0,  # 29
    ], dtype=torch.float32)

    b = torch.zeros(N, dtype=torch.float32)  # b_n = 0 ensures -b_n/a_n = 0 >= G_min = 0

    # ----------------------------------------------------------------
    # PRICE PREFERENCES c_{mn} and INTERCONNECTION CAPACITY k_{mn}
    #
    # c_{mn}: per-unit cost agent n assigns to trading with m.
    #   - Trades with root (node 0) are more expensive (grid tariff).
    #   - Peer trades are cheaper (incentivize local exchange).
    #
    # k_{mn}: max capacity on link (m,n).
    #   - Links to/from root: high capacity.
    #   - Links between peers: moderate capacity.
    # ----------------------------------------------------------------
    c = {}
    k = {}

    for n in range(N):
        for m in neighbors[n]:
            if m == 0 or n == 0:
                # Grid-connected link: higher price (import tariff)
                c[(m, n)] = torch.tensor(0.5, dtype=torch.float32)
                k[(m, n)] = torch.tensor(20.0, dtype=torch.float32)
            elif m <= 2 or n <= 2:
                # Links adjacent to main hubs (nodes 1, 2): medium
                c[(m, n)] = torch.tensor(0.2, dtype=torch.float32)
                k[(m, n)] = torch.tensor(10.0, dtype=torch.float32)
            else:
                # Peer-to-peer links deeper in the tree: cheaper, lower capacity
                c[(m, n)] = torch.tensor(0.1, dtype=torch.float32)
                k[(m, n)] = torch.tensor(5.0, dtype=torch.float32)
    return N, gamma, max_iter, tol, neighbors, D_min, D_max, G_min, G_max, G_res, D_star, a_tilde, a, b, c, k
