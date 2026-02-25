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

