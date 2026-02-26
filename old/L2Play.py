import math
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch.utils.data import Dataset, DataLoader
from modelsLearning import LearnedUpdate

from old.utils_torch import F_torch, flatten_state, project_all_agents_global_torch, project_all_agents_global_approx , unflatten_state, build_edge_list, initialize_DGq_torch, projection_loss_torch, cost
from plot_results import plot_fig, plot_convergence

# ---------------------------------------------------------
#  Device & seeds
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)
np.random.seed(1234)

# =============== Initialize problem parameters ===============

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

for n in neighbors:
    print(f"Node {n} neighbors: {neighbors[n]} D_min={D_min[n].item()} D_max={D_max[n].item()} G_min={G_min[n].item()} G_max={G_max[n].item()} G_res={G_res[n].item():.2f} D_star={D_star[n].item():.2f} a_tilde={a_tilde[n].item():.2f} a={a[n].item():.2f} b={b[n].item():.2f}")

# ======================= Dataset =======================

def forward_backward(D, G, q,  training,  gamma=0.1, max_iter=100, tol=1e-6):
    
        
    # --- FORWARD ---
    F_D, F_G, F_q = F_torch(D, G, N, a_tilde, a, b, D_star, c, neighbors, device)
    D_tmp = D - gamma * F_D
    G_tmp = G - gamma * F_G
    q_tmp = {key: q[key] - gamma * F_q[key] for key in q}


    # --- PROJECTION  ---
    if training:
        D_new, G_new, q_new = project_all_agents_global_approx(D_tmp,G_tmp,q_tmp, N, neighbors, G_res, D_min, D_max, G_min, G_max, k, device)
    else:
        D_new, G_new, q_new = project_all_agents_global_torch(D_tmp,G_tmp,q_tmp, N, neighbors, G_res, D_min, D_max, G_min, G_max, k, device)

    # Convergence Condition
    x_new = flatten_state(D_new, G_new, q_new, neighbors)
  

    return x_new


class ProjectionDataset(Dataset):
    """
    Fixed dataset: points in [0,2)^3, stored once in self.data.
    Every epoch we iterate over the SAME points.
    """

    def __init__(self, num_samples, device):
        self.num_samples = num_samples
        self.device = device
        self.samples = []
        for _ in range(num_samples):
            D, G, q = initialize_DGq_torch(D_min, D_max, G_min, G_max, neighbors, k, N)
            x = flatten_state(D, G, q, neighbors)
            self.samples.append(x)

        self.data = torch.stack(self.samples).to(device)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

# =============== Pure FB step (no learning) ===============

def FB_step(x, training:bool):
    D, G, q = unflatten_state(x, N, neighbors)
    x_next = forward_backward(D, G, q, training)

    return x_next

# =============== Learned FB step (no learning) ===============

def learned_FB_step(x, model, t:int, training:bool):
    """
    Learned FB step, Option A:

        x_{k+1} = T(x_k) + u_k,
        u_k = model(x_k, t)

    where T is the pure FB operator (no dual increments).
    """
    x_base = FB_step(x, training)#.float().to(device)
    
    if not x_base.requires_grad and training:
        x_base.requires_grad_(True)

    x_in = x_base.unsqueeze(0).unsqueeze(-1)   # (1, d, 1)
    update = model(x_in, t).squeeze(0)    # (d,)
    if training:
        x_next =  x_base +  update
    else:
        
        x_next =  0.8 * x_base + 0.2 * update
    
    """
    if t<13:
        x_next =  x_base +  update
    else:
        x_next = x_base.clone() # pure FB after 13 steps
    
        
    if t%10 == 0:
        print("Step", t)
        print("||x_base||", x_base.norm().item())
        print("||update||", update.norm().item())
        print("||x_next||", x_next.norm().item())
    """

    D, G, q = unflatten_state(x_next, N, neighbors)
    
    D_proj, G_proj, q_proj = project_all_agents_global_approx(D, G, q, N, neighbors, G_res, D_min, D_max, G_min, G_max, k, device)
    
    x_next_proj = flatten_state(D_proj, G_proj, q_proj, neighbors)
    
    return x_next_proj

# =============== Meta-training (L2O, NO true projection) ===============   
def meta_training_proj(model, dataloader, meta_optimizer,
                       T, device, epochs):
    """
    Meta-train a learned inexact baDR step (Option A) for A∩B.

    For each v in the batch:
      - Start x_0 = v.
      - Run T learned steps.
      - Loss is a feasibility-like loss using step differences:
            sum_t 0.95^(T-t) * ||x_{t+1} - x_t||.
    """

    model.train()
    meta_loss_epoch_list = []

    for epoch in range(epochs):
        meta_loss_epoch = 0.0

        for v_batch in dataloader:   
            v_batch = v_batch.to(device)
            meta_optimizer.zero_grad()

            x = v_batch.squeeze(0).clone() 
            meta_loss = torch.zeros(1, device=device)

            for t in range(T):
                x_old = x.clone()
                x_next = learned_FB_step(x, model, t, True)  # x is the next iterate after the learned FB step

                # step-difference residual as in your current meta-loss 
                D_n, G_n, q_n = unflatten_state(x_next, N, neighbors)
                cost_next = cost(N, D_n, G_n, q_n, a_tilde, a, b, D_star, c, neighbors)
                meta_loss = meta_loss + cost_next + torch.norm(x_next-x_old) *0.5 ** (T - t)  # (T-t) non c'era nel test T300e150)
                x = x_next.clone()  # update x for the next iteration

            meta_loss.backward()
            meta_optimizer.step()
            meta_loss_epoch += meta_loss.item()
        meta_loss_epoch_list.append(meta_loss_epoch)
        print("------------------------------------------------------------")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {meta_loss_epoch:.6f}")
        print("------------------------------------------------------------")
    
    plot_convergence(meta_loss_epoch_list, [],  "meta_training_loss.png", "Meta-Training Loss over Epochs")

    return model



def main():
    # Fixed dataset
    d = 2 * N + len(build_edge_list(neighbors))  # dimension of the flattened state
    training_samples = 20
    batch_size = 1
    training_dataset = ProjectionDataset(training_samples, device) #d*training_samples
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    learned_update = LearnedUpdate(d, q=0, rho=0.95).to(device) 
    meta_optimizer = torch.optim.Adam(learned_update.parameters(), lr=2e-3)

    T = 50
    epochs = 20
    load = True # se True, salta la meta-training e carica un modello pre-trained (assicurati di avere il file .pt nella cartella)
    if load: 
        # Load the pre-trained learned optimizer parameters
        directory_path = './'
        file_path = os.path.join(directory_path, 'learned_update_T'+str(T)+'e'+str(epochs)+'.pt')
        if os.path.exists(file_path):
            print(f"Loading pre-trained learned optimizer from {file_path}")
            learned_update.load_state_dict(torch.load(file_path))
            print("Pre-trained learned optimizer loaded successfully.")
        else:
            print(f"Pre-trained model not found at {file_path}.")
    else:
        # Meta-training with pure baDR base operator
        learned_update = meta_training_proj(learned_update, training_dataloader, meta_optimizer, T, device, epochs)
        directory_path = './'
        os.makedirs(directory_path, exist_ok=True)

        file_path = os.path.join(directory_path, f'learned_update_T'+str(T)+'e'+str(epochs)+'.pt')
        torch.save(learned_update.state_dict(), file_path)
        print(f"Trained learned optimizer saved to {file_path}")

    # =============== Real exe =============== 
    
    D, G, q = initialize_DGq_torch(D_min, D_max, G_min, G_max, neighbors, k, N)
    x = flatten_state(D, G, q, neighbors)
    x_base = x.clone()

    
    x_in = x.unsqueeze(0).unsqueeze(-1)   # (1, d, 1)
    update = learned_update(x_in, 0).squeeze(0)   # (d,)
    x = x + update
    

    residuals_l2p = []
    residuals = []

    for it in range(max_iter):
        x_base_new = FB_step(x_base, False)
        err_l = torch.sum((x_base_new - x_base) ** 2) #torch.norm(x_next - x).item()
        residuals.append(err_l.detach().cpu().numpy())

        x_next  = learned_FB_step(x, learned_update, it, False)
        err = torch.sum((x_next - x) ** 2) #torch.norm(x_next - x).item()
        residuals_l2p.append(err.detach().cpu().numpy())
        
        x = x_next.clone()     
        x_base = x_base_new.clone()

         
        if err < tol or err_l < tol:
            print(f"Convergenza raggiunta a iterazione {it}")
            break
        

    Dl, Gl, ql = unflatten_state(x, N, neighbors)
    D, G, q = unflatten_state(x_base, N, neighbors)

    # final results plot
    Ql = np.zeros(N)
    print("Risultati finali:")
    print("D =", Dl)
    print("G =", Gl)
    print("q =\n", ql)
    for n in range(N):
        Qn = sum(ql[(m, n)] for m in neighbors[n])
        Ql[n]=Qn
    print("COST L2Play: ", cost(N, Dl, Gl, ql, a_tilde, a, b, D_star, c, neighbors))
    plot_fig(N,Dl,Gl,ql,Ql, G_res, k, "graph_visualization_LFB.png")

    Q = np.zeros(N)
    for n in range(N):
        Qn = sum(q[(m, n)] for m in neighbors[n])
        Q[n]=Qn
        
    print("COST FB: ", cost(N, D, G, q, a_tilde, a, b, D_star, c, neighbors))
    plot_fig(N,D,G,q,Q, G_res, k)

    #plot_convergence(residuals_l2p, "plot_convergence_LFB_x+u.png", "Convergence of Learned FB x+u")
    plot_convergence(residuals, residuals_l2p, "plot_convergence.png", "Convergence FB base vs with learned update")

if __name__ == "__main__":
    main()