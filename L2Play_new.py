import torch
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
from modelsLearning import LearnedUpdate


from energy_network import EnergyNetwork

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
max_iter = 18
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

def build_edge_list(neighbors):
        """
        Build a fixed, ordered list of directed edges (m, n)
        such that m in neighbors[n].
        """
        edge_list = []
        for n in sorted(neighbors.keys()):
            for m in sorted(neighbors[n]):
                edge_list.append((m, n))
        return edge_list
# ======================= Dataset =======================


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
            network = EnergyNetwork(N, neighbors, D_min, D_max, G_min, G_max, G_res, a_tilde, a, b, D_star, c, k, device)
            
            self.samples.append(network.__getflattenstate__())

        self.data = torch.stack(self.samples).to(device)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

# =============== Pure FB step (no learning) ===============
def FB_step(network,  training,  gamma=0.1, max_iter=100, tol=1e-6):
    
    n = network.clone()
    # --- FORWARD ---
    F_D, F_G, F_q = n.F()
    D_tmp = n.D - gamma * F_D
    G_tmp = n.G - gamma * F_G
    q_tmp = {key: n.q[key] - gamma * F_q[key] for key in n.q}

    n.__setstate__(D_tmp, G_tmp, q_tmp)

    # --- PROJECTION  ---
    if training:
        n.project_all_agents_global_approx()
    else:
        n.project_all_agents_global_torch()

    return n

# =============== Learned FB step (no learning) ===============

def learned_FB_step(n:EnergyNetwork, model, t:int, training:bool):
    """
    Learned FB step, Option A:

        x_{k+1} = T(x_k) + u_k,
        u_k = model(x_k, t)

    where T is the pure FB operator and u_k is the learned update.
    """
    n_new = FB_step(n, training)
    
    if not n_new.x.requires_grad and training:
        n_new.x.requires_grad_(True)

    x_in = n_new.x.unsqueeze(0).unsqueeze(-1)   # (1, d, 1)
    update = model(x_in, t).squeeze(0)    # (d,)
    
    n_new.__setflattenstate__(n_new.x +  update)
    

    n_new.project_all_agents_global_approx()
    return n_new, update.norm().item()
    

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

            n = EnergyNetwork(N, neighbors, D_min, D_max, G_min, G_max, G_res, a_tilde, a, b, D_star, c, k, device)
            n.__setflattenstate__(v_batch.squeeze(0))  # inizializza lo network con lo stato flattenizzato del batch (qui batch_size=1, quindi squeeze(0) per rimuovere la dimensione del batch)
            
            meta_loss = torch.zeros(1, device=device)

            for t in range(T):
                n_old = n.clone()
                n_new,_ = learned_FB_step(n, model, t, True)  # x is the next iterate after the learned FB step

                meta_loss = meta_loss + torch.norm(n_new.x-n_old.x) * 0.95 ** (T - t)  # (T-t) non c'era nel test T300e150)
                n = n_new.clone()  # update n for the next iteration

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
    training_samples = 50
    batch_size = 1
    training_dataset = ProjectionDataset(training_samples, device) #d*training_samples
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    learned_update = LearnedUpdate(d, q=0, rho=0.5).to(device) #esempio 0 -> 0.3 random->0.5
    meta_optimizer = torch.optim.Adam(learned_update.parameters(), lr=2e-3) #esempio 0 -> 1e-1  random -> 2e-3

    T = 15
    epochs = 50
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
    
    n_base = EnergyNetwork(N, neighbors, D_min, D_max, G_min, G_max, G_res, a_tilde, a, b, D_star, c, k, device)
    
    n_pure_base = n_base.clone()

    
    x_in = n_base.x.unsqueeze(0).unsqueeze(-1)   # (1, d, 1)
    update = learned_update(x_in, 0).squeeze(0)   # (d,)
    n_base.__setflattenstate__(update)
    

    residuals_l2p = []
    residuals = []
    upd = []

    for it in range(max_iter):
        n_pure_new = FB_step(n_pure_base, False) # pure FB step
        
        err_l = torch.sum((n_pure_new.x - n_pure_base.x) ** 2) #torch.norm(x_next - x).item()
        residuals.append(err_l.detach().cpu().numpy())

        n_l_new, update  = learned_FB_step(n_base, learned_update, it, False)
        err = torch.sum((n_l_new.x - n_base.x) ** 2) #torch.norm(x_next - x).item()
        residuals_l2p.append(err.detach().cpu().numpy())
        upd.append(update)
        
        n_base = n_l_new.clone()     
        n_pure_base = n_pure_new.clone()

             
        if err < tol or err_l < tol:
            print(f"Convergenza raggiunta a iterazione {it}")
            break
        

    # final results plot
    print("COST L2Play: ", n_base.cost())
    Dl,Gl,ql = n_base.__getstate__()
    plot_fig(N,Dl,Gl,ql,n_base.getQ(), G_res, k, "graph_visualization_LFB.png")

    print("COST FB: ", n_pure_base.cost())
    D,G,q = n_pure_base.__getstate__()
    plot_fig(N,D,G,q,n_pure_base.getQ(), G_res, k)

    #plot_convergence(residuals_l2p, "plot_convergence_LFB_x+u.png", "Convergence of Learned FB x+u")
    
    plot_convergence(upd, [], "increment update", "", "update_convergence.png", "Convergence learned update")
    plot_convergence(residuals, residuals_l2p, "pure FB", "FB with learned update", "plot_convergence.png", "Convergence FB base vs with learned update")

if __name__ == "__main__":
    main()