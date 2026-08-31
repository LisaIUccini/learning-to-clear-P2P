"""
diagnose_mono_penalty.py
===========================

Before committing ~2h to a gamma_mono=0 retrain, check cheaply whether the
mono penalty is even a SIGNIFICANT contributor to the loss on this market
using the ALREADY-TRAINED checkpoint (mono=2.0, standard) -- if its
contribution is tiny relative to the main decay term, gamma_mono=0 would
not be expected to change much, and the hypothesis is weaker than assumed.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch

import config
import market
import algorithm
import gnn as gnn_mod
import train as train_mod

CKPT = "runs/cross_density/seed0_density0.8_gamma8_unroll750_rho9925_asymmetric/phi_w.pt"
GAMMA, RHO, SCALE, UNROLL = 8.0, 0.9925, 1.0, 750

device = torch.device("cpu")
mc = market.MarketConfig(density=0.8)
_, _, base_spec = market.sample_base(config.BASE_MARKET_N, mc, seed=0)

model = gnn_mod.PerturbationGNN(gnn_mod.GNNConfig.from_market_config(
    mc, rho=RHO, scale=SCALE, rho_slow=config.GNN_RHO_SLOW, scale_slow=config.GNN_SCALE_SLOW))
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(blob["state_dict"])
model.eval()

train_cfg = train_mod.TrainConfig(gamma=GAMMA, lam=config.DR_LAMBDA, bisect=config.DR_BISECT_STEPS,
                                   unroll=UNROLL, gamma_mono=2.0, eta=0.15, epochs=1,
                                   w_behind=2.0, w_ahead=1.0, loss_mode="asymmetric")

for seed in [1000, 1003, 1006]:
    gen = torch.Generator(device=device); gen.manual_seed(config.TRAIN_SEED * 7919 + seed)
    net = market.build_network(base_spec, device=device, rng=gen)
    x0 = net.flatten().detach()

    _, residuals = train_mod.rollout(net, model, train_cfg, x0=x0, differentiable=False)
    net_dr = market.build_network(base_spec, device=device)
    net_dr.unflatten(x0)
    _, residuals_dr = train_mod.rollout(net_dr, None, train_cfg, x0=x0, differentiable=False)

    scale_vec = model.physical_scale(net)
    denom = torch.linalg.vector_norm(scale_vec) + 1e-8
    rho = torch.stack(residuals) / denom
    rho_dr = (torch.stack(residuals_dr) / denom)
    a = rho - rho_dr
    decay = (2.0 * torch.relu(a) - 1.0 * torch.relu(-a)).mean()
    mono_raw = torch.relu(rho[1:] - rho[:-1]).mean()
    mono_contribution = 2.0 * mono_raw  # gamma_mono=2.0 applied

    n_increases = (rho[1:] > rho[:-1]).sum().item()
    print(f"seed={seed}: decay_term={decay.item():+.6f}  mono_raw={mono_raw.item():.6f}  "
          f"mono_contribution(x2.0)={mono_contribution.item():+.6f}  "
          f"total_loss={decay.item()+mono_contribution.item():+.6f}  "
          f"steps_with_increase={n_increases}/{len(rho)-1}  "
          f"max_single_increase={torch.relu(rho[1:] - rho[:-1]).max().item():.6f}")
