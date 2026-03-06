"""
L2Play — Main entry point

Runs the full pipeline:
    1. Build EnergyNetwork from parameters
    2. Generate dataset (GNE computed offline)
    3. Train GNNUpdate model
    4. Run inference: pure FB vs FB+GNN
    5. Plot and save results

Usage:
    python main.py [--N {3,10,30}] [--load]

Arguments:
    --N     : network size (default: 10)
    --load  : skip training and load pre-trained model
"""

import os
import argparse
import torch
import numpy as np

import init_parameters
from energy_network import EnergyNetwork
from model import GNNUpdate
from dataset import EnergyDataset, compute_gne
from train import meta_train, learned_FB_step
from inference import pure_FB_step, run_comparison, plot_results


def main(N_choice: int = 3, load: bool = True):

    # ------------------------------------------------------------------
    # Device and seeds
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Problem parameters
    # ------------------------------------------------------------------
    init_fn = {3: init_parameters.initN3,
               10: init_parameters.initN10,
               30: init_parameters.initN30}

    if N_choice not in init_fn:
        raise ValueError(f"N must be 3, 10, or 30. Got {N_choice}.")

    (N, gamma, max_iter, tol,
     neighbors, D_min, D_max, G_min, G_max, G_res,
     D_star, a_tilde, a, b, c, k) = init_fn[N_choice]()

    # ------------------------------------------------------------------
    # Build network template
    # ------------------------------------------------------------------
    net_template = EnergyNetwork(
        N, neighbors, D_min, D_max, G_min, G_max, G_res,
        a_tilde, a, b, D_star, c, k, device
    )
    print(net_template)
    print(f"State dimension: {net_template.dim}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    num_samples = 50
    batch_size  = 1
    T           = 15       # unrolling steps
    epochs      = 50
    lr          = 2e-3
    alpha       = 0.95     # discount factor in loss
    d_h         = 12#len(net_template.flatten())  # GNN hidden dimension
    rho         = 0.5      # summability decay

    save_path = (
        f"checkpoints/N3_T{T}_e{epochs}_s{num_samples}_learned_update"
        f"_rho{rho}_lr{lr}.pt"
    )

    print("\nsave_path:", save_path)
    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = GNNUpdate(d_h=d_h, rho=rho).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GNNUpdate — trainable parameters: {n_params}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if load and os.path.exists(save_path):
        print(f"\nLoading pre-trained model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))

    else:
        if load:
            print(f"Warning: model not found at {save_path}. Training from scratch.")

        print("\n--- Dataset generation ---")
        dataset = EnergyDataset(
            num_samples  = num_samples,
            net_template = net_template,
            gamma        = gamma,
            gne_max_iter = 200,
            gne_tol      = 1e-6,
            device       = device,
        )

        print("\n--- Meta-training ---")
        model, loss_history = meta_train(
            model        = model,
            dataset      = dataset,
            net_template = net_template,
            gamma        = gamma,
            T            = T,
            epochs       = epochs,
            lr           = lr,
            alpha        = alpha,
            batch_size   = batch_size,
            device       = device,
            save_path    = save_path,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("\n--- Inference ---")

    # Use a different seed for test instances
    torch.manual_seed(0)
    np.random.seed(0)

    n_test      = 5
    all_results = []

    os.makedirs("figures", exist_ok=True)

    for i in range(n_test):
        print(f"\nTest instance {i+1}/{n_test}:")

        # Random initial state (unseen during training)
        net_init = net_template.clone()
        net_init.D, net_init.G, net_init.Q = net_init._init_state()

        # Run pure FB vs FB+GNN
        results = run_comparison(
            net_init = net_init,
            model    = model,
            gamma    = gamma,
            max_iter = max_iter,
            tol      = tol,
            device   = device,
        )
        all_results.append(results)

        # Plot individual instance
        plot_results(
            results,  # only residuals
            save_path=f"figures/N{N}_test_instance_{i+1}.png"
        )
        print(f"Pure FB     — has violation? "
          f"{results['pure'].constraint_violation()}")
        print(f"FB + GNN    — has violation? "
          f"{results['learned'].constraint_violation()}")

    # ------------------------------------------------------------------
    # Aggregate plot over all test instances
    # ------------------------------------------------------------------
    max_len = max(len(r["residuals_pure"]) for r in all_results)

    def pad_and_mean(key):
        padded = []
        for r in all_results:
            seq = r[key]
            pad = [seq[-1]] * (max_len - len(seq))
            padded.append(seq + pad)
        return np.array(padded).mean(axis=0).tolist()

    avg_results = {
        key: pad_and_mean(key)
        for key in ["residuals_pure", "residuals_learned"]
    }

    plot_results(
        avg_results,
        save_path=f"figures/N{N}_average_comparison.png"
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"Pure FB     — final avg residual to GNE: "
          f"{avg_results['residuals_pure'][-1]:.6f}")
    print(f"FB + GNN    — final avg residual to GNE: "
          f"{avg_results['residuals_learned'][-1]:.6f}")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2Play — GNN accelerated FB")
    parser.add_argument("--N",    type=int,  default=10, choices=[3, 10, 30],
                        help="Network size (default: 3)")
    parser.add_argument("--load", action="store_true", default=False,
                        help="Load pre-trained model instead of training")
    args = parser.parse_args()

    main(N_choice=args.N, load=True)#args.load)