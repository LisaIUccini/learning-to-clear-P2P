"""
L2Play — Main entry point

Runs the full pipeline:
    1. Build EnergyNetwork from parameters
    2. Generate dataset
    3. Train GNNUpdate model
    4. Run inference: pure vs learned (FB or ADMM)
    5. Plot and save results

Usage:
    python main.py [--N {3,10,30}] [--load] [--algorithm {fb,admm}]

Arguments:
    --N         : network size (default: 3)
    --load      : skip training and load pre-trained model
    --algorithm : "fb" (default) or "admm"
"""

import os
import argparse
import torch
import numpy as np

import init_parameters
from model import GNNUpdate
from dataset import EnergyDataset
from train import meta_train
from inference import run_comparison
from plot import plot_results, plot_results_std, plot_fig


def main(N_choice: int = 3, N_train: int = 3, load: bool = True,
         algorithm: str = "fb"):

    # ------------------------------------------------------------------
    # Validate algorithm choice
    # ------------------------------------------------------------------
    if algorithm not in ("fb", "admm"):
        raise ValueError(f"algorithm must be 'fb' or 'admm', got '{algorithm}'")

    # ------------------------------------------------------------------
    # Device and seeds
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Algorithm: {algorithm.upper()}")
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Problem parameters
    # ------------------------------------------------------------------
    init_fn = {
        3:  init_parameters.initN3,
        10: init_parameters.initN10,
        30: init_parameters.initN30,
    }
    if N_choice not in init_fn:
        raise ValueError(f"N must be 3, 10, or 30. Got {N_choice}.")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    num_samples = 200
    batch_size  = 1
    T           = 30        # unrolling steps
    epochs      = 50
    lr          = 1e-3
    alpha       = 0.95      # kept for API compatibility
    d_h         = 12        # GNN hidden dimension
    rho_gnn     = 0.7       # GNN summability decay (rho in GNNUpdate)
    rho_admm    = 5      # ADMM penalty parameter

    # Build network for training (needed before model construction)
    net_train, gamma, max_iter, tol = init_fn[N_train](device=device)

    # Algorithm-specific parameter passed to training and inference
    algo_kwargs = {"gamma": gamma} if algorithm == "fb" else {"rho": rho_admm}

    save_path = (
        f"checkpoints/{algorithm}_N{N_train}_T{T}_e{epochs}"
        f"_s{num_samples}_learned_update_rho{rho_gnn}_lr{lr}.pt"
    )
    print(f"\nCheckpoint path: {save_path}")

    # ------------------------------------------------------------------
    # Model  (feat sizes depend on algorithm)
    # ------------------------------------------------------------------
    # FB:   node_feat_size=2 [D, G],        edge_feat_size=1 [q]
    # ADMM: node_feat_size=3 [D, G, u_avg], edge_feat_size=2 [q, z]
    node_feat_size = 3 if algorithm == "admm" else 2
    edge_feat_size = 2 if algorithm == "admm" else 1

    model = GNNUpdate(
        d_h            = d_h,
        rho            = rho_gnn,
        node_feat_size = node_feat_size,
        edge_feat_size = edge_feat_size,
    ).to(device)
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

        print(f"\n--- Dataset generation (N={N_train}) ---")
        dataset = EnergyDataset(
            num_samples  = num_samples,
            net_template = net_train,
            device       = device,
        )

        print(f"\n--- Meta-training ({algorithm.upper()}) ---")
        model, loss_history = meta_train(
            model        = model,
            dataset      = dataset,
            net_template = net_train,
            T            = T,
            epochs       = epochs,
            lr           = lr,
            alpha        = alpha,
            batch_size   = batch_size,
            device       = device,
            save_path    = save_path,
            algorithm    = algorithm,
            **algo_kwargs,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print(f"\n--- Inference ({algorithm.upper()}) ---")

    torch.manual_seed(0)
    np.random.seed(0)

    n_test      = 5
    all_results = []
    os.makedirs("figures", exist_ok=True)

    for i in range(n_test):
        print(f"\nTest instance {i+1}/{n_test}:")

        net_test, gamma_test, max_iter_test, tol_test = init_fn[N_choice](device=device)

        init_parameters.plot_graph(
            neighbors = net_test.neighbors,
            save_path = f"figures/N{net_test.N}_graph_{i+1}.png",
        )

        net_init = net_test.clone()
        net_init._init_state()

        # Inference algo_kwargs use test gamma/rho
        if algorithm == "fb":
            inf_kwargs = {"gamma": gamma_test}
        else:
            inf_kwargs = {"rho": rho_admm}

        results = run_comparison(
            net_init  = net_init,
            model     = model,
            max_iter  = max_iter_test,
            tol       = tol_test,
            device    = device,
            algorithm = algorithm,
            **inf_kwargs,
        )
        all_results.append(results)

        plot_results(
            results,
            save_path=f"figures/{algorithm}_N{N_choice}_test_istance_{i+1}.png"
        )

        print(f"Pure {algorithm.upper()}      — constraint violation: "
              f"{results['pure'].constraint_violation()}")
        print(f"{algorithm.upper()} + GNN    — constraint violation: "
              f"{results['learned'].constraint_violation()}")
        plot_fig(results['pure'], save_path=f"figures/{algorithm}_N{N_choice}_graph_result_pure_{i+1}.png")
        plot_fig(results['learned'], save_path=f"figures/{algorithm}_N{N_choice}_graph_result_learned_{i+1}.png")
        
    # ------------------------------------------------------------------
    # Aggregate plot
    # ------------------------------------------------------------------
    max_len = max(len(r["residuals_pure"]) for r in all_results)

    def pad_and_mean(key):
        padded = []
        for r in all_results:
            seq = r[key]
            pad = [seq[-1]] * (max_len - len(seq))
            padded.append(seq + pad)
        return np.array(padded).mean(axis=0).tolist()

    def pad_and_stack(key):
        padded = []
        for r in all_results:
            seq = r[key]
            pad = [seq[-1]] * (max_len - len(seq))
            padded.append(seq + pad)
        return np.array(padded)

    avg_results = {
        key: pad_and_mean(key)
        for key in ["residuals_pure", "residuals_learned"]
    }
    stacked = {
        key: pad_and_stack(key)
        for key in ["residuals_pure", "residuals_learned"]
    }
    std_results = {
        key: stacked[key].std(axis=0).tolist()
        for key in ["residuals_pure", "residuals_learned"]
    }

    plot_results(
        avg_results,
        save_path=f"figures/{algorithm}_N{N_choice}_average.png"
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(f"\n--- Summary ({algorithm.upper()}) ---")
    print(f"Pure {algorithm.upper()}   — final avg residual: "
          f"{avg_results['residuals_pure'][-1]:.6f}")
    print(f"{algorithm.upper()} + GNN — final avg residual: "
          f"{avg_results['residuals_learned'][-1]:.6f}")
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2Play — GNN accelerated FB/ADMM")

    parser.add_argument(
        "--N", type=int, default=3, choices=[3, 10, 30],
        help="Network size for inference (default: 3)"
    )
    parser.add_argument(
        "--N_train", type=int, default=3, choices=[3, 10, 30],
        help="Network size for training (default: 3)"
    )
    parser.add_argument(
        "--load", action="store_true", default=False,
        help="Load pre-trained model instead of training"
    )
    parser.add_argument(
        "--algorithm", type=str, default="admm", choices=["fb", "admm"],
        help="Algorithm to use: 'fb' (default) or 'admm'"
    )
    args = parser.parse_args()

    main(
        N_choice  = args.N,
        N_train   = args.N_train,
        load      = args.load,
        algorithm = args.algorithm,
    )