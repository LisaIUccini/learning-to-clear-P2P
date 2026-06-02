
"""
L2Play — Main entry point.
 
Pipeline:
    1. Build an EnergyNetwork from init_parameters.
    2. Generate a dataset of initial states.
    3. Train (or load) the GNNUpdate model.
    4. Run inference: pure ADMM vs learned ADMM, averaged over N_TEST instances.
    5. Save plots and print a summary.
 
All hyperparameters are defined in config.py.
"""
 
import os
import torch
import numpy as np
 
import config
import init_parameters
from model     import GNNUpdate
from dataset   import EnergyDataset
from train     import meta_train
from inference import run_comparison
from plot      import plot_results_std, plot_loss
 
 
# ------------------------------------------------------------------
# Init dispatch
# ------------------------------------------------------------------
 
_FIXED_INIT = {
    3:  init_parameters.initN3,
    4:  init_parameters.initN4,
    10: init_parameters.initN10,
    30: init_parameters.initN30,
}
 
 
def init_fn(N: int, device: torch.device, seed: int = 42) -> tuple:
    """Return (net, max_iter, tol, rho_admm) for any N."""
    if N in _FIXED_INIT:
        return _FIXED_INIT[N](device=device)
    return init_parameters.initN(N=N, seed=seed, device=device)
 
 
# ------------------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------------------
 
def _pad_and_stack(results_list: list, key: str, max_len: int) -> np.ndarray:
    """Pad each residual sequence with its last value, then stack."""
    padded = []
    for r in results_list:
        seq = r[key]
        padded.append(seq + [seq[-1]] * (max_len - len(seq)))
    return np.array(padded)
 
 
def _log_mean_lo_hi(results_list: list, key: str, max_len: int):
    """Compute log-mean and log-std bands for residual sequences."""
    arr      = _pad_and_stack(results_list, key, max_len)
    log_arr  = np.log10(np.clip(arr, 1e-30, None))
    log_mean = log_arr.mean(axis=0)
    log_std  = log_arr.std(axis=0)
    mean = 10 ** log_mean
    lo   = 10 ** (log_mean - log_std)
    hi   = 10 ** (log_mean + log_std)
    return mean.tolist(), lo.tolist(), hi.tolist()
 
 
# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
 
def main():
    device = config.DEVICE
    print(f"Device: {device}")
    torch.manual_seed(config.SEED_TRAIN)
    np.random.seed(config.SEED_TRAIN)
 
    # --- Network for training ---------------------------------------
    net_train, _, _, _,  rho_admm_train = init_fn(config.N_TRAIN, device=device)
 
    save_path = config.checkpoint_path()
    print(f"\nCheckpoint path: {save_path}")
 
    # --- Model ------------------------------------------------------
    model = GNNUpdate(
        d_h            = config.D_H,
        rho            = config.RHO_GNN,
        node_feat_size = config.NODE_FEAT_SIZE,
        edge_feat_size = config.EDGE_FEAT_SIZE,
    ).to(device)
    print(f"GNNUpdate — trainable parameters: "
          f"{sum(p.numel() for p in model.parameters())}")
 
    # --- Training ---------------------------------------------------
    if config.LOAD and os.path.exists(save_path):
        print(f"\nLoading pre-trained model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        if config.LOAD:
            print(f"Warning: model not found at {save_path}. "
                  f"Training from scratch.")
 
        print(f"\n--- Dataset generation "
              f"(N={config.N_TRAIN}, rho_admm={rho_admm_train}) ---")
        dataset = EnergyDataset(
            num_samples  = config.NUM_SAMPLES,
            net_template = net_train,
            device       = device,
        )
 
        print(f"\n--- Meta-training (ADMM) ---")
        model, loss_history = meta_train(
            model        = model,
            dataset      = dataset,
            net_template = net_train,
            T            = config.T_UNROLL,
            epochs       = config.EPOCHS,
            lr           = config.LR,
            alpha        = config.ALPHA,
            batch_size   = config.BATCH_SIZE,
            device       = device,
            save_path    = save_path,
            rho          = rho_admm_train,
        )
        plot_loss(loss_history, config.loss_history_path())
 
    phi_m = float(torch.nn.functional.softplus(model.phi_m_raw))
    print(f"phi_m (GNN amplitude) = {phi_m:.4f}")
 
    # --- Inference --------------------------------------------------
    print(f"\n--- Inference (ADMM) ---")
    torch.manual_seed(config.SEED_TEST)
    np.random.seed(config.SEED_TEST)
 
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    all_results = []
 
    for i in range(config.N_TEST):
        print(f"\nTest instance {i+1}/{config.N_TEST}:")
 
        net_test, _, max_iter_test, tol_test, rho_test = init_fn(
            config.N_CHOICE, device=device,
            seed=config.SEED_TRAIN + i,
        )
        net_init = net_test.clone()
        net_init._init_state()
 
        results = run_comparison(
            net_init = net_init,
            model    = model,
            max_iter = max_iter_test,
            tol      = tol_test,
            device   = device,
            rho      = rho_test,
        )
        all_results.append(results)
 
        print(f"Pure ADMM   — constraint violation: "
              f"{results['pure'].constraint_violation()}")
        print(f"ADMM + GNN  — constraint violation: "
              f"{results['learned'].constraint_violation()}")
 
    # --- Aggregate plot ---------------------------------------------
    max_len = max(len(r["residuals_pure"]) for r in all_results)
    avg_results, lo_results, hi_results = {}, {}, {}
    for key in ("residuals_pure", "residuals_learned"):
        m, l, h = _log_mean_lo_hi(all_results, key, max_len)
        avg_results[key] = m
        lo_results[key]  = l
        hi_results[key]  = h
 
    plot_results_std(
        avg_results,
        lo        = lo_results,
        hi        = hi_results,
        save_path = config.aggregate_figure_path(),
    )
 
    # --- Summary ----------------------------------------------------
    print(f"\n--- Summary (ADMM) ---")
    print(f"Pure ADMM   — final avg residual: "
          f"{avg_results['residuals_pure'][-1]:.6f}")
    print(f"ADMM + GNN  — final avg residual: "
          f"{avg_results['residuals_learned'][-1]:.6f}")
 
 
if __name__ == "__main__":
    main()
 