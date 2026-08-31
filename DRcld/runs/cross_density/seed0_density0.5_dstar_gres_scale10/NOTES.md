# cross_density/seed0_density0.5_dstar_gres_scale10

**Test**: zero-shot inference (NO retraining). Weights: `runs/cross_density/seed0_density0.5_gamma6_unroll305_scale10_asymmetric/phi_w.pt`.

**Dataset**: base market density=0.5, base_seed=0 (N=10).
Graph, a/a_tilde/b/c, kappa, and box bounds (D_min/D_max/G_min/G_max) ALL
held exactly fixed. Per instance, D_star *= f_D and G_res *= f_G, with
f_D, f_G independently drawn ~ U(0.90, 1.10), each SHARED
across all N nodes (one f_D, one f_G per instance). D_star is clipped back
into [D_min, D_max] after scaling. 0/30 instances
flagged infeasible by the Assumption-1 check.

**Question answered**: how does a model trained only on x0-diversity (this
exact market, unperturbed) perform when D* and G_res individually drift by
up to 10%, with everything else -- including the boxes it was
implicitly calibrated against -- held fixed?
