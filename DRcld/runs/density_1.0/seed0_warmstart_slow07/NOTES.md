# density_1.0/seed0_warmstart_slow07

**Test**: warm-start dual-channel experiment. Model initialised from
`runs/density_1.0/epochs30_gamma10_unroll373_scale07/phi_w.pt` (fast channel + shared encoder already trained, slow-channel
heads at random init since scale_slow=0 gave them zero gradient in that
run). Shared encoder (phi_msg, phi_upd) and fast-channel heads (phi_node,
phi_edge, phi_gate) FROZEN (requires_grad=False) for this run -- only the
slow-channel heads (phi_node_slow, phi_edge_slow, phi_gate_slow) are
trained. Because the two channels are additive and the fast channel's
weights are unchanged, its own contribution is mathematically identical to
the checkpoint's; only the slow channel's ADDITIVE contribution differs.

**Config**: density=1.0, base_seed=0, gamma=10.0,
unroll=750, rho_fast=0.9729, scale_fast=0.7,
rho_slow=0.994, scale_slow=0.7, epochs=30, eta=0.15,
25 train instances, 30 test instances.
