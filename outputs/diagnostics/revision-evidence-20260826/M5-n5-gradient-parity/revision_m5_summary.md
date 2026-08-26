# Revision M5 N=5 Gradient Parity

- Predeclared thresholds: rel_error <= 0.0001 or abs_error <= 0.0001; near-zero |grad| < 1e-08 uses abs_error <= 1e-06
- Primary FD step fraction: 0.001
- Cases: 4 (one per phase); pairs: 36 = 4 cases × 3 parameters × 3 outputs
- Residual quality: all four N=5 JAX solves have residual_max_abs in 6.15–20.06 (reported beside every pair; not used to select cases)
- Failed pairs: 1/36 — phase 1 `coal_mass_flow → CO_mol_s` (rel_error=1.33e-4, abs_error=6.91e-4). Remaining 35 pairs pass. This is selected-path consistency, not a reason to hide the miss.
- Interpretation: selected-path gradient consistency on the N=5 JAX/IFT path, not a full-solver differentiability proof. The failed pair remains visible and limits a blanket trust-layer claim that every plant-facing axis matches FD on every case.

Evidence boundary: revision evidence only; not production control, learned multi-alpha, z_d value, oxidation-channel updating, cross-gasifier generalization, or manuscript claim text
