# Pilot fixed-physics diagnostic baseline

This is a diagnostic smoke/baseline artifact, not calibrated accuracy evidence.

- Rows: 2
- N_cells coverage: 3, 5
- Solver path: GasifierSystem.solve(solver_method='minimize', jacobian_mode='scipy')
- Poor convergence threshold recorded, not relaxed: cost > 1e-4

| case_id | IG | N | runtime_s | poor_cells | T_out_C | CO | H2 | CO2 | dry_flow | dCO | dH2 | dCO2 | d_flow |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sy3_trial_0928_feed_product__modelcase_001 | IG001 | 3 | 2.386 | 0 | 1315.35 | 65.80 | 22.67 | 10.39 | 1.244 | 4.09 | -5.50 | 1.00 | -0.304 |
| sy3_trial_0928_feed_product__modelcase_001 | IG001 | 5 | 2.096 | 0 | 1209.13 | 66.69 | 23.85 | 8.31 | 1.322 | 4.98 | -4.32 | -1.07 | -0.225 |

Evidence boundary: reproducible pilot fixed-physics diagnostic baseline. Do not interpret as calibrated validation or industrial generalization.
