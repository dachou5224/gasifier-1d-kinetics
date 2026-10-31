# Cell-0 residual decomposition diagnostic

This is a fixed-physics diagnostic artifact, not a solver fix or calibration result.
It is generated from the current corrected ModelCase preview; earlier round-27 three-case outputs are superseded/not directly comparable after the coal-flow boundary correction.

- Rows: 3
- N_cells coverage: 1, 3, 5
- Poor-quality threshold: residual_cost > 0.0001
- Residual cost definition: `0.5 * sum(scaled_residual_i^2)`
- Residual normalization: Cell.residuals returns scaled residuals: gas species except N2 are divided by cell.ref_flow; N2 is divided by max(abs(inlet_N2)+abs(src_N2), 1e-3); solid_mass by cell.ref_solid; carbon_fraction by max(char_Xc0, 0.1); energy by max(ref_flow*35*200, 5e5). residual_cost = 0.5*sum(scaled_residual_i^2); poor-quality threshold is cost > 1e-4.
- Dominant residual counts: gas_H2=3

| case_id | IG | N | cost | max_abs | top1 | top1_value | top2 | top2_value | top3 | top3_value | poor_cells | dCO | dH2 | dCO2 | d_flow |
|---|---|---:|---:|---:|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|
| sy3_trial_0928_feed_product__modelcase_001 | IG001 | 1 | 0.00620302 | 0.0862631 | gas_H2 | 0.0862631 | gas_CO2 | -0.0684322 | gas_CO | 0.010672 | 0 | 17.6 | -28.2 | 8.45 | -0.851 |
| sy3_trial_0928_feed_product__modelcase_001 | IG001 | 3 | 0.00666669 | 0.0907763 | gas_H2 | 0.0907763 | gas_CO2 | -0.0683774 | gas_O2 | 0.0114413 | 0 | 4.09 | -5.5 | 1 | -0.304 |
| sy3_trial_0928_feed_product__modelcase_001 | IG001 | 5 | 0.00743858 | 0.0949349 | gas_H2 | 0.0949349 | gas_CO2 | -0.0733066 | gas_O2 | 0.011975 | 0 | 4.98 | -4.32 | -1.07 | -0.225 |

Interpretation boundary: a persistent cell-0 residual-quality flag means the first-cell scaled residual is above the recorded threshold; it does not indicate fallback or hidden solver threshold relaxation. Outlet deltas are included so reviewers can separate local residual quality from fixed-physics outlet mismatch.
