# Huayi influence-tracing diagnostic

This bundle connects 22-case fixed-physics sensitivity, fixed-branch FD-vs-IFT/JAX availability, and cell0 residual localization.

- Cases: 22
- Phases: phase1, phase2, phase3, phase4
- N_cells: 5
- Long-form rows: 440
- Sensitivity OK rows: 440

| rank | axis | mean_abs_norm_sens | max_abs_norm_sens | gradient_support | parity_ok_rows | dominant_cell0_status |
|---:|---|---:|---:|---|---:|---|
| 1 | oxygen_flow | 0.659251 | 1.06346 | supported_fixed_branch_not_tested_full_model | 5 | persistent_cell0_residual_quality_flag |
| 2 | coal_mass_flow | 0.482625 | 1.29491 | supported_fixed_branch_not_tested_full_model | 5 | persistent_cell0_residual_quality_flag |
| 3 | carrier_gas_flow | 0.172194 | 0.683473 | supported_fixed_branch_not_tested_full_model | 5 | persistent_cell0_residual_quality_flag |
| 4 | HeatLossPercent | 0.0645132 | 0.509486 | supported_fixed_branch_not_tested_full_model | 5 | persistent_cell0_residual_quality_flag |

Interpretation: axes are candidates for online-correction design because they are sensitivity-ranked over the Huayi dry-powder fixed-physics cases and have fixed-branch gradient parity evidence for matching feed/carrier/heat-loss directions. This is not a trained hybrid, calibrated accuracy result, causal proof, production online deployment, or full-reactor 22-case IFT parity.

Evidence boundary: fixed-physics influence-tracing diagnostic only; candidate online-correction axis selection support, not trained hybrid, calibrated accuracy, causal proof, production online deployment, full-reactor 22-case IFT parity, or Yulin/slurry generalization.

## Sensitivity subtable

# Fixed-physics sensitivity ranking diagnostic

This is finite-difference fixed-physics sensitivity evidence, not calibration accuracy evidence.

- Cases: 22
- Long-form rows: 440
- OK rows: 440
- Non-OK rows: 0
- N_cells: 5

| rank | parameter | mean_abs_norm_sens | median_abs_norm_sens | max_abs_norm_sens | ok_rows | status |
|---:|---|---:|---:|---:|---:|---|
| 1 | oxygen_flow | 0.659251 | 0.676603 | 1.06346 | 110 | ok |
| 2 | coal_mass_flow | 0.482625 | 0.444278 | 1.29491 | 110 | ok |
| 3 | carrier_gas_flow | 0.172194 | 0.0329369 | 0.683473 | 110 | ok |
| 4 | HeatLossPercent | 0.0645132 | 0.0296628 | 0.509486 | 110 | ok |

Evidence boundary: pilot representative ModelCases, centered finite differences, existing fixed-physics solver only. Use as a ranking/diagnostic input for later hybrid-correction design, not as fitted model accuracy.
