# Fixed-physics sensitivity ranking diagnostic

This is finite-difference fixed-physics sensitivity evidence, not calibration accuracy evidence.

- Cases: 1
- Long-form rows: 20
- OK rows: 20
- Non-OK rows: 0
- N_cells: 5

| rank | parameter | mean_abs_norm_sens | median_abs_norm_sens | max_abs_norm_sens | ok_rows | status |
|---:|---|---:|---:|---:|---:|---|
| 1 | oxygen_flow | 0.665223 | 0.681462 | 0.893642 | 5 | ok |
| 2 | coal_mass_flow | 0.500906 | 0.443185 | 1.11063 | 5 | ok |
| 3 | carrier_gas_flow | 0.197629 | 0.0420459 | 0.611162 | 5 | ok |
| 4 | HeatLossPercent | 0.0598263 | 0.0283972 | 0.195386 | 5 | ok |

Evidence boundary: pilot representative ModelCases, centered finite differences, existing fixed-physics solver only. Use as a ranking/diagnostic input for later hybrid-correction design, not as fitted model accuracy.
