# gasifier-1d-kinetic：`jax_newton` vs expected（按 UI KPI 口径）

- N_cells: 6
- baseline: `minimize`（UI 默认）
- jax solver: `jax_newton`

## 出口 KPI 与 expected 对比（含 wall time）
| Case | exp T(°C) | baseline T | jax T | exp yCO | baseline yCO | jax yCO | exp yH2 | baseline yH2 | jax yH2 | exp yCO2 | baseline yCO2 | jax yCO2 | baseline time(s) | jax time(s) | max|ΔT|(K) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 1370.0 | 1709.5 | 1709.5 | 61.70 | 70.37 | 70.37 | 30.30 | 21.84 | 21.84 | 1.30 | 7.21 | 7.21 | 1.54 | 1.50 | 0.00 |

## expected 差值（正数=高估；负数=低估）
| Case | dT_baseline | dT_jax | dyCO_baseline | dyCO_jax | dyH2_baseline | dyH2_jax | dyCO2_baseline | dyCO2_jax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 339.46 | 339.46 | 8.67 | 8.67 | -8.46 | -8.46 | 5.91 | 5.91 |
