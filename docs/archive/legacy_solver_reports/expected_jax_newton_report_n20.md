# gasifier-1d-kinetic：`jax_newton` vs expected（按 UI KPI 口径）

- N_cells: 20
- baseline: `minimize`（UI 默认）
- jax solver: `jax_newton`

## 出口 KPI 与 expected 对比（含 wall time）
| Case | exp T(°C) | baseline T | jax T | exp yCO | baseline yCO | jax yCO | exp yH2 | baseline yH2 | jax yH2 | exp yCO2 | baseline yCO2 | jax yCO2 | baseline time(s) | jax time(s) | max|ΔT|(K) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 1370.0 | 1686.9 | 1621.3 | 61.70 | 65.80 | 64.78 | 30.30 | 25.83 | 26.74 | 1.30 | 7.83 | 7.67 | 24.63 | 24.96 | 82.71 |
| Paper_Case_1 | 1333.0 | 1718.5 | 1673.3 | 59.90 | 66.13 | 64.63 | 29.50 | 25.37 | 25.89 | - | 7.97 | 7.74 | 36.29 | 19.64 | 48.31 |
| Paper_Case_2 | 1452.0 | 2200.8 | 2452.5 | 61.80 | 71.32 | 64.50 | 29.70 | 17.40 | 18.07 | - | 10.72 | 8.59 | 24.90 | 21.09 | 270.04 |
| LuNan_Texaco_Slurry | 1335.0 | 1412.8 | 1405.3 | 51.50 | 52.89 | 52.75 | 26.50 | 25.72 | 25.85 | - | 20.08 | 20.08 | 12.26 | 25.76 | 399.32 |

## expected 差值（正数=高估；负数=低估）
| Case | dT_baseline | dT_jax | dyCO_baseline | dyCO_jax | dyH2_baseline | dyH2_jax | dyCO2_baseline | dyCO2_jax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 316.89 | 251.29 | 4.10 | 3.08 | -4.47 | -3.56 | 6.53 | 6.37 |
| Paper_Case_1 | 385.52 | 340.27 | 6.23 | 4.73 | -4.13 | -3.61 | - | - |
| Paper_Case_2 | 748.76 | 1000.49 | 9.52 | 2.70 | -12.30 | -11.63 | - | - |
| LuNan_Texaco_Slurry | 77.77 | 70.35 | 1.39 | 1.25 | -0.78 | -0.65 | - | - |
