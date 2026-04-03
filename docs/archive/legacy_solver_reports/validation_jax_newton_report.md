# gasifier-1d-kinetic：验证 `jax_newton` vs baseline(newton)

- N_cells: 20
- baseline solver: `newton`
- jax solver: `jax_newton`
- jax_warmup: 仅对第一个案例启用（用于摊销编译开销）

## 汇总对比（出口 KPI + 运行时间 + profile 最大误差）
| Case | baseline T(°C) | baseline yCO(%) | baseline yH2(%) | baseline yCO2(%) | baseline time(s) | jax_newton T(°C) | jax_newton yCO(%) | jax_newton yH2(%) | jax_newton yCO2(%) | jax time(s) | max|ΔT|(K) | max|ΔCO|(mol/s) | max|ΔCO2|(mol/s) | max|ΔH2|(mol/s) | max|ΔCH4|(mol/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 1686.9 | 65.80 | 25.83 | 7.83 | 14.61 | 1621.3 | 64.78 | 26.74 | 7.67 | 21.76 | 82.709 | 1.07e+01 | 4.69e+00 | 1.43e+01 | 2.10e+00 |
| Paper_Case_1 | 1718.5 | 66.13 | 25.37 | 7.97 | 44.38 | 1673.3 | 64.63 | 25.89 | 7.74 | 22.56 | 48.308 | 2.70e+00 | 1.15e+00 | 1.07e+01 | 9.92e+00 |
| Paper_Case_2 | 2200.8 | 71.32 | 17.40 | 10.72 | 28.55 | 2452.5 | 64.50 | 18.07 | 8.59 | 25.77 | 270.041 | 1.34e+02 | 3.62e+01 | 7.09e+01 | 8.37e+01 |
| LuNan_Texaco_Slurry | 1412.8 | 52.89 | 25.72 | 20.08 | 22.20 | 1405.3 | 52.75 | 25.85 | 20.08 | 18.48 | 8.607 | 1.03e+00 | 2.51e-01 | 1.95e+00 | 2.91e-03 |

## 与预期（VALIDATION_CASES.expected）对齐情况
| Case | expected T(°C) | baseline ΔT(K) | jax_newton ΔT(K) | expected yCO(%) | baseline ΔyCO(%) | jax_newton ΔyCO(%) | expected yH2(%) | baseline ΔyH2(%) | jax_newton ΔyH2(%) | expected yCO2(%) | baseline ΔyCO2(%) | jax_newton ΔyCO2(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 1370.0 | 316.890 | 251.289 | 61.70 | 4.105 | 3.077 | 30.30 | -4.467 | -3.562 | 1.30 | 6.532 | 6.371 |
| Paper_Case_1 | 1333.0 | 385.517 | 340.271 | 59.90 | 6.227 | 4.728 | 29.50 | -4.126 | -3.609 | - | - | - |
| Paper_Case_2 | 1452.0 | 748.763 | 1000.490 | 61.80 | 9.519 | 2.698 | 29.70 | -12.301 | -11.627 | - | - | - |
| LuNan_Texaco_Slurry | 1335.0 | 77.784 | 70.347 | 51.50 | 1.390 | 1.254 | 26.50 | -0.784 | -0.650 | - | - | - |
