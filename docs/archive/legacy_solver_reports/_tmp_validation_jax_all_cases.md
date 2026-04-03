# gasifier-1d-kinetic：验证 `jax_newton` vs baseline(newton)

- N_cells: 6
- baseline solver: `newton`
- jax solver: `jax_newton`
- jax_warmup: 仅对第一个案例启用（用于摊销编译开销）

## 汇总对比（出口 KPI + 运行时间 + profile 最大误差）
| Case | baseline T(°C) | baseline yCO(%) | baseline yH2(%) | baseline yCO2(%) | baseline time(s) | jax_newton T(°C) | jax_newton yCO(%) | jax_newton yH2(%) | jax_newton yCO2(%) | jax time(s) | max|ΔT|(K) | max|ΔyCO|(mol%) | max|ΔyCO2|(mol%) | max|ΔyH2|(mol%) | max|ΔyCH4|(mol%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 1709.5 | 70.37 | 21.84 | 7.21 | 2.35 | 1709.5 | 70.37 | 21.84 | 7.21 | 2.03 | 0.000 | 2.23e-05 | 8.17e-07 | 7.88e-06 | 3.11e-08 |

## 与预期（VALIDATION_CASES.expected）对齐情况
| Case | expected T(°C) | baseline ΔT(K) | jax_newton ΔT(K) | expected yCO(%) | baseline ΔyCO(%) | jax_newton ΔyCO(%) | expected yH2(%) | baseline ΔyH2(%) | jax_newton ΔyH2(%) | expected yCO2(%) | baseline ΔyCO2(%) | jax_newton ΔyCO2(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 1370.0 | 339.458 | 339.458 | 61.70 | 8.672 | 8.672 | 30.30 | -8.461 | -8.461 | 1.30 | 5.913 | 5.913 |
