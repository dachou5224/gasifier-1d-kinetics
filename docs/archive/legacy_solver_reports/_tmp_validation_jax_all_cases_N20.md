# gasifier-1d-kinetic：验证 `jax_newton` vs baseline(newton)

- N_cells: 20
- baseline solver: `newton`
- jax solver: `jax_newton`
- jax_warmup: 仅对第一个案例启用（用于摊销编译开销）

## 汇总对比（出口 KPI + 运行时间 + profile 最大误差）
| Case | baseline T(°C) | baseline yCO(%) | baseline yH2(%) | baseline yCO2(%) | baseline time(s) | jax_newton T(°C) | jax_newton yCO(%) | jax_newton yH2(%) | jax_newton yCO2(%) | jax time(s) | max|ΔT|(K) | max|ΔyCO|(mol%) | max|ΔyCO2|(mol%) | max|ΔyH2|(mol%) | max|ΔyCH4|(mol%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 1609.3 | 70.45 | 22.18 | 6.78 | 2.16 | 1609.3 | 70.45 | 22.18 | 6.78 | 2.15 | 0.000 | 4.35e-07 | 5.79e-08 | 6.06e-08 | 7.42e-08 |

## 与预期（VALIDATION_CASES.expected）对齐情况
| Case | expected T(°C) | baseline ΔT(K) | jax_newton ΔT(K) | expected yCO(%) | baseline ΔyCO(%) | jax_newton ΔyCO(%) | expected yH2(%) | baseline ΔyH2(%) | jax_newton ΔyH2(%) | expected yCO2(%) | baseline ΔyCO2(%) | jax_newton ΔyCO2(%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper_Case_6 | 1370.0 | 239.278 | 239.278 | 61.70 | 8.746 | 8.746 | 30.30 | -8.116 | -8.116 | 1.30 | 5.482 | 5.482 |
