# `cell.residuals` 纯 jnp 化路线图（阶段 C）

目标：让热路径可在 JAX/XLA 上 trace，从而支持 `jit`/`vmap`、GPU 与更少 Python 开销。当前生产路径 `solver_method=jax_pure` 已改为 **host float64 残差 + 中心差分 Jacobian**（见 `jax_solver.newton_solve_cell_pure_jax_ad`），本文件描述后续迁移优先级。

## 优先级（建议顺序）

1. **物性与焓（`MaterialService` 相关）**  
   高频调用、分支相对少；先提供 `jnp` 版本或 `jax.pure_callback` 批量接口，避免逐标量 Python。

2. **均相反应速率（`KineticsService.calc_homogeneous_rates`）**  
   已在 `jax_kinetics` 有部分函数；补齐与 NumPy 路径一致的接口，并在 `Cell` 中按开关切换。

3. **异相反应与碳消耗**  
   与颗粒/气相耦合强，迁移时优先保证与 NumPy 残差逐维对齐的单元测试。

4. **能量与固体项**  
   最后迁移：调试成本高，且常依赖上面几项。

## 里程碑

- **M1**：单测覆盖「同一 `x` 下 NumPy 残差 vs jnp 残差」`max|Δ| < tol`（按分量缩放）。
- **M2**：`make_cell_residuals_jax` 改为真实 `jacfwd`（无 custom 数值 JVP）在代表性状态上可用。
- **M3**：整炉 `jax_pure` 或新开关 `jax_jit_cell` 端到端计时不低于当前 host FD（在 CPU 上至少不更慢）。

## 非目标（短期内）

- 强行在 GPU 上跑完整炉膛而不做批处理设计；
- 为追求可微而保留每步数十次 `pure_callback` 的旧路径作为默认生产配置。
