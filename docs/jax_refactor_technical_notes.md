# JAX 求解器重构技术文档 (2026-04-03)

## 1. 概述

本轮重构将气化炉 1D 模型的热路径迁移到 JAX，并把公开求解器接口收口到：

- `minimize`
- `newton_fd`
- `jax_jit`

其中 `jax_jit` 已完成整炉 `lax.scan` 路线和 19-case 可运行验证。

## 2. 新增模块

- **`jax_physics.py`**
  - 纯 JAX 热力学与物性计算
- **`jax_kinetics.py`**
  - 纯 JAX 均相/异相反应速率
- **`jax_residuals.py`**
  - 12 维状态向量残差实现（9 气相 + `Ws` + `Xc` + `T`）
- **`jax_solver.py`**
  - `reactor_solve_v4()` 整炉 JIT 路线
  - 兼容层与首格多初值求解

## 3. 关键修正与对齐

- 接口收口：
  - 旧 `newton` / `jax_newton` / `jax_pure` 统一并入 `newton_fd`
  - 旧 `use_jax_jacobian` 收口为 `jacobian_mode='centered_fd'`
- 单位与映射对齐：
  - 修正异相反应速率 kmol -> mol 偏差
  - 统一主线 8 组分与 JAX 9 组分映射
- 热损边界对齐：
  - 修正 `jax_jit` 错误按几何 `L` 归一化热损
  - 改为与主线一致按 `L_heatloss_norm` / `sum(dz)` 分摊
- 首格燃烧区对齐：
  - 修正挥发分瞬时燃烧后的 O2 剩余账本
  - 消除 `Texaco_I-*` / `Texaco_Exxon` 首格几十 K 的系统性偏差
- JAX tracing 修复：
  - 把 `jax_residuals` 导入移出 `@jit` 函数体，消除 `UnexpectedTracerError`

## 4. 使用指南

启用 `jax_jit`：

```python
res, z = system.solve(N_cells=100, solver_method="jax_jit")
```

建议命令行显式开启 x64：

```bash
JAX_ENABLE_X64=1 ./.venv/bin/python scripts/compare_sim_minimize_vs_jax_jit_all_cases.py --N_cells 20 --out docs/minimize_vs_jax_jit_report.md
```

## 5. 当前状态

- 原始文献 benchmark 大写工况：`jax_jit` 已基本贴齐 `minimize`
- 小写 `texaco i-*` / `slurry western`：来自另一套 pilot 数据口径，不建议混入主 benchmark 汇总
- 剩余误差主要表现为少数 case 的 profile 差异，而不是全局单向温度偏差

## 6. 后续建议

- 把主 benchmark 与 pilot duplicate 集彻底分表  
- 若继续压 profile 误差，优先看少数 remaining case 的逐格反应路径  
- `scripts/precompile_jax_solver.py` 应纳入 CI/CD（`.github/workflows/deploy-to-vps.yml` 在每次 push/pull 后运行 `python scripts/precompile_jax_solver.py --cases Paper_Case_6 Texaco_I-2 --n-cells 20`），确保部署到 VPS 里的 1d kinetic 模块已经完成 `jax_jit` 编译，portal 无需再重复编译。  
- `chem_portal` 配置应仅把 calculation 服务指向这条热态 `jax_jit` 计算模块，由 portal 负责路由/认证，保持模型逻辑与 equilibrium `gasifier-model` 并列。  
- 服务端部署时继续调用 `scripts/precompile_jax_solver.py`，并在 readiness probe 里验证热态后再接流量，必要时用不同 `N_cells` 组合覆盖常见 shape。  
