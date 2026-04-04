# 1D Kinetic Gasifier Model

中文 | [English](#english)

## 中文

基于 Wen & Chaung (1979) 的 1D 气流床气化炉动力学模型，面向 Texaco / Shell 型 entrained-flow gasifier，包含气固耦合、异相反应、均相反应、WGS 与轴向能量平衡。

当前这个项目的重点，不只是“模型能不能跑”，而是把它整理成一个更清晰的工程对象：

- `minimize` 作为稳健基线
- `newton_fd` 作为 CPU 侧工程折中路线
- `jax_jit` 作为面向热态服务的目标执行路径

## 项目定位

这个模型更适合回答的问题是：

- 在给定停留时间、动力学速率和轴向推进路径下，系统实际能走到哪里
- `cell0` 点火、高温跃升和刚性方程会如何影响全炉解
- 在保持接近基线精度的前提下，如何把动力学模型做成可在线调用的 calculation engine

相较于平衡模型，1D 动力学模型更适合：

- 轴向 profile 解释
- 反应区诊断
- 点火和高温区机理分析
- 热态高吞吐在线求解

## 当前状态

- 求解器公开接口已经收口到 `minimize` / `newton_fd` / `jax_jit`
- `jax_jit` 路线已基本贴齐 canonical benchmark 上的 `minimize` 主线结果
- 正式验证数据源统一为 `data/validation_cases_final.json`
- 当前正式数据集按 `Pilot / Industrial`、`Dry-fed / Slurry-fed` 分层，总计 35 个 canonical cases
- UI、回归测试、precompile 脚本和在线服务说明已基本成体系

## 为什么 `jax_jit` 重要

这个项目里最难求的一段，往往不是后续温和区，而是 `cell0`：

- 点火
- 氧气快速耗尽
- 温度陡升
- 反应速率突然放大
- 后续全炉路径分支选择

这会形成非常典型的刚性非线性方程组。`jax_jit` 的价值不只是“更快”，而是：

- 更适合承接这类刚性求解主路径
- 更适合把主计算图编译成可重复执行的热态路径
- 在完成 warmup / precompile 后，可以显著缩短重复求解的 wall time

关键点要区分清楚：

- 冷启动时，`jax_jit` 不一定快
- 热态重复运行时，`jax_jit` 才体现优势

当前仓库内记录的代表性单 case 实测：

- `minimize`: `1.3035s`
- `jax_jit` 第一次（含 compile）: `9.5113s`
- `jax_jit` 第二次（热态）: `0.6205s`

因此，`jax_jit` 的正确定位不是“首跑更快的按钮”，而是：

**完成编译后，适合在线热态服务的目标执行路径。**

## 在线服务价值

一旦动力学模型的稳定性和热态速度都过线，它的应用边界会明显扩大。当前项目特别适合朝这些方向发展：

- `soft sensor` 的机理内核
- `digital twin` 的 calculation engine
- `advanced control` / `real-time optimization` 的 mechanism model

这也是为什么本仓库现在强调：

- 部署阶段先 `precompile`
- readiness 以预编译完成为前提
- 在线请求默认直接走热态 `jax_jit`

相关说明见：

- [docs/online_jax_jit_service_cn.md](docs/online_jax_jit_service_cn.md)
- [docs/solver_comparison_cn.md](docs/solver_comparison_cn.md)
- [docs/wechat_article_gasifier_1d_kinetic_2026-04.md](docs/wechat_article_gasifier_1d_kinetic_2026-04.md)

## 项目结构

```text
gasifier-1d-kinetic/
├── src/model/
│   ├── gasifier_system.py
│   ├── cell.py
│   ├── solver.py
│   ├── jax_solver.py
│   ├── jax_residuals.py
│   └── ...
├── data/
│   └── validation_cases_final.json
├── scripts/
│   ├── run_all_validation_cases.py
│   ├── compare_sim_minimize_vs_jax_jit_all_cases.py
│   ├── precompile_jax_solver.py
│   └── diagnostics/
├── docs/
│   ├── physics_and_algorithms.md
│   ├── validation_newton_fd_report.md
│   ├── minimize_vs_jax_jit_report.md
│   ├── solver_comparison_cn.md
│   └── online_jax_jit_service_cn.md
└── README.md
```

## 快速开始

### 运行 `newton_fd` 验证

```bash
cd gasifier-1d-kinetic
PYTHONPATH=src python scripts/run_all_validation_cases.py
```

### 运行 `minimize` vs `jax_jit` 对比

```bash
cd gasifier-1d-kinetic
JAX_ENABLE_X64=1 ./.venv/bin/python scripts/compare_sim_minimize_vs_jax_jit_all_cases.py --N_cells 20 --out docs/minimize_vs_jax_jit_report.md
```

### 运行集成测试

```bash
cd gasifier-1d-kinetic
PYTHONPATH=src pytest tests/integration/ -q
```

### 仅验证 JAX 契约

```bash
PYTHONPATH=src pytest tests/integration/test_jax_jit_contracts.py -q
```

## 常用配置

- `solver_method='minimize'`
- `solver_method='newton_fd'`
- `solver_method='jax_jit'`
- `jacobian_mode='scipy'`
- `jacobian_mode='centered_fd'`

常用参数：

- `Combustion_CO2_Fraction`
- `HeatLossPercent`
- `WGS_CatalyticFactor`

## benchmark 说明

| 数据组 | 代表工况 | 推荐用途 | 当前状态 |
| :--- | :--- | :--- | :--- |
| canonical benchmark | `Texaco_I-*`, `LuNan_Texaco`, `Paper_Case_6`, `Fluid_Coke` | 主对齐集 | `jax_jit` 已基本贴齐主线 |
| pilot / duplicate 口径 | 小写 `texaco i-*` 等 | 单独诊断 | 不建议与 canonical benchmark 混合统计 |

---
*Updated: 2026-04-04*

## English

This repository contains a Wen & Chaung (1979)-style 1D kinetic model for entrained-flow coal gasifiers, targeting Texaco / Shell-type gasifiers with gas-solid coupling, heterogeneous reactions, homogeneous reactions, WGS, and axial energy balance.

The project is no longer just about “making the model run.” Its current engineering structure is:

- `minimize` as the robust baseline
- `newton_fd` as the CPU-side engineering compromise
- `jax_jit` as the target hot-path for online service

## Project Scope

This model is intended to answer questions such as:

- Given residence time, kinetics, and axial marching, where does the system actually end up?
- How do ignition, sharp temperature rise, and stiffness in `cell0` affect the full-furnace solution?
- How can a kinetic model become an online calculation engine while staying close to baseline accuracy?

Compared with an equilibrium model, this 1D kinetic model is better suited for:

- axial profile interpretation
- reaction-zone diagnostics
- ignition / high-temperature-zone mechanism analysis
- hot-path high-throughput online solving

## Current Status

- Public solver APIs are consolidated to `minimize` / `newton_fd` / `jax_jit`
- `jax_jit` is now broadly aligned with the `minimize` mainline on canonical benchmark cases
- The formal validation source is unified to `data/validation_cases_final.json`
- The current formal dataset is organized by `Pilot / Industrial` and `Dry-fed / Slurry-fed`, totaling 35 canonical cases
- UI, regression tests, precompile scripts, and service docs are now part of one coherent path

## Why `jax_jit` Matters

The hardest part of this model is usually not the mild downstream region but `cell0`, where several things happen together:

- ignition
- fast oxygen depletion
- sharp temperature rise
- sudden amplification of reaction source terms
- branch selection for the downstream cells

That makes `cell0` a highly nonlinear stiff system. The value of `jax_jit` is not only “speed”; it is also:

- better suited to carrying the stiff main solve path
- better suited to turning the main numerical path into a compiled hot path
- able to reduce repeated-solve wall time after warmup / precompile

The key distinction is:

- `jax_jit` is not necessarily faster on cold start
- `jax_jit` becomes valuable on repeated hot-path runs

Representative single-case timing already recorded in this repo:

- `minimize`: `1.3035s`
- `jax_jit` first run (with compile): `9.5113s`
- `jax_jit` second run (hot): `0.6205s`

So the correct product framing is not “the button that is always faster on first click,” but:

**the target execution path for online hot service after compilation is ready.**

## Online-Service Value

Once both stability and hot-path speed improve enough, the kinetic model becomes more suitable for real online use, such as:

- mechanism core for a `soft sensor`
- calculation engine for a `digital twin`
- mechanism model for `advanced control` / `real-time optimization`

That is why this repository emphasizes:

- precompile during deployment
- readiness gated by successful precompile
- serving online traffic through hot `jax_jit`

See also:

- [docs/online_jax_jit_service_cn.md](docs/online_jax_jit_service_cn.md)
- [docs/solver_comparison_cn.md](docs/solver_comparison_cn.md)
- [docs/wechat_article_gasifier_1d_kinetic_2026-04.md](docs/wechat_article_gasifier_1d_kinetic_2026-04.md)

## Project Layout

```text
gasifier-1d-kinetic/
├── src/model/
├── data/
├── scripts/
├── docs/
└── README.md
```

## Quick Start

### Run `newton_fd` validation

```bash
cd gasifier-1d-kinetic
PYTHONPATH=src python scripts/run_all_validation_cases.py
```

### Run `minimize` vs `jax_jit`

```bash
cd gasifier-1d-kinetic
JAX_ENABLE_X64=1 ./.venv/bin/python scripts/compare_sim_minimize_vs_jax_jit_all_cases.py --N_cells 20 --out docs/minimize_vs_jax_jit_report.md
```

### Run integration tests

```bash
cd gasifier-1d-kinetic
PYTHONPATH=src pytest tests/integration/ -q
```

### JAX-only contract test

```bash
PYTHONPATH=src pytest tests/integration/test_jax_jit_contracts.py -q
```

## Common Configuration

- `solver_method='minimize'`
- `solver_method='newton_fd'`
- `solver_method='jax_jit'`
- `jacobian_mode='scipy'`
- `jacobian_mode='centered_fd'`

Common parameters:

- `Combustion_CO2_Fraction`
- `HeatLossPercent`
- `WGS_CatalyticFactor`

## Benchmark Notes

| Dataset | Representative cases | Recommended use | Current status |
| :--- | :--- | :--- | :--- |
| canonical benchmark | `Texaco_I-*`, `LuNan_Texaco`, `Paper_Case_6`, `Fluid_Coke` | main alignment set | `jax_jit` is broadly aligned |
| pilot / duplicate cases | lowercase `texaco i-*`, etc. | separate diagnostics | should not be mixed into canonical statistics |
