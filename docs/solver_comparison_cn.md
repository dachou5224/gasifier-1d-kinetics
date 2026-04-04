# 三种 Solver 对比说明

本文说明当前项目中 3 种常见求解路径的差异：

- `minimize`
- `newton`
- `jax_jit`

注意：在当前代码里，`newton` 不是独立实现，而是兼容别名，实际会映射到 `newton_fd`。

相关实现入口：

- `src/model/gasifier_system.py`
- `src/model/solver.py`
- `src/model/jax_solver.py`

## 0. 当前推荐结论

如果目标是：

- 保持与 `minimize` 同级别的结果精度
- 同时缩短生产环境中的 wall time

那么当前项目应优先把 `jax_jit` 作为目标热路径。

更准确地说，这里的结论不是：

- “`jax_jit` 每次单跑都一定最快”

而是：

- “`jax_jit` 在完成 warmup / precompile 之后，最有希望在保持 `minimize` 级别精度的同时，把重复求解的 wall time 压下来”

这也是当前仓库里 JAX 路线的工程定位：

- `minimize`：精度与稳健性基线
- `jax_jit`：面向热态服务和重复 shape 的高吞吐路径
- `newton_fd`：位于两者之间的过渡/折中方案

### 0.1 现有证据

已有精度证据来自：

- [`docs/minimize_vs_jax_jit_report.md`](/Users/liuzhen/AI-projects/gasifier-1d-kinetic/docs/minimize_vs_jax_jit_report.md)
- [`docs/jax_refactor_technical_notes.md`](/Users/liuzhen/AI-projects/gasifier-1d-kinetic/docs/jax_refactor_technical_notes.md)
- [`docs/release_summary_2026-04-03.md`](/Users/liuzhen/AI-projects/gasifier-1d-kinetic/docs/release_summary_2026-04-03.md)

其中需要注意：

- `minimize_vs_jax_jit_report.md` 是较早的 19-case 报告
- 其中混入了小写 `texaco i-*` / `slurry western` 这类另一套 pilot 数据口径
- 对“正式 benchmark”做判断时，应优先看大写 canonical 工况

按这份报告里的 canonical benchmark 工况重新看，`jax_jit` 与 `minimize` 的出口误差已经很小：

- 平均出口温度差 `|ΔT_out|` 约 `0.523 K`
- 中位出口温度差 `|ΔT_out|` 约 `0.119 K`
- 最大出口温度差 `|ΔT_out|` 约 `2.402 K`
- `max|ΔyCO|` 平均约 `0.736 mol%`
- `max|ΔyH2|` 平均约 `0.662 mol%`
- `max|ΔyCO2|` 平均约 `0.674 mol%`

这组数字支持的结论是：

- 在主 benchmark 口径下，`jax_jit` 已经不是“另一套明显不同的解”
- 它更接近“与 `minimize` 同级别精度的等价实现”

### 0.2 关于 wall time 的正确表述

当前仓库对 `jax_jit` 的 wall time 结论应写得很精确：

- 冷启动首跑：不一定快，往往会被 JIT/XLA 编译成本拖慢
- 热态重复运行：目标就是比 `minimize` 更短的 wall time

本轮在本机 `.venv` 中对 `Paper_Case_6`（`N_cells=20`）做了一次同进程测试，结果是：

- `minimize`: `1.3035s`
- `jax_jit` 第一次（含 warmup / compile）: `9.5113s`
- `jax_jit` 第二次（热态、同 shape）: `0.6205s`

同一组测试里的出口 KPI 分别是：

- `minimize`: `T_out_C=1419.0385`, `yCO=68.8869`, `yH2=22.6217`, `yCO2=7.7989`
- `jax_jit` 热态: `T_out_C=1416.0004`, `yCO=68.8460`, `yH2=22.5115`, `yCO2=7.8845`

这组数据说明两件事：

1. 冷启动时，`jax_jit` 的确会因为编译成本显著慢于 `minimize`
2. 一旦进入热态，`jax_jit` 在这个代表性 case 上 wall time 约为 `0.6205 / 1.3035 ≈ 0.48x`，也就是大约快一倍，同时 KPI 仍保持同量级

项目内已有多处实现和文档都在围绕这个目标设计：

- `scripts/precompile_jax_solver.py`
- `docs/jax_refactor_technical_notes.md`
- `docs/plans/jax_solver_upgrade_plan.md`

尤其是：

- 文档明确建议把 `jax_jit` 作为“热态服务路径”
- 通过 precompile / readiness probe 避免把首次编译时间暴露给线上请求

因此，当前最准确的产品表述应该是：

> `jax_jit` 的价值不在于“无条件比 `minimize` 快”，而在于“完成预热后，在保持接近 `minimize` 精度的前提下，为重复求解提供更短的 wall time 和更好的吞吐”。

### 0.3 当前环境限制

需要明确区分两类证据来源：

- canonical benchmark 的“全量精度贴齐”结论，主要来自仓库中已存在的对比报告
- `Paper_Case_6` 的冷启动/热态 wall time 结论，来自本轮在 `.venv` 中的实时重跑

因此应把当前状态理解为：

- 精度方向：已有 benchmark 报告支持，本轮也补了单 case 实测
- 热态性能方向：本轮已确认单 case 热态收益存在；全量服务场景仍应继续保留基准脚本验证

## 1. 总结版

| Solver | 当前真实实现 | 核心方法 | 优点 | 缺点 | 适合场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `minimize` | SciPy `least_squares(method="trf")` | 把 cell residual 当作最小二乘问题求解 | 稳健，历史基线最清楚，失败时最容易诊断 | 慢，尤其是全炉多 cell 时 | 基线结果、回归验证、先求稳 |
| `newton` | `newton_fd` 的别名 | 阻尼 Newton + 有限差分 Jacobian | 比 `minimize` 更快，逻辑仍在 Python/NumPy 侧，容易接现有代码 | 对初值和 Jacobian 更敏感，失败时会 fallback | 日常加速、和基线做工程折中 |
| `jax_jit` | JAX 全炉扫描求解 | JIT 编译的 12 维状态 Newton 迭代 | 热启动后最快，适合重复运行同形状问题 | 首次编译成本高，调试门槛高，对 JAX/数值稳定性要求更高 | 服务预热后高吞吐、批量重复求解 |

## 2. API 层的真实行为

在 `GasifierSystem.solve(...)` 中，当前允许的 `solver_method` 只有：

- `minimize`
- `newton_fd`
- `jax_jit`

旧名字会被自动归一化：

- `newton` -> `newton_fd`
- `jax_newton` -> `newton_fd`
- `jax_pure` -> `newton_fd`

也就是说，如果代码里写了：

```python
system.solve(..., solver_method="newton")
```

当前实际走的是：

```python
system.solve(..., solver_method="newton_fd")
```

因此本文后面提到的 “`newton`” 都应理解为当前项目中的 `newton_fd` 路径。

## 3. `minimize` 是什么

### 3.1 算法

`minimize` 路径本质上不是手写 Newton，而是对每个 cell 调用 SciPy：

```python
scipy.optimize.least_squares(method="trf")
```

它把 cell residual 向量 `F(x)` 当作一个最小二乘问题，目标是让：

```text
0.5 * ||F(x)||^2
```

尽可能小。

### 3.2 当前项目中的特点

- cell 级顺序推进：前一格的出口作为后一格入口
- 第 0 格会用多组初始温度做多起点尝试：`3000 / 2000 / 1500 / 1000 / 400 K`
- 可以使用两种 Jacobian 口径：
  - `jacobian_mode="scipy"`：默认 SciPy 数值 Jacobian
  - `jacobian_mode="centered_fd"`：项目内提供的中心差分 Jacobian

### 3.3 优缺点

优点：

- 是当前最稳的基线
- 许多现有对比脚本默认拿它当 reference
- 第 0 格多起点处理更成熟，点火支路更容易找到

缺点：

- 每格都在 host 侧做数值求解，速度最慢
- 全炉 cell 数增多后，累计开销明显

### 3.4 什么时候用

- 写报告时需要“基线口径”
- 新 solver 上线前做对照
- 某工况数值不稳定，先要求解成功率

## 4. `newton` / `newton_fd` 是什么

### 4.1 算法

`newton_fd` 使用的是阻尼 Newton 迭代：

```text
J(x) * delta = -F(x)
x_new = x + damper * delta
```

其中 Jacobian `J(x)` 不是解析导数，而是有限差分近似：

- 前向差分
- 或中心差分

对应实现主要在：

- `src/model/solver.py`
- `src/model/jax_solver.py`

### 4.2 为什么名字里有 `fd`

`fd` 是 finite difference。

也就是说，当前这条路径的关键不是 “纯 JAX 自动微分 Newton”，而是：

- Newton 迭代框架
- Jacobian 由有限差分构造

### 4.3 当前项目中的真实执行方式

在 `GasifierSystem._solve_sequential(...)` 中：

- 第 0 格：
  - 走 `newton_solve_multistart_numpy(...)`
  - 同样做多起点温度扫描
- 其余格：
  - 走 `newton_solve_cell_numpy(...)`
  - 以前一格出口作为初值

如果 `newton_fd` 解得不好，会 fallback 回 SciPy `least_squares`。

当前 fallback 条件大致是：

- `sol is None`
- `not sol.success`
- `sol.cost > 1e-4`

所以它不是一个“失败就彻底报错”的路径，而是一个“先快后稳，必要时退回基线”的工程方案。

### 4.4 和 `minimize` 的关键差异

`minimize`：

- 直接把问题交给 SciPy `least_squares`
- 更像稳健优化器

`newton_fd`：

- 自己显式构造 Newton 步
- 自己控制 damping、迭代次数、差分 Jacobian
- 收敛快时更快
- 不稳时会退回 `least_squares`

### 4.5 优缺点

优点：

- 比 `minimize` 更快
- 仍然运行在 NumPy/Python 侧，调试比 `jax_jit` 容易
- 已经和项目当前 residual/physics 结构接起来了

缺点：

- 对初值、差分步长、阻尼参数更敏感
- 局部病态 Jacobian 时更容易退化
- 严格说它并不是完全独立于 `minimize` 的成熟求解路线，因为首格 seed 和 fallback 都借助基线思路

### 4.6 什么时候用

- 想要比 `minimize` 更快，但还不想承担 `jax_jit` 的调试复杂度
- 日常批量验证
- 需要保留 host 侧可解释性

## 5. `jax_jit` 是什么

### 5.1 算法

`jax_jit` 走的是 JAX 编译后的全炉扫描求解。

核心特征：

- 使用 9 组分气相表示
- 单格状态扩展成 12 维
- 用 `lax.scan` 沿炉长推进
- 用 `jacfwd` 在 JAX 图内拿 Jacobian
- 整条路径通过 `jit` 编译

主要入口在：

- `GasifierSystem._solve_jax_v4(...)`
- `jax_solver.reactor_solve_v4(...)`

### 5.2 为什么它通常最快

因为它把“每一格怎么解”的大量 Python 开销提前编译掉了。

首次运行时会有：

- JAX runtime 初始化
- XLA 编译

这一步很慢，但同一类 shape 之后重复跑会明显更快。

### 5.3 当前项目中的实现特点

- 首格优先使用 `minimize` 派生 seed
  - 这是为了避免纯启发式初值落到冷态支路
- 后续格在 JAX 图内做 Newton 迭代
- 明确启用 `jax_enable_x64=True`
  - 因为工业工况下 float32 容易溢出成 `NaN`
- 热损归一化专门对齐了主线 `_solve_sequential(...)`
  - 避免 `jax_jit` 因热损口径不同而系统性偏热

### 5.4 和 `newton_fd` 的关键差异

两者都带有 Newton 风格，但不是同一层面的实现：

`newton_fd`：

- host 侧 NumPy Newton
- Jacobian 主要靠有限差分
- 每格都是 Python 调度

`jax_jit`：

- JAX 图内 Newton
- `jacfwd` 直接在 JAX residual 上求导
- 整炉 scan + jit 编译

所以 `jax_jit` 不是 “把 `newton_fd` 简单换成 JAX”，而是另一条更深的执行路径。

### 5.5 优缺点

优点：

- 预热后最快
- 对重复同形状问题吞吐量最好
- 适合服务场景的批量求解

缺点：

- 首次编译成本高
- 调试困难，报错链更远
- 对 JAX runtime、dtype、shape 稳定性要求更高
- 当前仍依赖一些主线路径对首格 seed 和物理口径做对齐

### 5.6 什么时候用

- 服务启动后已完成 precompile / warmup
- 相同 case 结构要反复批量求解
- 追求吞吐量而不是单次首跑延迟

## 6. 三者的收敛与回退关系

这 3 条路径不是完全平行、互不依赖的关系。

当前项目中的真实层级更像：

1. `minimize`
   - 稳定基线
   - 很多对照和 seed 逻辑都默认参考它
2. `newton_fd`
   - 优先尝试更快的 Newton
   - 不稳时回退到 `least_squares`
3. `jax_jit`
   - 独立的 JAX 编译路线
   - 但首格仍借助 `minimize` 派生 seed 来避免错误支路

所以工程上可以理解为：

- `minimize` 负责稳
- `newton_fd` 负责在现有主线里加速
- `jax_jit` 负责在稳定 shape 上追求最高吞吐

## 7. 性能直觉

一般规律如下：

- 单次首跑：
  - 通常 `minimize` 或 `newton_fd` 更有优势
  - `jax_jit` 可能因为编译而最慢
- 重复同 shape 跑很多次：
  - `jax_jit` 通常会变成最快
- 需要最高成功率：
  - `minimize` 通常最稳
- 需要“比基线快，但失败时还能兜底”：
  - `newton_fd` 最合适

仓库里已有若干对比脚本可以辅助实测：

- `scripts/benchmark_solver_variants.py`
- `scripts/validate_jax_solver_all_cases.py`
- `scripts/compare_sim_minimize_vs_jax_jit_all_cases.py`

## 8. 选型建议

### 8.1 做功能开发或排查 bug

优先：

```python
solver_method="minimize"
```

理由：

- 结果最容易和历史基线对齐
- 失败时更容易看 residual 和逐格行为

### 8.2 做日常验证和提速

优先：

```python
solver_method="newton_fd"
```

配合：

```python
jacobian_mode="centered_fd"
```

理由：

- 速度和稳健性折中最好
- 与当前项目的验证脚本适配最多

### 8.3 做服务化或高吞吐批量求解

优先：

```python
solver_method="jax_jit"
```

前提：

- 已安装可用的 `jax` / `jaxlib`
- 已完成 warmup / precompile
- 接受更高的数值调试成本

## 9. 推荐记法

为了减少歧义，建议今后文档和代码里统一写：

- `minimize`
- `newton_fd`
- `jax_jit`

不再用：

- `newton`
- `jax_newton`
- `jax_pure`

因为这些名字在当前代码中都不是独立 solver，只是兼容别名，继续使用会让讨论变得模糊。
