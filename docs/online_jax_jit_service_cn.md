# `jax_jit` 在线快速计算服务实现说明

本文说明如何基于当前仓库里的现有能力，把 `solver_method="jax_jit"` 做成一个适合在线请求的快速计算服务。

目标不是让每次调用都“立即更快”，而是：

- 在部署阶段完成 JIT/XLA 编译
- 让线上请求尽量只走“热态”路径
- 在保持接近 `minimize` 精度的前提下，缩短重复计算的 wall time

## 1. 结论先行

如果要把 `jax_jit` 用在在线服务里，推荐架构是：

1. 服务进程启动前或启动时执行 precompile
2. readiness probe 只在 precompile 完成后才返回 ready
3. 在线请求固定走 `solver_method="jax_jit"`
4. 常见 `N_cells` / shape 在部署时一次性编译完

不要采用的方式是：

- 等第一条用户请求来了再触发 JAX 编译

那样用户会直接承担 compile latency。

## 2. 仓库里已经具备的基础

### 2.1 预编译脚本

仓库已经提供：

- [`scripts/precompile_jax_solver.py`](/Users/liuzhen/AI-projects/gasifier-1d-kinetic/scripts/precompile_jax_solver.py)

它会：

- 读取 `data/validation_cases_final.json`
- 选择代表性 case
- 对指定 `N_cells` 调用 `system.solve(..., solver_method="jax_jit")`
- 触发对应 shape 的 JIT/XLA 编译

默认预编译参数是：

```bash
--cases Paper_Case_6 Texaco_I-2 Coal_Water_Slurry_Western
--n-cells 20
```

### 2.2 求解器入口

统一入口在：

- [`src/model/gasifier_system.py`](/Users/liuzhen/AI-projects/gasifier-1d-kinetic/src/model/gasifier_system.py)

当调用：

```python
system.solve(N_cells=20, solver_method="jax_jit")
```

会走：

- `GasifierSystem._solve_jax_v4(...)`
- `jax_solver.reactor_solve_v4(...)`

### 2.3 轻量 warmup

JAX 轻量预热在：

- [`src/model/jax_solver.py`](/Users/liuzhen/AI-projects/gasifier-1d-kinetic/src/model/jax_solver.py)

对应函数：

```python
warmup_jax()
```

这个 warmup 很轻，只能帮助初始化 JAX runtime，不能替代真正的 shape 级 precompile。

### 2.4 现有部署流程

当前 GitHub Actions 已经把 precompile 放进部署流程：

- [`.github/workflows/deploy-to-vps.yml`](/Users/liuzhen/AI-projects/gasifier-1d-kinetic/.github/workflows/deploy-to-vps.yml)

它做了这些事：

1. 在 VPS 上拉最新代码
2. 创建或复用 `.venv`
3. `pip install -r requirements.txt`
4. 执行：

```bash
JAX_ENABLE_X64=1 python scripts/precompile_jax_solver.py --cases Paper_Case_6 Texaco_I-2 --n-cells 20
```

这已经是正确方向。

## 3. 为什么在线服务必须 precompile

本轮实测 `Paper_Case_6, N_cells=20`：

- `minimize`: `1.3035s`
- `jax_jit` 第一次（含 compile）: `9.5113s`
- `jax_jit` 第二次（热态）: `0.6205s`

这说明：

- 冷启动时，`jax_jit` 不适合直接暴露给用户
- 热态时，`jax_jit` 才体现出优势

因此在线服务的关键不是“支持 `jax_jit`”，而是“让用户尽量只遇到热态 `jax_jit`”。

## 4. 推荐上线架构

### 4.1 职责划分

推荐拆成两层：

- portal / API gateway 层
  - 鉴权
  - 参数校验
  - 路由
- 计算服务层
  - 实际创建 `GasifierSystem`
  - 调用 `solver_method="jax_jit"`
  - 维持热态编译缓存

仓库里的文档也已经倾向这个结构：

- `chem_portal` 只做外层流量入口
- 本仓库对应的服务进程保持为专门的 calculation service

### 4.2 启动时序

推荐的服务启动顺序：

1. 激活 `.venv`
2. 设置 `JAX_ENABLE_X64=1`
3. 运行 precompile
4. 通过 readiness 标记服务可用
5. 开始接流量

伪代码：

```bash
source .venv/bin/activate
export JAX_ENABLE_X64=1
python scripts/precompile_jax_solver.py --cases Paper_Case_6 Texaco_I-2 --n-cells 20 40
exec uvicorn your_service:app --host 0.0.0.0 --port 8000
```

### 4.3 readiness probe

推荐规则：

- precompile 成功前：readiness = false
- precompile 成功后：readiness = true

不要用“进程已启动”作为 ready 条件，因为那不能保证 JAX shape 已经编译完成。

## 5. 如何选择 precompile 的 shape

JAX 编译和 shape 强相关。对这个项目而言，至少要覆盖：

- 常见 `N_cells`
- 常见工况类型

建议最小覆盖：

```bash
JAX_ENABLE_X64=1 ./.venv/bin/python scripts/precompile_jax_solver.py \
  --cases Paper_Case_6 Texaco_I-2 Coal_Water_Slurry_Western \
  --n-cells 20
```

如果线上会出现多个 mesh 档位，建议扩展为：

```bash
JAX_ENABLE_X64=1 ./.venv/bin/python scripts/precompile_jax_solver.py \
  --cases Paper_Case_6 Texaco_I-2 Coal_Water_Slurry_Western \
  --n-cells 20 40 60
```

选择原则：

- `Paper_Case_6`：主线基线 case
- `Texaco_I-2`：Dry-fed 代表
- `Coal_Water_Slurry_Western`：Slurry-fed 代表

## 6. 在线请求应如何调用

在线请求不应再重复 warmup，而应直接使用热态路径：

```python
result, z = system.solve(
    N_cells=20,
    solver_method="jax_jit",
    jax_warmup=False,
)
```

原因：

- `jax_warmup=True` 适合启动期或脚本场景
- 线上请求期应尽量避免做额外初始化工作

## 7. 失败与回退策略

`jax_jit` 在线化时建议保留降级方案。

推荐顺序：

1. 优先使用 `jax_jit`
2. 如果 JAX runtime 不可用或进程未完成预热，返回 “warming up” / 503
3. 如果业务要求必须返回结果，可选回退到 `minimize`

两种策略都可以，但要选清楚：

- 强一致低延迟：未预热时直接拒绝
- 高可用优先：未预热时回退 `minimize`

如果回退：

```python
solver_method="minimize"
```

必须在日志里明确记录，因为这会改变 wall time 特征。

## 8. 部署建议

### 8.1 VPS / GitHub Actions

当前 workflow 已经基本正确，建议继续保留：

```bash
JAX_ENABLE_X64=1 python scripts/precompile_jax_solver.py --cases Paper_Case_6 Texaco_I-2 --n-cells 20
```

如果线上常用多个 mesh，建议把 workflow 里的 `--n-cells` 扩成真实生产集合。

### 8.2 Python 环境

建议统一使用仓库内：

```bash
.venv
```

不要依赖系统 Python。

原因：

- 当前仓库要求 `jax[cpu]>=0.4.28`
- 系统 Python 在 Debian/Ubuntu 上常受 PEP 668 限制
- `.venv` 更适合部署时稳定复现

### 8.3 x64

建议服务环境固定：

```bash
export JAX_ENABLE_X64=1
```

原因：

- 这个模型里流量、焓、热损量级较大
- 文档和代码都明确指出 float32 更容易出现数值问题

## 9. 最小可执行方案

如果现在就要做一个最小但正确的在线服务方案，我建议：

### 方案 A：单进程热态服务

启动脚本：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /path/to/gasifier-1d-kinetic
. .venv/bin/activate
export JAX_ENABLE_X64=1

python scripts/precompile_jax_solver.py --cases Paper_Case_6 Texaco_I-2 Coal_Water_Slurry_Western --n-cells 20 40

exec uvicorn service:app --host 0.0.0.0 --port 8000
```

请求处理：

```python
system.solve(N_cells=n_cells, solver_method="jax_jit", jax_warmup=False)
```

### 方案 B：多 worker 服务

如果是多 worker：

- 每个 worker 启动时都要各自 precompile
- readiness 必须按 worker 粒度判断

因为 JIT cache 在进程内，不会自动跨进程共享。

## 10. 还可以继续优化的地方

如果你后续要把这条服务线做得更稳，可以继续加：

- 明确的 `/healthz` 与 `/readyz`
- 记录当前已编译 `N_cells` 集合
- 启动期后台异步补编译冷门 shape
- 统计 `jax_jit` 命中率、平均 wall time、回退到 `minimize` 的次数

## 11. 推荐的一句话描述

对外说明时，建议这样表述：

> 在线服务采用 `jax_jit` 热态求解路径：部署时预编译常见工况与网格 shape，服务 readiness 以编译完成为前提，从而在保持接近 `minimize` 精度的前提下，把重复请求的 wall time 压到更低。
