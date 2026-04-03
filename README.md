# 1D Kinetic Gasifier Model (Refactored & Verified)

基于 Wen & Chaung (1979) 的 1D 气流床气化炉动力学模型，面向 Texaco/Shell 型气化炉，含气固耦合与异相反应动力学。

## 最新验证进展 (2026-04-03)

本轮收尾完成了两类关键工作：

- 求解器接口收口到 `minimize` / `newton_fd` / `jax_jit`
- `jax_jit` 路线完成了 19-case 全量可运行验证，并修正了两处核心偏差来源：
  - 热损归一化长度错误
  - 首格燃烧区 O2 bookkeeping 与主线不一致

当前状态：

- `newton_fd`：适合作为 CPU 端有限差分 Newton 路线
- `minimize`：仍是最稳的基线
- `jax_jit`：对原始文献 benchmark 大写工况已基本贴齐 `minimize`

已知保留项：

- `validation_cases_final.json` 中仍混有一组来自 `validation_cases_pilot.json` 的小写重复工况
- 这些小写工况输入口径与对应大写文献工况并不相同，不应与主 benchmark 混为一组解读

## 项目结构

```text
gasifier-1d-kinetic/
├── src/model/
│   ├── gasifier_system.py
│   ├── cell.py
│   ├── jax_solver.py
│   ├── jax_residuals.py
│   └── ...
├── data/
│   └── validation_cases_final.json  # 当前 benchmark 数据集 (19 例)
├── scripts/
│   ├── run_all_validation_cases.py
│   ├── compare_sim_minimize_vs_jax_jit_all_cases.py
│   └── diagnostics/
├── docs/
│   ├── physics_and_algorithms.md
│   ├── validation_newton_fd_report.md
│   ├── minimize_vs_jax_jit_report.md
│   └── jax_refactor_technical_notes.md
└── README.md
```

## 物理模型核心要点

详见 [docs/physics_and_algorithms.md](docs/physics_and_algorithms.md)。

- 热化学基准：基于 `HHV_d` 反算 `Hf_coal`
- 辐射分摊：热损失按 `(T/1800)^4` 分配权重
- 进料判定：自动识别 `Slurry-fed` 与 `Dry-fed`

## 快速开始

### 运行 `newton_fd` 全量验证

```bash
cd gasifier-1d-kinetic
PYTHONPATH=src python scripts/run_all_validation_cases.py
```

### 运行 `minimize` vs `jax_jit` 全量对比

```bash
cd gasifier-1d-kinetic
JAX_ENABLE_X64=1 ./.venv/bin/python scripts/compare_sim_minimize_vs_jax_jit_all_cases.py --N_cells 20 --out docs/minimize_vs_jax_jit_report.md
```

## 配置说明

- 求解器公开接口：`solver_method='minimize'`、`solver_method='newton_fd'`、`solver_method='jax_jit'`
- `jacobian_mode`：`scipy` 或 `centered_fd`
- 常用参数：
  - `Combustion_CO2_Fraction`
  - `HeatLossPercent`
  - `WGS_CatalyticFactor`

## benchmark 说明

| 数据组 | 代表工况 | 推荐用途 | 当前状态 |
| :--- | :--- | :--- | :--- |
| 原始文献 benchmark | `Texaco_I-*`, `LuNan_Texaco`, `Paper_Case_6` | 主对齐集 | `jax_jit` 已基本贴齐 |
| pilot duplicate 集 | `texaco i-*`, `slurry western` | 单独诊断 | 不建议与主 benchmark 混合统计 |

## 部署与服务预热建议

`gasifier-1d-kinetic` 是 `chem_portal` 下的一个计算模块，和 equilibrium `gasifier-model` 在线服务保持同级，`chem_portal` 仅需在配置里把 `calculation_service_url` 指向本模块的 REST/GRPC endpoint（即编译完成的 `jax_jit` 服务），用户在入口访问时就能直接走“热态快速”路线。

- 在 `chem_portal` 所在的镜像里，先 canvass 参考 `../gasifier-model` 的 Docker/entrypoint，然后在入口脚本 `/ENTRYPOINT` 中执行 `scripts/precompile_jax_solver.py`（按需传 `--cases`/`--n-cells`）以完成 `jax_jit` 编译，再启动服务接受流量；这样 portal 无需再重复这些调用。  
- `.github/workflows/deploy-to-vps.yml` 的 CI/CD 部署步骤也会在每次 git push / VPS pull 后自动运行上述预热脚本（见 `python scripts/precompile_jax_solver.py --cases Paper_Case_6 Texaco_I-2 --n-cells 20`），确保部署到 VPS 的 `jax_jit` 计算 service 始终是热态。  
- 每个 worker 的 readiness probe 或 supervisor 启动流程里可以再调用一次预热命令，避免第一次请求触发编译延迟；对于尚未预热的 `N_cells`/shape，可临时回退 `solver_method="minimize"` 或异步编译后再切到 `jax_jit`。

### 参考流程

在 `chem_portal` 的 Dockerfile/entrypoint 中加入：

```bash
python scripts/precompile_jax_solver.py --cases Paper_Case_6 Texaco_I-2 --n-cells 20 40
```

然后再启动计算服务进程（例如 `uvicorn`/`gunicorn`），确保 `precompile_jax_solver.py` 走完后入口才感知清爽的 `jax_jit` 性能。

---
*Updated: 2026-04-03*
