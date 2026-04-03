# JAX/JIT 收尾总结 (2026-04-03)

## 本轮完成项

- 公开求解器接口收口为：
  - `minimize`
  - `newton_fd`
  - `jax_jit`
- `validation_cases_final.json` 成为当前统一 benchmark 数据源
- `jax_jit` 路线完成可运行、可对比、可回归的全量脚本
- 关键偏差根因已处理：
  - 热损归一化长度不一致
  - 首格燃烧区 O2 bookkeeping 不一致
  - `@jit` 内导入导致的 `UnexpectedTracerError`

## 当前验证状态

- `tests/integration/test_jax_jit_contracts.py`：通过
- `tests/integration/test_jax_solver.py`：通过
- `docs/minimize_vs_jax_jit_report.md`：已更新为最新 19-case 对比
- `docs/validation_newton_fd_report.md`：已生成

## 结果解读

- 原始文献 benchmark 大写工况已基本贴齐 `minimize`
- 小写 `texaco i-*` / `slurry western` 等工况来自另一套 pilot 数据口径
- 这些小写工况不宜继续与主 benchmark 混合统计

## 提交建议

- 保留：
  - `docs/minimize_vs_jax_jit_report.md`
  - `docs/validation_newton_fd_report.md`
  - `docs/jax_refactor_technical_notes.md`
- 归档或忽略：
  - 历史旧命名报告
  - 本地虚拟环境目录
  - 临时 `_tmp_` 报告

## 建议提交信息

```text
refactor: consolidate solver APIs and align jax_jit with mainline energy and combustion logic
```
