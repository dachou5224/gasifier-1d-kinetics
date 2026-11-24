# 复现命令

```bash
cd /Users/liuzhen/AI-projects/gasifier-1d-kinetic
git rev-parse HEAD          # 期望 c7cde49
./.venv/bin/python scripts/p06a_controlled_design_and_separation.py
cd outputs/diagnostics/operating-response-comparison/06A
shasum -a 256 -c sha256.txt
```

## 口径说明

- 求解入口：`model.p1_5_round_20261030.run_case`，`n_cells = 182`，`solver_method = minimize`，`jacobian_mode = scipy`。
- 名义工况：`/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases` 下 P1/P3/P4 预览 parquet 的**入炉侧**首行；未读取任何装置出口组成作为真值。
- 响应配对：同一条名义工况下只改 O2 流量 ±2.5%，得 `rel_u ≈ 4.1%`。
- 预设测量分辨率：σ_y ∈ {0.10, 0.25, 0.50} pp，属设计假设，不是实测标定值。
- 求解器确定性：同一输入重复求解逐位相同（见 `baseline_equivalence.csv` 的 `historical_no_key` 行，dCO = 0）。
