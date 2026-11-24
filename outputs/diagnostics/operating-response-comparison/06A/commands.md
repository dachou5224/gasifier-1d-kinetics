# 复现命令

```bash
cd /Users/liuzhen/AI-projects/gasifier-1d-kinetic
git rev-parse HEAD          # 期望 c7cde49
./.venv/bin/python scripts/p06a_controlled_design_and_separation.py
cd outputs/diagnostics/operating-response-comparison/06A
shasum -a 256 -c sha256.txt
```

## 底座（P2.2-A 冻结配置）

| 项 | 值 | 来源 |
|---|---|---|
| MeshFamily | `wsr_plus_pfr` | `generate_p2_2a_20261118.build_op` |
| WSRLength_m | 1.0 | 同上 |
| PFRCellLength_m | 0.0515 | 同上 |
| n_cells | 182 | `generate_p2_0a_20261113.mesh_n()` |
| HeatLossModel | `total_percent` | 同上 |
| HeatLossPercentTotal | 1.0 | 同上 |
| GasificationRateFactor | 1.0 | 同上 |
| CharCO2RateFactor | 1.0 | 同上 |
| OxygenMixingModel | `instant` | 同上 |
| carrier_gas_flow | `p18a._carrier_kg_s(co2_carrier_measured_nm3_h)` | P1.8-D 实测口径 |
| coal_flow | 显式干煤 `coal_dry_recon_phase_kg_s` | P2.2-D |

## 输入（只读）

- `windows_142.csv` sha256 `a24b5fe636752d8b7872ebacd569e40325244f90b851ae4ff1adaeee6e989119`
- `baseline_142.csv` sha256 `0c1fa8e2f7d7843295369a75a72c7460db65ecd3a11986bc02e9781dc4247c26`

## 口径

- 响应配对：同一窗口 O2 流量 ×0.975 / ×1.025。
- 预设 σ_y ∈ {0.10, 0.25, 0.50} pp，**设计假设，非实测标定**。
- 误差传播分开：组成用 σ_y；响应用 σ_r = ln(1.05)·√2·σ_y/Δln u。
- 未读取任何装置出口组成作为真值。
