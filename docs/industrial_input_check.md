# 工业工况输入数据检查

> 来源：`run_gasifier_model_cases.py` INDUSTRIAL_CASES、`validation_cases_industrial.json`、`chemistry.py` COAL_DATABASE  
> **已合并**：原 4 个 Paper 工况（同一基础改 O2/Coal）合并为 1 个 Paper_Case_6，工业工况现仅 2 例

---

## 1. 工况与操作条件汇总

### 1.1 run_gasifier_model_cases.py（合并后 2 例）

| 工况 | 煤种 | FeedRate (kg/h) | SlurryConc | O2/Coal | Steam/Coal | P (MPa) | T_in (K) | HeatLoss | L (m) | D (m) |
|------|------|-----------------|------------|---------|------------|---------|----------|----------|-------|-------|
| Paper_Case_6 | Paper_Base_Coal | 41670 | 60% | 1.05 | 0.08 | 4.08 | 300 | 8.0% | 6.0 | 2.0 |
| LuNan_Texaco | LuNan_Coal | 17917 | 66% | 0.872 | 0.0 | 4.0 | 400 | 5.0% | 6.87 | 1.68 |

### 1.2 validation_cases_industrial.json（仅 LuNan）

| 项目 | JSON 值 | run 脚本值 | 一致性 |
|------|---------|------------|--------|
| coal_feed_rate_kg_hr | 17917 | 17917 | ✓ |
| slurry_concentration_pct | 66 | 66 | ✓ |
| O2_to_fuel_ratio | 0.872 | 0.872 | ✓ |
| steam_to_fuel_ratio | 0 | 0 | ✓ |
| pressure_Pa | 4e6 | 4e6 | ✓ |
| inlet_temperature_K | 400 | 400 | ✓ |
| heat_loss_percent | **2** | **5.0** | ⚠️ 不同 |
| L_reactor_m | 6.87 | 6.87 | ✓ |
| D_reactor_m | 1.68 | 1.68 | ✓ |

**说明**：run 脚本 LuNan 用 HeatLossPercent=5.0%（热损扫描优化），JSON 为 2%。以 run 脚本为准。

---

## 2. 煤质数据 (COAL_DATABASE)

### 2.1 Paper_Base_Coal

| 项目 | 值 | 说明 |
|------|-----|------|
| Cd | 80.19 | 干基碳 |
| Hd | 4.83 | 干基氢 |
| Od | 9.76 | 干基氧 |
| Nd | 0.85 | 干基氮 |
| Sd | 0.41 | 干基硫 |
| Ad | 7.35 | 干基灰 |
| Vd | 31.24 | 干基挥发分 |
| FCd | 61.41 | 干基固定碳 |
| Mt | 4.53 | 水分 |
| HHV_d | 29800 kJ/kg | 干基高位热值 |

**元素守恒**：Cd+Hd+Od+Nd+Sd+Ad ≈ 103.39（>100，需核对；可能 Od 为差减项）

### 2.2 LuNan_Coal

| 项目 | 值 | 说明 |
|------|-----|------|
| Cd | 71.5 | 干基碳 |
| Hd | 4.97 | 干基氢 |
| Od | 11.15 | 干基氧 |
| Nd | 1.07 | 干基氮 |
| Sd | 2.16 | 干基硫 |
| Ad | 9.15 | 干基灰 |
| Vd | 32.0 | 干基挥发分 |
| FCd | 58.85 | 干基固定碳 |
| Mt | 0.0 | 水分 |
| HHV_d | 27800 kJ/kg | 鲁南北宿+落陵混煤 |

**与 JSON 一致**：validation_cases_industrial.json 中 ultimate_analysis 与 COAL_DATABASE 一致。

---

## 3. 调参项（WGS/燃烧等）

| 工况 | WGS_CatalyticFactor | WGS_K_Factor | HeatLoss | CharCombustionRateFactor | UseFortranDiffusion | P_O2_Combustion_atm |
|------|---------------------|--------------|----------|---------------------------|---------------------|---------------------|
| Paper_Case_6 | 0.5 | 0.2 | 8.0% | 默认 | 默认 | 默认 0.05 |
| LuNan_Texaco | 默认 0.2 | 0.5 | 5.0% | 0.35 | True | 0.03 |

---

## 4. 期望值（合并后 2 例）

| 工况 | T (°C) | CO (%) | H2 (%) | CO2 (%) |
|------|--------|--------|--------|---------|
| Paper_Case_6 | 1370 | 61.7 | 30.3 | 1.3 |
| LuNan_Texaco | 1350 | 48.82 | 36.58 | 14.41 |

---

## 5. 已知问题与建议

| 项目 | 问题 | 建议 |
|------|------|------|
| Paper CO2=1.3% | 极低，典型 Texaco 5–20% | 核对文献定义 |
| LuNan heat_loss | JSON 2% vs run 1.5% | 以 run 脚本 1.5% 为准 |

---

## 6. 数据来源一致性

- **Paper 工况**：来自 `run_gasifier_model_cases.py`，煤种取自 `chemistry.py` COAL_DATABASE
- **LuNan**：`run_gasifier_model_cases.py` 与 `validation_cases_industrial.json` 操作条件基本一致（除 heat_loss）
- **几何**：Paper 默认 L=6 m, D=2 m；LuNan 用 L=6.87 m, D=1.68 m（鲁南厂实际）
