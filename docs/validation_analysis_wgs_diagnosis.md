# 验证算例模拟结果分析：出口温度偏低与 CO 偏低

## 1. 数据汇总

### 1.1 工业规模算例 (validation_results_industrial.json)

| 算例 | 出口 T 预测 | 期望 T | ΔT (°C) | CO 预测 | 期望 CO | ΔCO (pct) | CO2 预测 | H2 预测 |
|------|-------------|--------|---------|---------|---------|-----------|----------|---------|
| Paper_Case_6 (Base) | 1099 | 1370 | **-271** | 48.0 | 61.7 | **-13.7** | 19.4 | 32.2 |
| Paper_Case_6 (Calibrated) | 1234 | 1370 | -136 | 50.2 | 61.7 | -11.5 | 17.9 | 30.2 |
| Paper_Case_1 | 1148 | 1333 | -185 | 49.2 | 59.9 | -10.7 | 19.2 | 31.2 |
| Paper_Case_2 | 1482 | 1452 | +30 | 53.3 | 61.8 | -8.5 | 20.6 | 25.0 |

### 1.2 原始论文算例 (validation_cases_merged.json 期望值)

| 算例 | 期望 T (°C) | 期望 CO | 期望 CO2 | 期望 H2 |
|------|-------------|---------|----------|---------|
| Texaco_I-1 | 1370 | 57.57 | 2.95 | 39.13 |
| Texaco_I-2 | 1333 | 53.06 | 5.15 | 41.0 |
| Texaco_I-10 | - | 55.18 | 4.69 | 39.24 |
| Texaco_Exxon | - | 45.6 (湿基) | 7.52 | 33.8 |
| Coal_Water_Slurry_Western | - | 50.71 | 13.14 | 35.79 |
| Coal_Water_Slurry_Eastern | - | 41.55 | 20.64 | 36.15 |

---

## 2. 偏差模式分析

### 2.1 共性规律

1. **出口温度偏低**：工业算例中 3/4 工况 T 偏低 136–271°C；仅 Paper_Case_2（O2/Coal=1.22 高氧）略高。
2. **CO 系统性偏低**：所有工况 CO 偏低约 8–14 pct，与期望差距明显。
3. **CO2 系统性偏高**：工业算例 CO2 约 18–21%，而 Texaco_I-1 期望仅 2.95%，说明 CO→CO2 转化偏多。
4. **H2 变化不一**：部分工况 H2 略高（32% vs 30%），与 CO+H2O→CO2+H2 的化学计量一致。

### 2.2 化学计量关系

水煤气变换 (WGS) 反应：
```
CO + H2O ⇌ CO2 + H2    (ΔH ≈ -41 kJ/mol，正向略放热)
```

- **若 WGS 正向速率过高**：CO 被消耗 → CO2、H2 增加。
- **现象**：CO 偏低、CO2 偏高、H2 略增，与当前偏差一致。
- **温度**：WGS 正向放热，理论上应略升 T；但若同时存在 C+H2O/C+CO2 吸热、浆液蒸发、热损失等，净效应可能为降温。

---

## 3. WGS 反应速率过高的证据与机制

### 3.1 现象支持

| 证据 | 说明 |
|------|------|
| CO 系统性偏低 | 所有工业工况 CO 偏低 8–14 pct |
| CO2 系统性偏高 | 工业 CO2 18–21% vs Texaco 期望 2.95–5% |
| 化学计量一致 | CO 减少量 ≈ CO2 增加量，符合 WGS 计量 |
| 文献诊断 | fortran_analysis.md、texaco_i1_vs_exxon_analysis.md 均指出 WGS 过度反应 |

### 3.2 当前 Python WGS 实现 (kinetics_service.py)

```python
# 净速：r_net = k_fwd × (C_CO·C_H2O - C_CO2·C_H2/K_eq)
k_fwd = 2.88e5 * exp(-116148/(R*T))
K_eq = exp(4578/T - 4.33)
r_net = k_fwd * (C['CO']*C['H2O'] - C['CO2']*C['H2']/(K_eq+1e-12))
```

- 已实现：Ts≤1000K 关闭、平衡驱动力 (C_CO·C_H2O - C_CO2·C_H2/K_eq)。
- 潜在问题：**k_fwd 或等效速率可能偏大**，导致在相同驱动力下净速率过高。

### 3.3 Fortran 与 Python 差异 (Source1_副本.for, wgshift)

| 项目 | Fortran | Python |
|------|---------|--------|
| 催化因子 f | **f = 0.2** | 无对应项 |
| 指前因子 | 2.877e5 × f = 5.75e4 | 2.88e5 |
| 额外因子 rat | **rat = exp(-8.91 + 5553/ts)** | 无 |
| 压力修正 pf | pf = pt^ep, ep=0.5-pt/250 | 无 |

**rat 因子影响**（ts=1500K）：
```
rat = exp(-8.91 + 5553/1500) = exp(-5.21) ≈ 0.0055
```
Fortran 在高温下对 WGS 有约 0.5% 的额外抑制，Python 未实现该因子。

**催化因子 f=0.2**：Fortran 有效指前因子约为 Python 的 1/5。

---

## 4. 问题定位结论

### 4.1 主要假设：WGS 速率偏高

1. **催化因子缺失**：Fortran 使用 f=0.2，Python 未引入，等效指前因子可能偏大 5 倍。
2. **rat 因子缺失**：Fortran 的 `exp(-8.91+5553/ts)` 在 1200–1600K 将速率压低约 1–2 个数量级。
3. **综合效应**：Python 的 WGS 净速率可能比 Fortran 高一个数量级以上，导致 CO 过度转化为 CO2。

### 4.2 次要因素

- **浆液蒸发**：工业工况浆液浓度 60%，蒸发吸热可能加剧出口温度偏低。
- **热损失**：HeatLossPercent 3% 对温度有影响。
- **起燃与 Cell 0**：高流量下 dz_cell0 已达 0.5 m 上限，起燃可能仍不充分。

---

## 5. 建议的修正方向

### 5.1 优先：WGS 速率缩放

在 `kinetics_service.py` 中引入与 Fortran 一致的因子：

```python
# 催化因子 (Fortran f=0.2)
WGS_CATALYTIC_FACTOR = 0.2

# rat 因子 (Fortran: exp(-8.91+5553/ts))
# 需用 Ts_particle 或 T
rat = np.exp(-8.91 + 5553.0 / T_wgs_check) if T_wgs_check > 1000 else 0.0

# 修正后
k_eff = self.A_homo['WGS'] * WGS_CATALYTIC_FACTOR * rat * np.exp(-self.E_homo['WGS']/(R_CONST*T))
r_net = k_eff * (C['CO']*C['H2O'] - C['CO2']*C['H2']/(K_eq+1e-12))
```

### 5.2 可选：压力修正

Fortran 使用 `pf = pt^ep`，可在高压工况下补充压力修正。

### 5.3 验证结果（已实施）

| 配置 | Paper_Case_6 Base T | CO | 说明 |
|------|---------------------|-----|------|
| 修正前 | 1099 °C | 48.0% | 基准 |
| **催化因子 0.2** | **1309 °C** | **51.5%** | 温度 +210°C，CO +3.5 pct ✓ |
| 催化因子 + rat | 1198 °C | 40.8% | rat 过度抑制，CO 反降 |

**结论**：仅用催化因子 f=0.2 效果最佳；rat 因子在工业工况下过度压低 WGS，可能抑制了 WGS 逆向（CO2+H2→CO+H2O）产 CO，导致 CO 反而下降。当前默认 `_use_rat = False`。

---

## 6. 催化因子 f 与 rat 因子的物理意义与文献来源

### 6.1 代码与文档中的记载

| 来源 | 催化因子 f=0.2 | rat = exp(-8.91+5553/ts) |
|------|----------------|---------------------------|
| **Fortran (Source1_副本.for)** | `f = 0.2`，无注释 | `rat = exp(-8.91+5553./ts)`，无注释 |
| **validation_cases_OriginalPaper.json** | `catalytic_reactivity_factor: 0.2`，无说明 | 未出现 |
| **fortran_analysis.md** | 仅列出与 Python 差异 | 未提及 |
| **1D_Gasifier_Model_Manual_cn.md** | 未提及 | 未提及 |

### 6.2 结论：**无明确物理意义或文献引用**

- **催化因子 f=0.2**：Fortran 与 OriginalPaper 中均只给出数值，**无物理含义说明**，也未引用 Wen & Chaung (1979) 或其它文献。常见推测为“催化活性分数”或“有效反应面积比例”，但属推断，非代码/文档明示。
- **rat 因子**：Fortran 中**无任何注释**，项目内文档均未解释。形式 `exp(-8.91 + 5553/T)` 类似 Arrhenius（5553≈E/R，E≈11 kcal/mol），但与 WGS 主活化能 27760 cal/mol 不符，可能为经验修正或来自未标注的文献。

### 6.3 建议

在缺乏文献依据时，引入这些因子应视为**与 Fortran 的数值对齐**，而非有明确物理依据的修正。若需发表或严格溯源，建议查阅 Wen & Chaung (1979) 原文及后续引用该模型的工作，确认 f 与 rat 的原始定义与出处。

---

## 7. 参考文献

- `docs/fortran_analysis.md`：WGS 平衡驱动力、温度阈值
- `docs/texaco_i1_vs_exxon_analysis.md`：WGS 逆向吸热诊断
- `reference_fortran/Source1_副本.for`：wgshift 子程序 (Line 1264–1305)
- `data/validation_cases_OriginalPaper.json`：model_parameters.water_gas_shift
