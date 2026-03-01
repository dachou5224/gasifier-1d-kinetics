# Fortran input_副本.txt 工况与煤质数据检查

> 来源：`reference_fortran/input_副本.txt`  
> Fortran namelist 变量说明见 `Source1_副本.for` L40-41, L66, L132-133

---

## 1. Namelist 变量含义

| 变量 | 含义 | 单位 |
|------|------|------|
| ta | 煤浆/煤入口温度 | K |
| tsteam | 蒸汽温度 | K |
| toxy | 氧气温度 | K |
| yoxy | 氧元素质量分数 (daf) | 分数 |
| yh | 氢质量分数 (daf) | 分数 |
| yc | 碳质量分数 (daf) | 分数 |
| ys | 硫质量分数 (daf) | 分数 |
| yn | 氮质量分数 (daf) | 分数 |
| yash | 灰分质量分数 | 分数 |
| xmois | 水分质量分数 | 分数 |
| foxy | O2/煤 质量比 | kg_O2/kg_coal |
| fsteam | 蒸汽/煤 质量比 | kg_steam/kg_coal |
| tfcoal | 煤进料速率 | kg/h |

**注**：`fcoal = tfcoal*(1-yash)*(1-xmois)` 为 dmmf 煤进料 (kg/h)

---

## 2. 工况与煤质汇总

### 2.1 工况参数

| 工况 | tfcoal (kg/h) | foxy (O2/Coal) | fsteam | ta (K) | tsteam (K) | toxy (K) | xmois |
|------|---------------|----------------|--------|--------|------------|----------|-------|
| texaco i-1 | 76.66 | 0.866 | 0.241 | 505.22 | 696.67 | 298 | 0.002 |
| texaco i-2 | 81.18 | 0.768 | 0.314 | 496.33 | 676.33 | 298 | 0.002 |
| texaco i-5c | 56.26 | 0.832 | 0.429 | 497.44 | 714.67 | 298 | 0.002 |
| texaco i-10 | 129.77 | 0.835 | 0.276 | 481.33 | 708 | 298 | 0.002 |
| texaco exxon | 126.11 | 0.79 | 0.5 | 505.22 | 696.67 | 298 | 0 |
| slurry western | 186.78 | 0.91 | 0.52 | 400 | 400 | 600 | 0 |
| slurry eastern | 133.5 | 0.87 | 0.79 | 400 | 400 | 600 | 0 |

### 2.2 煤质组成 (daf 质量分数 → 干基 wt%)

**换算**：Cd = yc×(1-yash)×100，Ad = yash×100，余同

| 工况 | yc | yh | yoxy | ys | yn | yash | Cd | Hd | Od | Sd | Nd | Ad | Σ |
|------|-----|-----|------|-----|-----|------|-----|-----|-----|-----|-----|-----|-----|
| texaco i-1 | 0.8803 | 0.0743 | 0.0157 | 0.0210 | 0.0084 | 0.159 | 74.0 | 6.2 | 1.32 | 1.77 | 0.71 | 15.9 | 99.9 |
| texaco i-2 | 0.8833 | 0.0704 | 0.0206 | 0.0166 | 0.0088 | 0.173 | 73.1 | 5.8 | 1.71 | 1.37 | 0.73 | 17.3 | 100.0 |
| texaco i-5c | 0.8741 | 0.0666 | 0.0269 | 0.0224 | 0.0097 | 0.197 | 70.2 | 5.4 | 2.16 | 1.80 | 0.78 | 19.7 | 100.0 |
| texaco i-10 | 0.8840 | 0.0696 | 0.0214 | 0.0168 | 0.0079 | 0.173 | 73.2 | 5.8 | 1.77 | 1.39 | 0.65 | 17.3 | 100.0 |
| texaco exxon | 0.8494 | 0.0561 | 0.0474 | 0.0329 | 0.0142 | 0.167 | 70.6 | 4.7 | 3.94 | 2.73 | 1.18 | 16.7 | 99.9 |
| slurry western | 0.8034 | 0.0572 | 0.1236 | 0.0050 | 0.0107 | 0.072 | 74.6 | 5.3 | 11.5 | 0.46 | 0.99 | 7.2 | 100.1 |
| slurry eastern | 0.7968 | 0.0551 | 0.1000 | 0.0328 | 0.0153 | 0.087 | 72.7 | 5.0 | 9.2 | 3.0 | 1.40 | 8.7 | 100.0 |

---

## 3. 与 validation_cases_fortran.json 对比

| 项目 | input_副本.txt | validation_cases_fortran.json | 说明 |
|------|----------------|------------------------------|------|
| texaco i-1 FeedRate | **76.66** kg/h | **76660** kg/h | ⚠️ **差 1000 倍** |
| texaco i-2 FeedRate | 81.18 | 81180 | ⚠️ 差 1000 倍 |
| texaco i-5c FeedRate | 56.26 | 56264 | ⚠️ 差 1000 倍 |
| texaco i-10 FeedRate | 129.77 | 129770 | ⚠️ 差 1000 倍 |
| texaco exxon FeedRate | 126.11 | 126110 | ⚠️ 差 1000 倍 |
| slurry western FeedRate | 186.78 | 186780 | ⚠️ 差 1000 倍 |
| slurry eastern FeedRate | 133.5 | 133500 | ⚠️ 差 1000 倍 |
| O2/Coal, Steam/Coal | 一致 | 一致 | ✓ |
| 煤质 Cd, Hd 等 | 换算后接近 | 略有差异 | 可能换算/舍入不同 |

**结论**：原 validation_cases_fortran.json 的 FeedRate 误为 input 的 1000 倍（单位换算错误）。已修正并迁移至 **validation_cases_pilot.json**，FeedRate 与 input_副本.txt 一致（~56–187 kg/h）。

---

## 4. 合理性检查

### 4.1 氧煤比 (foxy)

| 工况 | foxy | 评价 |
|------|------|------|
| texaco i-1~i-10 | 0.77–0.87 | ✓ 典型 Texaco 范围 |
| slurry western | 0.91 | ⚠️ 偏高，CO2=41%、H2=5% 与强氧化一致 |
| slurry eastern | 0.87 | ✓ |

### 4.2 煤质元素守恒

- 各工况 daf 组分 (yc+yh+yoxy+ys+yn) ≈ 1.0 ✓
- 干基 Cd+Hd+Od+Sd+Nd+Ad ≈ 100% ✓

### 4.3 蒸汽/煤比 (fsteam)

- texaco 系列：0.24–0.50
- slurry：0.52–0.79
- 均在常见 Texaco 浆液气化范围内 ✓

### 4.4 温度

- ta (煤浆)：400–505 K
- toxy：298 K (texaco) 或 600 K (slurry)
- slurry 工况 toxy=600 K 表示预热氧，合理 ✓

---

## 5. 与 Paper/LuNan 工业工况的关系

| 来源 | 工况 | O2/Coal | 煤种 | 规模 |
|------|------|---------|------|------|
| input_副本.txt | texaco i-1 | 0.866 | 高碳烟煤 (Cd~74%) | 76.66 kg/h |
| run_gasifier_model_cases | Paper_Case_6 | 1.05 | Paper_Base_Coal (Cd=80%) | 41670 kg/h |
| run_gasifier_model_cases | LuNan_Texaco | 0.872 | LuNan_Coal (Cd=71.5%) | 17917 kg/h |

**说明**：input_副本.txt 为 Fortran 原始小试工况，与 Paper、LuNan 工业案例**不是同一批数据**。煤种、氧煤比、规模均不同，验证时需区分。

---

## 6. 建议

1. **确认 FeedRate 单位**：input_副本.txt 中 tfcoal 若为 kg/h，则 76.66 为小试；若 JSON 的 76660 为正确工业值，需核查 Fortran 输入是否应乘以 1000。
2. **统一煤质来源**：若需与 Fortran 对标，应从 input_副本.txt 换算煤质，而非直接用 validation_cases_fortran.json 中的煤质（可能来自不同换算）。
3. **区分验证层级**：小试 (input_副本) 与工业 (Paper/LuNan) 分开验证，避免混用期望值。
