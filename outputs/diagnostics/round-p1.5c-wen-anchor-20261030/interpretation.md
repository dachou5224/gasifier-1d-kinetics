# P1.5-C Wen/Fortran 锚点 —— interpretation

## 1. Fortran 是否成功运行

gfortran 16.1.0 `-std=legacy -O2 -ffixed-form` 编译通过；唯一补丁是把只用于旧绘图标签的 `ic1..ic5` 从 INTEGER 改成 CHARACTER*4（DATA 语句类型错误）。7 个算例全部跑完，进程退出码 2 来自文件末尾多读一次 namelist，不影响结果。出口值见 `fortran_outlet.csv`。

## 2. 模式 F 是否达到锚定判据

5 个干粉算例中，模式 F 全部**可评估**判据通过的算例数：**0/5**。逐算例逐判据见 `anchor_criteria.csv`。

- `texaco exxon`：未通过 X_C_pp、CO_vol_pct、CO2_vol_pct、T_gas_K
- `texaco i-1`：未通过 X_C_pp、CO2_vol_pct、T_gas_K
- `texaco i-10`：未通过 X_C_pp、T_gas_K
- `texaco i-2`：未通过 X_C_pp、T_gas_K
- `texaco i-5c`：未通过 CO_vol_pct

O2 降到入口 1% 的位置不可评估：Fortran 只在收敛后打印少数气室，第一次打印时 O2 已为 0；Python 在 cell 0 就已低于阈值。该判据在两侧都没有分辨率，未计入上面的通过数。

## 3. 模式 M 与 Fortran 的差距

- `texaco i-1`：X_C 87.7 vs 92.4 pp，CO 52.7 vs 54.9，H2 37.9 vs 39.3，CO2 7.5 vs 4.8，T 1228 vs 1373 K
- `texaco i-2`：X_C 82.2 vs 84.7 pp，CO 49.8 vs 51.8，H2 41.2 vs 40.9，CO2 8.0 vs 6.4，T 1219 vs 1354 K
- `texaco i-5c`：X_C 99.9 vs 98.5 pp，CO 54.5 vs 52.0，H2 38.2 vs 39.6，CO2 6.4 vs 7.4，T 1429 vs 1401 K
- `texaco i-10`：X_C 91.8 vs 88.4 pp，CO 54.5 vs 53.3，H2 37.8 vs 39.0，CO2 6.3 vs 6.9，T 1257 vs 1309 K
- `texaco exxon`：X_C 99.9 vs 92.6 pp，CO 54.5 vs 49.1，H2 37.3 vs 38.8，CO2 6.9 vs 10.8，T 1442 vs 1286 K
- `slurry weste`：X_C 99.9 vs 97.4 pp，CO 53.2 vs 48.6，H2 35.5 vs 37.5，CO2 10.7 vs 13.4，T 1412 vs 1212 K
- `slurry easte`：X_C 99.9 vs 96.4 pp，CO 46.7 vs 44.0，H2 38.0 vs 39.0，CO2 13.7 vs 15.5，T 1366 vs 1234 K

按平均 |ΔX_C| 排序的归因项（模式 M 下一次性切换）：UseFortranDiffusion（0.36 pp）、WGS_RatFactor（0.03 pp）、CharCombustionRateFactor（0.00 pp）。
没有开关的差异项（WGS 平衡常数、热损闭合、颗粒温度反应热、热解分配、氢气限幅、入口温度分裂）无法逐项切换，剩余偏差主要来自它们，尤其热损闭合（Fortran 燃烧区按 30% 反应热、气化区按气壁换热，本模型按 2% HHV × T⁴）。

### 温度差距的来源（B9 敏感性，模式 F，N=160）

- `texaco i-1`：带 `Hf_coal` 1225 K → 去掉 `Hf_coal` 1531 K（Δ+306 K），Fortran 1373 K；X_C 87.7% → 99.9%，Fortran 92.4%。
- `texaco i-2`：带 `Hf_coal` 1215 K → 去掉 `Hf_coal` 1315 K（Δ+99 K），Fortran 1354 K；X_C 82.0% → 94.8%，Fortran 84.7%。
- `texaco i-5c`：带 `Hf_coal` 1425 K → 去掉 `Hf_coal` 1515 K（Δ+90 K），Fortran 1401 K；X_C 99.9% → 99.9%，Fortran 98.5%。
- `texaco i-10`：带 `Hf_coal` 1256 K → 去掉 `Hf_coal` 1443 K（Δ+188 K），Fortran 1309 K；X_C 90.8% → 99.9%，Fortran 88.4%。
- `texaco exxon`：带 `Hf_coal` 1442 K → 去掉 `Hf_coal` 1451 K（Δ+9 K），Fortran 1286 K；X_C 99.9% → 99.9%，Fortran 92.6%。

说明：`Hf_coal`（固相焓按原煤生成焓计，审计 B9/D-HUAYI-02）是出口温度差距的主导项——去掉它会把温度抬高 300 K 量级并让 X_C 撞到 0.999，说明本模型现在落在 Fortran 与“无形成焓”之间。热损不是主因：把 `HeatLossPercent` 从 2% 降到 0 只抬高约 22 K（texaco i-1），升到 5% 只降低约 32 K。该差异没有开关，按卡片要求本轮不新建开关、不改方程。

## 4. 两模式的碳转化率

- `texaco i-1`：Fortran 92.4%，模式 M 87.7%，模式 F 87.7%
- `texaco i-2`：Fortran 84.7%，模式 M 82.2%，模式 F 82.0%
- `texaco i-5c`：Fortran 98.5%，模式 M 99.9%，模式 F 99.9%
- `texaco i-10`：Fortran 88.4%，模式 M 91.8%，模式 F 90.8%
- `texaco exxon`：Fortran 92.6%，模式 M 99.9%，模式 F 99.9%
- `slurry weste`：Fortran 97.4%，模式 M 99.9%，模式 F 99.9%
- `slurry easte`：Fortran 96.4%，模式 M 99.9%，模式 F 99.9%

## 5. 台账

共 14 项，其中有开关且模式 F 已切换的 6 项；其余记为 `none`，按卡片要求本轮不新建开关。
