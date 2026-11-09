# P1.12-A 整炉总量热损 + 全混区解耦扫描 — 判读

Status: `EXECUTED — model repo`

依据：`manuscript/model-side-handoff-p1.12a-20261106.md`；上游 `manuscript/model-evidence-verification-20261106-p111.md` I2–I5。

## 0. 结构与热损模型

结构冻结为 `wsr_plus_pfr`、`dz_down` = 0.0515 m、`OxygenMixingModel = instant`。热损改为**整炉总量**：`Q_tot = HL_tot/100 × m_dry × HHV_d`（**干煤**口径），逐格按 `w_i = dz_i·clip((T_i/1800)^4, 0.1, 5)` 归一化分配。逐格推进的求解器在解第 i 格时不知道下游温度，因此 T 权重来自**上一轮全炉温度场**，外层不动点迭代至 `|ΣQ−Q_tot|/Q_tot < 1e-6` 且 `max|Δw/w| < 1e-4`，最多 30 轮。

`total_percent_uniform` 是同总量的对照：`w_i = dz_i`，无需迭代，`ΣQ` 逐位等于 `Q_tot`（第 2 节检查 2）。

## 1. 单元检查（卡第 2 节）

- 共 19 项，全部通过：**False**
  - 检查 1：1/1 通过
  - 检查 2：4/4 通过
  - 检查 3：10/10 通过
  - 检查 3b：1/1 通过
  - 检查 4：0/3 通过

### 迭代轮数分布（主扫描全量）

| L_wsr | HL_tot | min | median | max |
|---|---|---|---|---|
| 0 | 0 | 3 | 4 | 6 |
| 0 | 1 | 4 | 4 | 6 |
| 0 | 2 | 5 | 5 | 7 |
| 0 | 3 | 5 | 5 | 6 |
| 0 | 4 | 6 | 6 | 10 |
| 1 | 0 | 4 | 4 | 6 |
| 1 | 1 | 4 | 4 | 5 |
| 1 | 2 | 4 | 5 | 7 |
| 1 | 3 | 5 | 5 | 6 |
| 1 | 4 | 5 | 6 | 8 |
| 2 | 0 | 4 | 4 | 6 |
| 2 | 1 | 4 | 4 | 7 |
| 2 | 2 | 4 | 5 | 5 |
| 2 | 3 | 5 | 5 | 5 |
| 2 | 4 | 5 | 6 | 6 |
| 4 | 0 | 3 | 4 | 5 |
| 4 | 1 | 4 | 4 | 6 |
| 4 | 2 | 4 | 4 | 5 |
| 4 | 3 | 4 | 5 | 5 |
| 4 | 4 | 5 | 5 | 5 |
| 6 | 0 | 3 | 3 | 3 |
| 6 | 1 | 3 | 4 | 4 |
| 6 | 2 | 4 | 4 | 4 |
| 6 | 3 | 4 | 4 | 4 |
| 6 | 4 | 5 | 5 | 5 |

- 最终 `|ΣQ−Q_tot|/Q_tot` 最大 5.61e-16（450 行全部须 < 1e-06）
- 状态分布：{'completed': 450}

## 2. K1–K8 结论

### K1 纯混合杠杆（HL_tot = 1%）

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | dX_C_pp | -0.7737 | nan | None | HL_tot=1%，L_wsr 0.2575→6；9 窗中位，范围 -1.11…1.998e-13 |
| nan | dT_out_K | 27.1899 | nan | None | HL_tot=1%，L_wsr 0.2575→6；9 窗中位，范围 5.727…32.76 |
| nan | dCO2_vol_pct | 0.3515 | nan | None | HL_tot=1%，L_wsr 0.2575→6；9 窗中位，范围 0.06196…0.464 |
| nan | dT_wsr_K | -7.9452 | nan | None | HL_tot=1%，L_wsr 0.2575→6；9 窗中位，范围 -15.41…3.097 |

### K1 与 P1.11-A 的对比：被热损耦合解释掉的份额

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | dX_C_pp__explained_by_coupling | 0.7891 | nan | None | P1.11-A（hhv_percent, HLP=1%）同档 -3.669；本卡（HL_tot=1%）-0.774；被热损耦合解释掉的比例 78.9% |
| nan | dT_out_K__explained_by_coupling | 0.6138 | nan | None | P1.11-A（hhv_percent, HLP=1%）同档 -70.396；本卡（HL_tot=1%）+27.190；被热损耦合解释掉的比例 61.4% |

### K2 热损杠杆（每 1% HL_tot）

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 1.0000 | dX_C_pp_per_pct | -1.1149 | nan | None | 每 1% HL_tot 的变化（45 档线拟合，9 窗池化） |
| 1.0000 | dT_out_K_per_pct | -53.3653 | nan | None | 每 1% HL_tot 的变化（45 档线拟合，9 窗池化） |
| 1.0000 | dCO2_vol_pct_per_pct | 0.6856 | nan | None | 每 1% HL_tot 的变化（45 档线拟合，9 窗池化） |
| 4.0000 | dX_C_pp_per_pct | -1.5394 | nan | None | 每 1% HL_tot 的变化（45 档线拟合，9 窗池化） |
| 4.0000 | dT_out_K_per_pct | -41.5297 | nan | None | 每 1% HL_tot 的变化（45 档线拟合，9 窗池化） |
| 4.0000 | dCO2_vol_pct_per_pct | 0.8566 | nan | None | 每 1% HL_tot 的变化（45 档线拟合，9 窗池化） |

### K3 可信热损下的到达（HL_tot ≤ 1%）

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 0.2575 | n_windows_within_1pp::HL0.5 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +7.32 pp；|偏差|≤1 pp 的窗数 0/9 |
| 0.2575 | n_windows_within_1pp::HL1 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +7.03 pp；|偏差|≤1 pp 的窗数 0/9 |
| 1.0000 | n_windows_within_1pp::HL0.5 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +7.32 pp；|偏差|≤1 pp 的窗数 0/9 |
| 1.0000 | n_windows_within_1pp::HL1 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +6.64 pp；|偏差|≤1 pp 的窗数 0/9 |
| 2.0000 | n_windows_within_1pp::HL0.5 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +7.32 pp；|偏差|≤1 pp 的窗数 0/9 |
| 2.0000 | n_windows_within_1pp::HL1 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +6.53 pp；|偏差|≤1 pp 的窗数 0/9 |
| 4.0000 | n_windows_within_1pp::HL0.5 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +7.29 pp；|偏差|≤1 pp 的窗数 0/9 |
| 4.0000 | n_windows_within_1pp::HL1 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +6.33 pp；|偏差|≤1 pp 的窗数 0/9 |
| 6.0000 | n_windows_within_1pp::HL0.5 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +6.90 pp；|偏差|≤1 pp 的窗数 0/9 |
| 6.0000 | n_windows_within_1pp::HL1 | 0.0000 | 0.0000 | 0.0000 | 9 窗中位偏差 +5.93 pp；|偏差|≤1 pp 的窗数 0/9 |

### K4 所需热量移除 HL_tot*

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 0.2575 | HL_tot_star_pct::-28D_008__W003 | 3.7609 | nan | None | 偏差范围 -0.23…+2.48 pp；区间内插值 |
| 1.0000 | HL_tot_star_pct::-28D_008__W003 | 2.6687 | nan | None | 偏差范围 -1.98…+2.47 pp；区间内插值 |
| 2.0000 | HL_tot_star_pct::-28D_008__W003 | 2.3852 | nan | None | 偏差范围 -2.87…+2.44 pp；区间内插值 |
| 4.0000 | HL_tot_star_pct::-28D_008__W003 | 2.1997 | nan | None | 偏差范围 -3.58…+2.36 pp；区间内插值 |
| 6.0000 | HL_tot_star_pct::-28D_008__W003 | 1.9669 | nan | None | 偏差范围 -4.06…+2.16 pp；区间内插值 |
| 0.2575 | HL_tot_star_pct::22-09-28__W001 | nan | nan | None | 偏差范围 +2.19…+3.49 pp；区间内无零点 |
| 1.0000 | HL_tot_star_pct::22-09-28__W001 | nan | nan | None | 偏差范围 +0.61…+3.49 pp；区间内无零点 |
| 2.0000 | HL_tot_star_pct::22-09-28__W001 | 3.8590 | nan | None | 偏差范围 -0.24…+3.49 pp；区间内插值 |
| 4.0000 | HL_tot_star_pct::22-09-28__W001 | 3.5450 | nan | None | 偏差范围 -0.89…+3.49 pp；区间内插值 |
| 6.0000 | HL_tot_star_pct::22-09-28__W001 | 3.2895 | nan | None | 偏差范围 -1.41…+3.49 pp；区间内插值 |
| 0.2575 | HL_tot_star_pct::-03D_011__W001 | nan | nan | None | 偏差范围 +7.66…+7.67 pp；区间内无零点 |
| 1.0000 | HL_tot_star_pct::-03D_011__W001 | nan | nan | None | 偏差范围 +6.78…+7.67 pp；区间内无零点 |
| 2.0000 | HL_tot_star_pct::-03D_011__W001 | nan | nan | None | 偏差范围 +6.04…+7.67 pp；区间内无零点 |
| 4.0000 | HL_tot_star_pct::-03D_011__W001 | nan | nan | None | 偏差范围 +5.42…+7.67 pp；区间内无零点 |
| 6.0000 | HL_tot_star_pct::-03D_011__W001 | nan | nan | None | 偏差范围 +4.96…+7.67 pp；区间内无零点 |
| 0.2575 | HL_tot_star_pct::-03N_013__W001 | nan | nan | None | 偏差范围 +7.01…+7.32 pp；区间内无零点 |
| 1.0000 | HL_tot_star_pct::-03N_013__W001 | nan | nan | None | 偏差范围 +5.74…+7.32 pp；区间内无零点 |
| 2.0000 | HL_tot_star_pct::-03N_013__W001 | nan | nan | None | 偏差范围 +4.94…+7.32 pp；区间内无零点 |
| 4.0000 | HL_tot_star_pct::-03N_013__W001 | nan | nan | None | 偏差范围 +4.29…+7.32 pp；区间内无零点 |
| 6.0000 | HL_tot_star_pct::-03N_013__W001 | nan | nan | None | 偏差范围 +3.83…+7.32 pp；区间内无零点 |
| 0.2575 | HL_tot_star_pct::-09D_014__W004 | nan | nan | None | 偏差范围 +5.57…+7.48 pp；区间内无零点 |
| 1.0000 | HL_tot_star_pct::-09D_014__W004 | nan | nan | None | 偏差范围 +3.90…+7.48 pp；区间内无零点 |
| 2.0000 | HL_tot_star_pct::-09D_014__W004 | nan | nan | None | 偏差范围 +3.02…+7.48 pp；区间内无零点 |
| 4.0000 | HL_tot_star_pct::-09D_014__W004 | nan | nan | None | 偏差范围 +2.35…+7.48 pp；区间内无零点 |
| 6.0000 | HL_tot_star_pct::-09D_014__W004 | nan | nan | None | 偏差范围 +1.85…+7.45 pp；区间内无零点 |
| 0.2575 | HL_tot_star_pct::-09D_015__W001 | nan | nan | None | 偏差范围 +5.16…+7.74 pp；区间内无零点 |
| 1.0000 | HL_tot_star_pct::-09D_015__W001 | nan | nan | None | 偏差范围 +3.43…+7.73 pp；区间内无零点 |
| 2.0000 | HL_tot_star_pct::-09D_015__W001 | nan | nan | None | 偏差范围 +2.53…+7.70 pp；区间内无零点 |
| 4.0000 | HL_tot_star_pct::-09D_015__W001 | nan | nan | None | 偏差范围 +1.83…+7.64 pp；区间内无零点 |
| 6.0000 | HL_tot_star_pct::-09D_015__W001 | nan | nan | None | 偏差范围 +1.34…+7.44 pp；区间内无零点 |
| 0.2575 | HL_tot_star_pct::-10N_006__W001 | nan | nan | None | 偏差范围 +3.61…+7.69 pp；区间内无零点 |
| 1.0000 | HL_tot_star_pct::-10N_006__W001 | nan | nan | None | 偏差范围 +1.66…+7.52 pp；区间内无零点 |
| 2.0000 | HL_tot_star_pct::-10N_006__W001 | nan | nan | None | 偏差范围 +0.71…+7.46 pp；区间内无零点 |
| 4.0000 | HL_tot_star_pct::-10N_006__W001 | 3.9703 | nan | None | 偏差范围 -0.07…+7.29 pp；区间内插值 |
| 6.0000 | HL_tot_star_pct::-10N_006__W001 | 3.7525 | nan | None | 偏差范围 -0.56…+6.90 pp；区间内插值 |
| 0.2575 | HL_tot_star_pct::-11N_009__W002 | nan | nan | None | 偏差范围 +2.53…+6.60 pp；区间内无零点 |
| 1.0000 | HL_tot_star_pct::-11N_009__W002 | nan | nan | None | 偏差范围 +0.59…+6.42 pp；区间内无零点 |
| 2.0000 | HL_tot_star_pct::-11N_009__W002 | 3.8139 | nan | None | 偏差范围 -0.36…+6.36 pp；区间内插值 |
| 4.0000 | HL_tot_star_pct::-11N_009__W002 | 3.4842 | nan | None | 偏差范围 -1.13…+6.19 pp；区间内插值 |
| 6.0000 | HL_tot_star_pct::-11N_009__W002 | 3.2732 | nan | None | 偏差范围 -1.63…+5.79 pp；区间内插值 |
| 0.2575 | HL_tot_star_pct::22-10-11__W001 | nan | nan | None | 偏差范围 +2.53…+6.54 pp；区间内无零点 |
| 1.0000 | HL_tot_star_pct::22-10-11__W001 | nan | nan | None | 偏差范围 +0.60…+6.37 pp；区间内无零点 |
| 2.0000 | HL_tot_star_pct::22-10-11__W001 | 3.8222 | nan | None | 偏差范围 -0.34…+6.29 pp；区间内插值 |
| 4.0000 | HL_tot_star_pct::22-10-11__W001 | 3.4950 | nan | None | 偏差范围 -1.10…+6.12 pp；区间内插值 |
| 6.0000 | HL_tot_star_pct::22-10-11__W001 | 3.2803 | nan | None | 偏差范围 -1.61…+5.72 pp；区间内插值 |

### K5 输入等效

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | unit_hloss_dX_C_pp_per_pct | 1.3272 | nan | None | K2 各档斜率 |ΔX_C| 的中位，用作输入扰动的等价换算 |
| nan | dX_C_pp::Mt_4 | -0.0782 | nan | None | ΔT_out 中位 -25.34 K；ΔCO2 中位 +0.753 vol% |
| nan | equivalent_HL_tot_pct::Mt_4 | 0.0589 | nan | None | 等效热损变化 = −ΔX_C(扰动)/单位热损 ΔX_C；正值表示该扰动相当于**多**移除这些 % 的热损 |
| nan | dX_C_pp::coal_x0.97 | 0.2670 | nan | None | ΔT_out 中位 +95.02 K；ΔCO2 中位 +0.769 vol% |
| nan | equivalent_HL_tot_pct::coal_x0.97 | -0.2012 | nan | None | 等效热损变化 = −ΔX_C(扰动)/单位热损 ΔX_C；正值表示该扰动相当于**多**移除这些 % 的热损 |
| nan | dX_C_pp::coal_x1.03 | -2.5064 | nan | None | ΔT_out 中位 -27.99 K；ΔCO2 中位 +0.084 vol% |
| nan | equivalent_HL_tot_pct::coal_x1.03 | 1.8885 | nan | None | 等效热损变化 = −ΔX_C(扰动)/单位热损 ΔX_C；正值表示该扰动相当于**多**移除这些 % 的热损 |
| nan | dX_C_pp::o2_x0.97 | -2.5884 | nan | None | ΔT_out 中位 -32.74 K；ΔCO2 中位 +0.225 vol% |
| nan | equivalent_HL_tot_pct::o2_x0.97 | 1.9503 | nan | None | 等效热损变化 = −ΔX_C(扰动)/单位热损 ΔX_C；正值表示该扰动相当于**多**移除这些 % 的热损 |
| nan | dX_C_pp::o2_x1.03 | 0.2670 | nan | None | ΔT_out 中位 +96.31 K；ΔCO2 中位 +0.611 vol% |
| nan | equivalent_HL_tot_pct::o2_x1.03 | -0.2012 | nan | None | 等效热损变化 = −ΔX_C(扰动)/单位热损 ΔX_C；正值表示该扰动相当于**多**移除这些 % 的热损 |

### K6 可辨识性

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 2.0000 | singular_value::SV1 | 62.8559 | nan | None | nan |
| 2.0000 | singular_value::SV2 | 1.8170 | nan | None | nan |
| 2.0000 | singular_value::SV3 | 0.5002 | nan | None | nan |
| 2.0000 | singular_value::SV4 | 0.0022 | nan | None | nan |
| 2.0000 | condition_number::cond(J) | 28493.6018 | nan | None | 最大/最小奇异值 |
| 2.0000 | weakest_direction::L_wsr | 0.9769 | nan | None | 与最小奇异值对应的参数组合（Vt 最后一行） |
| 2.0000 | weakest_direction::HL_tot | -0.0140 | nan | None | 与最小奇异值对应的参数组合（Vt 最后一行） |
| 2.0000 | weakest_direction::ln_f_gas | 0.2133 | nan | None | 与最小奇异值对应的参数组合（Vt 最后一行） |
| 2.0000 | weakest_direction::o2_ratio | 0.0001 | nan | None | 与最小奇异值对应的参数组合（Vt 最后一行） |

### K7 组成与氢账本

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 1.0000 | median_X_C_dev_pp | 6.6411 | nan | None | HL_tot=1% |
| 1.0000 | median_CO_vol_pct | 72.8881 | nan | None |  |
| 1.0000 | median_H2_vol_pct | 23.3726 | nan | None |  |
| 4.0000 | median_X_C_dev_pp | 6.3333 | nan | None | HL_tot=1% |
| 4.0000 | median_CO_vol_pct | 72.8201 | nan | None |  |
| 4.0000 | median_H2_vol_pct | 23.1769 | nan | None |  |
| nan | Mt_alt_median_dH2_vol_pct | 0.6476 | nan | None | Mt 4% 相对基准的 H2 变化（水汽带入氢） |

### K8 分配形式

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 0.2575 | median_abs_dX_C_pp | 0.0646 | nan | None | ΔT 中位 +2.62 K；ΔCO2 中位 +0.033 vol% |
| 1.0000 | median_abs_dX_C_pp | 0.1823 | nan | None | ΔT 中位 +6.46 K；ΔCO2 中位 +0.084 vol% |
| 2.0000 | median_abs_dX_C_pp | 0.2450 | nan | None | ΔT 中位 +8.03 K；ΔCO2 中位 +0.107 vol% |
| 4.0000 | median_abs_dX_C_pp | 0.2581 | nan | None | ΔT 中位 +7.76 K；ΔCO2 中位 +0.107 vol% |
| 6.0000 | median_abs_dX_C_pp | 0.2212 | nan | None | ΔT 中位 +6.17 K；ΔCO2 中位 +0.089 vol% |

## 2.5 关键读数

- **K1 解耦效果**：`L_wsr` 0.2575→6 m 的 ΔX_C 在旧热损式下为 P1.11-A 的值，在整炉总量式下为 K1 的值；**78.9%** 的杠杆可由「`L_wsr` 改变高温区长度→ 改变按 T⁴ 权重分配的热损」解释掉。这正是 I2 指出的构造性混杂。
- **K3 可信热损下能否到达**：**不存在** `HL_tot ≤ 1%` 且 `L_wsr ≤ 6 m` 使某窗落在 ±1 pp。
- **K4 所需热量移除**：16 / 45 个（窗, `L_wsr`）组合在 0.5–4% 区间内存在零点；其余落在扫描边界。
  - `HL_tot*` 中位 3.49%（范围 1.97–3.97%）
- **K5 输入等效**（扰动换算成等效 `HL_tot`；正 = 相当于多移除热）：
  - `Mt_4` → +0.06% HL_tot
  - `coal_x0.97` → -0.20% HL_tot
  - `coal_x1.03` → +1.89% HL_tot
  - `o2_x0.97` → +1.95% HL_tot
  - `o2_x1.03` → -0.20% HL_tot

- **K5 判定**：K4 需要的热损中位 **3.49%**；O2 −3% 与煤量 +3% **同向叠加**的等效量约 **3.84%**（单扰动线性外推，未做组合实测）。因此「K4 的过剩热量能否由 ≤3% 的 O2/煤量比偏差解释」的答案是 **能**，但要求两者同时处在各自的 3% 极限且方向一致——这是边界条件，不是典型工况。
  - 单看任一变量都不够：O2 −3% 约 1.95%，煤量 +3% 约 1.89%。
  - **这条只按单扰动线性外推**，本轮没有做组合扰动的实测；两个 3% 同时成立是边界情形，且方向必须与「少氧 / 多煤」一致。它的意义是：能量过剩**并非必然不可解释**，但需要 O2 与煤量两个独立的 3% 级偏差同向叠加——而这两个量里只有煤量做了协调，O2 流量计至今未核（见 P1.11-A 核查 I3 第 2 项）。把这条当作「已解释」是不对的，正确表述是「可被边界级输入偏差覆盖」。
- **K6 可辨识性**：cond(J) = 2.85e+04；最小奇异值 0.00221。
  - 最弱方向由 **L_wsr** 主导（载荷 +0.977），即在该工作点上它最难以独立辨识。

## 4. 失败与异常

- 主扫描非 completed 行：0 / 450
- `no_root`：0；`hloss_not_converged`：0
- 多火焰根行：0

## 5. 复现边界

- 只改热损模型与扫描参数，不改动力学常数、不采纳任何参数值、不改合同。
- K1–K8 阈值在运行前写入脚本常量，运行后未改。
- 整炉总量热损在 JAX 路径上未实现，遇非 `hhv_percent` 直接报错。
- `total_percent` 的收敛判据以 `max|Δw/w|` 为主：非自洽权重下 ΣQ 恒等于 Q_tot（分母就是同一轮权重和），所以 ΣQ 那一条几乎恒成立，保留它只为符合卡面文字核对。

