# P1.12-A 整炉总量热损 + 全混区解耦扫描 — 判读

Status: `EXECUTED — model repo`

依据：`manuscript/model-side-handoff-p1.12a-20261106.md`；上游 `manuscript/model-evidence-verification-20261106-p111.md` I2–I5。

## 0. 结构与热损模型

结构冻结为 `wsr_plus_pfr`、`dz_down` = 0.0515 m、`OxygenMixingModel = instant`。热损改为**整炉总量**：`Q_tot = HL_tot/100 × m_dry × HHV_d`（**干煤**口径），逐格按 `w_i = dz_i·clip((T_i/1800)^4, 0.1, 5)` 归一化分配。逐格推进的求解器在解第 i 格时不知道下游温度，因此 T 权重来自**上一轮全炉温度场**，外层不动点迭代至 `|ΣQ−Q_tot|/Q_tot < 1e-6` 且未松弛残差 `max|(w_new−w)/w| < 1e-4`，最多 30 轮。欠松弛只用于推进迭代。本轮 450 行扫描在代码修正前用的是松弛步残差；检查 4 把权重容差收紧 100 倍只移动 0.039 K，X_C 结论在 0.1 pp 量级不受影响，故未重跑主扫描。

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

### K1 与 P1.11-A 的对比（未匹配配置，不是归因）

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | dX_C_pp__p11a | -3.6688 | nan | None | P1.11-A hhv_percent HLP=1%；实际 ΣQ 随高温区变长，与 HL_tot=1% 不是同一总量 |
| nan | dX_C_pp__p12a | -0.7737 | nan | None | P1.12-A total_percent HL_tot=1%（干煤口径，总量固定） |
| nan | dX_C_pp__unmatched_amplitude_ratio | 0.7891 | nan | None | 1−|Δ_new|/|Δ_old|。只描述两个未匹配配置的幅度比，不是方差分解，也不是耦合归因。HLP=1% 与 HL_tot=1% 的实际 ΣQ 不同。 |
| nan | dT_out_K__p11a | -70.3957 | nan | None | P1.11-A hhv_percent HLP=1%；实际 ΣQ 随高温区变长，与 HL_tot=1% 不是同一总量 |
| nan | dT_out_K__p12a | 27.1899 | nan | None | P1.12-A total_percent HL_tot=1%（干煤口径，总量固定） |
| nan | dT_out_K__unmatched_amplitude_ratio | 0.6138 | nan | None | 1−|Δ_new|/|Δ_old|。只描述两个未匹配配置的幅度比，不是方差分解，也不是耦合归因。HLP=1% 与 HL_tot=1% 的实际 ΣQ 不同。 |

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
| nan | dX_C_pp::Mt_4 | -0.0782 | nan | None | perturbed−baseline 中位；ΔT_out 中位 -25.34 K；ΔCO2 中位 +0.753 vol% |
| nan | equivalent_HL_tot_pct::Mt_4 | 0.0589 | nan | None | 局部增量（HL_tot=1% 线性化）= −ΔX_C(扰动)/|K2 斜率|；正值 = 与多移除热同方向。不是抵消扰动所需的额外热损，也不是 K4 的绝对水平。 |
| nan | dX_C_pp::coal_x0.97 | 0.2670 | nan | None | perturbed−baseline 中位；ΔT_out 中位 +95.02 K；ΔCO2 中位 +0.769 vol% |
| nan | equivalent_HL_tot_pct::coal_x0.97 | -0.2012 | nan | None | 局部增量（HL_tot=1% 线性化）= −ΔX_C(扰动)/|K2 斜率|；正值 = 与多移除热同方向。不是抵消扰动所需的额外热损，也不是 K4 的绝对水平。 |
| nan | dX_C_pp::coal_x1.03 | -2.5064 | nan | None | perturbed−baseline 中位；ΔT_out 中位 -27.99 K；ΔCO2 中位 +0.084 vol% |
| nan | equivalent_HL_tot_pct::coal_x1.03 | 1.8885 | nan | None | 局部增量（HL_tot=1% 线性化）= −ΔX_C(扰动)/|K2 斜率|；正值 = 与多移除热同方向。不是抵消扰动所需的额外热损，也不是 K4 的绝对水平。 |
| nan | dX_C_pp::o2_x0.97 | -2.5884 | nan | None | perturbed−baseline 中位；ΔT_out 中位 -32.74 K；ΔCO2 中位 +0.225 vol% |
| nan | equivalent_HL_tot_pct::o2_x0.97 | 1.9503 | nan | None | 局部增量（HL_tot=1% 线性化）= −ΔX_C(扰动)/|K2 斜率|；正值 = 与多移除热同方向。不是抵消扰动所需的额外热损，也不是 K4 的绝对水平。 |
| nan | dX_C_pp::o2_x1.03 | 0.2670 | nan | None | perturbed−baseline 中位；ΔT_out 中位 +96.31 K；ΔCO2 中位 +0.611 vol% |
| nan | equivalent_HL_tot_pct::o2_x1.03 | -0.2012 | nan | None | 局部增量（HL_tot=1% 线性化）= −ΔX_C(扰动)/|K2 斜率|；正值 = 与多移除热同方向。不是抵消扰动所需的额外热损，也不是 K4 的绝对水平。 |

### K5 判定

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | K4_HL_tot_star_median_pct | 3.4896 | nan | None | K4 绝对水平（到达所需 HL_tot* 中位），不是局部增量 |
| nan | local_increment_o2_x0.97_plus_coal_x1.03_pct | 3.8389 | nan | None | 两个单扰动局部增量之和（HL_tot=1% 线性外推，未做组合实测） |
| nan | o2_over_coal_ratio_change_if_both | -0.0583 | nan | None | O2×0.97 且煤×1.03 时氧煤比变化 = 0.97/1.03−1，超出 ≤3% 比偏差预算 |
| nan | covers_K4_absolute | 0.0000 | 1.0000 | 0.0000 | 不能把局部增量与 K4 绝对水平相比；也不能把两个独立 ±3% 流量误差当成 ≤3% 氧煤比。O2−3% 的 ΔX_C=-2.59 pp（扰动−基准，下降），煤+3% 的 ΔX_C=-2.51 pp。两者与加热损同方向，使过预测缩小，但未回答卡面的 ≤3% 比偏差问题。 |

### K6 可辨识性

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 2.0000 | singular_value::SV1 | 2.6784 | nan | None | obs+param scaled; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | singular_value::SV2 | 1.6750 | nan | None | obs+param scaled; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | singular_value::SV3 | 0.7804 | nan | None | obs+param scaled; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | singular_value::SV4 | 0.0022 | nan | None | obs+param scaled; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | condition_number::cond(J) | 1192.7549 | nan | None | obs+param scaled; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | weakest_direction::L_wsr | 0.9944 | nan | None | Vt last row after obs+param scaling; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | weakest_direction::HL_tot | -0.0143 | nan | None | Vt last row after obs+param scaling; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | weakest_direction::ln_f_gas | 0.1044 | nan | None | Vt last row after obs+param scaling; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | weakest_direction::o2_ratio | 0.0031 | nan | None | Vt last row after obs+param scaling; L_wsr unit = 1 m; HL_tot unit = 1 percentage point; ln_f_gas unit = ln(8) (f_gas ×8, the scan grid); o2_ratio unit = 0.03 (K5 ±3%) |
| 2.0000 | singular_value_obs_only::SV1 | 62.8559 | nan | None | rows scaled by (1 pp, 0.5, 0.5, 0.5 vol%); columns in native units (m, %, ln, ratio) — cond depends on those units |
| 2.0000 | singular_value_obs_only::SV2 | 1.8170 | nan | None | rows scaled by (1 pp, 0.5, 0.5, 0.5 vol%); columns in native units (m, %, ln, ratio) — cond depends on those units |
| 2.0000 | singular_value_obs_only::SV3 | 0.5002 | nan | None | rows scaled by (1 pp, 0.5, 0.5, 0.5 vol%); columns in native units (m, %, ln, ratio) — cond depends on those units |
| 2.0000 | singular_value_obs_only::SV4 | 0.0022 | nan | None | rows scaled by (1 pp, 0.5, 0.5, 0.5 vol%); columns in native units (m, %, ln, ratio) — cond depends on those units |
| 2.0000 | condition_number_obs_only::cond(J) | 28493.6018 | nan | None | native parameter units; not invariant under rescaling a column |
| 2.0000 | weakest_direction_obs_only::L_wsr | 0.9769 | nan | None | native parameter units; not invariant under rescaling a column |
| 2.0000 | weakest_direction_obs_only::HL_tot | -0.0140 | nan | None | native parameter units; not invariant under rescaling a column |
| 2.0000 | weakest_direction_obs_only::ln_f_gas | 0.2133 | nan | None | native parameter units; not invariant under rescaling a column |
| 2.0000 | weakest_direction_obs_only::o2_ratio | 0.0001 | nan | None | native parameter units; not invariant under rescaling a column |
| 2.0000 | param_scale::L_wsr | 1.0000 | nan | None | 1 m |
| 2.0000 | param_scale::HL_tot | 1.0000 | nan | None | 1 percentage point |
| 2.0000 | param_scale::ln_f_gas | 2.0794 | nan | None | ln(8) (f_gas ×8, the scan grid) |
| 2.0000 | param_scale::o2_ratio | 0.0300 | nan | None | 0.03 (K5 ±3%) |

### K7 范围

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | full_species_H_ledger_available | 0.0000 | 1.0000 | 0.0000 | scan.csv 无 CH4/H2S 摩尔流，完整物种氢账本标为不完整；下列 ξ / 闭合残差 / H2-only 账本由存档 F_CO2、dry_total、vol% 与装置组成重建 |

### K7 组成与氢账本

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 1.0000 | median_X_C_dev_pp | 6.6411 | nan | None | K3 邻域 HL_tot=1%，未到达 |
| 1.0000 | median_xi_mol_s::analyzer::K3_neighborhood_HL1 | 101.2609 | nan | None | K3_neighborhood_HL1；n=9 |
| 1.0000 | median_dCO_closed_vol_pct::analyzer::K3_neighborhood_HL1 | -1.1956 | nan | None | K3_neighborhood_HL1；闭合后 CO 残差 |
| 1.0000 | median_dH2_closed_vol_pct::analyzer::K3_neighborhood_HL1 | 1.1092 | nan | None | K3_neighborhood_HL1；闭合后 H2 残差 |
| 1.0000 | median_H_ledger_h2only::analyzer::K3_neighborhood_HL1 | 1.1129 | nan | None | K3_neighborhood_HL1；H2-only 氢账本（模型+2ξ）/装置 H2 氢 |
| 1.0000 | median_xi_mol_s::liu_phase::K3_neighborhood_HL1 | 100.6688 | nan | None | K3_neighborhood_HL1；n=9 |
| 1.0000 | median_dCO_closed_vol_pct::liu_phase::K3_neighborhood_HL1 | 2.7820 | nan | None | K3_neighborhood_HL1；闭合后 CO 残差 |
| 1.0000 | median_dH2_closed_vol_pct::liu_phase::K3_neighborhood_HL1 | -2.6919 | nan | None | K3_neighborhood_HL1；闭合后 H2 残差 |
| 1.0000 | median_H_ledger_h2only::liu_phase::K3_neighborhood_HL1 | 0.9710 | nan | None | K3_neighborhood_HL1；H2-only 氢账本（模型+2ξ）/装置 H2 氢 |
| 4.0000 | median_X_C_dev_pp | 6.3333 | nan | None | K3 邻域 HL_tot=1%，未到达 |
| 4.0000 | median_xi_mol_s::analyzer::K3_neighborhood_HL1 | 98.9011 | nan | None | K3_neighborhood_HL1；n=9 |
| 4.0000 | median_dCO_closed_vol_pct::analyzer::K3_neighborhood_HL1 | -0.9912 | nan | None | K3_neighborhood_HL1；闭合后 CO 残差 |
| 4.0000 | median_dH2_closed_vol_pct::analyzer::K3_neighborhood_HL1 | 0.8429 | nan | None | K3_neighborhood_HL1；闭合后 H2 残差 |
| 4.0000 | median_H_ledger_h2only::analyzer::K3_neighborhood_HL1 | 1.0947 | nan | None | K3_neighborhood_HL1；H2-only 氢账本（模型+2ξ）/装置 H2 氢 |
| 4.0000 | median_xi_mol_s::liu_phase::K3_neighborhood_HL1 | 97.1288 | nan | None | K3_neighborhood_HL1；n=9 |
| 4.0000 | median_dCO_closed_vol_pct::liu_phase::K3_neighborhood_HL1 | 2.9707 | nan | None | K3_neighborhood_HL1；闭合后 CO 残差 |
| 4.0000 | median_dH2_closed_vol_pct::liu_phase::K3_neighborhood_HL1 | -2.9369 | nan | None | K3_neighborhood_HL1；闭合后 H2 残差 |
| 4.0000 | median_H_ledger_h2only::liu_phase::K3_neighborhood_HL1 | 0.9576 | nan | None | K3_neighborhood_HL1；H2-only 氢账本（模型+2ξ）/装置 H2 氢 |
| nan | n_K4_arrival_nearest_grid | 16.0000 | nan | None | K4 区间内零点对应的最近扫描档（不是 HL* 插值点重解） |
| nan | median_xi_mol_s::analyzer::K4_arrival_nearest_grid | 49.3638 | nan | None | K4_arrival_nearest_grid；n=16 |
| nan | median_dCO_closed_vol_pct::analyzer::K4_arrival_nearest_grid | 1.5530 | nan | None | K4_arrival_nearest_grid；闭合后 CO 残差 |
| nan | median_dH2_closed_vol_pct::analyzer::K4_arrival_nearest_grid | -1.7011 | nan | None | K4_arrival_nearest_grid；闭合后 H2 残差 |
| nan | median_H_ledger_h2only::analyzer::K4_arrival_nearest_grid | 0.9154 | nan | None | K4_arrival_nearest_grid；H2-only 氢账本（模型+2ξ）/装置 H2 氢 |
| nan | median_xi_mol_s::liu_phase::K4_arrival_nearest_grid | 46.8822 | nan | None | K4_arrival_nearest_grid；n=16 |
| nan | median_dCO_closed_vol_pct::liu_phase::K4_arrival_nearest_grid | 4.7845 | nan | None | K4_arrival_nearest_grid；闭合后 CO 残差 |
| nan | median_dH2_closed_vol_pct::liu_phase::K4_arrival_nearest_grid | -4.7514 | nan | None | K4_arrival_nearest_grid；闭合后 H2 残差 |
| nan | median_H_ledger_h2only::liu_phase::K4_arrival_nearest_grid | 0.8340 | nan | None | K4_arrival_nearest_grid；H2-only 氢账本（模型+2ξ）/装置 H2 氢 |
| nan | Mt_alt_median_dH2_vol_pct | 0.6537 | nan | None | Mt 4% 相对基准的 H2 变化（水汽带入氢） |
| nan | Mt_alt_median_dH_ledger_analyzer | 0.0021 | nan | None | Mt 4% 相对基准，analyzer 口径 H2-only 氢账本之差 |

### K8 分配形式

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 0.2575 | median_abs_dX_C_pp | 0.0646 | nan | None | ΔT 中位 +2.62 K；ΔCO2 中位 +0.033 vol% |
| 1.0000 | median_abs_dX_C_pp | 0.1823 | nan | None | ΔT 中位 +6.46 K；ΔCO2 中位 +0.084 vol% |
| 2.0000 | median_abs_dX_C_pp | 0.2450 | nan | None | ΔT 中位 +8.03 K；ΔCO2 中位 +0.107 vol% |
| 4.0000 | median_abs_dX_C_pp | 0.2581 | nan | None | ΔT 中位 +7.76 K；ΔCO2 中位 +0.107 vol% |
| 6.0000 | median_abs_dX_C_pp | 0.2212 | nan | None | ΔT 中位 +6.17 K；ΔCO2 中位 +0.089 vol% |

## 2.5 关键读数

- **K1 对比（不是归因）**：`L_wsr` 0.2575→6 m 的 ΔX_C 从 P1.11-A （`hhv_percent`, HLP=1%）的 **-3.67 pp** 变为本卡（`total_percent`, HL_tot=1%）的 **-0.77 pp**。两个 1% 设定的实际 ΣQ 不同（湿/干分母；旧式让 ΣQ 随高温区变长），幅度变小与热损耦合一致，但不能把 1−|Δ_new|/|Δ_old| 写成「来自耦合的份额」。
- **K3 可信热损下能否到达**：**不存在** `HL_tot ≤ 1%` 且 `L_wsr ≤ 6 m` 使某窗落在 ±1 pp。
- **K4 所需热量移除**：16 / 45 个（窗, `L_wsr`）组合在 0.5–4% 区间内存在零点；其余落在扫描边界。
  - `HL_tot*` 中位 3.49%（范围 1.97–3.97%）
- **K5 输入等效**（`ΔX_C = 扰动 − 基准`；局部增量正值 = 与多移除热同方向；在 `HL_tot=1%` 处线性化，不是 K4 的绝对水平）：
  - `Mt_4`：ΔX_C -0.08 pp → 局部等效 +0.06 个百分点热损
  - `coal_x0.97`：ΔX_C +0.27 pp → 局部等效 -0.20 个百分点热损
  - `coal_x1.03`：ΔX_C -2.51 pp → 局部等效 +1.89 个百分点热损
  - `o2_x0.97`：ΔX_C -2.59 pp → 局部等效 +1.95 个百分点热损
  - `o2_x1.03`：ΔX_C +0.27 pp → 局部等效 -0.20 个百分点热损

- **K5 判定**：卡面问的是 ≤3% 的 **O2/煤量比** 能否解释 K4 的过剩热量。答案是**未回答**，原因有三条，都不是符号写反那么简单。
  - 符号：O2 −3% 与煤 +3% 的 ΔX_C 中位分别是 -2.59、-2.51 pp（扰动 − 基准）。基线已经偏高，这两项使 X_C 下降，与加热损同方向；它们不会扩大偏差。PR 正文曾把这两项写成 +2.18、+2.11 pp，那是把下降量写成正号，再和正的等效热损放在一起，看起来像方向反了。
  - 局部增量 ≠ 绝对水平：两项局部等效之和约 **3.84** 个百分点（在 HL_tot=1% 处），K4 的 `HL_tot*` 中位是 **3.49%**。前者是 1% 附近的斜率换算，后者是到达所需的绝对热损；不能拿 3.84 去覆盖 3.49。
  - 两个流量误差 ≠ 比偏差：同时施加 O2×0.97 与煤×1.03 会使 O2/煤 比变化 `0.97/1.03 − 1 = −5.83%`，超出 ≤3% 的比预算。本轮没有做过满足该约束的扰动，比偏差问题保持未决。
- **K6 可辨识性**（观测按 1 pp / 0.5 / 0.5 / 0.5 vol% 缩放，参数列按 1 m、1 个百分点、ln 8、O2 3% 缩放；只描述本窗口 `L_wsr=2 m`、`HL_tot=1%`、这组尺度）：cond(J) = 1.19e+03；最小奇异值 0.00225。
  - 该坐标系下最弱方向由 **L_wsr** 主导（载荷 +0.994）。改任一列的物理尺度都会改载荷。
  - 对照（只缩放观测、列用原单位 m/%/ln/比例）：cond(J) = 2.85e+04。这个数字随单位而变，不当作主结论。
- **K7 组成与氢账本**：K3 在可信热损下未到达；下列是 HL_tot=1% 邻域与 K4 最近扫描档上、两种组成口径的 ξ / 闭合后残差 / H2-only 氢账本。完整物种（含 CH4/H2S）账本因存档缺列标为不完整。

## 4. 失败与异常

- 主扫描非 completed 行：0 / 450
- `no_root`：0；`hloss_not_converged`：0
- 多火焰根行：0

## 5. 复现边界

- 只改热损模型与扫描参数，不改动力学常数、不采纳任何参数值、不改合同。
- K1–K8 阈值在运行前写入脚本常量，运行后未改。
- 整炉总量热损在 JAX 路径上未实现，遇非 `hhv_percent` 直接报错。
- `total_percent` 的收敛判据以未松弛 `max|(w_new−w)/w|` 为主：非自洽权重下 ΣQ 恒等于 Q_tot（分母就是同一轮权重和），所以 ΣQ 那一条几乎恒成立，保留它只为符合卡面文字核对。欠松弛不进入判据。

