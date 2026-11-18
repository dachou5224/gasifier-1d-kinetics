# P2.2-A 现有模型在测量不确定度尺度上的重新基线

口径：κ = 1、钩子关闭、Wen 1979 动力学、逐窗协调干煤量、实测载气 C-m、热损 1%。本轮不反演、不加钩子、不改合同。

> **已触发卡面停止条件，总判定未给出。** B3 装置斜率核对不通过：phase2 用新窗口表重算的最大 |Δbeta_ln| = 4.907403e-01 > 1e-6（同一函数用旧窗口表重算时为 0.000e+00）

## 第 0 节与复现检查

- `PAIRING` · 煤质集合 ↔ 窗口表（第 0 节第 1 行）：通过（值 a24b5fe636752d8b7872ebacd569e40325244f90b851ae4ff1adaeee6e989119，阈值 a24b5fe636752d8b7872ebacd569e40325244f90b851ae4ff1adaeee6e989119）
- `COAL_HV` · 四阶段热值等于所选集合的值：通过（值 P1=28179;P2=28440;P3=28579;P4=28588，阈值 P1=28179;P2=28440;P3=28579;P4=28588）
- `EXPORT_HV_PROVENANCE` · config/model_case_contract*.yaml → coal_heating_value_dry_kj_kg.source_status：通过（值 model_case_contract.yaml=literature_derived;model_case_contract_huayi_p2.yaml=literature_derived;model_case_contract_huayi_p3.yaml=literature_derived;model_case_contract_huayi_p4.yaml=literature_derived，阈值 literature_derived）
- `EXPORT_HV_PROVENANCE` · （参考，不参与断言）yulin 合同：仅记录（值 model_case_contract_yulin_wgs2.yaml=assumed，阈值 nan）
- `U1` · 不适用：U1 须 legacy 煤质 + p1.14d 旧窗口表：仅记录（值 nan，阈值 1e-09）
- `U2` · C 闭合残差量级（9 窗最大绝对残差比，口径 A）：通过（值 5.220669135224923，阈值 10.0）
- `U2-REF` · C 逐窗比值（参考口径 B，不作判据）：不通过（值 132959.0574320918，阈值 10.0）
- `U2` · H 闭合残差量级（9 窗最大绝对残差比，口径 A）：通过（值 3.3427692618862888，阈值 10.0）
- `U2-REF` · H 逐窗比值（参考口径 B，不作判据）：不通过（值 42.13352718321627，阈值 10.0）
- `U2` · O 闭合残差量级（9 窗最大绝对残差比，口径 A）：通过（值 0.42438439718871146，阈值 10.0）
- `U2-REF` · O 逐窗比值（参考口径 B，不作判据）：通过（值 2.382997700351819，阈值 10.0）
- `PAIRING` · 煤质集合 ↔ 窗口表（第 0 节第 2 行）：通过（值 25f3a7b4d403adc83ab888c893bc23e555af2121b1d4fa04cb53c93d57e0f991，阈值 25f3a7b4d403adc83ab888c893bc23e555af2121b1d4fa04cb53c93d57e0f991）
- `COAL_HV` · 四阶段热值等于所选集合的值：通过（值 P1=27720;P2=28094;P3=27976;P4=28103，阈值 P1=27720;P2=28094;P3=27976;P4=28103）
- `EXPORT_HV_PROVENANCE` · config/model_case_contract*.yaml → coal_heating_value_dry_kj_kg.source_status：通过（值 model_case_contract.yaml=literature_derived;model_case_contract_huayi_p2.yaml=literature_derived;model_case_contract_huayi_p3.yaml=literature_derived;model_case_contract_huayi_p4.yaml=literature_derived，阈值 literature_derived）
- `EXPORT_HV_PROVENANCE` · （参考，不参与断言）yulin 合同：仅记录（值 model_case_contract_yulin_wgs2.yaml=assumed，阈值 nan）
- `U1` · 9 参照窗 C0 逐窗 X_C_pp vs P2.0-A perturb_snaps.json：通过（值 0.0，阈值 1e-09）
- `U1` · phase1 参照窗中位 vs P2.0-A comparison_points.csv C0 行：通过（值 0.0，阈值 1e-09）
- `U1` · phase2 参照窗中位 vs P2.0-A comparison_points.csv C0 行：通过（值 0.0，阈值 1e-09）
- `U1` · phase3 参照窗中位 vs P2.0-A comparison_points.csv C0 行：通过（值 0.0，阈值 1e-09）
- `U1` · phase4 参照窗中位 vs P2.0-A comparison_points.csv C0 行：通过（值 0.0，阈值 1e-09）
- `U2` · 不适用：U2 须 current 煤质 + P2.2-D 新窗口表：仅记录（值 nan，阈值 10.0）

## B1 碳转化率（142 窗全窗中位 vs FINAL 目标）

| 阶段 | 模型中位 % | FINAL 目标 % | 1σ pp | z | 判定 |
|---|---:|---:|---:|---:|---|
| P1 | 100.0000 | n/a | n/a | n/a | 通过 |
| P2 | 100.0000 | 94.5957 | 3.63909 | 1.4850690694650024 | 通过 |
| P3 | 100.0000 | 93.3362 | 3.5531 | 1.8754890095969596 | 不通过 |
| P4 | 99.1509 | 90.7684 | 3.50251 | 2.3932777146998445 | 不通过 |

## B2 出口温度（主口径参照窗中位）

| 阶段 | T_out °C | FT °C | T25 °C | 判定 |
|---|---:|---:|---:|---|
| P1 | 1279.81 | 1280.0 | 1346.4 | marginal |
| P2 | 1411.77 | 1290.0 | 1327.5 | pass |
| P3 | 1266.65 | 1270.0 | 1282.4 | marginal |
| P4 | 1142.52 | 1250.0 | 1277.0 | fail |

## B3 响应方向（8 个有符号组合）

| 阶段 | 观测量 | 输入 | 模型 per3 | 装置 per3 | 同号 |
|---|---|---|---:|---:|---|
| P1 | CO2_vol_pct | ln_coal | -1.0095 | -0.2935 | True |
| P2 | CO_vol_pct | ln_coal | 0.3209 | 0.6967 | True |
| P2 | CO2_vol_pct | ln_coal | -0.9695 | -0.7469 | True |
| P3 | CO_vol_pct | ln_coal | 0.4170 | 0.2814 | True |
| P3 | CO2_vol_pct | ln_coal | -0.9946 | -0.3094 | True |
| P4 | CO_vol_pct | ln_coal | -0.1782 | 0.5277 | False |
| P4 | H2_vol_pct | ln_O2 | -0.3554 | -0.4850 | True |
| P4 | CO2_vol_pct | ln_coal | -0.2624 | -0.6235 | True |

同号 7/8。phase2 装置斜率核对：新窗口表最大 |Δbeta_ln| = 4.907e-01（旧窗口表 0.000e+00）。

## B4 组成水平（ξ 闭合后 142 窗中位偏差，模型−装置）

| 阶段 | CO pp | H2 pp | CH4 pp |
|---|---:|---:|---:|
| P1 | 0.489 | -0.781 | 0.114 |
| P2 | 0.077 | -0.087 | 0.097 |
| P3 | -1.642 | 1.470 | 0.140 |
| P4 | -2.735 | 2.431 | 0.247 |

## B5 弹性（模型 vs P2.0-A 装置 95% 区间）

| 阶段 | e_煤 模型 | 装置区间 | e_氧 模型 | 装置区间 |
|---|---:|---|---:|---|
| P1 | 0.996 | [-0.419, 0.199] | 0.006 | [0.531, 1.313] |
| P2 | 1.000 | [0.323, 0.621] | 0.000 | [0.131, 1.132] |
| P3 | 1.000 | [0.187, 0.566] | 0.001 | [-0.102, 0.616] |
| P4 | 0.397 | [0.195, 0.730] | 0.618 | [1.160, 2.064] |

## B6 封顶

| 阶段 | X_C≥99.9% 基准 | 含扰动 | 阶段中位 X_C % | 达 99% 所需氧量倍数 |
|---|---:|---:|---:|---:|
| P1 | 0.919 | 1.000 | 100.000 | 1.000 |
| P2 | 1.000 | 1.000 | 100.000 | 1.000 |
| P3 | 0.977 | 1.000 | 100.000 | 1.000 |
| P4 | 0.118 | 0.882 | 99.151 | 1.000 |

## B7 热值影响与 U1→主口径拆分

| 阶段 | ΔT_out（刘文−化验表）K | ΔX_C pp | ΔCO pp | ΔH2 pp |
|---|---:|---:|---:|---:|
| P1 | 126.4 | 0.145 | 0.528 | -0.139 |
| P2 | 131.9 | 0.000 | 0.440 | -0.167 |
| P3 | 123.2 | 0.505 | 0.603 | -0.043 |
| P4 | 55.3 | 3.028 | 1.633 | -0.075 |

| 阶段 | ΔT_out 合计 | = 煤质互换（含热值） | + 干煤量常数 | 其中热值 | 其中组成余量 |
|---|---:|---:|---:|---:|---:|
| P1 | 126.4 | 126.4 | 0.0 | 126.4 | 0.0 |
| P2 | 162.4 | 191.6 | -29.1 | 130.9 | 60.7 |
| P3 | 103.0 | 74.0 | 29.0 | 103.7 | -29.6 |
| P4 | 55.3 | 55.3 | 0.0 | 55.3 | 0.0 |

说明：「煤质互换（含热值）」= B−A；「干煤量常数」= D−B，两者相加等于 D−A。「其中热值」= B−C（同一 current 组成下把刘文热值换成化验表值）；「其中组成余量」= (B−A)−(B−C)，含 legacy 化验表值 28094 与 current 化验表值 27976 之间约 10 K 的未单列残差，故真实组成项略大于此值。阶段 1、4 的煤质成分未变，A→B 与 B→C 是同一热值效应的两半。

## 判定

**HALTED（B3 核对未通过）**

求解失败记录见 `solve_failures.csv`，共 0 条。
