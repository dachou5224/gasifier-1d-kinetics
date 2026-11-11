# P1.14-A 可辨识性设计判读

结构冻结 `wsr_plus_pfr`，`L_wsr = 1 m`，`dz_down = 0.0515 m`，`total_percent` `HL_tot = 1%`，`OxygenMixingModel = instant`。阶段 θ 取自 P1.13-A 分析仪口径（`L_wsr=1 m`，`HL_tot=1%`）；θ0 = (κ=1, ξ=0, f_gas=1)。ξ 只作用于出口组成，不进求解器。水冷壁诊断量 `Q_wall_conv` / `Q_wall_rad` 不进求解；对流与辐射分列（`S3_conv`/`S3_rad`、`S4_conv`/`S4_rad`），各自带 ln U，禁止把两种壁面代理堆进同一个 Fisher。对数残差对 ln U 的未缩放列为 1，缩放后为 1/0.10。本轮**不采纳**任何参数值，不做反演。

观测缩放（预声明，跑完未改）：X_C 1.5 pp；CO/H2/CO2 0.5 vol%；CH4 0.05 vol%；干气/干煤 2%（相对）；Q_wall 10%（对数）；T_out 20 K 为**理想设计假设**，不是装置出口温测量不确定度（用于 `S4_conv` / `S4_rad` / `S4_no_wall`）。

## 覆盖
- 142 窗基线：θ 成功 142/142，雅可比有限 142/142，失败 0。
- θ 下 X_C 模型−装置 (pp)：中位 -0.0127，范围 -4.25 … 3.51（n=142）
- θ 下 CO (vol%)：中位 -0.297，范围 -2.37 … 1.45（n=142）
- θ 下 H2 (vol%)：中位 -0.246，范围 -1.04 … 0.574（n=142）
- θ 下 CO2 (vol%)：中位 0.0318，范围 -1.49 … 2.18（n=142）
- θ0 下 X_C 模型−装置 (pp)：中位 6.91，范围 -3.8 … 10（n=142）

冻结切分：P1 全部 train（37），P2 全部 train（44），P3 全部 validation（44），P4 全部 test（17）。主分析 `window_set=train` **仅 P1/P2**。P3/P4 的 `phase_all` 标 exploratory：超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证。`S4_no_wall` 为探索变体，不是卡面 S4。

## Q1 局部设计门槛（不是统计可辨识）
逆 Fisher 是固定阶段 θ、共享 κ/ln f_gas/ξ、对角预声明权重、窗间独立假设下的曲率尺度。装置 X_C 与 CO/CO2/CH4 及校准干气流量共享原始数据，阶段内还共享标定系数。通过的是局部设计门槛，不是统计可辨识。
判据：|ρ(κ, ln f_gas)| < 0.9 **且** σ(ln f_gas) < ln 2。主结论只用 P1/P2 train。
- phase1 S0 [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.2314 (thr 0.9); σ(ln f_gas)=0.01837 (thr ln2=0.6931); n=37; 通过=局部设计门槛，不是统计可辨识
- phase1 S1 [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.3749 (thr 0.9); σ(ln f_gas)=0.01591 (thr ln2=0.6931); n=37; 通过=局部设计门槛，不是统计可辨识
- phase1 S2 [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.2496 (thr 0.9); σ(ln f_gas)=0.01721 (thr ln2=0.6931); n=37; 通过=局部设计门槛，不是统计可辨识
- phase1 S3_conv [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.2307 (thr 0.9); σ(ln f_gas)=0.01831 (thr ln2=0.6931); n=37; 通过=局部设计门槛，不是统计可辨识
- phase1 S3_rad [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.2299 (thr 0.9); σ(ln f_gas)=0.01826 (thr ln2=0.6931); n=37; 通过=局部设计门槛，不是统计可辨识
- phase1 S4_conv [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.3053 (thr 0.9); σ(ln f_gas)=0.01396 (thr ln2=0.6931); n=37; 通过=局部设计门槛，不是统计可辨识
- phase1 S4_rad [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.3048 (thr 0.9); σ(ln f_gas)=0.01394 (thr ln2=0.6931); n=37; 通过=局部设计门槛，不是统计可辨识
- phase1 S4_no_wall [train]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.3059 (thr 0.9); σ(ln f_gas)=0.01399 (thr ln2=0.6931); n=37; 探索变体，不是卡面 S4；无壁面、无 ln U; 通过=局部设计门槛，不是统计可辨识
- phase2 S0 [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.07213 (thr 0.9); σ(ln f_gas)=0.01171 (thr ln2=0.6931); n=44; 通过=局部设计门槛，不是统计可辨识
- phase2 S1 [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.08754 (thr 0.9); σ(ln f_gas)=0.01082 (thr ln2=0.6931); n=44; 通过=局部设计门槛，不是统计可辨识
- phase2 S2 [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.1046 (thr 0.9); σ(ln f_gas)=0.01093 (thr ln2=0.6931); n=44; 通过=局部设计门槛，不是统计可辨识
- phase2 S3_conv [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.07169 (thr 0.9); σ(ln f_gas)=0.01169 (thr ln2=0.6931); n=44; 通过=局部设计门槛，不是统计可辨识
- phase2 S3_rad [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.07133 (thr 0.9); σ(ln f_gas)=0.01168 (thr ln2=0.6931); n=44; 通过=局部设计门槛，不是统计可辨识
- phase2 S4_conv [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.001986 (thr 0.9); σ(ln f_gas)=0.009757 (thr ln2=0.6931); n=44; 通过=局部设计门槛，不是统计可辨识
- phase2 S4_rad [train]：**通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.001834 (thr 0.9); σ(ln f_gas)=0.009748 (thr ln2=0.6931); n=44; 通过=局部设计门槛，不是统计可辨识
- phase2 S4_no_wall [train]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.00217 (thr 0.9); σ(ln f_gas)=0.009768 (thr ln2=0.6931); n=44; 探索变体，不是卡面 S4；无壁面、无 ln U; 通过=局部设计门槛，不是统计可辨识
- phase3 S0 [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase3 S1 [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase3 S2 [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase3 S3_conv [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase3 S3_rad [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase3 S4_conv [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase3 S4_rad [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase3 S4_no_wall [train]：**探索 / 无训练窗**。无训练窗，不作 train 集局部设计结论
- phase3 S0 [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.1803 (thr 0.9); σ(ln f_gas)=0.05435 (thr ln2=0.6931); n=44; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase3 S1 [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.7557 (thr 0.9); σ(ln f_gas)=0.02176 (thr ln2=0.6931); n=44; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase3 S2 [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.2998 (thr 0.9); σ(ln f_gas)=0.05184 (thr ln2=0.6931); n=44; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase3 S3_conv [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.1802 (thr 0.9); σ(ln f_gas)=0.05427 (thr ln2=0.6931); n=44; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase3 S3_rad [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.1799 (thr 0.9); σ(ln f_gas)=0.05418 (thr ln2=0.6931); n=44; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase3 S4_conv [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.6102 (thr 0.9); σ(ln f_gas)=0.01762 (thr ln2=0.6931); n=44; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase3 S4_rad [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.6101 (thr 0.9); σ(ln f_gas)=0.01761 (thr ln2=0.6931); n=44; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase3 S4_no_wall [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.6102 (thr 0.9); σ(ln f_gas)=0.01762 (thr ln2=0.6931); n=44; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 探索变体，不是卡面 S4；无壁面、无 ln U; 通过=局部设计门槛，不是统计可辨识
- phase4 S0 [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase4 S1 [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase4 S2 [train]：**无训练窗**。无训练窗，不作 train 集局部设计结论
- phase4 S4_no_wall [train]：**探索 / 无训练窗**。无训练窗，不作 train 集局部设计结论
- phase4 S0 [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.05785 (thr 0.9); σ(ln f_gas)=0.05691 (thr ln2=0.6931); n=17; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase4 S1 [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.4637 (thr 0.9); σ(ln f_gas)=0.03322 (thr ln2=0.6931); n=17; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase4 S2 [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.09743 (thr 0.9); σ(ln f_gas)=0.05355 (thr ln2=0.6931); n=17; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 通过=局部设计门槛，不是统计可辨识
- phase4 S4_no_wall [phase_all]：**探索 / 通过局部设计门槛**。|ρ(κ,ln f_gas)|=0.3099 (thr 0.9); σ(ln f_gas)=0.02935 (thr ln2=0.6931); n=17; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证; 探索变体，不是卡面 S4；无壁面、无 ln U; 通过=局部设计门槛，不是统计可辨识

## Q2 信息来源（相对 S0 的 σ(ln f_gas) 缩小；S3_conv / S3_rad 分开）
- phase1 `shrink_S1_over_S0` = 1.1545076760617103。σ_S0=0.01837; σ_S1=0.01591
- phase1 `shrink_S2_over_S0` = 1.0675125208677918。σ_S0=0.01837; σ_S2=0.01721
- phase1 `shrink_S3_conv_over_S0` = 1.003260648067019。σ_S0=0.01837; σ_S3_conv=0.01831
- phase1 `shrink_S3_rad_over_S0` = 1.005942139442612。σ_S0=0.01837; σ_S3_rad=0.01826
- phase1 `largest_contributor` = 1.1545076760617103。S1
- phase2 `shrink_S1_over_S0` = 1.0826930267922508。σ_S0=0.01171; σ_S1=0.01082
- phase2 `shrink_S2_over_S0` = 1.0714952248635183。σ_S0=0.01171; σ_S2=0.01093
- phase2 `shrink_S3_conv_over_S0` = 1.0016553647279973。σ_S0=0.01171; σ_S3_conv=0.01169
- phase2 `shrink_S3_rad_over_S0` = 1.0029717161471594。σ_S0=0.01171; σ_S3_rad=0.01168
- phase2 `largest_contributor` = 1.0826930267922508。S1
- 探索 / phase3 `shrink_S1_over_S0` = 2.4975542011655634。σ_S0=0.05435; σ_S1=0.02176; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `shrink_S2_over_S0` = 1.048428528760997。σ_S0=0.05435; σ_S2=0.05184; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `shrink_S3_conv_over_S0` = 1.0013919613207847。σ_S0=0.05435; σ_S3_conv=0.05427; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `shrink_S3_rad_over_S0` = 1.0030797568224272。σ_S0=0.05435; σ_S3_rad=0.05418; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `largest_contributor` = 2.4975542011655634。S1; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `shrink_S1_over_S0` = 1.7128620848922982。σ_S0=0.05691; σ_S1=0.03322; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `shrink_S2_over_S0` = 1.0627184197845947。σ_S0=0.05691; σ_S2=0.05355; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `largest_contributor` = 1.7128620848922982。S1; 超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证

## Q3 扩窗（S0）：堆叠总信息、每窗归一 σ√n、ρ 变化
扩窗同时改变样本数 n 与工况。堆叠总尺度下降不能归因于 O/C 多样性；相同阶段 θ 下参考窗 S0 已通过 Q1 局部门槛。下列分列 stacked / per-window / ρ，不跑等 n 对照反应器。
- phase1 `stacked_sigma_ref` [p113a_ref] = 0.061210706700953424。堆叠总信息; n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase1 `stacked_sigma_exp` [train] = 0.01837226012985676。堆叠总信息; n=37; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase1 `stacked_ratio_ref_over_exp` [train] = 3.331691706317608。σ_ref/σ_exp 堆叠总信息; σ_ref=0.06121 → σ_exp=0.01837; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase1 `per_window_sigma_sqrt_n_ref` [p113a_ref] = 0.08656501157893003。σ√n; n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase1 `per_window_sigma_sqrt_n_exp` [train] = 0.11175409551478459。σ√n; n=37; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase1 `per_window_ratio_ref_over_exp` [train] = 0.7746025877635764。σ√n_ref/σ√n_exp; 0.08657 → 0.1118; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase1 `rho_kappa_lnfg_ref` [p113a_ref] = 0.37045840792271983。n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase1 `rho_kappa_lnfg_exp` [train] = -0.23142362754880427。n=37; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase1 `rho_kappa_lnfg_delta_exp_minus_ref` [train] = -0.6018820354715241。ρ_exp−ρ_ref; ρ_ref=0.3705 → ρ_exp=-0.2314; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `stacked_sigma_ref` [p113a_ref] = 0.048202041038247476。堆叠总信息; n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `stacked_sigma_exp` [train] = 0.011711766735151995。堆叠总信息; n=44; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `stacked_ratio_ref_over_exp` [train] = 4.115693398637513。σ_ref/σ_exp 堆叠总信息; σ_ref=0.0482 → σ_exp=0.01171; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `per_window_sigma_sqrt_n_ref` [p113a_ref] = 0.0681679801703541。σ√n; n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `per_window_sigma_sqrt_n_exp` [train] = 0.07768707178532966。σ√n; n=44; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `per_window_ratio_ref_over_exp` [train] = 0.8774687808895747。σ√n_ref/σ√n_exp; 0.06817 → 0.07769; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `rho_kappa_lnfg_ref` [p113a_ref] = 0.3565201900381535。n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `rho_kappa_lnfg_exp` [train] = 0.07213332814311449。n=44; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- phase2 `rho_kappa_lnfg_delta_exp_minus_ref` [train] = -0.28438686189503903。ρ_exp−ρ_ref; ρ_ref=0.3565 → ρ_exp=0.07213; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛
- 探索 / phase3 `stacked_sigma_ref` [p113a_ref] = 0.34552070187275696。堆叠总信息; n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `stacked_sigma_exp` [phase_all] = 0.054346981602779086。堆叠总信息; n=44; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `stacked_ratio_ref_over_exp` [phase_all] = 6.357679703321158。σ_ref/σ_exp 堆叠总信息; σ_ref=0.3455 → σ_exp=0.05435; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `per_window_sigma_sqrt_n_ref` [p113a_ref] = 0.4886400626691238。σ√n; n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `per_window_sigma_sqrt_n_exp` [phase_all] = 0.3604970929295319。σ√n; n=44; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `per_window_ratio_ref_over_exp` [phase_all] = 1.3554618671075958。σ√n_ref/σ√n_exp; 0.4886 → 0.3605; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `rho_kappa_lnfg_ref` [p113a_ref] = -0.6269170947629623。n=2; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `rho_kappa_lnfg_exp` [phase_all] = -0.18031735717962374。n=44; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase3 `rho_kappa_lnfg_delta_exp_minus_ref` [phase_all] = 0.4465997375833386。ρ_exp−ρ_ref; ρ_ref=-0.6269 → ρ_exp=-0.1803; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `stacked_sigma_ref` [p113a_ref] = 0.17092553511544747。堆叠总信息; n=3; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `stacked_sigma_exp` [phase_all] = 0.056906660709411624。堆叠总信息; n=17; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `stacked_ratio_ref_over_exp` [phase_all] = 3.003612107697941。σ_ref/σ_exp 堆叠总信息; σ_ref=0.1709 → σ_exp=0.05691; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `per_window_sigma_sqrt_n_ref` [p113a_ref] = 0.29605171113085327。σ√n; n=3; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `per_window_sigma_sqrt_n_exp` [phase_all] = 0.23463217290609056。σ√n; n=17; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `per_window_ratio_ref_over_exp` [phase_all] = 1.261769464366446。σ√n_ref/σ√n_exp; 0.2961 → 0.2346; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `rho_kappa_lnfg_ref` [p113a_ref] = 0.20387360171001576。n=3; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `rho_kappa_lnfg_exp` [phase_all] = -0.05785248543603289。n=17; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证
- 探索 / phase4 `rho_kappa_lnfg_delta_exp_minus_ref` [phase_all] = -0.26172608714604867。ρ_exp−ρ_ref; ρ_ref=0.2039 → ρ_exp=-0.05785; 扩窗同时改变 n 与工况；总尺度下降不能归因于 O/C；参考窗 S0 已过 Q1 局部门槛；超出训练窗主分析；阶段 θ 已用同阶段观测，不是独立验证

## Q4 符号核对 ∂T_out/∂κ 与 ∂T_out/∂ln f_gas
- **同号窗 22.5% > 20%，单列**：第 0 节的物理差别在这些窗上不普遍成立。
- `n_same_sign_dT`  = 32.0. 32/142 = 0.225; >20% 则物理差别不普遍，单列
- `frac_same_sign_dT`  = 0.22535211267605634. 
- `n_same_sign_dT` phase1 = 19.0. 19/37
- `n_same_sign_dT` phase2 = 9.0. 9/44
- `n_same_sign_dT` phase3 = 4.0. 4/44
- `n_same_sign_dT` phase4 = 0.0. 0/17
  - `09-28N_009__W004` phase1 ∂T/∂κ=651.7; ∂T/∂ln f_gas=6.982
  - `09-28N_009__W005` phase1 ∂T/∂κ=662.8; ∂T/∂ln f_gas=56.02
  - `09-28N_009__W003` phase1 ∂T/∂κ=658.4; ∂T/∂ln f_gas=3.389
  - `09-28N_009__W007` phase1 ∂T/∂κ=752.8; ∂T/∂ln f_gas=131.2
  - `09-28N_009__W001` phase1 ∂T/∂κ=831.5; ∂T/∂ln f_gas=67.08
  - `09-28N_009__W002` phase1 ∂T/∂κ=591.9; ∂T/∂ln f_gas=109.4
  - `09-29N_013__W002` phase1 ∂T/∂κ=811; ∂T/∂ln f_gas=10.89
  - `09-29N_014__W001` phase1 ∂T/∂κ=760.7; ∂T/∂ln f_gas=1.369
  - `09-29N_013__W001` phase1 ∂T/∂κ=1340; ∂T/∂ln f_gas=5.471
  - `09-30D_015__W002` phase1 ∂T/∂κ=727.5; ∂T/∂ln f_gas=28.01
  - `09-29N_012__W002` phase1 ∂T/∂κ=975.5; ∂T/∂ln f_gas=61.11
  - `2022-09-29__W001` phase1 ∂T/∂κ=736.6; ∂T/∂ln f_gas=35.6
  - `09-30D_017__W002` phase1 ∂T/∂κ=580.4; ∂T/∂ln f_gas=113.3
  - `09-29D_010__W001` phase1 ∂T/∂κ=1960; ∂T/∂ln f_gas=5.646
  - `2022-09-30__W001` phase1 ∂T/∂κ=603.3; ∂T/∂ln f_gas=15.41
  - `09-30N_018__W001` phase1 ∂T/∂κ=720.3; ∂T/∂ln f_gas=24.7
  - `09-30N_018__W003` phase1 ∂T/∂κ=1797; ∂T/∂ln f_gas=88.98
  - `09-30N_018__W006` phase1 ∂T/∂κ=1816; ∂T/∂ln f_gas=35.99
  - `10-02D_007__W001` phase2 ∂T/∂κ=626; ∂T/∂ln f_gas=0.0737
  - `10-02D_007__W002` phase2 ∂T/∂κ=657.3; ∂T/∂ln f_gas=30.33
  - `09-30N_018__W002` phase1 ∂T/∂κ=1591; ∂T/∂ln f_gas=44.07
  - `2022-10-02__W001` phase2 ∂T/∂κ=717.9; ∂T/∂ln f_gas=24.99
  - `10-02N_009__W005` phase2 ∂T/∂κ=601.9; ∂T/∂ln f_gas=4.588
  - `2022-10-03__W001` phase2 ∂T/∂κ=638.8; ∂T/∂ln f_gas=19.47
  - `10-03N_013__W004` phase2 ∂T/∂κ=584.6; ∂T/∂ln f_gas=12.65
  - `10-03N_013__W006` phase2 ∂T/∂κ=597.3; ∂T/∂ln f_gas=205
  - `10-04D_014__W002` phase2 ∂T/∂κ=579.2; ∂T/∂ln f_gas=74.2
  - `10-04D_014__W005` phase2 ∂T/∂κ=546.6; ∂T/∂ln f_gas=31.06
  - `2022-10-06__W001` phase3 ∂T/∂κ=988.4; ∂T/∂ln f_gas=16.36
  - `10-06D_008__W002` phase3 ∂T/∂κ=852.4; ∂T/∂ln f_gas=1.981
  - `10-09N_017__W001` phase3 ∂T/∂κ=791.7; ∂T/∂ln f_gas=1.364
  - `10-09N_017__W002` phase3 ∂T/∂κ=826; ∂T/∂ln f_gas=31.65

## Q5 数据可用性（P1.14-D 第 2 问，未重算）
卡面：若汽包产汽与 O/C、煤量的 |ρ| 都 < 0.3，**或** 窗间 CV < 窗内 CV，则标注「S3 的信息在理论上存在，但数据中可能被慢变量淹没」。
- phase1 `steam_rho_oc` = 0.254。P1.14-D Q2; not recomputed
- phase1 `steam_rho_coal` = -0.29。P1.14-D Q2; not recomputed
- phase1 `steam_cv_between_over_within` = 1.425。P1.14-D Q2; >1 窗间不小于窗内
- phase1 `s3_may_be_drowned` = 1.0。S3 的信息在理论上存在，但数据中可能被慢变量淹没
- phase2 `steam_rho_oc` = 0.586。P1.14-D Q2; not recomputed
- phase2 `steam_rho_coal` = -0.527。P1.14-D Q2; not recomputed
- phase2 `steam_cv_between_over_within` = 1.677。P1.14-D Q2; >1 窗间不小于窗内
- phase2 `s3_may_be_drowned` = 0.0。卡面条件未触发（需双 |ρ|<0.3 或 CV间<CV内）
- phase3 `steam_rho_oc` = 0.478。P1.14-D Q2; not recomputed
- phase3 `steam_rho_coal` = -0.229。P1.14-D Q2; not recomputed
- phase3 `steam_cv_between_over_within` = 1.271。P1.14-D Q2; >1 窗间不小于窗内
- phase3 `s3_may_be_drowned` = 0.0。卡面条件未触发（需双 |ρ|<0.3 或 CV间<CV内）

## Fisher 摘要
- phase1 S0 [train] n=37: ρ(κ,ln f_gas)=-0.23142362754880427; σ(ln f_gas)=0.01837226012985676; cond_J=617.5947098929624; min_eig=0.969109205990801; 方向 `+0.0017 kappa + -0.0012 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S1 [train] n=37: ρ(κ,ln f_gas)=-0.37486063102110484; σ(ln f_gas)=0.015913501928829732; cond_J=644.7567633551015; min_eig=1.3431048290432819; 方向 `+0.0012 kappa + -0.0024 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S2 [train] n=37: ρ(κ,ln f_gas)=-0.24956476421688037; σ(ln f_gas)=0.017210346268277735; cond_J=580.4446076645414; min_eig=1.2562386363917601; 方向 `+0.0014 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S3_conv [train] n=37: ρ(κ,ln f_gas)=-0.23065051858014748; σ(ln f_gas)=0.01831254935121254; cond_J=620.6192817641409; min_eig=0.9698172182568995; 方向 `-0.0017 kappa + +0.0012 ln_f_gas + -1.0000 xi_mol_s + +0.0017 ln_U`. Q1=是
- phase1 S3_rad [train] n=37: ρ(κ,ln f_gas)=-0.22988923627877775; σ(ln f_gas)=0.01826373447287609; cond_J=630.5142981850025; min_eig=0.9705700679828735; 方向 `-0.0017 kappa + +0.0012 ln_f_gas + -1.0000 xi_mol_s + +0.0036 ln_U`. Q1=是
- phase1 S4_conv [train] n=37: ρ(κ,ln f_gas)=-0.30531220686330285; σ(ln f_gas)=0.013960792937226445; cond_J=664.0785069065183; min_eig=1.603908009395051; 方向 `+0.0009 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s + -0.0009 ln_U`. Q1=是
- phase1 S4_rad [train] n=37: ρ(κ,ln f_gas)=-0.30480898444404886; σ(ln f_gas)=0.013939121474389476; cond_J=669.7628706791826; min_eig=1.60410444421029; 方向 `+0.0009 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s + -0.0018 ln_U`. Q1=是
- phase1 S4_no_wall [train] n=37: ρ(κ,ln f_gas)=-0.3058610262310301; σ(ln f_gas)=0.01398719907970299; cond_J=662.2915866949776; min_eig=1.60372052018121; 方向 `+0.0009 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S0 [phase_all] n=37: ρ(κ,ln f_gas)=-0.23142362754880427; σ(ln f_gas)=0.01837226012985676; cond_J=617.5947098929624; min_eig=0.969109205990801; 方向 `+0.0017 kappa + -0.0012 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S1 [phase_all] n=37: ρ(κ,ln f_gas)=-0.37486063102110484; σ(ln f_gas)=0.015913501928829732; cond_J=644.7567633551015; min_eig=1.3431048290432819; 方向 `+0.0012 kappa + -0.0024 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S2 [phase_all] n=37: ρ(κ,ln f_gas)=-0.24956476421688037; σ(ln f_gas)=0.017210346268277735; cond_J=580.4446076645414; min_eig=1.2562386363917601; 方向 `+0.0014 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S3_conv [phase_all] n=37: ρ(κ,ln f_gas)=-0.23065051858014748; σ(ln f_gas)=0.01831254935121254; cond_J=620.6192817641409; min_eig=0.9698172182568995; 方向 `-0.0017 kappa + +0.0012 ln_f_gas + -1.0000 xi_mol_s + +0.0017 ln_U`. Q1=是
- phase1 S3_rad [phase_all] n=37: ρ(κ,ln f_gas)=-0.22988923627877775; σ(ln f_gas)=0.01826373447287609; cond_J=630.5142981850025; min_eig=0.9705700679828735; 方向 `-0.0017 kappa + +0.0012 ln_f_gas + -1.0000 xi_mol_s + +0.0036 ln_U`. Q1=是
- phase1 S4_conv [phase_all] n=37: ρ(κ,ln f_gas)=-0.30531220686330285; σ(ln f_gas)=0.013960792937226445; cond_J=664.0785069065183; min_eig=1.603908009395051; 方向 `+0.0009 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s + -0.0009 ln_U`. Q1=是
- phase1 S4_rad [phase_all] n=37: ρ(κ,ln f_gas)=-0.30480898444404886; σ(ln f_gas)=0.013939121474389476; cond_J=669.7628706791826; min_eig=1.60410444421029; 方向 `+0.0009 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s + -0.0018 ln_U`. Q1=是
- phase1 S4_no_wall [phase_all] n=37: ρ(κ,ln f_gas)=-0.3058610262310301; σ(ln f_gas)=0.01398719907970299; cond_J=662.2915866949776; min_eig=1.60372052018121; 方向 `+0.0009 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S0 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.37045840792271983; σ(ln f_gas)=0.061210706700953424; cond_J=611.1047658204623; min_eig=0.048162948584671295; 方向 `+0.0017 kappa + -0.0016 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S1 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3069943857890655; σ(ln f_gas)=0.054571956536601025; cond_J=624.6735078177159; min_eig=0.06353537179133818; 方向 `+0.0011 kappa + -0.0031 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S2 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.43933324631162796; σ(ln f_gas)=0.05718328610405256; cond_J=566.827112007429; min_eig=0.06414703562563963; 方向 `+0.0014 kappa + -0.0015 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S3_conv [p113a_ref] n=2: ρ(κ,ln f_gas)=0.36825922157033825; σ(ln f_gas)=0.061039800139869815; cond_J=613.7473710828166; min_eig=0.048165857669116144; 方向 `+0.0017 kappa + -0.0016 ln_f_gas + +1.0000 xi_mol_s + -0.0016 ln_U`. Q1=是
- phase1 S3_rad [p113a_ref] n=2: ρ(κ,ln f_gas)=0.36634605996073366; σ(ln f_gas)=0.06089618506847463; cond_J=623.1991917003788; min_eig=0.04816860245919096; 方向 `+0.0017 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0034 ln_U`. Q1=是
- phase1 S4_conv [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3927831078241355; σ(ln f_gas)=0.04888297511822084; cond_J=622.6143476048412; min_eig=0.07832126028273019; 方向 `+0.0008 kappa + -0.0023 ln_f_gas + +1.0000 xi_mol_s + -0.0009 ln_U`. Q1=是
- phase1 S4_rad [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3917659543611386; σ(ln f_gas)=0.048812608657224685; cond_J=628.3543208714703; min_eig=0.07832141769358456; 方向 `+0.0008 kappa + -0.0023 ln_f_gas + +1.0000 xi_mol_s + -0.0018 ln_U`. Q1=是
- phase1 S4_no_wall [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3939569258585654; σ(ln f_gas)=0.048966622687572965; cond_J=621.0120202861341; min_eig=0.07832073256198785; 方向 `+0.0008 kappa + -0.0023 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S0 [train] n=44: ρ(κ,ln f_gas)=0.07213332814311449; σ(ln f_gas)=0.011711766735151995; cond_J=755.6164368381066; min_eig=0.9925011375777523; 方向 `+0.0015 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S1 [train] n=44: ρ(κ,ln f_gas)=-0.08754373991041728; σ(ln f_gas)=0.010817255164052398; cond_J=746.9242302171316; min_eig=1.2018334818218612; 方向 `+0.0013 kappa + -0.0025 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S2 [train] n=44: ρ(κ,ln f_gas)=0.10463066043479545; σ(ln f_gas)=0.010930302313427277; cond_J=694.1309867054824; min_eig=1.354443490095416; 方向 `+0.0012 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S3_conv [train] n=44: ρ(κ,ln f_gas)=0.07169020502251283; σ(ln f_gas)=0.011692411529520797; cond_J=757.2537735358319; min_eig=0.9925434120714367; 方向 `+0.0015 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s + -0.0012 ln_U`. Q1=是
- phase2 S3_rad [train] n=44: ρ(κ,ln f_gas)=0.07132898131920304; σ(ln f_gas)=0.011677065810132581; cond_J=764.4207965335089; min_eig=0.9925768015459471; 方向 `+0.0015 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s + -0.0027 ln_U`. Q1=是
- phase2 S4_conv [train] n=44: ρ(κ,ln f_gas)=0.001985509836737897; σ(ln f_gas)=0.009757258238415166; cond_J=717.4009000920399; min_eig=1.5545340495571518; 方向 `+0.0010 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0008 ln_U`. Q1=是
- phase2 S4_rad [train] n=44: ρ(κ,ln f_gas)=0.0018337681093714168; σ(ln f_gas)=0.009748463575487894; cond_J=722.2252994454243; min_eig=1.554571749507333; 方向 `+0.0010 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0018 ln_U`. Q1=是
- phase2 S4_no_wall [train] n=44: ρ(κ,ln f_gas)=0.002169914199007744; σ(ln f_gas)=0.009768340652014037; cond_J=716.3021882249672; min_eig=1.5544836847348964; 方向 `+0.0010 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S0 [phase_all] n=44: ρ(κ,ln f_gas)=0.07213332814311449; σ(ln f_gas)=0.011711766735151995; cond_J=755.6164368381066; min_eig=0.9925011375777523; 方向 `+0.0015 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S1 [phase_all] n=44: ρ(κ,ln f_gas)=-0.08754373991041728; σ(ln f_gas)=0.010817255164052398; cond_J=746.9242302171316; min_eig=1.2018334818218612; 方向 `+0.0013 kappa + -0.0025 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S2 [phase_all] n=44: ρ(κ,ln f_gas)=0.10463066043479545; σ(ln f_gas)=0.010930302313427277; cond_J=694.1309867054824; min_eig=1.354443490095416; 方向 `+0.0012 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S3_conv [phase_all] n=44: ρ(κ,ln f_gas)=0.07169020502251283; σ(ln f_gas)=0.011692411529520797; cond_J=757.2537735358319; min_eig=0.9925434120714367; 方向 `+0.0015 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s + -0.0012 ln_U`. Q1=是
- phase2 S3_rad [phase_all] n=44: ρ(κ,ln f_gas)=0.07132898131920304; σ(ln f_gas)=0.011677065810132581; cond_J=764.4207965335089; min_eig=0.9925768015459471; 方向 `+0.0015 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s + -0.0027 ln_U`. Q1=是
- phase2 S4_conv [phase_all] n=44: ρ(κ,ln f_gas)=0.001985509836737897; σ(ln f_gas)=0.009757258238415166; cond_J=717.4009000920399; min_eig=1.5545340495571518; 方向 `+0.0010 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0008 ln_U`. Q1=是
- phase2 S4_rad [phase_all] n=44: ρ(κ,ln f_gas)=0.0018337681093714168; σ(ln f_gas)=0.009748463575487894; cond_J=722.2252994454243; min_eig=1.554571749507333; 方向 `+0.0010 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0018 ln_U`. Q1=是
- phase2 S4_no_wall [phase_all] n=44: ρ(κ,ln f_gas)=0.002169914199007744; σ(ln f_gas)=0.009768340652014037; cond_J=716.3021882249672; min_eig=1.5544836847348964; 方向 `+0.0010 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S0 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3565201900381535; σ(ln f_gas)=0.048202041038247476; cond_J=686.4374423332321; min_eig=0.045600158133054576; 方向 `+0.0016 kappa + -0.0009 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S1 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.23804902773217457; σ(ln f_gas)=0.04440105466062769; cond_J=681.5043255158316; min_eig=0.056154597435287334; 方向 `+0.0012 kappa + -0.0024 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S2 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.420783816900442; σ(ln f_gas)=0.04497758882671372; cond_J=631.3084675203473; min_eig=0.06213881069116109; 方向 `+0.0013 kappa + -0.0009 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S3_conv [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3556337702978235; σ(ln f_gas)=0.04812115713102232; cond_J=688.0877004055834; min_eig=0.045600118061950384; 方向 `+0.0016 kappa + -0.0009 ln_f_gas + +1.0000 xi_mol_s + -0.0012 ln_U`. Q1=是
- phase2 S3_rad [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3548930964633592; σ(ln f_gas)=0.04805738247052394; cond_J=695.6368601265068; min_eig=0.045599901928803456; 方向 `+0.0016 kappa + -0.0009 ln_f_gas + +1.0000 xi_mol_s + -0.0028 ln_U`. Q1=是
- phase2 S4_conv [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3464543016324863; σ(ln f_gas)=0.039990561392672735; cond_J=657.5895751216135; min_eig=0.07197458741459567; 方向 `+0.0009 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0008 ln_U`. Q1=是
- phase2 S4_rad [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3460010372395182; σ(ln f_gas)=0.03995491910821592; cond_J=662.5892755400608; min_eig=0.07197532637079393; 方向 `+0.0009 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0018 ln_U`. Q1=是
- phase2 S4_no_wall [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3470007313289387; σ(ln f_gas)=0.040035853113187594; cond_J=656.5070182135706; min_eig=0.07197318321901577; 方向 `+0.0009 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase3 S0 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase3 S1 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase3 S2 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase3 S3_conv [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase3 S3_rad [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase3 S4_conv [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase3 S4_rad [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase3 S4_no_wall [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase3 S0 [phase_all] n=44: ρ(κ,ln f_gas)=-0.18031735717962374; σ(ln f_gas)=0.054346981602779086; cond_J=632.7658387255451; min_eig=1.0770339526125932; 方向 `+0.0018 kappa + -0.0059 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase3 S1 [phase_all] n=44: ρ(κ,ln f_gas)=-0.7557095270963484; σ(ln f_gas)=0.02176008095336483; cond_J=778.8732539824667; min_eig=1.1339513403202628; 方向 `+0.0017 kappa + -0.0131 ln_f_gas + +0.9999 xi_mol_s`. Q1=是
- phase3 S2 [phase_all] n=44: ρ(κ,ln f_gas)=-0.2997821013018883; σ(ln f_gas)=0.05183661080551175; cond_J=597.8510141713992; min_eig=1.4042069438831049; 方向 `+0.0015 kappa + -0.0117 ln_f_gas + +0.9999 xi_mol_s`. Q1=是
- phase3 S3_conv [phase_all] n=44: ρ(κ,ln f_gas)=-0.18017220041804805; σ(ln f_gas)=0.054271437860453964; cond_J=635.0688693357183; min_eig=1.0771337436315154; 方向 `-0.0018 kappa + +0.0059 ln_f_gas + -1.0000 xi_mol_s + +0.0017 ln_U`. Q1=是
- phase3 S3_rad [phase_all] n=44: ρ(κ,ln f_gas)=-0.17993485710954624; σ(ln f_gas)=0.05418012000854285; cond_J=644.4346056421064; min_eig=1.0773086829965783; 方向 `-0.0018 kappa + +0.0058 ln_f_gas + -1.0000 xi_mol_s + +0.0038 ln_U`. Q1=是
- phase3 S4_conv [phase_all] n=44: ρ(κ,ln f_gas)=-0.6101574994325681; σ(ln f_gas)=0.017615416176216786; cond_J=713.2848549367621; min_eig=1.642189021214711; 方向 `+0.0011 kappa + -0.0086 ln_f_gas + +1.0000 xi_mol_s + -0.0013 ln_U`. Q1=是
- phase3 S4_rad [phase_all] n=44: ρ(κ,ln f_gas)=-0.6100765282870957; σ(ln f_gas)=0.017612052825781006; cond_J=718.7280760030468; min_eig=1.6423181560271274; 方向 `+0.0011 kappa + -0.0086 ln_f_gas + +1.0000 xi_mol_s + -0.0028 ln_U`. Q1=是
- phase3 S4_no_wall [phase_all] n=44: ρ(κ,ln f_gas)=-0.6102114430700601; σ(ln f_gas)=0.017617980232223905; cond_J=711.9566748152857; min_eig=1.642102551262396; 方向 `+0.0011 kappa + -0.0086 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase3 S0 [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.6269170947629623; σ(ln f_gas)=0.34552070187275696; cond_J=665.9951035985011; min_eig=0.046651466054692156; 方向 `+0.0020 kappa + -0.0112 ln_f_gas + +0.9999 xi_mol_s`. Q1=是
- phase3 S1 [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.7968056911133269; σ(ln f_gas)=0.10179555057641844; cond_J=789.702589370632; min_eig=0.04689319878318446; 方向 `+0.0020 kappa + -0.0127 ln_f_gas + +0.9999 xi_mol_s`. Q1=是
- phase3 S2 [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.7417198198418464; σ(ln f_gas)=0.3371102170110813; cond_J=631.7659381041834; min_eig=0.06008584688680777; 方向 `+0.0019 kappa + -0.0221 ln_f_gas + +0.9998 xi_mol_s`. Q1=是
- phase3 S3_conv [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.626814226974949; σ(ln f_gas)=0.34543628658634595; cond_J=667.9577318263182; min_eig=0.04665205802837106; 方向 `-0.0020 kappa + +0.0112 ln_f_gas + -0.9999 xi_mol_s + +0.0019 ln_U`. Q1=是
- phase3 S3_rad [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.6267811037781879; σ(ln f_gas)=0.34541533499594557; cond_J=676.3453321409422; min_eig=0.04665182493374522; 方向 `-0.0020 kappa + +0.0112 ln_f_gas + -0.9999 xi_mol_s + +0.0042 ln_U`. Q1=是
- phase3 S4_conv [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.6932416216331525; σ(ln f_gas)=0.08416235253400724; cond_J=709.4975474442272; min_eig=0.07023906434860844; 方向 `+0.0014 kappa + -0.0088 ln_f_gas + +1.0000 xi_mol_s + -0.0014 ln_U`. Q1=是
- phase3 S4_rad [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.6932383339990962; σ(ln f_gas)=0.08416187015222609; cond_J=714.7123698703647; min_eig=0.07023879594102295; 方向 `+0.0014 kappa + -0.0088 ln_f_gas + +1.0000 xi_mol_s + -0.0030 ln_U`. Q1=是
- phase3 S4_no_wall [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.6932487103873721; σ(ln f_gas)=0.08416373909789779; cond_J=708.2816078380614; min_eig=0.07023874990861723; 方向 `+0.0014 kappa + -0.0088 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S0 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase4 S1 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase4 S2 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase4 S4_no_wall [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 ``. Q1=否
- phase4 S0 [phase_all] n=17: ρ(κ,ln f_gas)=-0.05785248543603289; σ(ln f_gas)=0.056906660709411624; cond_J=725.2620231603214; min_eig=0.3919273770520011; 方向 `+0.0016 kappa + -0.0040 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S1 [phase_all] n=17: ρ(κ,ln f_gas)=-0.4637226078249845; σ(ln f_gas)=0.033223142254905955; cond_J=759.9298789138192; min_eig=0.4042547719065891; 方向 `+0.0016 kappa + -0.0085 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S2 [phase_all] n=17: ρ(κ,ln f_gas)=-0.09742625203087539; σ(ln f_gas)=0.05354820209190144; cond_J=666.8976753225272; min_eig=0.5353093423708076; 方向 `+0.0013 kappa + -0.0058 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S4_no_wall [phase_all] n=17: ρ(κ,ln f_gas)=-0.3099189630701488; σ(ln f_gas)=0.0293539960446207; cond_J=693.465568468881; min_eig=0.5870805104244643; 方向 `+0.0012 kappa + -0.0059 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S0 [p113a_ref] n=3: ρ(κ,ln f_gas)=0.20387360171001576; σ(ln f_gas)=0.17092553511544747; cond_J=745.8509452027101; min_eig=0.06957412774257846; 方向 `+0.0015 kappa + -0.0049 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S1 [p113a_ref] n=3: ρ(κ,ln f_gas)=-0.3793818699348325; σ(ln f_gas)=0.08374391029948876; cond_J=787.7112491319439; min_eig=0.07072851152675896; 方向 `+0.0014 kappa + -0.0093 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S2 [p113a_ref] n=3: ρ(κ,ln f_gas)=0.19064871987527668; σ(ln f_gas)=0.16150204266414947; cond_J=687.9200911382343; min_eig=0.09444781005357353; 方向 `+0.0012 kappa + -0.0080 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S4_no_wall [p113a_ref] n=3: ρ(κ,ln f_gas)=-0.18214073382666376; σ(ln f_gas)=0.07404971820128788; cond_J=715.9471578996432; min_eig=0.10365290155199995; 方向 `+0.0011 kappa + -0.0064 ln_f_gas + +1.0000 xi_mol_s`. Q1=是

## 必须回答
1. 求解状态与模型−装置偏差见 `baseline_142.csv` 与上文覆盖节。
2. Q1–Q5 见上，逐阶段数值见 `readings.csv` / `fisher_subsets.csv`。
3. 求解失败行在 `baseline_142.csv` 的 `ok_theta` / `ok_jacobian` / `error`。

本轮不采纳 κ、ξ、`f_gas`、U 的任何数值，不改合同。

