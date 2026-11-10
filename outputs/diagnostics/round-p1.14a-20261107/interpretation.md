# P1.14-A 可辨识性设计判读

结构冻结 `wsr_plus_pfr`，`L_wsr = 1 m`，`dz_down = 0.0515 m`，`total_percent` `HL_tot = 1%`，`OxygenMixingModel = instant`。阶段 θ 取自 P1.13-A 分析仪口径（`L_wsr=1 m`，`HL_tot=1%`）；θ0 = (κ=1, ξ=0, f_gas=1)。ξ 只作用于出口组成，不进求解器。水冷壁诊断量 `Q_wall_conv` / `Q_wall_rad` 不进求解；Fisher 中用逐阶段冗余参数 ln U，对数残差对 ln U 的未缩放列为 1，缩放后为 1/0.10。本轮**不采纳**任何参数值，不做反演。

观测缩放（预声明，跑完未改）：X_C 1.5 pp；CO/H2/CO2 0.5 vol%；CH4 0.05 vol%；干气/干煤 2%（相对）；Q_wall 10%（对数）；T_out 20 K（仅 S4）。

## 覆盖
- 142 窗基线：θ 成功 142/142，雅可比有限 142/142，失败 0。
- θ 下 X_C 模型−装置 (pp)：中位 -0.0127，范围 -4.25 … 3.51（n=142）
- θ 下 CO (vol%)：中位 -0.297，范围 -2.37 … 1.45（n=142）
- θ 下 H2 (vol%)：中位 -0.246，范围 -1.04 … 0.574（n=142）
- θ 下 CO2 (vol%)：中位 0.0318，范围 -1.49 … 2.18（n=142）
- θ0 下 X_C 模型−装置 (pp)：中位 6.91，范围 -3.8 … 10（n=142）

冻结切分：P1 全部 train（37），P2 全部 train（44），P3 全部 validation（44），P4 全部 test（17）。主 Fisher（Q1–Q3）P1/P2 用 train；P3/P4 无 train 窗，用 `phase_all` 并在表中标明 split_role，不是 train 集宣称。

## Q1 可辨识 f_gas
判据：|ρ(κ, ln f_gas)| < 0.9 **且** σ(ln f_gas) < ln 2。
- phase1 S0 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.2314 (thr 0.9); σ(ln f_gas)=0.01837 (thr ln2=0.6931); n=37; 
- phase1 S1 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.3749 (thr 0.9); σ(ln f_gas)=0.01591 (thr ln2=0.6931); n=37; 
- phase1 S2 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.2496 (thr 0.9); σ(ln f_gas)=0.01721 (thr ln2=0.6931); n=37; 
- phase1 S3 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.2269 (thr 0.9); σ(ln f_gas)=0.01819 (thr ln2=0.6931); n=37; 
- phase1 S4 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.3033 (thr 0.9); σ(ln f_gas)=0.01391 (thr ln2=0.6931); n=37; 
- phase2 S0 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.07213 (thr 0.9); σ(ln f_gas)=0.01171 (thr ln2=0.6931); n=44; 
- phase2 S1 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.08754 (thr 0.9); σ(ln f_gas)=0.01082 (thr ln2=0.6931); n=44; 
- phase2 S2 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.1046 (thr 0.9); σ(ln f_gas)=0.01093 (thr ln2=0.6931); n=44; 
- phase2 S3 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.07171 (thr 0.9); σ(ln f_gas)=0.01166 (thr ln2=0.6931); n=44; 
- phase2 S4 [train]：**可辨识**。|ρ(κ,ln f_gas)|=0.002291 (thr 0.9); σ(ln f_gas)=0.009737 (thr ln2=0.6931); n=44; 
- phase3 S0 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase3 S1 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase3 S2 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase3 S3 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase3 S4 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase3 S0 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.1803 (thr 0.9); σ(ln f_gas)=0.05435 (thr ln2=0.6931); n=44; 
- phase3 S1 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.7557 (thr 0.9); σ(ln f_gas)=0.02176 (thr ln2=0.6931); n=44; 
- phase3 S2 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.2998 (thr 0.9); σ(ln f_gas)=0.05184 (thr ln2=0.6931); n=44; 
- phase3 S3 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.17 (thr 0.9); σ(ln f_gas)=0.05387 (thr ln2=0.6931); n=44; 
- phase3 S4 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.6076 (thr 0.9); σ(ln f_gas)=0.01756 (thr ln2=0.6931); n=44; 
- phase4 S0 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase4 S1 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase4 S2 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase4 S4 [train]：**无训练窗**。无训练窗，不作 train 集可辨识结论
- phase4 S0 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.05785 (thr 0.9); σ(ln f_gas)=0.05691 (thr ln2=0.6931); n=17; 
- phase4 S1 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.4637 (thr 0.9); σ(ln f_gas)=0.03322 (thr ln2=0.6931); n=17; 
- phase4 S2 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.09743 (thr 0.9); σ(ln f_gas)=0.05355 (thr ln2=0.6931); n=17; 
- phase4 S4 [phase_all]：**可辨识**。|ρ(κ,ln f_gas)|=0.3099 (thr 0.9); σ(ln f_gas)=0.02935 (thr ln2=0.6931); n=17; 

## Q2 信息来源（相对 S0 的 σ(ln f_gas) 缩小倍数）
- phase1 `shrink_S1_over_S0` = 1.1545076760617103。σ_S0=0.01837; σ_S1=0.01591
- phase1 `shrink_S2_over_S0` = 1.0675125208677918。σ_S0=0.01837; σ_S2=0.01721
- phase1 `shrink_S3_over_S0` = 1.0097436126349901。σ_S0=0.01837; σ_S3=0.01819
- phase1 `largest_contributor` = 1.1545076760617103。S1
- phase2 `shrink_S1_over_S0` = 1.0826930267922508。σ_S0=0.01171; σ_S1=0.01082
- phase2 `shrink_S2_over_S0` = 1.0714952248635183。σ_S0=0.01171; σ_S2=0.01093
- phase2 `shrink_S3_over_S0` = 1.0046366800362405。σ_S0=0.01171; σ_S3=0.01166
- phase2 `largest_contributor` = 1.0826930267922508。S1
- phase3 `shrink_S1_over_S0` = 2.4975542011655634。σ_S0=0.05435; σ_S1=0.02176
- phase3 `shrink_S2_over_S0` = 1.048428528760997。σ_S0=0.05435; σ_S2=0.05184
- phase3 `shrink_S3_over_S0` = 1.0089316136433348。σ_S0=0.05435; σ_S3=0.05387
- phase3 `largest_contributor` = 2.4975542011655634。S1
- phase4 `shrink_S1_over_S0` = 1.7128620848922982。σ_S0=0.05691; σ_S1=0.03322
- phase4 `shrink_S2_over_S0` = 1.0627184197845947。σ_S0=0.05691; σ_S2=0.05355
- phase4 `largest_contributor` = 1.7128620848922982。S1

## Q3 工况多样性（S0：P1.13-A 参考窗 → 该阶段展开窗）
- phase1 `sigma_ln_f_gas_p113a_ref` [p113a_ref] = 0.061210706700953424。n=2
- phase1 `sigma_ln_f_gas_expanded` [train] = 0.01837226012985676。n=37
- phase1 `sigma_shrink_ref_to_expanded` [train] = 3.331691706317608。σ_ref=0.06121 → σ_exp=0.01837; >1 means expansion helped
- phase2 `sigma_ln_f_gas_p113a_ref` [p113a_ref] = 0.048202041038247476。n=2
- phase2 `sigma_ln_f_gas_expanded` [train] = 0.011711766735151995。n=44
- phase2 `sigma_shrink_ref_to_expanded` [train] = 4.115693398637513。σ_ref=0.0482 → σ_exp=0.01171; >1 means expansion helped
- phase3 `sigma_ln_f_gas_p113a_ref` [p113a_ref] = 0.34552070187275696。n=2
- phase3 `sigma_ln_f_gas_expanded` [phase_all] = 0.054346981602779086。n=44
- phase3 `sigma_shrink_ref_to_expanded` [phase_all] = 6.357679703321158。σ_ref=0.3455 → σ_exp=0.05435; >1 means expansion helped
- phase4 `sigma_ln_f_gas_p113a_ref` [p113a_ref] = 0.17092553511544747。n=3
- phase4 `sigma_ln_f_gas_expanded` [phase_all] = 0.056906660709411624。n=17
- phase4 `sigma_shrink_ref_to_expanded` [phase_all] = 3.003612107697941。σ_ref=0.1709 → σ_exp=0.05691; >1 means expansion helped

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
- phase1 S3 [train] n=37: ρ(κ,ln f_gas)=-0.2269450100097758; σ(ln f_gas)=0.018194975338257577; cond_J=631.5286109004825; min_eig=0.977601796770021; 方向 `-0.0017 kappa + +0.0011 ln_f_gas + -1.0000 xi_mol_s + +0.0027 ln_U`. Q1=是
- phase1 S4 [train] n=37: ρ(κ,ln f_gas)=-0.30332298848397393; σ(ln f_gas)=0.013908370226966796; cond_J=671.2387452220985; min_eig=1.605850360358076; 方向 `-0.0009 kappa + +0.0011 ln_f_gas + -1.0000 xi_mol_s + +0.0013 ln_U`. Q1=是
- phase1 S0 [phase_all] n=37: ρ(κ,ln f_gas)=-0.23142362754880427; σ(ln f_gas)=0.01837226012985676; cond_J=617.5947098929624; min_eig=0.969109205990801; 方向 `+0.0017 kappa + -0.0012 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S1 [phase_all] n=37: ρ(κ,ln f_gas)=-0.37486063102110484; σ(ln f_gas)=0.015913501928829732; cond_J=644.7567633551015; min_eig=1.3431048290432819; 方向 `+0.0012 kappa + -0.0024 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S2 [phase_all] n=37: ρ(κ,ln f_gas)=-0.24956476421688037; σ(ln f_gas)=0.017210346268277735; cond_J=580.4446076645414; min_eig=1.2562386363917601; 方向 `+0.0014 kappa + -0.0011 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S3 [phase_all] n=37: ρ(κ,ln f_gas)=-0.2269450100097758; σ(ln f_gas)=0.018194975338257577; cond_J=631.5286109004825; min_eig=0.977601796770021; 方向 `-0.0017 kappa + +0.0011 ln_f_gas + -1.0000 xi_mol_s + +0.0027 ln_U`. Q1=是
- phase1 S4 [phase_all] n=37: ρ(κ,ln f_gas)=-0.30332298848397393; σ(ln f_gas)=0.013908370226966796; cond_J=671.2387452220985; min_eig=1.605850360358076; 方向 `-0.0009 kappa + +0.0011 ln_f_gas + -1.0000 xi_mol_s + +0.0013 ln_U`. Q1=是
- phase1 S0 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.37045840792271983; σ(ln f_gas)=0.061210706700953424; cond_J=611.1047658204623; min_eig=0.048162948584671295; 方向 `+0.0017 kappa + -0.0016 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S1 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3069943857890655; σ(ln f_gas)=0.054571956536601025; cond_J=624.6735078177159; min_eig=0.06353537179133818; 方向 `+0.0011 kappa + -0.0031 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S2 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.43933324631162796; σ(ln f_gas)=0.05718328610405256; cond_J=566.827112007429; min_eig=0.06414703562563963; 方向 `+0.0014 kappa + -0.0015 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase1 S3 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3638316306910186; σ(ln f_gas)=0.060709861110364596; cond_J=623.9250396514346; min_eig=0.04847797679438257; 方向 `-0.0017 kappa + +0.0017 ln_f_gas + -1.0000 xi_mol_s + +0.0025 ln_U`. Q1=是
- phase1 S4 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3906251680755382; σ(ln f_gas)=0.048725431502941174; cond_J=629.6436128627917; min_eig=0.07840589558419409; 方向 `+0.0008 kappa + -0.0023 ln_f_gas + +1.0000 xi_mol_s + -0.0014 ln_U`. Q1=是
- phase2 S0 [train] n=44: ρ(κ,ln f_gas)=0.07213332814311449; σ(ln f_gas)=0.011711766735151995; cond_J=755.6164368381066; min_eig=0.9925011375777523; 方向 `+0.0015 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S1 [train] n=44: ρ(κ,ln f_gas)=-0.08754373991041728; σ(ln f_gas)=0.010817255164052398; cond_J=746.9242302171316; min_eig=1.2018334818218612; 方向 `+0.0013 kappa + -0.0025 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S2 [train] n=44: ρ(κ,ln f_gas)=0.10463066043479545; σ(ln f_gas)=0.010930302313427277; cond_J=694.1309867054824; min_eig=1.354443490095416; 方向 `+0.0012 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S3 [train] n=44: ρ(κ,ln f_gas)=0.07170823620817243; σ(ln f_gas)=0.011657713647016664; cond_J=764.1298169966151; min_eig=0.997733983492401; 方向 `-0.0015 kappa + +0.0010 ln_f_gas + -1.0000 xi_mol_s + +0.0019 ln_U`. Q1=是
- phase2 S4 [train] n=44: ρ(κ,ln f_gas)=0.0022913833692633752; σ(ln f_gas)=0.009736960686400824; cond_J=722.8174367496546; min_eig=1.556880930189512; 方向 `+0.0010 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0013 ln_U`. Q1=是
- phase2 S0 [phase_all] n=44: ρ(κ,ln f_gas)=0.07213332814311449; σ(ln f_gas)=0.011711766735151995; cond_J=755.6164368381066; min_eig=0.9925011375777523; 方向 `+0.0015 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S1 [phase_all] n=44: ρ(κ,ln f_gas)=-0.08754373991041728; σ(ln f_gas)=0.010817255164052398; cond_J=746.9242302171316; min_eig=1.2018334818218612; 方向 `+0.0013 kappa + -0.0025 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S2 [phase_all] n=44: ρ(κ,ln f_gas)=0.10463066043479545; σ(ln f_gas)=0.010930302313427277; cond_J=694.1309867054824; min_eig=1.354443490095416; 方向 `+0.0012 kappa + -0.0010 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S3 [phase_all] n=44: ρ(κ,ln f_gas)=0.07170823620817243; σ(ln f_gas)=0.011657713647016664; cond_J=764.1298169966151; min_eig=0.997733983492401; 方向 `-0.0015 kappa + +0.0010 ln_f_gas + -1.0000 xi_mol_s + +0.0019 ln_U`. Q1=是
- phase2 S4 [phase_all] n=44: ρ(κ,ln f_gas)=0.0022913833692633752; σ(ln f_gas)=0.009736960686400824; cond_J=722.8174367496546; min_eig=1.556880930189512; 方向 `+0.0010 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0013 ln_U`. Q1=是
- phase2 S0 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.3565201900381535; σ(ln f_gas)=0.048202041038247476; cond_J=686.4374423332321; min_eig=0.045600158133054576; 方向 `+0.0016 kappa + -0.0009 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S1 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.23804902773217457; σ(ln f_gas)=0.04440105466062769; cond_J=681.5043255158316; min_eig=0.056154597435287334; 方向 `+0.0012 kappa + -0.0024 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S2 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.420783816900442; σ(ln f_gas)=0.04497758882671372; cond_J=631.3084675203473; min_eig=0.06213881069116109; 方向 `+0.0013 kappa + -0.0009 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase2 S3 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.35375085177147175; σ(ln f_gas)=0.04796598717316786; cond_J=695.3563334679612; min_eig=0.04585978082124815; 方向 `+0.0016 kappa + -0.0009 ln_f_gas + +1.0000 xi_mol_s + -0.0020 ln_U`. Q1=是
- phase2 S4 [p113a_ref] n=2: ρ(κ,ln f_gas)=0.34553894386532463; σ(ln f_gas)=0.0399076038243625; cond_J=663.2476809056567; min_eig=0.07207365993753834; 方向 `+0.0009 kappa + -0.0017 ln_f_gas + +1.0000 xi_mol_s + -0.0013 ln_U`. Q1=是
- phase3 S0 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase3 S1 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase3 S2 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase3 S3 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase3 S4 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase3 S0 [phase_all] n=44: ρ(κ,ln f_gas)=-0.18031735717962374; σ(ln f_gas)=0.054346981602779086; cond_J=632.7658387255451; min_eig=1.0770339526125932; 方向 `+0.0018 kappa + -0.0059 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase3 S1 [phase_all] n=44: ρ(κ,ln f_gas)=-0.7557095270963484; σ(ln f_gas)=0.02176008095336483; cond_J=778.8732539824667; min_eig=1.1339513403202628; 方向 `+0.0017 kappa + -0.0131 ln_f_gas + +0.9999 xi_mol_s`. Q1=是
- phase3 S2 [phase_all] n=44: ρ(κ,ln f_gas)=-0.2997821013018883; σ(ln f_gas)=0.05183661080551175; cond_J=597.8510141713992; min_eig=1.4042069438831049; 方向 `+0.0015 kappa + -0.0117 ln_f_gas + +0.9999 xi_mol_s`. Q1=是
- phase3 S3 [phase_all] n=44: ρ(κ,ln f_gas)=-0.1700459843679653; σ(ln f_gas)=0.05386587244156983; cond_J=644.0932491082224; min_eig=1.0864671308281058; 方向 `-0.0018 kappa + +0.0054 ln_f_gas + -1.0000 xi_mol_s + +0.0027 ln_U`. Q1=是
- phase3 S4 [phase_all] n=44: ρ(κ,ln f_gas)=-0.6075842754945471; σ(ln f_gas)=0.017564765535456172; cond_J=719.054519660744; min_eig=1.6470676794611077; 方向 `-0.0011 kappa + +0.0085 ln_f_gas + -1.0000 xi_mol_s + +0.0021 ln_U`. Q1=是
- phase3 S0 [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.6269170947629623; σ(ln f_gas)=0.34552070187275696; cond_J=665.9951035985011; min_eig=0.046651466054692156; 方向 `+0.0020 kappa + -0.0112 ln_f_gas + +0.9999 xi_mol_s`. Q1=是
- phase3 S1 [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.7968056911133269; σ(ln f_gas)=0.10179555057641844; cond_J=789.702589370632; min_eig=0.04689319878318446; 方向 `+0.0020 kappa + -0.0127 ln_f_gas + +0.9999 xi_mol_s`. Q1=是
- phase3 S2 [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.7417198198418464; σ(ln f_gas)=0.3371102170110813; cond_J=631.7659381041834; min_eig=0.06008584688680777; 方向 `+0.0019 kappa + -0.0221 ln_f_gas + +0.9998 xi_mol_s`. Q1=是
- phase3 S3 [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.6144565954136072; σ(ln f_gas)=0.34030537707113473; cond_J=674.7323762127764; min_eig=0.047155673647841255; 方向 `-0.0019 kappa + +0.0100 ln_f_gas + -0.9999 xi_mol_s + +0.0029 ln_U`. Q1=是
- phase3 S4 [p113a_ref] n=2: ρ(κ,ln f_gas)=-0.6908289143483636; σ(ln f_gas)=0.08388247960787358; cond_J=714.6176359660819; min_eig=0.07050101918340844; 方向 `-0.0014 kappa + +0.0087 ln_f_gas + -1.0000 xi_mol_s + +0.0022 ln_U`. Q1=是
- phase4 S0 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase4 S1 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase4 S2 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase4 S4 [train] n=0: ρ(κ,ln f_gas)=nan; σ(ln f_gas)=nan; cond_J=nan; min_eig=nan; 方向 `nan`. Q1=否
- phase4 S0 [phase_all] n=17: ρ(κ,ln f_gas)=-0.05785248543603289; σ(ln f_gas)=0.056906660709411624; cond_J=725.2620231603214; min_eig=0.3919273770520011; 方向 `+0.0016 kappa + -0.0040 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S1 [phase_all] n=17: ρ(κ,ln f_gas)=-0.4637226078249845; σ(ln f_gas)=0.033223142254905955; cond_J=759.9298789138192; min_eig=0.4042547719065891; 方向 `+0.0016 kappa + -0.0085 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S2 [phase_all] n=17: ρ(κ,ln f_gas)=-0.09742625203087539; σ(ln f_gas)=0.05354820209190144; cond_J=666.8976753225272; min_eig=0.5353093423708076; 方向 `+0.0013 kappa + -0.0058 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S4 [phase_all] n=17: ρ(κ,ln f_gas)=-0.3099189630701488; σ(ln f_gas)=0.0293539960446207; cond_J=693.465568468881; min_eig=0.5870805104244643; 方向 `+0.0012 kappa + -0.0059 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S0 [p113a_ref] n=3: ρ(κ,ln f_gas)=0.20387360171001576; σ(ln f_gas)=0.17092553511544747; cond_J=745.8509452027101; min_eig=0.06957412774257846; 方向 `+0.0015 kappa + -0.0049 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S1 [p113a_ref] n=3: ρ(κ,ln f_gas)=-0.3793818699348325; σ(ln f_gas)=0.08374391029948876; cond_J=787.7112491319439; min_eig=0.07072851152675896; 方向 `+0.0014 kappa + -0.0093 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S2 [p113a_ref] n=3: ρ(κ,ln f_gas)=0.19064871987527668; σ(ln f_gas)=0.16150204266414947; cond_J=687.9200911382343; min_eig=0.09444781005357353; 方向 `+0.0012 kappa + -0.0080 ln_f_gas + +1.0000 xi_mol_s`. Q1=是
- phase4 S4 [p113a_ref] n=3: ρ(κ,ln f_gas)=-0.18214073382666376; σ(ln f_gas)=0.07404971820128788; cond_J=715.9471578996432; min_eig=0.10365290155199995; 方向 `+0.0011 kappa + -0.0064 ln_f_gas + +1.0000 xi_mol_s`. Q1=是

## 必须回答
1. 求解状态与模型−装置偏差见 `baseline_142.csv` 与上文覆盖节。
2. Q1–Q5 见上，逐阶段数值见 `readings.csv` / `fisher_subsets.csv`。
3. 求解失败行在 `baseline_142.csv` 的 `ok_theta` / `ok_jacobian` / `error`。

本轮不采纳 κ、ξ、`f_gas`、U 的任何数值，不改合同。

