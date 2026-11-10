# P1.13-A 试点非线性反演判读

结构冻结 `wsr_plus_pfr`，`L_wsr = 1 m`（名义），`dz_down = 0.0515 m`，`total_percent` `HL_tot = 1%`，`OxygenMixingModel = instant`。反演参数：κ（入口 O2 乘数，整体数据协调，不归因到仪表）、ξ（反应器后变换，不进求解器）、ln `f_gas`。观测：装置 X_C（实测载气、标定合成气、协调煤量主口径）与干基 CO/H2/CO2；σ = [1.5, 0.5, 0.5, 0.5]（预声明试点权重，不是测量不确定度）。算法：`least_squares` / `trf`，`x_scale` = [0.05, 50.0, 1.0]，有限差分相对步长 0.0001（ξ 步长外底 10.0 mol/s）。P2 主报告用 `J @ diag(x_scale)`；(JᵀJ)⁻¹ 是局部曲率诊断。本轮**不采纳**任何参数值。

## P1 拟合质量
σ 是预声明试点权重，不是测量不确定度；RMS≤1 记为达到预设拟合容差。
- `window_n_success::analyzer` analyzer = 9.0. /9
- `window_median_rms::analyzer` analyzer = 0.3728251980471264 **达到预设拟合容差**. 缩放残差均方根；≤1 记为达到预设拟合容差（试点权重 1.5 pp / 0.5 vol%，不是测量不确定度）
- `window_max_rms::analyzer` analyzer = 0.598576410520924 **达到预设拟合容差**. 
- `window_n_rms_le_1::analyzer` analyzer = 9.0. 
- `window_n_success::liu_phase` liu_phase = 9.0. /9
- `window_median_rms::liu_phase` liu_phase = 1.6474236102456001 **未达预设拟合容差**. 缩放残差均方根；≤1 记为达到预设拟合容差（试点权重 1.5 pp / 0.5 vol%，不是测量不确定度）
- `window_max_rms::liu_phase` liu_phase = 2.1714780253637413 **未达预设拟合容差**. 
- `window_n_rms_le_1::liu_phase` liu_phase = 2.0. 
- `phase_n_success::analyzer` analyzer = 4.0. /4
- `phase_median_rms::analyzer` analyzer = 0.5054809638546925 **达到预设拟合容差**. 缩放残差均方根；≤1 记为达到预设拟合容差（试点权重 1.5 pp / 0.5 vol%，不是测量不确定度）
- `phase_max_rms::analyzer` analyzer = 1.086246880450095 **未达预设拟合容差**. 
- `phase_n_rms_le_1::analyzer` analyzer = 3.0. 
- `phase_n_success::liu_phase` liu_phase = 4.0. /4
- `phase_median_rms::liu_phase` liu_phase = 1.6755639394686306 **未达预设拟合容差**. 缩放残差均方根；≤1 记为达到预设拟合容差（试点权重 1.5 pp / 0.5 vol%，不是测量不确定度）
- `phase_max_rms::liu_phase` liu_phase = 2.153784647960606 **未达预设拟合容差**. 
- `phase_n_rms_le_1::liu_phase` liu_phase = 1.0. 

## P2 可辨识性
主报告：参数列按优化器 `x_scale`=[0.05, 50.0, 1.0] 无量纲化后 cond(J@S) < 100.0 且全部 |ρ| < 0.9。原 cond(J)（ξ 以 mol/s）保留为历史读法，不推广。局部共线（|ρ|）、贴边界与全局条件数分开列。
- 无量纲尺度下可分别辨识：14 / 26（逐窗 6/18；逐阶段 8/8）
- 历史（原生单位）可分别辨识：0 / 26（不推广）
- 局部共线（无量纲 |ρ|≥0.9）：12 / 26
- 贴边界：6 / 26（单独列，不并入 cond 判据）
  - 不可辨识 analyzer phase1 `09-28D_008__W003`：J@diag(0.05,50,1); cond=424.7; max|ρ|=1.000; pair=kappa-lnfg
  - 不可辨识 liu_phase phase1 `09-28D_008__W003`：J@diag(0.05,50,1); cond=59.38; max|ρ|=0.997; pair=kappa-lnfg
  - 不可辨识 analyzer phase1 `2022-09-28__W001`：J@diag(0.05,50,1); cond=94.06; max|ρ|=0.992; pair=kappa-lnfg
  - 不可辨识 analyzer phase2 `10-03D_011__W001`：J@diag(0.05,50,1); cond=55.45; max|ρ|=0.994; pair=kappa-lnfg
  - 不可辨识 liu_phase phase2 `10-03D_011__W001`：J@diag(0.05,50,1); cond=419.3; max|ρ|=1.000; pair=kappa-lnfg
  - 不可辨识 liu_phase phase2 `10-03N_013__W001`：J@diag(0.05,50,1); cond=335; max|ρ|=0.999; pair=kappa-lnfg
  - 不可辨识 liu_phase phase3 `10-09D_014__W004`：J@diag(0.05,50,1); cond=383.4; max|ρ|=1.000; pair=kappa-lnfg
  - 不可辨识 liu_phase phase3 `10-09D_015__W001`：J@diag(0.05,50,1); cond=375.1; max|ρ|=1.000; pair=kappa-lnfg
  - 不可辨识 liu_phase phase4 `10-10N_006__W001`：J@diag(0.05,50,1); cond=42.87; max|ρ|=0.988; pair=kappa-lnfg
  - 不可辨识 liu_phase phase4 `10-11N_009__W002`：J@diag(0.05,50,1); cond=180.4; max|ρ|=1.000; pair=kappa-lnfg
  - 不可辨识 analyzer phase4 `2022-10-11__W001`：J@diag(0.05,50,1); cond=14.06; max|ρ|=0.929; pair=kappa-lnfg
  - 不可辨识 liu_phase phase4 `2022-10-11__W001`：J@diag(0.05,50,1); cond=297; max|ρ|=1.000; pair=kappa-lnfg

## P3 一致性
- `within_phase_std::kappa` analyzer phase1  = 0.016962399258633246 (threshold 0.5209398935010485) median curvature 1σ=0.5209; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::xi_mol_s` analyzer phase1  = 3.025656736954309 (threshold 6.664199219435869) median curvature 1σ=6.664; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::ln_f_gas` analyzer phase1  = 0.022839510765431833 (threshold 6.511452989129679) median curvature 1σ=6.511; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::f_gas` analyzer phase1  = 0.06689273949736416 (threshold nan) median curvature 1σ=nan; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::kappa` analyzer phase2  = 0.00889753527340254 (threshold 0.07890172461797128) median curvature 1σ=0.0789; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::xi_mol_s` analyzer phase2  = 0.880077023891573 (threshold 6.686953461658912) median curvature 1σ=6.687; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::ln_f_gas` analyzer phase2  = 0.29907009134299184 (threshold 0.9838979510216569) median curvature 1σ=0.9839; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::f_gas` analyzer phase2  = 0.13303260660980679 (threshold nan) median curvature 1σ=nan; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::kappa` analyzer phase3  = 0.003030511264186387 (threshold 0.017166044104842505) median curvature 1σ=0.01717; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::xi_mol_s` analyzer phase3  = 0.4374275562697235 (threshold 6.545228617046176) median curvature 1σ=6.545; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::ln_f_gas` analyzer phase3  = 0.03966496912269881 (threshold 0.4942665081108941) median curvature 1σ=0.4943; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::f_gas` analyzer phase3  = 0.00424816660021419 (threshold nan) median curvature 1σ=nan; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::kappa` analyzer phase4  = 0.0061787749168499785 (threshold 0.017551351678550705) median curvature 1σ=0.01755; n=3; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::xi_mol_s` analyzer phase4  = 1.796743252794793 (threshold 6.594220769874649) median curvature 1σ=6.594; n=3; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::ln_f_gas` analyzer phase4  = 0.13426121746567654 (threshold 0.5974052271152) median curvature 1σ=0.5974; n=3; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::f_gas` analyzer phase4  = 0.02517719850935509 (threshold nan) median curvature 1σ=nan; n=3; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::kappa` liu_phase phase1  = 0.01576819075895833 (threshold 0.07921880684816981) median curvature 1σ=0.07922; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::xi_mol_s` liu_phase phase1  = 2.9847268391461554 (threshold 6.764148882779036) median curvature 1σ=6.764; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::ln_f_gas` liu_phase phase1  = 1.0426652414591676e-08 (threshold 4.038899136187545) median curvature 1σ=4.039; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::f_gas` liu_phase phase1  = 4.170660871631123e-08 (threshold nan) median curvature 1σ=nan; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::kappa` liu_phase phase2  = 0.0019434211200064515 (threshold 0.6780832009720256) median curvature 1σ=0.6781; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::xi_mol_s` liu_phase phase2  = 0.19697645946479098 (threshold 8.686424367175889) median curvature 1σ=8.686; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::ln_f_gas` liu_phase phase2  = 0.2523980560197983 (threshold 17.589879204177695) median curvature 1σ=17.59; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::f_gas` liu_phase phase2  = 0.81408418538413 (threshold nan) median curvature 1σ=nan; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::kappa` liu_phase phase3  = 0.0019541640647281365 (threshold 0.9872516809968397) median curvature 1σ=0.9873; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::xi_mol_s` liu_phase phase3  = 2.0939505573302037 (threshold 7.279895251493432) median curvature 1σ=7.28; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::ln_f_gas` liu_phase phase3  = 0.4614549732169421 (threshold 9.212176168319639) median curvature 1σ=9.212; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::f_gas` liu_phase phase3  = 1.3186841459128187 (threshold nan) median curvature 1σ=nan; n=2; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::kappa` liu_phase phase4  = 0.005505259881892691 (threshold 0.42839986256715185) median curvature 1σ=0.4284; n=3; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::xi_mol_s` liu_phase phase4  = 4.091494434144619 (threshold 7.285116267402258) median curvature 1σ=7.285; n=3; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::ln_f_gas` liu_phase phase4  = 0.18830836146193783 (threshold 5.156167588818063) median curvature 1σ=5.156; n=3; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `within_phase_std::f_gas` liu_phase phase4  = 0.6377406371605786 (threshold nan) median curvature 1σ=nan; n=3; (JᵀJ)⁻¹ 是局部曲率，不是统计 1σ
- `basis_diff::kappa`   09-28D_008__W003 = 0.0030503291463409576 (threshold 0.9415788902541158) σ_an=0.9416; σ_liu=0.1384
- `basis_diff::xi_mol_s`   09-28D_008__W003 = 16.825948747698845 (threshold 6.542745841161109) σ_an=6.299; σ_liu=6.543
- `basis_diff::ln_f_gas`   09-28D_008__W003 = 0.29558937737502755 (threshold 5.692166271840502) σ_an=5.692; σ_liu=3.347
- `basis_diff::kappa`   2022-09-28__W001 = 0.004739195002882446 (threshold 0.10030089674798134) σ_an=0.1003; σ_liu=0.02
- `basis_diff::xi_mol_s`   2022-09-28__W001 = 16.76806513111201 (threshold 7.029002066296689) σ_an=7.029; σ_liu=6.986
- `basis_diff::ln_f_gas`   2022-09-28__W001 = 0.3278893085119543 (threshold 7.330739706418855) σ_an=7.331; σ_liu=4.731
- `basis_diff::kappa`   10-03D_011__W001 = 0.010596065854306391 (threshold 1.030961938883531) σ_an=0.1359; σ_liu=1.031
- `basis_diff::xi_mol_s`   10-03D_011__W001 = 20.571510106301673 (threshold 7.0318148554399835) σ_an=6.684; σ_liu=7.032
- `basis_diff::ln_f_gas`   10-03D_011__W001 = 1.5933325822190816 (threshold 8.78598531037297) σ_an=0.8838; σ_liu=8.786
- `basis_diff::kappa`   10-03N_013__W001 = 0.00076146330428295 (threshold 0.32520446306052025) σ_an=0.02191; σ_liu=0.3252
- `basis_diff::xi_mol_s`   10-03N_013__W001 = 19.6054600236246 (threshold 10.341033878911793) σ_an=6.69; σ_liu=10.34
- `basis_diff::ln_f_gas`   10-03N_013__W001 = 2.3732263154363036 (threshold 26.393773097982418) σ_an=1.084; σ_liu=26.39
- `basis_diff::kappa`   10-09D_014__W004 = 0.006249807903379145 (threshold 0.9831742511459872) σ_an=0.01529; σ_liu=0.9832
- `basis_diff::xi_mol_s`   10-09D_014__W004 = 28.682661400099676 (threshold 7.051363067345904) σ_an=6.509; σ_liu=7.051
- `basis_diff::ln_f_gas`   10-09D_014__W004 = 3.6207799953217217 (threshold 8.634695296153893) σ_an=0.4978; σ_liu=8.635
- `basis_diff::kappa`   10-09D_015__W001 = 0.0007995875507983419 (threshold 0.9913291108476922) σ_an=0.01905; σ_liu=0.9913
- `basis_diff::xi_mol_s`   10-09D_015__W001 = 25.102752140352237 (threshold 7.508427435640961) σ_an=6.582; σ_liu=7.508
- `basis_diff::ln_f_gas`   10-09D_015__W001 = 2.912089376489378 (threshold 9.789657040485384) σ_an=0.4907; σ_liu=9.79
- `basis_diff::kappa`   10-10N_006__W001 = 0.0020719291724937694 (threshold 0.07189963247073285) σ_an=0.0129; σ_liu=0.0719
- `basis_diff::xi_mol_s`   10-10N_006__W001 = 40.91573535495862 (threshold 7.215312066199139) σ_an=6.587; σ_liu=7.215
- `basis_diff::ln_f_gas`   10-10N_006__W001 = 3.0856377304974045 (threshold 2.870015736344158) σ_an=0.5974; σ_liu=2.87
- `basis_diff::kappa`   10-11N_009__W002 = 9.778114756664458e-06 (threshold 0.42839986256715185) σ_an=0.01755; σ_liu=0.4284
- `basis_diff::xi_mol_s`   10-11N_009__W002 = 43.18992166118166 (threshold 7.741591660186973) σ_an=6.659; σ_liu=7.742
- `basis_diff::ln_f_gas`   10-11N_009__W002 = 3.1838930008733355 (threshold 10.969787752144354) σ_an=0.5741; σ_liu=10.97
- `basis_diff::kappa`   2022-10-11__W001 = 0.0017640865906852765 (threshold 0.6954653424914767) σ_an=0.04245; σ_liu=0.6955
- `basis_diff::xi_mol_s`   2022-10-11__W001 = 46.58715027467568 (threshold 7.285116267402258) σ_an=6.594; σ_liu=7.285
- `basis_diff::ln_f_gas`   2022-10-11__W001 = 2.598760265974173 (threshold 5.156167588818063) σ_an=0.5981; σ_liu=5.156
- `phase_median::kappa` analyzer   = 0.9367465738374177 (threshold nan) range 0.8998 … 0.9636; n=4
- `phase_median::xi_mol_s` analyzer   = 84.34465163577353 (threshold nan) range 56.14 … 104.8; n=4
- `phase_median::f_gas` analyzer   = 0.28707172629833844 (threshold nan) range 0.1079 … 0.6531; n=4
- `phase_median::kappa` liu_phase   = 0.9365372418833774 (threshold nan) range 0.9045 … 0.9685; n=4
- `phase_median::xi_mol_s` liu_phase   = 107.26613086084586 (threshold nan) range 99.52 … 121.6; n=4
- `phase_median::f_gas` liu_phase   = 2.187908250207499 (threshold nan) range 1.286 … 3.23; n=4

## P4 物理合理性（只标记，不判通过）
- `n_bound_hit_windows` = 6.0. >6 窗贴边界则判读单列；不停止
- `n_needs_quench_water` = 15.0. /18；T_eq 用反应器出口 H2O，假定激冷水恰好补 ξ mol
- `T_eq_is_proxy` = 1.0. 表观平衡温度是该水量假设下的代理量，不是测得的激冷后平衡温度
- `n_kappa_in_phys` = 14.0. /18 收敛窗（两口径合计）
- `n_fgas_in_phys` = 7.0. /18 收敛窗（两口径合计）
- `n_Teq_in_phys` = 18.0. /18 收敛窗（两口径合计）
  - `09-28D_008__W003` liu_phase hits `ln_f_gas:hi` κ=0.979 ξ=120 f_gas=4
  - `2022-09-28__W001` liu_phase hits `ln_f_gas:hi` κ=0.9567 ξ=124 f_gas=4
  - `10-03N_013__W001` liu_phase hits `ln_f_gas:hi` κ=0.9066 ξ=106 f_gas=3.835
  - `10-09D_014__W004` liu_phase hits `ln_f_gas:hi` κ=0.9279 ξ=111 f_gas=3.891
  - `10-10N_006__W001` liu_phase hits `ln_f_gas:hi` κ=0.944 ξ=94.8 f_gas=4
  - `10-11N_009__W002` liu_phase hits `ln_f_gas:hi` κ=0.9529 ξ=101 f_gas=3.933

## P5 稳健性
逐阶段拟合在 `L_wsr=4 m`、`HL_tot` ∈ {0.5, 1.5}% 下重做，报告 Δθ / 1σ。
- `d_over_sigma::kappa` analyzer phase1 = 0.6944. L_wsr=4.0 HL=0.5; Δ=0.008358; σ_nom=0.01204
- `d_over_sigma::xi_mol_s` analyzer phase1 = 0.424. L_wsr=4.0 HL=0.5; Δ=1.932; σ_nom=4.557
- `d_over_sigma::ln_f_gas` analyzer phase1 = 18.56. L_wsr=4.0 HL=0.5; Δ=1.136; σ_nom=0.06121
- `d_over_sigma::kappa` analyzer phase1 = 0.9905. L_wsr=4.0 HL=1.5; Δ=0.01192; σ_nom=0.01204
- `d_over_sigma::xi_mol_s` analyzer phase1 = 0.5477. L_wsr=4.0 HL=1.5; Δ=2.496; σ_nom=4.557
- `d_over_sigma::ln_f_gas` analyzer phase1 = 17.35. L_wsr=4.0 HL=1.5; Δ=1.062; σ_nom=0.06121
- `d_over_sigma::kappa` liu_phase phase1 = 1.249. L_wsr=4.0 HL=0.5; Δ=0.01599; σ_nom=0.0128
- `d_over_sigma::xi_mol_s` liu_phase phase1 = 0.1215. L_wsr=4.0 HL=0.5; Δ=0.5771; σ_nom=4.75
- `d_over_sigma::ln_f_gas` liu_phase phase1 = 2.62. L_wsr=4.0 HL=0.5; Δ=0.5247; σ_nom=0.2003
- `d_over_sigma::kappa` liu_phase phase1 = 0.8323. L_wsr=4.0 HL=1.5; Δ=0.01066; σ_nom=0.0128
- `d_over_sigma::xi_mol_s` liu_phase phase1 = 0.497. L_wsr=4.0 HL=1.5; Δ=2.361; σ_nom=4.75
- `d_over_sigma::ln_f_gas` liu_phase phase1 = 0.4301. L_wsr=4.0 HL=1.5; Δ=0.08614; σ_nom=0.2003
- `d_over_sigma::kappa` analyzer phase2 = 0.1751. L_wsr=4.0 HL=0.5; Δ=0.001968; σ_nom=0.01124
- `d_over_sigma::xi_mol_s` analyzer phase2 = 0.5833. L_wsr=4.0 HL=0.5; Δ=2.731; σ_nom=4.683
- `d_over_sigma::ln_f_gas` analyzer phase2 = 5.807. L_wsr=4.0 HL=0.5; Δ=0.2799; σ_nom=0.0482
- `d_over_sigma::kappa` analyzer phase2 = 1.095. L_wsr=4.0 HL=1.5; Δ=0.0123; σ_nom=0.01124
- `d_over_sigma::xi_mol_s` analyzer phase2 = 0.7047. L_wsr=4.0 HL=1.5; Δ=3.3; σ_nom=4.683
- `d_over_sigma::ln_f_gas` analyzer phase2 = 12.26. L_wsr=4.0 HL=1.5; Δ=0.591; σ_nom=0.0482
- `d_over_sigma::kappa` liu_phase phase2 = 1.03. L_wsr=4.0 HL=0.5; Δ=0.009659; σ_nom=0.009373
- `d_over_sigma::xi_mol_s` liu_phase phase2 = 0.04991. L_wsr=4.0 HL=0.5; Δ=0.2455; σ_nom=4.919
- `d_over_sigma::ln_f_gas` liu_phase phase2 = 2.139. L_wsr=4.0 HL=0.5; Δ=1.135; σ_nom=0.5307
- `d_over_sigma::kappa` liu_phase phase2 = 1.019. L_wsr=4.0 HL=1.5; Δ=0.009547; σ_nom=0.009373
- `d_over_sigma::xi_mol_s` liu_phase phase2 = 0.9056. L_wsr=4.0 HL=1.5; Δ=4.455; σ_nom=4.919
- `d_over_sigma::ln_f_gas` liu_phase phase2 = 2.139. L_wsr=4.0 HL=1.5; Δ=1.135; σ_nom=0.5307
- `d_over_sigma::kappa` analyzer phase3 = 0.101. L_wsr=4.0 HL=0.5; Δ=0.001365; σ_nom=0.01352
- `d_over_sigma::xi_mol_s` analyzer phase3 = 0.4632. L_wsr=4.0 HL=0.5; Δ=2.145; σ_nom=4.63
- `d_over_sigma::ln_f_gas` analyzer phase3 = 0.2401. L_wsr=4.0 HL=0.5; Δ=0.08295; σ_nom=0.3455
- `d_over_sigma::kappa` analyzer phase3 = 1.192. L_wsr=4.0 HL=1.5; Δ=0.01611; σ_nom=0.01352
- `d_over_sigma::xi_mol_s` analyzer phase3 = 0.861. L_wsr=4.0 HL=1.5; Δ=3.986; σ_nom=4.63
- `d_over_sigma::ln_f_gas` analyzer phase3 = 1.297. L_wsr=4.0 HL=1.5; Δ=0.448; σ_nom=0.3455
- `d_over_sigma::kappa` liu_phase phase3 = 0.534. L_wsr=4.0 HL=0.5; Δ=0.006396; σ_nom=0.01198
- `d_over_sigma::xi_mol_s` liu_phase phase3 = 0.34. L_wsr=4.0 HL=0.5; Δ=1.685; σ_nom=4.955
- `d_over_sigma::ln_f_gas` liu_phase phase3 = 3.988. L_wsr=4.0 HL=0.5; Δ=0.7753; σ_nom=0.1944
- `d_over_sigma::kappa` liu_phase phase3 = 1.044. L_wsr=4.0 HL=1.5; Δ=0.01251; σ_nom=0.01198
- `d_over_sigma::xi_mol_s` liu_phase phase3 = 0.6379. L_wsr=4.0 HL=1.5; Δ=3.161; σ_nom=4.955
- `d_over_sigma::ln_f_gas` liu_phase phase3 = 3.097. L_wsr=4.0 HL=1.5; Δ=0.6023; σ_nom=0.1944
- `d_over_sigma::kappa` analyzer phase4 = 0.3559. L_wsr=4.0 HL=0.5; Δ=0.00287; σ_nom=0.008063
- `d_over_sigma::xi_mol_s` analyzer phase4 = 0.6123. L_wsr=4.0 HL=0.5; Δ=2.321; σ_nom=3.791
- `d_over_sigma::ln_f_gas` analyzer phase4 = 0.3137. L_wsr=4.0 HL=0.5; Δ=0.05362; σ_nom=0.1709
- `d_over_sigma::kappa` analyzer phase4 = 1.488. L_wsr=4.0 HL=1.5; Δ=0.012; σ_nom=0.008063
- `d_over_sigma::xi_mol_s` analyzer phase4 = 1.093. L_wsr=4.0 HL=1.5; Δ=4.142; σ_nom=3.791
- `d_over_sigma::ln_f_gas` analyzer phase4 = 5.154. L_wsr=4.0 HL=1.5; Δ=0.881; σ_nom=0.1709
- `d_over_sigma::kappa` liu_phase phase4 = 0.8459. L_wsr=4.0 HL=0.5; Δ=0.007437; σ_nom=0.008792
- `d_over_sigma::xi_mol_s` liu_phase phase4 = 0.4466. L_wsr=4.0 HL=0.5; Δ=1.871; σ_nom=4.191
- `d_over_sigma::ln_f_gas` liu_phase phase4 = 3.571. L_wsr=4.0 HL=0.5; Δ=0.2138; σ_nom=0.05988
- `d_over_sigma::kappa` liu_phase phase4 = 1.534. L_wsr=4.0 HL=1.5; Δ=0.01349; σ_nom=0.008792
- `d_over_sigma::xi_mol_s` liu_phase phase4 = 0.6899. L_wsr=4.0 HL=1.5; Δ=2.891; σ_nom=4.191
- `d_over_sigma::ln_f_gas` liu_phase phase4 = 3.571. L_wsr=4.0 HL=1.5; Δ=0.2138; σ_nom=0.05988

## P6 多起点
- `n_nonunique_window_fits` = 0.0. 代价差<1% 且至少一参数相差超过各自 1σ
- `n_converged_starts` = 126.0. /126 起点

## P7 留一窗预测（只报告）
对照：(a) 不修正 θ=(1,0,1)；(b) 训练窗常数残差补丁。同阶段跨窗转移；刘文口径在阶段内组成目标相同，不作独立组成预测验证。
- `median_test_rms::inversion` analyzer 中位测试 rms = 0.5936648823576116
- `median_test_rms::residual_patch` analyzer 中位测试 rms = 0.39105365749592247
- `median_test_rms::uncorrected` analyzer 中位测试 rms = 10.612650302147651
- `median_test_rms::inversion` liu_phase 中位测试 rms = 2.161288975595957
- `median_test_rms::residual_patch` liu_phase 中位测试 rms = 0.3954570566114537
- `median_test_rms::uncorrected` liu_phase 中位测试 rms = 14.488179690673238

## 必须回答
1. 每次拟合的 θ̂±曲率 1σ、相关、条件数与缩放残差见 `fits_window.csv` / `fits_phase.csv`；全部起点见 `multistart.csv`；加权残差雅可比见 `jacobians.csv`（含 `start_name`/`selected`）。主报告条件数为 `cond_J_xscale`。
2. P1–P7 见上。
3. 求解失败、贴边界、非唯一解见 `multistart.csv` 的 `success` / `bound_hits` 与 fits 表的 `nonunique`。

本轮不采纳 κ、ξ、`f_gas` 的任何数值，不改合同。

