# P1.8-A 锚点 Fortran 热损模型对齐 + 装置九窗二维辨识性扫描 —— interpretation

本轮在模型仓按 2026-11-04 卡执行：①新增 `HeatLossModel ∈ {hhv_percent, fortran_wall}`，把 `Source1_副本.for` 的两区壁面热损逐字转写；②锚点模式 F′ = P1.7-A 模式 F + `fortran_wall`；③装置九窗做 `f_gas`（C+H2O 与 C+CO2 共用指前因子倍数）× `HeatLossPercent` 的二维扫描。冻结项一项未动：全部 Arrhenius 常数（除第 2 节扫描的倍数）、`phi`、`f_co2 = 1.0`、0.05 atm、`CharCombustionRateFactor`、氧化因子、热解产率、负向 WGS 截断（只统计绑定）、`USE_RK_GILL_COMBUSTION = False`、`fortran_constant`、蒸汽进料 = 0、低位热值口径、`uniform_full_length`、`SolverStrictRoot = True`。本轮**只扫描、不采纳**任何参数值。

## 摘要（对应卡第 3 节的五问）

1. **符号表与单元检查**：`fortran_heatloss_symbols.md` 逐符号列出定义/单位/行号/是否重赋值；单元检查 48/48 通过。`hhv_percent` 下模式 F 的 5 个算例出口**逐位复现** P1.7-A；`fortran_wall` 下 `HHV_d` 30→34 的温变**逐位为 0**。
2. **模式 F′**：全部判据通过的算例仍是 **0/5**；温度平均绝对差 55.6 → **235.1 K**，X_C 平均绝对差 3.66 → **8.68 pp**。方向是**变差**：F′ 的温度整体偏高 165–309 K、5/5 个算例的 X_C 贴到 100%。原因是燃烧区只有 1 个单元（2.06 cm vs Fortran 75 cm）拿到 30% 壁损，而气化区的固定壁温剖面反过来给气体加热。
3. **二维扫描**：R1 只有 **4/9** 窗存在 |ΔX_C| ≤ 1 pp 的 (f_gas, HLP) 组合（其余 5 窗：指定角点 f_gas=0.03125、HLP=6% 处残差 1.19–4.87 pp，整个搜索池的最小值 7.3–7.7 pp——两者不可混用）；R2 `analyzer` ξ_min 中位 25.8 mol/s；`liu_phase` ξ_min 中位 23.2 mol/s → `analyzer` → 不需要反应器后变换；`liu_phase` → 不需要反应器后变换（该判据只在这 4 窗上成立；其余窗根本没有 R1 组合）；R3 有 R1 组合的 4 窗上 cond(J) = 3.12–4.93、|corr| = 0.70–0.80，按预声明判据 **8/8 条记为可分别辨识**（所有最优点都落在 HLP 上边界 6%）；R4 `analyzer` P1 中位 0.1406、P4 中位 0.03125；`liu_phase` P1 中位 0.1406、P4 中位 0.03125（每个阶段只有 2 窗，不足以判跨阶段趋势）；R5 C+CO2 在净 CO2 消耗中占 84–95%，**C+H2O 与 C+CO2 共用指前因子应当拆开**（本轮不拆）。
4. **复现行**：卡面口径（旧载气 = `co2_carrier_t_nm3_h`）与 P1.7-A **不能**逐位一致——P1.7-A 的模型入口载气其实是冻结切分的 `0.24 × o2_flow_mean`，两者差 5%–7%。按冻结切分原值另跑的 9 行 **9/9 与 P1.7-A 逐位相同**，因此第 2 节继续执行（详见第 4.2 节）。
5. **`no_root` 与负向 WGS 绑定**：`no_root` **0/234**；负向 WGS 绑定长度占比 > 5% 的行 **0/234**（每行都是 1 个单元，整炉占比恒为 0.0250）。

## 0. 提交与工作树

- `protocol.json` 里的 `source_commit`（生成本轮产物的代码提交）：`9333fe4`。
- 产物随后以一次只含 `outputs/diagnostics/round-p1.8a-20261104/` 与脚本的提交落盘，两个哈希不同（`protocol.json` 要写入那个哈希，互为自指），与 2026-10-30 起各轮约定一致。
- 生成时的代码树状态（已排除本轮产物目录本身）：`clean`。

## 1. 第 1.1 节：符号表

完整表（含行号、是否被重赋值）见 `fortran_heatloss_symbols.md`，下面重列一遍。

| 符号 | 定义 / 单位 | 首次赋值行 | 是否被重新赋值 | Python 对应量 |
|---|---|---|---|---|
| `dt` | 反应器直径，**cm**。`data` 语句给 152.4 cm（= 1.524 m），与 `at = 18241.5 cm²` 满足 π/4·dt² = at，确认是直径 | 32 | **否**（`data` 初值，全程序未再赋值） | `FORTRAN_DT_CM`（常量，不取 Python 几何） |
| `height` | 反应器/积分高度，**cm**。`data` 给 330 cm（= 3.3 m）；第 787–788 行 `!change 325 into height-5` / `dhs = hs-(height-5.)` 说明积分到 `height−5` = 325 cm 为止，所以它是**总高度**而非有效长度 | 32 | **否** | `FORTRAN_HEIGHT_CM`（常量） |
| `at` | 反应器截面积，**cm²**（18241.5） | 32 | 否 | 无用（Python 用几何 D） |
| `pt` | 压力，**atm**（24.0） | 32 | 否 | `op_conds['P']` |
| `nc` | 一个气室内的子步数，无量纲。`nc = 3`（第 145 行注释：`nc is the number of step sizes in each gas compartment`） | 146 | **否** | 不单独出现；`nc·delh` → `dz` |
| `delh` | **一个子步的轴向长度，cm**。`data` 给 25，但 25 只用于**第一个气室**；第 852–858 行在每个新气室开头重设为 `0.5` 或 `1.0`（`hs.le.40 → 0.5`；`poxy.gt.0.01 → 0.5`；否则 `1.0`），并在迭代不收敛时被 `delh = delh/2`（第 367、523 行）折半 | 96 | **是**（852–858、367、523） | `dz`（Python 一个单元 = 一个气室） |
| `hs` | 颗粒已走过的轴向高度，**cm**；初值 `hsin`（第 221 行），每个子步 `hs = hs + delh`（第 277、763 行） | 221 | **是**（每子步累进；每气室由 `hsin` 重置） | 单元出口 `z + dz/2` |
| `fcoal` | 入炉煤流量，**g/s，dmmf 基**。`fcoal = tfcoal*(1.-yash)*(1.-xmois)`（第 66 行），`tfcoal` 来自 namelist | 66 | **是**（由 namelist 的 `tfcoal/yash/xmois` 算出） | 干煤流量（`W_dry`，g/s） |
| `rheat1` | **本气室挥发分项放热**，cal/g dmmf。由 `pyroly` 写入：`wl*(xh2*ht(1)/2 + xco*ht(2)/28 + xch4*ht(3)/16 + xtar*ht(4)/78)/100 − cop*ht(2)/28 − ch4p*ht(3)/16 − h2p*ht(1)/2`（第 1087–1088 行），其中 `ht(n, tg)` 是组分在**气相温度**下的焓（`heatup` 第 1131 行起，cal/mol） | 248（每气室清零） | **是**：每个气室开头清零（248、865），由第 314 行调用 `pyroly` 赋值 | 无直接对应；挥发分能量走 `PyrolysisSource.energy_src_W` |
| `rheat2` | 气相燃烧项，cal/g dmmf。`rheat2 += rhta+rhtb+rhtc-rhtd`（第 556 行）与 `+qmref*dch46/16`（第 587、743 行），`rhta/b/c/d` 由 `q1/qrh/ht` 与碳消耗量 `cwl/cwl2/cwl3/cop` 组合 | 249 | **是**（每气室清零 417/611，多处累加） | 由 `CO_Ox/H2_Ox/CH4_Ox/MSR` 的反应焓体现 |
| `rheat3` | WGS 项，cal/g dmmf。`rheat3 += qwg*wgl*(1-fwgs)`（第 588、744 行），`fwgs = 1.0`（第 100 行）是 WGS 放热被固相吸收的份额 | 250 | **是** | 由 `WGS/RWGS` 的反应焓体现 |
| `rhts` | 被固相吸收的热，cal/g dmmf：`rhts += cwl*qrh − cwl2*qcsm/12 − cwl3*qcbco2/12 − cwl5*qcbhy/12`（第 558 行）与 `−qwg*wgl*fwgs`（第 745 行） | 251 | **是**（每气室清零 417/611） | C+O2/C+H2O/C+CO2/C+H2 的反应焓（含固相） |
| `rheat` | **净放热**，cal/g dmmf：`rheat = rheat1+rheat2+rheat3 − rhts`（第 772 行） | 772 | **是**（每气室清零 869） | `reaction_heat_release_w()` 的返回值 ÷ 煤流量 |
| `hl` | 燃烧区传给壁面的放热份额，**%**。`data hl/30./`（第 30 行，注释 `combustion heat percent transfered to the wall`） | 30 | **否** | `FORTRAN_HL_PERCENT = 30` |
| `u0` | 气-壁总换热系数，**cal/(s·cm²·K)**，`u0 = 3.4e-3`（第 214 行） | 214 | **否** | `FORTRAN_U0_CAL_PER_S_CM2_K` |
| `tw` | 壁温，**K**：`tw = 2100. − 600.*hs/height`（第 348、777 行）。第 216 行的 `tw = 1700.` 只是**初始化占位**，两条 hloss 分支都会覆盖它 | 216（占位） | **是**（348/777 覆盖） | `gasification_zone_heat_loss_w()` 内的 `tw_K` |
| `tg` | 气室气相温度，**K**（待解未知量；由 Wegstein 迭代收敛，第 218–219、800 行） | 218 | **是**（迭代更新） | 单元 `current.T` |
| `poxyin` | 气室**入口** O2 分压，atm（第 141、147 行算出，每气室由 `poxyin = poxy` 更新，第 882 行）。`poxyin > 0.05` 是燃烧区判据 | 147 | **是**（每气室 882） | 入口 pO2（`Cell._inlet_pO2_is_combustion`） |
| `deltim` | 子步时长，s，由 `vsolid` 回填（`time = delh/us`，第 987 行） | 261 | **是**（每子步） | 不进入本轮的 hloss 式子 |



**两条 hloss 分支的被乘数不同**：第 346 行用 `rheat1`（挥发分项），第 780 行用 `rheat`（净放热）。卡第 1.1 节写的是 `|rheat|`，对应主循环 515→737 那条，本轮按此转写。

**`nc·delh` 与 `hs` 的映射**：`delh` 是子步长度，一个气室走 `nc` 个子步，故 `nc·delh` 就是一个气室的轴向长度；Fortran 里它随气室变化（第一个 75 cm，其后 1.5 或 3 cm），Python 一个单元就是一个气室，故 `nc·delh → dz`、`hs → z + dz/2`。`dt`、`height` 按卡取 Fortran 第 32 行的 152.4 cm / 330 cm，不取 Python 几何。

**JAX 路径**：`GasifierSystem._solve_jax_v4` 在 `HeatLossModel='fortran_wall'` 时直接抛 `ValueError`（不静默回退到 `hhv_percent`）；`GasificationRateFactor != 1` 同样抛错，因为 JAX 的 batch 残差里没有这两项的对应实现。两条都在 `unit_checks.csv` 的 `jax_path_guard` 组里验证（`raises=True`）。

## 2. 第 1.2 节：单元检查

判定项 **48**，通过 **48** → 整体**通过**。

| 组 | 上下文 | 指标 | 手算 | 代码 | 相对差 | 通过 |
|---|---|---|---:|---:|---:|---|
| single_cell_combustion | 合成值（rheat = 1.2345678 MW） | `hloss_W` | 370370.340000 | 370370.340000 | 0.000 | True |
| single_cell_combustion | 合成值（rheat = −1.2345678 MW） | `hloss_W_abs_sign` | 370370.340000 | 370370.340000 | 0.000 | True |
| single_cell_gasification | 合成值（tg = 1523.75 K, hs = 187.5 cm） | `hloss_W` | -13223.816310 | -13223.816310 | 0.000 | True |
| single_cell_gasification | 合成值（tg = 1523.75 K, hs = 187.5 cm） | `tw_K` | 1759.090909 | 1759.090909 | 0.000 | True |
| single_cell_gasification | 合成值（tg = 1200.0 K < tw = 1759.09 K） | `hloss_W_negative_preserved` | -31415.343431 | -31415.343431 | 0.000 | True |
| single_cell_combustion | texaco i-1 cell 0（z = 0.0103 m） | `hloss_W` | 219813.879140 | 219813.879140 | 0.000 | True |
| single_cell_combustion | texaco i-1 cell 0（z = 0.0103 m） | `reaction_heat_W` | 732712.930466 | 732712.930466 | 0.000 | True |
| single_cell_gasification | texaco i-1 cell 1（z = 0.0309 m） | `hloss_W` | -5713.151737 | -5713.151737 | 0.000 | True |
| single_cell_gasification | texaco i-1 cell 1（z = 0.0309 m） | `reaction_heat_W` | -92057.021927 | -92057.021927 | 0.000 | True |
| p17a_bitwise_parity | texaco i-1 模式 F N=160 | `T_gas_K` | 1310.869354 | 1310.869354 | 0.000 | True |
| p17a_bitwise_parity | texaco i-1 模式 F N=160 | `X_C_pct` | 96.304975 | 96.304975 | 0.000 | True |
| p17a_bitwise_parity | texaco i-1 模式 F N=160 | `CO_vol_pct` | 55.964609 | 55.964609 | 0.000 | True |
| p17a_bitwise_parity | texaco i-1 模式 F N=160 | `H2_vol_pct` | 38.332415 | 38.332415 | 0.000 | True |
| p17a_bitwise_parity | texaco i-1 模式 F N=160 | `CO2_vol_pct` | 4.337564 | 4.337564 | 0.000 | True |
| p17a_bitwise_parity | texaco i-1 模式 F N=160 | `CH4_vol_pct` | 0.539727 | 0.539727 | 0.000 | True |
| p17a_bitwise_parity | texaco i-2 模式 F N=160 | `T_gas_K` | 1245.130565 | 1245.130565 | 0.000 | True |
| p17a_bitwise_parity | texaco i-2 模式 F N=160 | `X_C_pct` | 85.871273 | 85.871273 | 0.000 | True |
| p17a_bitwise_parity | texaco i-2 模式 F N=160 | `CO_vol_pct` | 51.625673 | 51.625673 | 0.000 | True |
| p17a_bitwise_parity | texaco i-2 模式 F N=160 | `H2_vol_pct` | 40.970702 | 40.970702 | 0.000 | True |
| p17a_bitwise_parity | texaco i-2 模式 F N=160 | `CO2_vol_pct` | 6.437149 | 6.437149 | 0.000 | True |
| p17a_bitwise_parity | texaco i-2 模式 F N=160 | `CH4_vol_pct` | 0.198957 | 0.198957 | 0.000 | True |
| p17a_bitwise_parity | texaco i-5c 模式 F N=160 | `T_gas_K` | 1316.264891 | 1316.264891 | 0.000 | True |
| p17a_bitwise_parity | texaco i-5c 模式 F N=160 | `X_C_pct` | 99.701692 | 99.701692 | 0.000 | True |
| p17a_bitwise_parity | texaco i-5c 模式 F N=160 | `CO_vol_pct` | 52.940509 | 52.940509 | 0.000 | True |
| p17a_bitwise_parity | texaco i-5c 模式 F N=160 | `H2_vol_pct` | 38.863499 | 38.863499 | 0.000 | True |
| p17a_bitwise_parity | texaco i-5c 模式 F N=160 | `CO2_vol_pct` | 7.247408 | 7.247408 | 0.000 | True |
| p17a_bitwise_parity | texaco i-5c 模式 F N=160 | `CH4_vol_pct` | 0.080295 | 0.080295 | 0.000 | True |
| p17a_bitwise_parity | texaco i-10 模式 F N=160 | `T_gas_K` | 1300.347452 | 1300.347452 | 0.000 | True |
| p17a_bitwise_parity | texaco i-10 模式 F N=160 | `X_C_pct` | 95.024168 | 95.024168 | 0.000 | True |
| p17a_bitwise_parity | texaco i-10 模式 F N=160 | `CO_vol_pct` | 55.719993 | 55.719993 | 0.000 | True |
| p17a_bitwise_parity | texaco i-10 模式 F N=160 | `H2_vol_pct` | 38.432920 | 38.432920 | 0.000 | True |
| p17a_bitwise_parity | texaco i-10 模式 F N=160 | `CO2_vol_pct` | 4.889322 | 4.889322 | 0.000 | True |
| p17a_bitwise_parity | texaco i-10 模式 F N=160 | `CH4_vol_pct` | 0.257581 | 0.257581 | 0.000 | True |
| p17a_bitwise_parity | texaco exxon 模式 F N=160 | `T_gas_K` | 1272.816437 | 1272.816437 | 0.000 | True |
| p17a_bitwise_parity | texaco exxon 模式 F N=160 | `X_C_pct` | 98.000932 | 98.000932 | 0.000 | True |
| p17a_bitwise_parity | texaco exxon 模式 F N=160 | `CO_vol_pct` | 52.549537 | 52.549537 | 0.000 | True |
| p17a_bitwise_parity | texaco exxon 模式 F N=160 | `H2_vol_pct` | 37.656010 | 37.656010 | 0.000 | True |
| p17a_bitwise_parity | texaco exxon 模式 F N=160 | `CO2_vol_pct` | 8.303299 | 8.303299 | 0.000 | True |
| p17a_bitwise_parity | texaco exxon 模式 F N=160 | `CH4_vol_pct` | 0.142834 | 0.142834 | 0.000 | True |
| f_gas_neutrality | texaco i-1 模式 F N=160 | `T_gas_K` | 1310.869354 | 1310.869354 | 0.000 | True |
| f_gas_neutrality | texaco i-1 模式 F N=160 | `X_C_pct` | 96.304975 | 96.304975 | 0.000 | True |
| f_gas_neutrality | texaco i-1 模式 F N=160 | `CO_vol_pct` | 55.964609 | 55.964609 | 0.000 | True |
| f_gas_neutrality | texaco i-1 模式 F N=160 | `H2_vol_pct` | 38.332415 | 38.332415 | 0.000 | True |
| f_gas_neutrality | texaco i-1 模式 F N=160 | `CO2_vol_pct` | 4.337564 | 4.337564 | 0.000 | True |
| hhv_independence_fortran_wall | I-1 模式 F′ N=160，HHV_d 30 → 34 | `d_T_out_K` | 0.000000 | 0.000000 | 0.000 | True |
| hhv_independence_fortran_wall | I-1 模式 F′ N=160，HHV_d 30 → 34 | `d_X_C_pp` | 0.000000 | 0.000000 | 0.000 | True |
| jax_path_guard | texaco i-1, N=8, solver_method=jax_jit | `jax_rejects_fortran_wall` | 1.000000 | 1.000000 | 0.000 | True |
| jax_path_guard | texaco i-1, N=8, solver_method=jax_jit | `jax_rejects_gasification_rate_factor` | 1.000000 | 1.000000 | 0.000 | True |

- **`hhv_percent` 逐位复现**：模式 F、N=160 的 5 个干粉算例，出口温度与 X_C/CO/H2/CO2/CH4 全部与 P1.7-A `anchor_outlet.csv` **逐位相同**（判据是 `==`，不是容差）。这是「新开关没污染旧行为」的证据。
- **单格手算**：燃烧区 `0.30·|rheat|`、气化区 `u0·π·dt·dz·(tg−tw)` 与 `tw = 2100−600·hs/330` 都在检查脚本里独立重写一遍算式，相对差 < 1e-9；同时验证 `tg < tw` 时 hloss 为负且**不 clamp**。
- **`HHV_d` 无关性**：`fortran_wall` 下 30→34 的 I-1 出口温度变化**逐位为 0**（判据 < 1e-6 K）。P1.7-A 记录的 −4.55 K 全部来自 `2% × HHV` 热损基准，换成 Fortran 热损后这一依赖本身不存在。

## 3. 第 1.3 节：模式 F′ 的锚点结果

### 3.1 模式 F（`hhv_percent`，P1.7-A）

全部判据通过的算例：**0/5**。

| 算例 | 判据 | Python | Fortran | 绝对差 | 容差 | 通过 |
|---|---|---:|---:|---:|---:|---|
| texaco exxon | CO2_vol_pct | 8.30 | 10.75 | 2.45 | 2 | False |
| texaco exxon | CO_vol_pct | 52.55 | 49.06 | 3.49 | 2 | False |
| texaco exxon | H2_vol_pct | 37.66 | 38.75 | 1.09 | 2 | True |
| texaco exxon | T_gas_K | 1272.82 | 1286.36 | 13.54 | 50 | True |
| texaco exxon | X_C_pp | 98.00 | 92.62 | 5.38 | 2 | False |
| texaco i-1 | CO2_vol_pct | 4.34 | 4.80 | 0.46 | 2 | True |
| texaco i-1 | CO_vol_pct | 55.96 | 54.91 | 1.05 | 2 | True |
| texaco i-1 | H2_vol_pct | 38.33 | 39.32 | 0.99 | 2 | True |
| texaco i-1 | T_gas_K | 1310.87 | 1373.16 | 62.29 | 50 | False |
| texaco i-1 | X_C_pp | 96.30 | 92.37 | 3.93 | 2 | False |
| texaco i-10 | CO2_vol_pct | 4.89 | 6.88 | 1.99 | 2 | True |
| texaco i-10 | CO_vol_pct | 55.72 | 53.27 | 2.45 | 2 | False |
| texaco i-10 | H2_vol_pct | 38.43 | 39.03 | 0.59 | 2 | True |
| texaco i-10 | T_gas_K | 1300.35 | 1308.67 | 8.32 | 50 | True |
| texaco i-10 | X_C_pp | 95.02 | 88.39 | 6.63 | 2 | False |
| texaco i-2 | CO2_vol_pct | 6.44 | 6.42 | 0.02 | 2 | True |
| texaco i-2 | CO_vol_pct | 51.63 | 51.80 | 0.18 | 2 | True |
| texaco i-2 | H2_vol_pct | 40.97 | 40.91 | 0.06 | 2 | True |
| texaco i-2 | T_gas_K | 1245.13 | 1354.45 | 109.32 | 50 | False |
| texaco i-2 | X_C_pp | 85.87 | 84.69 | 1.18 | 2 | True |
| texaco i-5c | CO2_vol_pct | 7.25 | 7.42 | 0.18 | 2 | True |
| texaco i-5c | CO_vol_pct | 52.94 | 51.98 | 0.96 | 2 | True |
| texaco i-5c | H2_vol_pct | 38.86 | 39.62 | 0.75 | 2 | True |
| texaco i-5c | T_gas_K | 1316.26 | 1400.58 | 84.32 | 50 | False |
| texaco i-5c | X_C_pp | 99.70 | 98.52 | 1.18 | 2 | True |

### 3.2 模式 F′（`fortran_wall`）

全部判据通过的算例：**0/5**。

| 算例 | 判据 | Python | Fortran | 绝对差 | 容差 | 通过 |
|---|---|---:|---:|---:|---:|---|
| texaco exxon | CO2_vol_pct | 6.49 | 10.75 | 4.26 | 2 | False |
| texaco exxon | CO_vol_pct | 55.05 | 49.06 | 5.99 | 2 | False |
| texaco exxon | H2_vol_pct | 37.09 | 38.75 | 1.66 | 2 | True |
| texaco exxon | T_gas_K | 1595.17 | 1286.36 | 308.81 | 50 | False |
| texaco exxon | X_C_pp | 100.00 | 92.62 | 7.38 | 2 | False |
| texaco i-1 | CO2_vol_pct | 2.10 | 4.80 | 2.70 | 2 | False |
| texaco i-1 | CO_vol_pct | 58.57 | 54.91 | 3.66 | 2 | False |
| texaco i-1 | H2_vol_pct | 38.53 | 39.32 | 0.79 | 2 | True |
| texaco i-1 | T_gas_K | 1577.78 | 1373.16 | 204.62 | 50 | False |
| texaco i-1 | X_C_pp | 100.00 | 92.37 | 7.63 | 2 | False |
| texaco i-10 | CO2_vol_pct | 2.76 | 6.88 | 4.12 | 2 | False |
| texaco i-10 | CO_vol_pct | 58.26 | 53.27 | 4.99 | 2 | False |
| texaco i-10 | H2_vol_pct | 38.20 | 39.03 | 0.83 | 2 | True |
| texaco i-10 | T_gas_K | 1583.33 | 1308.67 | 274.66 | 50 | False |
| texaco i-10 | X_C_pp | 100.00 | 88.39 | 11.61 | 2 | False |
| texaco i-2 | CO2_vol_pct | 1.80 | 6.42 | 4.62 | 2 | False |
| texaco i-2 | CO_vol_pct | 57.02 | 51.80 | 5.21 | 2 | False |
| texaco i-2 | H2_vol_pct | 40.47 | 40.91 | 0.44 | 2 | True |
| texaco i-2 | T_gas_K | 1576.74 | 1354.45 | 222.29 | 50 | False |
| texaco i-2 | X_C_pp | 100.00 | 84.69 | 15.31 | 2 | False |
| texaco i-5c | CO2_vol_pct | 5.56 | 7.42 | 1.86 | 2 | True |
| texaco i-5c | CO_vol_pct | 55.59 | 51.98 | 3.61 | 2 | False |
| texaco i-5c | H2_vol_pct | 37.97 | 39.62 | 1.65 | 2 | True |
| texaco i-5c | T_gas_K | 1565.67 | 1400.58 | 165.08 | 50 | False |
| texaco i-5c | X_C_pp | 100.00 | 98.52 | 1.48 | 2 | True |

### 3.3 两条热损模型的逐判据对比

| 算例 | 判据 | Fortran | 模式 F | 模式 F′ | F 的绝对差 | F′ 的绝对差 | 变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| texaco i-1 | X_C_pp | 92.37 | 96.30 | 100.00 | 3.93 | 7.63 | +3.70 |
| texaco i-1 | CO_vol_pct | 54.91 | 55.96 | 58.57 | 1.05 | 3.66 | +2.61 |
| texaco i-1 | H2_vol_pct | 39.32 | 38.33 | 38.53 | 0.99 | 0.79 | -0.20 |
| texaco i-1 | CO2_vol_pct | 4.80 | 4.34 | 2.10 | 0.46 | 2.70 | +2.24 |
| texaco i-1 | T_gas_K | 1373.16 | 1310.87 | 1577.78 | 62.29 | 204.62 | +142.34 |
| texaco i-2 | X_C_pp | 84.69 | 85.87 | 100.00 | 1.18 | 15.31 | +14.13 |
| texaco i-2 | CO_vol_pct | 51.80 | 51.63 | 57.02 | 0.18 | 5.21 | +5.04 |
| texaco i-2 | H2_vol_pct | 40.91 | 40.97 | 40.47 | 0.06 | 0.44 | +0.38 |
| texaco i-2 | CO2_vol_pct | 6.42 | 6.44 | 1.80 | 0.02 | 4.62 | +4.60 |
| texaco i-2 | T_gas_K | 1354.45 | 1245.13 | 1576.74 | 109.32 | 222.29 | +112.97 |
| texaco i-5c | X_C_pp | 98.52 | 99.70 | 100.00 | 1.18 | 1.48 | +0.30 |
| texaco i-5c | CO_vol_pct | 51.98 | 52.94 | 55.59 | 0.96 | 3.61 | +2.65 |
| texaco i-5c | H2_vol_pct | 39.62 | 38.86 | 37.97 | 0.75 | 1.65 | +0.89 |
| texaco i-5c | CO2_vol_pct | 7.42 | 7.25 | 5.56 | 0.18 | 1.86 | +1.68 |
| texaco i-5c | T_gas_K | 1400.58 | 1316.26 | 1565.67 | 84.32 | 165.08 | +80.76 |
| texaco i-10 | X_C_pp | 88.39 | 95.02 | 100.00 | 6.63 | 11.61 | +4.98 |
| texaco i-10 | CO_vol_pct | 53.27 | 55.72 | 58.26 | 2.45 | 4.99 | +2.54 |
| texaco i-10 | H2_vol_pct | 39.03 | 38.43 | 38.20 | 0.59 | 0.83 | +0.23 |
| texaco i-10 | CO2_vol_pct | 6.88 | 4.89 | 2.76 | 1.99 | 4.12 | +2.13 |
| texaco i-10 | T_gas_K | 1308.67 | 1300.35 | 1583.33 | 8.32 | 274.66 | +266.34 |
| texaco exxon | X_C_pp | 92.62 | 98.00 | 100.00 | 5.38 | 7.38 | +2.00 |
| texaco exxon | CO_vol_pct | 49.06 | 52.55 | 55.05 | 3.49 | 5.99 | +2.50 |
| texaco exxon | H2_vol_pct | 38.75 | 37.66 | 37.09 | 1.09 | 1.66 | +0.56 |
| texaco exxon | CO2_vol_pct | 10.75 | 8.30 | 6.49 | 2.45 | 4.26 | +1.81 |
| texaco exxon | T_gas_K | 1286.36 | 1272.82 | 1595.17 | 13.54 | 308.81 | +295.27 |

- `T_gas_K`：平均绝对差 55.56 → 235.09（+179.54）；逐算例的符号与量级见上表。
- `X_C_pp`：平均绝对差 3.66 → 8.68（+5.02）；逐算例的符号与量级见上表。
- `CO_vol_pct`：平均绝对差 1.63 → 4.69（+3.07）；逐算例的符号与量级见上表。
- `CO2_vol_pct`：平均绝对差 1.02 → 3.51（+2.49）；逐算例的符号与量级见上表。

### 3.4 为什么 F′ 没有改善温度：燃烧区只覆盖 1 个单元

| 算例 | 燃烧区格数 | 燃烧区 hloss (MW) | 气化区格数 | 气化区 hloss (MW) | 合计 (MW) | T_out (K) |
|---|---:|---:|---:|---:|---:|---:|
| texaco exxon | 2 | 0.368 | 158 | -0.547 | -0.179 | 1595.2 |
| texaco i-1 | 1 | 0.220 | 159 | -0.320 | -0.100 | 1577.8 |
| texaco i-10 | 1 | 0.361 | 159 | -0.542 | -0.181 | 1583.3 |
| texaco i-2 | 1 | 0.214 | 159 | -0.414 | -0.200 | 1576.7 |
| texaco i-5c | 1 | 0.159 | 159 | -0.211 | -0.052 | 1565.7 |

三条事实合起来解释了 F′ 的方向：
1. Python 的 O2 在 **cell 0** 内就降到入口的 1% 以下（P1.5-C 已记录），所以 `poxyin > 0.05` 的判据只命中 **1 个单元**（N=160 时 2.06 cm，N=80 时 4.13 cm）。Fortran 的**第一个气室是 75 cm**（`nc=3`、首个 `delh=25`），同一条 30% 规则作用在 36 倍长的区间上。
2. 气化区的 `tw = 2100 − 600·hs/height` 是一条**被写死的边界条件**，它编码的是 Fortran 自身的降温史（入口 2100 K、出口 1500 K）。在 F′ 的算例里，绝大多数单元的气体温度低于这条剖面（例如 i-1 的 cell 1：tg = 1693 K vs tw = 2092 K），因此 `hloss` 为负——壁面向气体供热。全炉气化区合计是**净供热**，不是净散热。
3. 结果是 F′ 的净壁面热流接近零甚至为**净供热**（燃烧区散热 0.16–0.37 MW，气化区净供热 0.21–0.55 MW，合计 −0.05 ~ −0.20 MW），反应器实际上更接近绝热，温度因此**高于** Fortran、也高于模式 F。这**排除的是公式转写错误**，不构成对成因的识别：48/48 单元检查验证的是某一组假设下的算术。

**结论（第 1 节）**：转写与单元检查都通过，但模式 F′ **没有**缩小与 Fortran 的差距——温度从「偏低」翻成「偏高」，X_C 反而更贴顶。**成因是未验证的假说**（2026-11-05 审阅意见）：首要候选是燃烧区的离散尺度（Fortran 75 cm 气室 vs Python 一个单元），但本轮没有做同状态下 Fortran `rheat` 逐项对齐，也没有做固定其余条件的长度变化受控试验，因此不能说「真正的差别就在离散尺度」。若要确立该机制，需要补这两项；在此之前，「先解决燃烧区长度」只是假说驱动的建议。

### 3.5 与 Fortran 中间气室温度的逐点对比

逐点对比见 `anchor_profiles.csv`（Fortran 每个气室一个打印点，共 27 个匹配点）。温度差的统计：

| 算例 | 点数 | 平均差 (K) | 最小 | 最大 |
|---|---:|---:|---:|---:|
| texaco exxon | 5 | +301.2 | +300.1 | +302.2 |
| texaco i-1 | 8 | +236.2 | +200.8 | +271.5 |
| texaco i-10 | 4 | +269.3 | +268.6 | +270.2 |
| texaco i-2 | 5 | +221.1 | +218.6 | +224.7 |
| texaco i-5c | 5 | +164.6 | +161.4 | +169.3 |

## 4. 第 2 节：九窗二维扫描

网格行数 **180**（9 窗 × 5 × 4 = 180），另有复现行与 P1.7-A 对齐行见 `row_kind` 列。N = 40，煤量主口径（协调-阶段），模型入口载气 = `co2_carrier_measured_nm3_h`。

### 4.1 逐窗 X_C（模型 − 装置主口径，pp）

| 窗口 | f=0.0625/H=1 | f=0.0625/H=2 | f=0.0625/H=4 | f=0.0625/H=6 | f=0.125/H=1 | f=0.125/H=2 | f=0.125/H=4 | f=0.125/H=6 | f=0.25/H=1 | f=0.25/H=2 | f=0.25/H=4 | f=0.25/H=6 | f=0.5/H=1 | f=0.5/H=2 | f=0.5/H=4 | f=0.5/H=6 | f=1/H=1 | f=1/H=2 | f=1/H=4 | f=1/H=6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `plateau_2022-09-28D_00…W003` | 1.50 | 0.88 | -0.46 | -1.78 | 1.92 | 1.36 | 0.17 | -1.01 | 2.17 | 1.72 | 0.68 | -0.37 | 2.36 | 2.03 | 1.10 | 0.18 | 2.47 | 2.27 | 1.46 | 0.65 |
| `shift_morning_handover…W001` | 3.16 | 2.72 | 1.59 | 0.36 | 3.43 | 3.17 | 2.23 | 1.16 | 3.49 | 3.41 | 2.73 | 1.83 | 3.49 | 3.49 | 3.12 | 2.38 | 3.49 | 3.49 | 3.38 | 2.83 |
| `plateau_2022-10-03D_01…W001` | 7.67 | 7.59 | 6.96 | 5.98 | 7.67 | 7.67 | 7.46 | 6.75 | 7.67 | 7.67 | 7.66 | 7.29 | 7.67 | 7.67 | 7.67 | 7.61 | 7.67 | 7.67 | 7.67 | 7.67 |
| `plateau_2022-10-03N_01…W001` | 7.28 | 7.06 | 6.19 | 5.07 | 7.32 | 7.30 | 6.79 | 5.89 | 7.32 | 7.32 | 7.16 | 6.50 | 7.32 | 7.32 | 7.31 | 6.96 | 7.32 | 7.32 | 7.32 | 7.25 |
| `plateau_2022-10-09D_01…W004` | 6.91 | 6.38 | 5.10 | 3.80 | 7.28 | 6.87 | 5.79 | 4.63 | 7.44 | 7.17 | 6.26 | 5.24 | 7.48 | 7.38 | 6.66 | 5.78 | 7.48 | 7.48 | 6.98 | 6.22 |
| `plateau_2022-10-09D_01…W001` | 6.81 | 6.18 | 4.80 | 3.46 | 7.24 | 6.70 | 5.49 | 4.28 | 7.47 | 7.04 | 5.97 | 4.91 | 7.65 | 7.33 | 6.39 | 5.46 | 7.73 | 7.55 | 6.74 | 5.91 |
| `plateau_2022-10-10N_00…W001` | 6.47 | 5.58 | 3.89 | 2.38 | 6.84 | 6.03 | 4.54 | 3.19 | 7.16 | 6.44 | 5.07 | 3.87 | 7.48 | 6.77 | 5.51 | 4.45 | 7.66 | 6.98 | 5.88 | 4.95 |
| `plateau_2022-10-11N_00…W002` | 5.29 | 4.42 | 2.73 | 1.23 | 5.69 | 4.88 | 3.40 | 2.06 | 6.02 | 5.31 | 3.95 | 2.77 | 6.36 | 5.66 | 4.41 | 3.35 | 6.56 | 5.89 | 4.79 | 3.86 |
| `shift_evening_handover…W001` | 5.16 | 4.31 | 2.66 | 1.17 | 5.58 | 4.79 | 3.33 | 2.00 | 5.93 | 5.22 | 3.89 | 2.70 | 6.26 | 5.59 | 4.35 | 3.30 | 6.50 | 5.83 | 4.74 | 3.80 |

装置主口径 X_C 中位 92.33%，模型 X_C 中位（全部网格行）98.51%；X_C ≥ 0.9999 的网格行 31/180。

### 4.2 第 3 节第 4 问：复现行与 P1.7-A 的逐位一致性

卡面要求「复现行 = `f_gas = 1`、`HeatLossPercent = 2`、**旧载气**（`co2_carrier_t_nm3_h`）」，并断言它与 P1.7-A 主口径应逐位一致。**这一条与仓库实际不符**，必须单列：

P1.7-A 的模型入口载气来自冻结切分的 `co2_carrier_volumetric_flow_value`（`revision_evidence_common.model_case_rows`：`max(o2_flow_mean × 0.24, 0)`），既不是 `co2_carrier_t_nm3_h`（运行数据页 T 列），也不是 `FI_1002`。因此按卡面口径（T 列）跑的复现行**不可能**与 P1.7-A 逐位一致。

为了把「新开关没污染旧行为」这件事真正验掉，本轮另跑了一组 `row_kind = parity_p17a` 的行（`f_gas = 1`、`HeatLossPercent = 2`、载气 = 冻结切分原值）。两组数并列如下：

| 窗口 | 载气口径 | 载气 (Nm³/h) | T_out (K) | X_C (%) | CO2 (vol%) | 与 P1.7-A 逐位相同 |
|---|---|---:|---:|---:|---:|---|
| `plateau_2022-09-28D_…W003` | card: T 列 | 9541.1 | 1395.02 | 99.78 | 3.621 | — |
| `plateau_2022-09-28D_…W003` | frozen split | 9014.6 | 1400.97 | 99.79 | 3.366 | True |
| `shift_morning_handov…W001` | card: T 列 | 10315.3 | 1433.80 | 100.00 | 4.008 | — |
| `shift_morning_handov…W001` | frozen split | 9723.7 | 1440.10 | 100.00 | 3.743 | True |
| `plateau_2022-10-03D_…W001` | card: T 列 | 126.6 | 1625.21 | 99.99 | 0.096 | — |
| `plateau_2022-10-03D_…W001` | frozen split | 9878.6 | 1512.26 | 100.00 | 3.830 | True |
| `plateau_2022-10-03N_…W001` | card: T 列 | 124.3 | 1616.36 | 99.84 | 0.027 | — |
| `plateau_2022-10-03N_…W001` | frozen split | 9920.5 | 1490.50 | 100.00 | 3.592 | True |
| `plateau_2022-10-09D_…W004` | card: T 列 | 10331.5 | 1431.53 | 100.00 | 3.021 | — |
| `plateau_2022-10-09D_…W004` | frozen split | 9646.8 | 1439.12 | 100.00 | 2.738 | True |
| `plateau_2022-10-09D_…W001` | card: T 列 | 10540.6 | 1415.04 | 99.81 | 2.977 | — |
| `plateau_2022-10-09D_…W001` | frozen split | 9655.5 | 1425.08 | 99.81 | 2.615 | True |
| `plateau_2022-10-10N_…W001` | card: T 列 | 10258.3 | 1353.61 | 96.17 | 4.211 | — |
| `plateau_2022-10-10N_…W001` | frozen split | 9666.3 | 1358.78 | 96.19 | 3.945 | True |
| `plateau_2022-10-11N_…W002` | card: T 列 | 10533.9 | 1354.14 | 96.45 | 4.250 | — |
| `plateau_2022-10-11N_…W002` | frozen split | 9796.9 | 1360.48 | 96.48 | 3.922 | True |
| `shift_evening_handov…W001` | card: T 列 | 10500.5 | 1357.47 | 97.04 | 4.191 | — |
| `shift_evening_handov…W001` | frozen split | 9786.6 | 1363.68 | 97.07 | 3.875 | True |

**结论**：`parity_p17a` 组与 P1.7-A `baseline.csv` 的 `recon_phase / N=40` 行逐位相同 → 缺省 `HeatLossModel='hhv_percent'` 与 `f_gas=1` 没有污染旧行为。卡面写的「T 列」那一组与 P1.7-A 不同是**输入不同**，不是回归，故第 2 节继续执行。阶段 2 两窗的 `co2_carrier_t_nm3_h` 无效（`t_carrier_status = invalid_phase2_tag_rotation`），复现行对这两窗**不可比**，表里如实保留。

### 4.3 载气口径本身的效应

| 窗口 | 载气 实测 / 冻结切分 (Nm³/h) | ΔX_C (pp) | ΔT_out (K) | ΔCO2 (vol%) |
|---|---|---:|---:|---:|
| `plateau_2022-09-28D_00…W003` | 9541 / 9015 | -0.01 | -5.95 | +0.255 |
| `shift_morning_handover…W001` | 10315 / 9724 | +0.00 | -6.30 | +0.266 |
| `plateau_2022-10-03D_01…W001` | 10588 / 9879 | -0.00 | -7.48 | +0.301 |
| `plateau_2022-10-03N_01…W001` | 10466 / 9921 | -0.00 | -5.75 | +0.229 |
| `plateau_2022-10-09D_01…W004` | 10332 / 9647 | +0.00 | -7.58 | +0.282 |
| `plateau_2022-10-09D_01…W001` | 10541 / 9655 | +0.00 | -10.03 | +0.361 |
| `plateau_2022-10-10N_00…W001` | 10258 / 9666 | -0.02 | -5.17 | +0.266 |
| `plateau_2022-10-11N_00…W002` | 10534 / 9797 | -0.03 | -6.34 | +0.327 |
| `shift_evening_handover…W001` | 10501 / 9787 | -0.02 | -6.21 | +0.316 |

实测载气比冻结切分的 0.24×O2 高约 5%–7%，多进来的 CO2 直接抬高气相的 CO2 分压，把 C+CO2 推快、同时稀释 CO/H2；净效果见上表的 ΔX_C 与 ΔCO2。

### 4.4 `no_root` 与负向 WGS 绑定

`no_root` / 未完成：**0/234**。
负向 WGS 截断绑定长度占比 > 5% 的行：**0/234**。
绑定格数（`wgs_neg_bind_cells`）分布：[1]；整炉绑定长度占比中位 0.0250（范围 0.0250–0.0250），与 P1.7-A 同一量级。

## 5. 预声明读法 R1–R5（跑完不改）

### R1 离开上限（|模型 X_C − 装置主口径 X_C| ≤ 1 pp）

9 窗中有 R1 组合的：**4/9**（`*` 表示线性插值得到的 HLP）。

| 窗口 | 阶段 | 存在 | 组合 |
|---|---|---|---|
| `plateau_2022-09-28D_008…W003` | P1 | True | f=0.03125:HLP=1,HLP=2；f=0.03125*:HLP≈2.121；f=0.0625:HLP=2,HLP=4；f=0.0625*:HLP≈3.311；f=0.125:HLP=4；f=0.125*:HLP≈4.281；f=0.25:HLP=4,HLP=6；f=0.25*:HLP≈5.294；f=0.5:HLP=6；f=1:HLP=6 |
| `shift_morning_handover_2…W001` | P1 | True | f=0.03125:HLP=4,HLP=6；f=0.03125*:HLP≈4.841；f=0.0625:HLP=6 |
| `plateau_2022-10-03D_011…W001` | P2 | False | 无 |
| `plateau_2022-10-03N_013…W001` | P2 | False | 无 |
| `plateau_2022-10-09D_014…W004` | P3 | False | 无 |
| `plateau_2022-10-09D_015…W001` | P3 | False | 无 |
| `plateau_2022-10-10N_006…W001` | P4 | False | 无 |
| `plateau_2022-10-11N_009…W002` | P4 | True | f=0.03125:HLP=6 |
| `shift_evening_handover_2…W001` | P4 | True | f=0.03125:HLP=6；f=0.03125*:HLP≈5.947 |

R1 只用 X_C，而 X_C 与组成口径无关（装置 X_C 由碳平衡给出，模型 X_C 不含组成），所以**两口径的 R1 完全相同**；下面 R2–R4 才分开报。

### R2 是否需要反应器后变换

**analyzer**：逐窗 ξ_min（mol/s）=

| 窗口 | 阶段 | ξ_min | 依据 |
|---|---|---:|---|
| `plateau_2022-09-28D_008…W003` | P1 | 61.8 | f=0.03125/HLP=1.000:ξ=75.5；f=0.03125/HLP=2.000:ξ=66.7；f=0.03125/HLP=2.121*:ξ=65.6；f=0.0625/HLP=2.000:ξ=77.3；f=0.0625/HLP=4.000:ξ=61.8；f=0.0625/HLP=3.311*:ξ=67.2；f=0.125/HLP=4.000:ξ=72.0；f=0.125/HLP=4.281*:ξ=70.2；f=0.25/HLP=4.000:ξ=80.1；f=0.25/HLP=6.000:ξ=68.3；f=0.25/HLP=5.294*:ξ=72.5；f=0.5/HLP=6.000:ξ=75.7… |
| `shift_morning_handover_2…W001` | P1 | 38.9 | f=0.03125/HLP=4.000:ξ=56.4；f=0.03125/HLP=6.000:ξ=38.9；f=0.03125/HLP=4.841*:ξ=49.0；f=0.0625/HLP=6.000:ξ=54.6 |
| `plateau_2022-10-03D_011…W001` | P2 | nan |  |
| `plateau_2022-10-03N_013…W001` | P2 | nan |  |
| `plateau_2022-10-09D_014…W004` | P3 | nan |  |
| `plateau_2022-10-09D_015…W001` | P3 | nan |  |
| `plateau_2022-10-10N_006…W001` | P4 | nan |  |
| `plateau_2022-10-11N_009…W002` | P4 | 12.8 | f=0.03125/HLP=6.000:ξ=12.8 |
| `shift_evening_handover_2…W001` | P4 | 9.0 | f=0.03125/HLP=6.000:ξ=9.0；f=0.03125/HLP=5.947*:ξ=9.5 |

**liu_phase**：逐窗 ξ_min（mol/s）=

| 窗口 | 阶段 | ξ_min | 依据 |
|---|---|---:|---|
| `plateau_2022-09-28D_008…W003` | P1 | 61.3 | f=0.03125/HLP=1.000:ξ=74.9；f=0.03125/HLP=2.000:ξ=66.1；f=0.03125/HLP=2.121*:ξ=65.1；f=0.0625/HLP=2.000:ξ=76.8；f=0.0625/HLP=4.000:ξ=61.3；f=0.0625/HLP=3.311*:ξ=66.6；f=0.125/HLP=4.000:ξ=71.5；f=0.125/HLP=4.281*:ξ=69.6；f=0.25/HLP=4.000:ξ=79.5；f=0.25/HLP=6.000:ξ=67.7；f=0.25/HLP=5.294*:ξ=71.9；f=0.5/HLP=6.000:ξ=75.1… |
| `shift_morning_handover_2…W001` | P1 | 33.9 | f=0.03125/HLP=4.000:ξ=51.2；f=0.03125/HLP=6.000:ξ=33.9；f=0.03125/HLP=4.841*:ξ=44.0；f=0.0625/HLP=6.000:ξ=49.4 |
| `plateau_2022-10-03D_011…W001` | P2 | nan |  |
| `plateau_2022-10-03N_013…W001` | P2 | nan |  |
| `plateau_2022-10-09D_014…W004` | P3 | nan |  |
| `plateau_2022-10-09D_015…W001` | P3 | nan |  |
| `plateau_2022-10-10N_006…W001` | P4 | nan |  |
| `plateau_2022-10-11N_009…W002` | P4 | 11.3 | f=0.03125/HLP=6.000:ξ=11.3 |
| `shift_evening_handover_2…W001` | P4 | 12.5 | f=0.03125/HLP=6.000:ξ=12.5；f=0.03125/HLP=5.947*:ξ=13.0 |

- `analyzer`：n=4/9 窗有 R1 组合；ξ_min 中位 25.8 mol/s；不需要反应器后变换
- `liu_phase`：n=4/9 窗有 R1 组合；ξ_min 中位 23.2 mol/s；不需要反应器后变换

**读法必须带上的限定**：R2 的判据只在「存在 R1 组合」的窗口上有定义（本轮 4/9 窗）。其余窗口在整个搜索池（f_gas ≥ 1/32、HLP ≤ 6%）里模型 X_C 仍高于装置值，ξ 作为一个把 CO2 拉回装置水平的量在那些窗上根本不存在。因此「不需要反应器后变换」这句只在**能到达装置 X_C 的那几窗**上成立；把这句读成「全部 9 窗都不需要」是超出证据的。

### R3 可辨识性（R1 组合里 RSS 最小的点，有限差分）

差分步长：`ln f_gas` ±0.05，`HeatLossPercent` ±0.05。观测按 1 pp、0.5、0.5、0.5 vol% 缩放。

| 窗口 | 口径 | f_gas* | HLP* | cond(J) | cond(JᵀJ) | corr | 可分别辨识 |
|---|---|---:|---:|---:|---:|---:|---|
| `plateau_2022-09-28D_00…W003` | analyzer | 0.25 | 6.000 | 3.12 | 9.74 | 0.700 | 是 |
| `plateau_2022-09-28D_00…W003` | liu_phase | 0.25 | 6.000 | 3.12 | 9.74 | 0.700 | 是 |
| `shift_morning_handover…W001` | analyzer | 0.03125 | 6.000 | 4.93 | 24.27 | 0.798 | 是 |
| `shift_morning_handover…W001` | liu_phase | 0.03125 | 6.000 | 4.93 | 24.27 | 0.798 | 是 |
| `plateau_2022-10-11N_00…W002` | analyzer | 0.03125 | 6.000 | 4.65 | 21.59 | 0.766 | 是 |
| `plateau_2022-10-11N_00…W002` | liu_phase | 0.03125 | 6.000 | 4.65 | 21.59 | 0.766 | 是 |
| `shift_evening_handover…W001` | analyzer | 0.03125 | 6.000 | 4.62 | 21.38 | 0.770 | 是 |
| `shift_evening_handover…W001` | liu_phase | 0.03125 | 6.000 | 4.62 | 21.38 | 0.770 | 是 |

判据：条件数 < 100 **且** |相关| < 0.95 才记为「可分别辨识」。逐窗雅可比元素见 `jacobian.csv`。

R1 无组合、无法求雅可比的窗-口径：10 条（plateau_2022-10-03/analyzer、plateau_2022-10-03/liu_phase、plateau_2022-10-03/analyzer、plateau_2022-10-03/liu_phase、plateau_2022-10-09/analyzer、plateau_2022-10-09/liu_phase、plateau_2022-10-09/analyzer、plateau_2022-10-09/liu_phase、plateau_2022-10-10/analyzer、plateau_2022-10-10/liu_phase）。

### R4 跨阶段一致性

| 口径 | 阶段 | 最优 f_gas 中位 | 说明 |
|---|---|---:|---|
| analyzer | P1 | 0.1406 | n=2；最优 f_gas 中位 0.1406，范围 0.03125–0.25 |
| analyzer | P4 | 0.0312 | n=2；最优 f_gas 中位 0.03125，范围 0.03125–0.03125 |
| analyzer | all | 0.0312 | n=4；中位 0.03125，极差 0.2188 |
| liu_phase | P1 | 0.1406 | n=2；最优 f_gas 中位 0.1406，范围 0.03125–0.25 |
| liu_phase | P4 | 0.0312 | n=2；最优 f_gas 中位 0.03125，范围 0.03125–0.03125 |
| liu_phase | all | 0.0312 | n=4；中位 0.03125，极差 0.2188 |

### R5 CO2 由谁消耗

| 窗口 | 阶段 | 点 | C+CO2 占比 | 细节 |
|---|---|---|---:|---|
| `plateau_2022-09-28D_00…W003` | P1 | f_gas=1 | 94.1% | f_gas=1, HLP=2.000；净 CO2 消耗 60.7 mol/s；C+CO2 412.6；逆 WGS 26.1；正 WGS 生成 111.6；C+CO2 占（C+CO2+逆WGS）94.1% |
| `plateau_2022-09-28D_00…W003` | P1 | R3_optimum | 92.6% | f_gas=0.25, HLP=6.000；净 CO2 消耗 35.5 mol/s；C+CO2 398.9；逆 WGS 31.9；正 WGS 生成 129.1；C+CO2 占（C+CO2+逆WGS）92.6% |
| `shift_morning_handover…W001` | P1 | f_gas=1 | 94.3% | f_gas=1, HLP=2.000；净 CO2 消耗 60.2 mol/s；C+CO2 442.2；逆 WGS 26.6；正 WGS 生成 121.0；C+CO2 占（C+CO2+逆WGS）94.3% |
| `shift_morning_handover…W001` | P1 | R3_optimum | 84.3% | f_gas=0.03125, HLP=6.000；净 CO2 消耗 11.2 mol/s；C+CO2 395.2；逆 WGS 73.6；正 WGS 生成 170.2；C+CO2 占（C+CO2+逆WGS）84.3% |
| `plateau_2022-10-03D_01…W001` | P2 | f_gas=1 | 90.3% | f_gas=1, HLP=2.000；净 CO2 消耗 60.1 mol/s；C+CO2 429.2；逆 WGS 46.2；正 WGS 生成 129.2；C+CO2 占（C+CO2+逆WGS）90.3% |
| `plateau_2022-10-03N_01…W001` | P2 | f_gas=1 | 91.1% | f_gas=1, HLP=2.000；净 CO2 消耗 63.0 mol/s；C+CO2 434.8；逆 WGS 42.7；正 WGS 生成 126.6；C+CO2 占（C+CO2+逆WGS）91.1% |
| `plateau_2022-10-09D_01…W004` | P3 | f_gas=1 | 90.7% | f_gas=1, HLP=2.000；净 CO2 消耗 75.0 mol/s；C+CO2 428.5；逆 WGS 44.1；正 WGS 生成 124.2；C+CO2 占（C+CO2+逆WGS）90.7% |
| `plateau_2022-10-09D_01…W001` | P3 | f_gas=1 | 91.0% | f_gas=1, HLP=2.000；净 CO2 消耗 77.9 mol/s；C+CO2 433.4；逆 WGS 43.1；正 WGS 生成 123.9；C+CO2 占（C+CO2+逆WGS）91.0% |
| `plateau_2022-10-10N_00…W001` | P4 | f_gas=1 | 95.2% | f_gas=1, HLP=2.000；净 CO2 消耗 54.4 mol/s；C+CO2 429.0；逆 WGS 21.7；正 WGS 生成 115.9；C+CO2 占（C+CO2+逆WGS）95.2% |
| `plateau_2022-10-11N_00…W002` | P4 | f_gas=1 | 95.2% | f_gas=1, HLP=2.000；净 CO2 消耗 56.1 mol/s；C+CO2 436.2；逆 WGS 22.0；正 WGS 生成 117.7；C+CO2 占（C+CO2+逆WGS）95.2% |
| `plateau_2022-10-11N_00…W002` | P4 | R3_optimum | 85.9% | f_gas=0.03125, HLP=6.000；净 CO2 消耗 -3.1 mol/s；C+CO2 393.8；逆 WGS 64.6；正 WGS 生成 177.3；C+CO2 占（C+CO2+逆WGS）85.9% |
| `shift_evening_handover…W001` | P4 | f_gas=1 | 95.0% | f_gas=1, HLP=2.000；净 CO2 消耗 56.9 mol/s；C+CO2 436.7；逆 WGS 22.9；正 WGS 生成 119.1；C+CO2 占（C+CO2+逆WGS）95.0% |
| `shift_evening_handover…W001` | P4 | R3_optimum | 85.6% | f_gas=0.03125, HLP=6.000；净 CO2 消耗 -2.1 mol/s；C+CO2 393.3；逆 WGS 66.4；正 WGS 生成 178.5；C+CO2 占（C+CO2+逆WGS）85.6% |

C+CO2 占比 > 50% 的窗-点：**13/13**，中位 91.1%。按卡面要求**单列**：C+H2O 与 C+CO2 共用指前因子可能需要拆开，供论文侧决定是否在下一轮增加 `f_CO2` 维度；本轮不拆。

## 6. 与卡面不一致 / 需要统筹方裁决的地方

1. **卡面第 2.1 节的「旧载气 = 运行数据页 T 列」与 P1.7-A 的实现不符。** P1.7-A 的模型入口载气是冻结切分的 `max(0.24 × o2_flow_mean, 0)`，与 T 列（`co2_carrier_t_nm3_h`）相差 5%–7%。因此卡面「复现行应与 P1.7-A 逐位一致」按字面执行必然不成立。本轮按「两行都跑」处理：卡面口径的复现行照跑并如实标注，另加一组 `parity_p17a`（用冻结切分原值）**确实逐位复现** P1.7-A，以此满足第 3 节第 4 问真正要验的东西（新开关没污染旧行为）。**这一处需要统筹方确认**：以后「旧载气」到底指 T 列、J 列还是 0.24×O2。
2. **卡面第 2.2 节的「载气 CO2 净去向」与 E10 的模型侧数字口径不同。** E10 说「模型消耗 43–61%」，那是 P1.7-A（2% HHV 热损、冻结切分载气）下的结果；本轮换了载气口径与热损扫描维度，模型侧消耗比例的数值会变，逐窗见 `scan.csv` 的 `model_co2_consumption_frac`，不能直接与 E10 的区间对照。
3. **燃烧区判据在两套代码里的有效长度差 36 倍**（Fortran 首个气室 75 cm vs Python 一个单元 2.06 cm）。卡面要求「按 `poxyin > 0.05` 逐字转写」，照做；但由此产生的能量模型差异不是公式问题，第 3.4 节已量化。这一点超出卡面预期（卡面说「在这一项对齐之前 Python 继承 Wen 1979 尚未成立」，而实际结果表明即使换了公式，锚点仍不对齐）。
4. **`hhv_percent` 的逐位复现只覆盖模式 F 的 5 个干粉算例**（卡面原文如此）；模式 M 与两个浆态算例未做逐位比对。第 2 节的 `parity_p17a` 组另外覆盖了 9 窗基线。
5. **R2 的「9 窗中位」在 R1 无解的窗上不参与统计**（本轮有窗缺 R1 组合，见 R1 表），因此中位数只代表有解的窗；已在 R2 明细里逐窗列出 `exists`。

## 7. 未做的部分

- 本轮**没有**把燃烧区的离散尺度对齐（Fortran 75 cm 气室）；卡未授权改燃烧区判据，只按 `poxyin > 0.05` 逐字转写。第 1 节的结论因此是「转写对了、模型没对齐」。
- 本轮**没有**为 `f_gas` 与 `HeatLossPercent` 之外再加维度（`f_CO2` 未拆，见 R5）。
- JAX 路径未实现 `fortran_wall`，遇到即报错；未做 JAX 与 NumPy 的 Fortran 热损并行验证。
- 组成两口径只用来判读残差；模型求解本身只用一个组成口径（装置组成不进入模型）。
