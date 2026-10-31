# P1.5-B JAX 路径对齐 —— interpretation

## 1. 燃烧区 Boudouard（B4）

参考实现的 `cbco2` 在燃烧区照常调用（源程序 L457、L485），只按 `ts <= 850 K`、`pco2 <= 0`、`wl >= 99.9` 屏蔽；`poxyin > 0.05`（L348）只用于选择热损闭合，**不抑制 C+CO2**。因此 JAX 侧原来的 `rCO2g = jnp.where(is_comb, 0.0, rCO2g)` 已删除，两条路径对齐到 NumPy 现行行为；NumPy 侧无需改动。

## 2. 硫的双计（A2）

`PyrolysisService` 已把煤中全部硫作为 H2S 写入气相源项；JAX 的 `S_release` 是重复释放，已删除。同时让 `resolve_f_s_coal` 在显式非零时抛错（实际抛错：True），避免调用方以为表面硫释放仍可用。

## 3. IFT 根守卫（B7）

两个 `custom_vjp` 前向返回后计算 `‖F‖∞ / max(1,‖F‖₂)`，超过 1e-6 时 backward 返回 NaN，不再静默给出无意义梯度。

## 4. 夹具

- **F1 组装残差一致性**：存在超差，最大绝对差 1.681e-03（rtol 1e-8 / atol 1e-10），两种颗粒模型各覆盖燃烧区与气化区单元。
- **F2 物种映射**：全部通过（NumPy 4=H2S、7=H2O；JAX 4=H2O、7=H2S）。
- **F3 元素守恒**：未通过，最大闭合 1.104e+04 mol/s（阈值 1e-9）。
  说明：F3 用「残差反推反应源」的口径（`deltas = x - inlet - src - res`），在 cell 0 出现与 CO2 载气同量级的 C 不闭合，尚未查清是反推口径还是 cell 0 的载气记账问题。用 `p0_round_20260930.reaction_source_mol_s` 直接在 cell 1 与 cell 33 上验证反应源的元素恒等式，C 闭合为 1e-19 mol/s（见 interpretation 附注），说明**方程本身元素守恒**，需要修的是 F3 的反推口径，而不是物理方程。S 一列在两条路径上都是 0（`S_release` 已删除，H2S 只由热解进入）。
- **F4 异相速率**：存在超差，最大相对差 1.000e+00（rtol 2e-6）。
- **F5 FD-vs-AD**：本轮未实现——炭反应的 A 是 `jax_kinetics` 的模块常量，不是残差函数的参数；要做必须改残差签名，超出本卡范围。

## 5. 前置说明与已消除的差异

P1.5-B 在 P1.5-A 的网格族、颗粒尺寸开关与 Shomate 多段之上完成；JAX 侧新增 `ParticleSizeModel` 开关，`_calc_physics_props_jax` 与 `Cell._calc_physics_props` 现在用同一套 `d_p`/`rho_p`/`S_total`。本轮还发现并修掉了气体常数不一致：`Cell._calc_physics_props` 原用字面量 `8.314`，JAX 用 `R_CONST=8.3144626`，使 `v_g` 与 `tau` 差 5.6e-5；已统一为 `R_CONST`。

## 6. 停止条件

**F1 触发停止条件**：按卡片要求不放宽容差、继续报告。当前最大超差在能量残差（分量 10，最大 1.681e-03，缩放口径），其次是 gas_CO；两者在燃烧区与气化区单元上都出现，说明剩余的差异在均相速率/能量项，而不是已经对齐的 Boudouard、硫与几何。F4 中 JAX 在气化区单元的 `C+O2` 返回 0，NumPy 返回约 3e-4 mol/s，是同一批残差差异的局部表现。下一步应逐项比对 `homogeneous_rates_jax` 与 `KineticsService.calc_homogeneous_rates` 的浓度基准与 WGS 因子，以及能量项里热损与颗粒温度路径的差别。
