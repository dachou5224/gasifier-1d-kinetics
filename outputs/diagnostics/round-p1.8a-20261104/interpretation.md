# P1.8-A 单元检查（unit stage）

整体：**通过**。

| 组 | 上下文 | 指标 | 手算 | 代码 | 相对差 | 通过 |
|---|---|---|---:|---:|---:|---|
| single_cell_combustion | 合成值（rheat = 1.2345678 MW） | `hloss_W` | 370370.340000 | 370370.340000 | 0.000 | True |
| single_cell_combustion | 合成值（rheat = −1.2345678 MW） | `hloss_W_abs_sign` | 370370.340000 | 370370.340000 | 0.000 | True |
| single_cell_gasification | 合成值（tg = 1523.75 K, hs = 187.5 cm） | `hloss_W` | -13223.816310 | -13223.816310 | 0.000 | True |
| single_cell_gasification | 合成值（tg = 1523.75 K, hs = 187.5 cm） | `tw_K` | 1759.090909 | 1759.090909 | 0.000 | True |
| single_cell_gasification | 合成值（tg = 1200.0 K < tw = 1759.09 K） | `hloss_W_negative_preserved` | -31415.343431 | -31415.343431 | 0.000 | True |
| single_cell_combustion | texaco i-1 cell 0（z = 0.0103 m） | `hloss_W` | 220643.651922 | 220643.651922 | 0.000 | True |
| single_cell_combustion | texaco i-1 cell 0（z = 0.0103 m） | `reaction_heat_W` | 735478.839740 | 735478.839740 | 0.000 | True |
| single_cell_gasification | texaco i-1 cell 1（z = 0.0309 m） | `hloss_W` | -5728.044124 | -5728.044124 | 0.000 | True |
| single_cell_gasification | texaco i-1 cell 1（z = 0.0309 m） | `reaction_heat_W` | -89939.756481 | -89939.756481 | 0.000 | True |
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
