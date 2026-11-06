# P1.11-A 烧嘴近场全混流区（固定物理长度 L_wsr）— 判读

Status: `EXECUTED — model repo`

依据：`manuscript/model-side-handoff-p1.11a-20261106.md`；上游 `manuscript/model-evidence-verification-20261106-p110.md` H2–H4。

结构：首格长度恰为 `L_wsr`（全混流区，体积 = A·L_wsr），其后 `[L_wsr, L]` 为等长 `dz_down` 的串联格。全部 O2 进入首格（`OxygenMixingModel` 冻结为 `instant`）。未改速率式、限幅、燃烧区判据或求解器容差。

# P1.11-A 第 1 节：现行单格模型的 dz 依赖性清单

判定标准：本模型每一格的残差都是 `n_out = n_in + src − Σrate`，其中 `Σrate` 用**该格出口状态**求值（CSTR 定义），容量项为体积 `V = A·dz` 或持料量 `ṁ·τ`。因此「全混流」在此指**格内无轴向梯度解**。若任一依赖项隐含格内轴向积分，则卡面第 1 节的停止条件触发，必须停止并报告。

### 逐项清单

| # | 依赖 dz 的项 | 代码位置 | 形式 | 格长为数米时是否仍是全混流 |
|---|---|---|---|---|
| 1 | 气相反应速率容量 | `kinetics_service.calc_homogeneous_rates`（`r·volume`） | `rate = k·C_i·C_j·V` | **是**。`C_i = y_i·P/(R·T)` 由摩尔分数给出，与 V 无关；乘 V 就是「单位体积速率 × 体积」，CSTR 的标准容量形式。大 V 使转化更完全，不引入格内梯度。 |
| 2 | 异相反应持料量 | `cell._calc_physics_props`（`m_hold = W_s·τ`） | `m_hold = W_s·dz/v_g` | **是**。`W_s·τ` 正是 CSTR 的持料量（流入速率 × 平均停留时间）。 |
| 3 | 颗粒表面积 | 同上（`S_total = 6·m_hold/(ρ_p·d_p)`） | `S_total ∝ m_hold ∝ dz` | **是**。表面积由持料量派生，随 dz 线性变化，与全混流一致。 |
| 4 | 颗粒驻留时间 | `cell._calc_particle_temperature`（`deltim = τ/nc`） | 时间步推进 `nc` 次，总时长 = τ | **是（但存在近似）**。CSTR 中 τ 是平均停留时间，用它对颗粒做瞬态传热是自洽的。见下方「必须声明的近似」。 |
| 5 | 热损分配 | `cell.hhv_percent_hloss`（`dz/L_norm × clip((T/1800)^4)`） | `Q_loss ∝ dz` | **是**。散热正比于体积（`dz/L` 是体积份额），CSTR 下用出口温度作为格温度是标准假设。 |
| 6 | 限幅可用量 | `cell._calc_balances`（`avail = inlet + gas_src`） | 与 dz 无关 | **是**。限幅以「该格进料」为上限，与格长无关；这正是全混流区内反应物被出口条件封顶的行为。 |
| 7 | 热解源 | `source_terms.PyrolysisSource`（`target_cell_idx=0`） | 只在首格，总量与 dz 无关 | **是**。全混流区内热解瞬间完成，与「整格均匀」一致。 |
| 8 | 蒸发源 | `source_terms.EvaporationSource._fraction_in_cell` | `[z−dz/2, z+dz/2]` 与 `L_evap` 的重叠 | **是**。这是几何区间取份额，不是轴向积分；默认 `L_evap = 1e-6 m` 时全部落在首格，与全混流一致。若 `L_evap` 与格长可比，则该段内隐含均匀分布假设——本卡不涉及。 |
| 9 | 碳表面损耗 | `cell._calc_balances`（`W_surf_loss = Σrates·0.012011`） | 与速率同源 | **是**。由反应量派生，不额外引入 dz。 |
| 10 | 燃烧区判据 | `cell._inlet_pO2_is_combustion` | 与 dz 无关 | **是**。用格入口 pO2 判定，阈值不变（卡面冻结项）。 |

### 结论：没有一项触发停止条件

清单中没有任何一项在格内做轴向积分。全部依赖项分两类：**容量项**（体积 `V`、持料量 `ṁτ`、表面积 `∝m_hold`、散热 `∝dz`）与**与 dz 无关项**（限幅、热解、判据）。容量项随 `dz` 线性变化，这正是全混流区的定义性质，而非格内梯度求解。

**因此卡面第 1 节的停止条件不触发。**

### 必须声明的三点

### 1. 下游「平推流」实际是串联全混流（tanks-in-series）

同一 `Cell` 类对每一格都按 CSTR 求速率，所以 `[L_wsr, L]` 段严格说是**串联 CSTR**，只有 `dz_down → 0` 才逼近平推流。这不是「非全混流处理」（每格都是全混流），但卡面「平推流格」的措辞应理解为串联近似。第 3 节的 W1 正是在检验这个近似是否已收敛。

### 2. 大格用单一 τ 会高估颗粒-气体热弛豫

`_calc_particle_temperature` 以 `τ = dz/v_g` 推进颗粒温度。全混流区内停留时间分布实际是宽的（有短路与回流），用单一平均 τ 会让颗粒温度比真实更接近气体温度。`L_wsr` 越大该近似越强——下表给出实测 τ 与持料量，供判读时参考。

### 3. 大格内温度被视为均匀

热损权重 `clip((T/1800)^4)` 与颗粒温度都用该格出口温度，即假设格内温度均匀。对 `L_wsr` = 数米的区域，真实温度场很可能不均匀；这是全混流假设的固有代价，不是实现缺陷。

### 实测：全混流区的尺寸、驻留时间与持料量

窗口取首个参考窗，`dz_down` = 0.2575 m，`HLP` = 1%。

| L_wsr | N_cells | tau_wsr_s | m_hold_wsr_kg | S_total_m2 | V_wsr_m3 | T_wsr_K |
|---|---|---|---|---|---|---|
| 0.2575 | 40.0000 | 0.4548 | 3.9557 | 295.6169 | 2.9203 | 2708.8375 |
| 0.5000 | 40.0000 | 0.8751 | 7.4888 | 568.8157 | 5.6706 | 2718.9943 |
| 1.0000 | 38.0000 | 1.7494 | 14.8194 | 1137.1168 | 11.3411 | 2711.2928 |
| 2.0000 | 34.0000 | 3.5340 | 29.7712 | 2297.1593 | 22.6823 | 2679.4746 |
| 4.0000 | 26.0000 | 7.2268 | 60.7042 | 4697.4853 | 45.3646 | 2618.2761 |

`τ` 随 `L_wsr` 线性增长而 `v_g` 变化不大，因此大 `L_wsr` 的 `τ` 可达秒级，上述第 2 点近似随之增强。这一点在 W6 的报告里会与限幅绑定比例一起给出。


## 2. 单元检查（卡第 2 节）

| check | context | metric | value | tolerance | pass |
|---|---|---|---|---|---|
| 1.000000 | L_wsr = dz_down = 0.2575 m | mesh_bitwise_equal_uniform_N40 | 1.000000 | 0.000000 | 1.000000 |
| 1.000000 | L_wsr = dz_down = 0.2575，9 窗出口量 | max_abs_outlet_diff_vs_uniform_N40 | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=0.2575, dz_down=0.2575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=0.2575, dz_down=0.103 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=0.2575, dz_down=0.0515 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=0.2575, dz_down=0.02575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=0.5, dz_down=0.2575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=0.5, dz_down=0.103 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=0.5, dz_down=0.0515 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=0.5, dz_down=0.02575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=1.0, dz_down=0.2575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=1.0, dz_down=0.103 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=1.0, dz_down=0.0515 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=1.0, dz_down=0.02575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=2.0, dz_down=0.2575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=2.0, dz_down=0.103 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=2.0, dz_down=0.0515 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=2.0, dz_down=0.02575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=4.0, dz_down=0.2575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=4.0, dz_down=0.103 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=4.0, dz_down=0.0515 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | L_wsr=4.0, dz_down=0.02575 | sum_dz_rel_error | 0.000000 | 0.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.107225 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.230082 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.799572 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.938190 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.267755 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 1.000000 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.256379 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.078325 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.166000 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P1__huayi_sy） | O_closure_rel_over_baseline_max | 0.596331 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.629028 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.205668 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.172616 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.360113 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.464497 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.505894 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.114125 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.161179 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.099185 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P2__huayi_sy） | O_closure_rel_over_baseline_max | 0.681788 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.186661 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.763326 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.649484 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.399518 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.551612 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.772263 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.537171 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.309595 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 0.112085 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P3__huayi_sy） | O_closure_rel_over_baseline_max | 1.414373 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.438868 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.372193 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 1.866537 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.974746 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.346520 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.569121 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 2.998084 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 4.339310 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 2.175141 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.098128 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.2575（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.223260 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=0.5（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.198493 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=1.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.977615 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=2.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 3.744516 | 10.000000 | 1.000000 |
| 2.000000 | 元素守恒 O，L_wsr=4.0（R52-C-75__P4__huayi_sy） | O_closure_rel_over_baseline_max | 0.394423 | 10.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P1__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P2__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P3__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.2575（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=0.5（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=1.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=2.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |
| 3.000000 | 多解，L_wsr=4.0（R52-C-75__P4__huayi_sy） | n_flame_roots_above_1500K | 1.000000 | 1.000000 | 1.000000 |

- 全部通过：**True**（112 项）
- 多解：共记录 45 条根，其中 >1500 K 的火焰根 45 条；(窗口, `L_wsr`) 组合中火焰根数 > 1 的有 0 个

## 3. W1 下游收敛（卡第 3 节）

| L_wsr | n_converged | n_windows | admitted | max_abs_dX_C_pp | max_abs_dT_K | max_abs_dCO2_vol | wsr_state_all_invariant |
|---|---|---|---|---|---|---|---|
| 0.258 | 9.000 | 9.000 | 1.000 | 0.022 | 0.404 | 0.005 | 1.000 |
| 0.500 | 9.000 | 9.000 | 1.000 | 0.017 | 0.341 | 0.005 | 1.000 |
| 1.000 | 9.000 | 9.000 | 1.000 | 0.016 | 0.306 | 0.005 | 1.000 |
| 2.000 | 9.000 | 9.000 | 1.000 | 0.017 | 0.237 | 0.005 | 1.000 |
| 4.000 | 9.000 | 9.000 | 1.000 | 0.018 | 0.359 | 0.007 | 1.000 |
| 6.000 | 9.000 | 9.000 | 1.000 | 0.025 | 0.605 | 0.011 | 1.000 |

- 准入档数 **6**；准入 `L_wsr`：[0.2575, 0.5, 1.0, 2.0, 4.0, 6.0]
- 全混区状态对 `dz_down` 逐位不变：**True**（卡面要求的结构性质：`L_wsr` 是物理参数，不随网格加密改变）

## 4. 杠杆与到达（W2–W6）

### W2 杠杆（准入档最小 vs 最大 `L_wsr`）

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | dX_C_pp | -3.669 | nan | None | L_wsr 0.2575→6，9 窗中位；范围 -4.748…-0.6394 |
| nan | dT_out_K | -70.396 | nan | None | L_wsr 0.2575→6，9 窗中位；范围 -143.9…-31.96 |
| nan | dCO2_vol_pct | 1.881 | nan | None | L_wsr 0.2575→6，9 窗中位；范围 0.7793…2.354 |
| nan | dT_wsr_K | -137.677 | nan | None | L_wsr 0.2575→6，9 窗中位；范围 -146.2…-131.5 |

### W3 方向

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | sign::R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003 | -1.000 | nan | None | ΔX_C=-3.669 pp（降），ΔCO2=+1.968 vol%（升） |
| nan | sign::R52-C-75__P1__huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-28__W001 | -1.000 | nan | None | ΔX_C=-2.234 pp（降），ΔCO2=+1.426 vol%（升） |
| nan | sign::R52-C-75__P2__huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03D_011__W001 | -1.000 | nan | None | ΔX_C=-0.639 pp（降），ΔCO2=+0.779 vol%（升） |
| nan | sign::R52-C-75__P2__huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03N_013__W001 | -1.000 | nan | None | ΔX_C=-1.253 pp（降），ΔCO2=+1.011 vol%（升） |
| nan | sign::R52-C-75__P3__huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_014__W004 | -1.000 | nan | None | ΔX_C=-3.061 pp（降），ΔCO2=+1.654 vol%（升） |
| nan | sign::R52-C-75__P3__huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_015__W001 | -1.000 | nan | None | ΔX_C=-3.690 pp（降），ΔCO2=+1.881 vol%（升） |
| nan | sign::R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-10N_006__W001 | -1.000 | nan | None | ΔX_C=-4.748 pp（降），ΔCO2=+2.354 vol%（升） |
| nan | sign::R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11N_009__W002 | -1.000 | nan | None | ΔX_C=-4.709 pp（降），ΔCO2=+2.338 vol%（升） |
| nan | sign::R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-11__W001 | -1.000 | nan | None | ΔX_C=-4.680 pp（降），ΔCO2=+2.319 vol%（升） |
| nan | n_windows_X_C_decreases | 9.000 | 9.000 | None | X_C 随 L_wsr 增大而下降 9 窗 / 上升 0 窗 |

### W4 到达（±1 pp，含线性插值）

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| 0.258 | X_C_dev_median_pp::HLP1 | 7.318 | 1.000 | 0.000 | 9 窗中位偏差 +7.32 pp；|偏差|≤1 pp 的窗数 0/9；边界档 |
| 0.500 | X_C_dev_median_pp::HLP1 | 7.318 | 1.000 | 0.000 | 9 窗中位偏差 +7.32 pp；|偏差|≤1 pp 的窗数 0/9；内部档 |
| 1.000 | X_C_dev_median_pp::HLP1 | 6.977 | 1.000 | 0.000 | 9 窗中位偏差 +6.98 pp；|偏差|≤1 pp 的窗数 0/9；内部档 |
| 2.000 | X_C_dev_median_pp::HLP1 | 6.155 | 1.000 | 0.000 | 9 窗中位偏差 +6.15 pp；|偏差|≤1 pp 的窗数 0/9；内部档 |
| 4.000 | X_C_dev_median_pp::HLP1 | 4.575 | 1.000 | 1.000 | 9 窗中位偏差 +4.58 pp；|偏差|≤1 pp 的窗数 1/9；边界档 |
| 6.000 | X_C_dev_median_pp::HLP1 | 2.953 | 1.000 | 0.000 | 9 窗中位偏差 +2.95 pp；|偏差|≤1 pp 的窗数 0/9；内部档 |
| 0.258 | X_C_dev_median_pp::HLP2 | 7.086 | 1.000 | 0.000 | 9 窗中位偏差 +7.09 pp；|偏差|≤1 pp 的窗数 0/9；边界档 |
| 0.500 | X_C_dev_median_pp::HLP2 | 6.627 | 1.000 | 0.000 | 9 窗中位偏差 +6.63 pp；|偏差|≤1 pp 的窗数 0/9；内部档 |
| 1.000 | X_C_dev_median_pp::HLP2 | 5.734 | 1.000 | 0.000 | 9 窗中位偏差 +5.73 pp；|偏差|≤1 pp 的窗数 0/9；内部档 |
| 2.000 | X_C_dev_median_pp::HLP2 | 4.215 | 1.000 | 1.000 | 9 窗中位偏差 +4.22 pp；|偏差|≤1 pp 的窗数 1/9；内部档 |
| 4.000 | X_C_dev_median_pp::HLP2 | 1.636 | 1.000 | 1.000 | 9 窗中位偏差 +1.64 pp；|偏差|≤1 pp 的窗数 3/9；边界档 |
| 6.000 | X_C_dev_median_pp::HLP2 | -0.596 | 1.000 | 1.000 | 9 窗中位偏差 -0.60 pp；|偏差|≤1 pp 的窗数 3/9；内部档 |
| 4.467 | interpolated_L_wsr_at_zero_dev::HLP1::R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003 | 4.467 | nan | 1.000 | 窗口 2-09-28D_008__W003 偏差在 4→6 m 之间由 +0.36→-1.19 pp 穿过 0；线性插值 L_wsr*=4.467 m（插值点，非实测档） |
| 2.040 | interpolated_L_wsr_at_zero_dev::HLP2::R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003 | 2.040 | nan | 1.000 | 窗口 2-09-28D_008__W003 偏差在 2→4 m 之间由 +0.05→-2.44 pp 穿过 0；线性插值 L_wsr*=2.040 m（插值点，非实测档） |
| 4.122 | interpolated_L_wsr_at_zero_dev::HLP2::R52-C-75__P1__huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-28__W001 | 4.122 | nan | 1.000 | 窗口 r_2022-09-28__W001 偏差在 4→6 m 之间由 +0.14→-2.09 pp 穿过 0；线性插值 L_wsr*=4.122 m（插值点，非实测档） |
| 5.466 | interpolated_L_wsr_at_zero_dev::HLP2::R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-10N_006__W001 | 5.466 | nan | 1.000 | 窗口 2-10-10N_006__W001 偏差在 4→6 m 之间由 +1.64→-0.60 pp 穿过 0；线性插值 L_wsr*=5.466 m（插值点，非实测档） |
| 4.539 | interpolated_L_wsr_at_zero_dev::HLP2::R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11N_009__W002 | 4.539 | nan | 1.000 | 窗口 2-10-11N_009__W002 偏差在 4→6 m 之间由 +0.60→-1.63 pp 穿过 0；线性插值 L_wsr*=4.539 m（插值点，非实测档） |
| 4.526 | interpolated_L_wsr_at_zero_dev::HLP2::R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-11__W001 | 4.526 | nan | 1.000 | 窗口 r_2022-10-11__W001 偏差在 4→6 m 之间由 +0.59→-1.64 pp 穿过 0；线性插值 L_wsr*=4.526 m（插值点，非实测档） |

### W5 动力学杠杆

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | P1.8A_uniform_median_abs_dX_C_pp | 0.494 | nan | None | P1.8-A 均匀 N=40、HLP=1%，f_gas 1→1/8 的 9 窗中位 |ΔX_C| |
| 0.258 | median_abs_dX_C_pp | 0.538 | nan | None | f_gas 1→1/8，9 窗中位 |
| 0.258 | ratio_vs_P1.8A_uniform | 1.089 | 3.000 | 0.000 | 中位比 ≥3 记为本征动力学恢复杠杆 |
| 0.500 | median_abs_dX_C_pp | 0.668 | nan | None | f_gas 1→1/8，9 窗中位 |
| 0.500 | ratio_vs_P1.8A_uniform | 1.352 | 3.000 | 0.000 | 中位比 ≥3 记为本征动力学恢复杠杆 |
| 1.000 | median_abs_dX_C_pp | 0.860 | nan | None | f_gas 1→1/8，9 窗中位 |
| 1.000 | ratio_vs_P1.8A_uniform | 1.741 | 3.000 | 0.000 | 中位比 ≥3 记为本征动力学恢复杠杆 |
| 2.000 | median_abs_dX_C_pp | 1.069 | nan | None | f_gas 1→1/8，9 窗中位 |
| 2.000 | ratio_vs_P1.8A_uniform | 2.163 | 3.000 | 0.000 | 中位比 ≥3 记为本征动力学恢复杠杆 |
| 4.000 | median_abs_dX_C_pp | 1.388 | nan | None | f_gas 1→1/8，9 窗中位 |
| 4.000 | ratio_vs_P1.8A_uniform | 2.809 | 3.000 | 0.000 | 中位比 ≥3 记为本征动力学恢复杠杆 |
| 6.000 | median_abs_dX_C_pp | 1.791 | nan | None | f_gas 1→1/8，9 窗中位 |
| 6.000 | ratio_vs_P1.8A_uniform | 3.625 | 3.000 | 1.000 | 中位比 ≥3 记为本征动力学恢复杠杆 |

### W6 全混区状态

| L_wsr | metric | value | threshold | pass | note |
|---|---|---|---|---|---|
| nan | T_wsr_min_K | 2515.504 | nan | None |  |
| nan | T_wsr_max_K | 2753.086 | nan | None |  |
| nan | n_rows_with_limit_binding | 54.000 | 54.000 | None | 全混区内有限幅绑定的行数（行数 = 准入档 × 9 窗） |
| nan | n_window_level_with_multiple_roots | 0.000 | 0.000 | None | (窗口, L_wsr) 组合中根数 > 1 的个数，明细见 roots.csv |
| nan | n_window_level_with_2plus_flame_roots | 0.000 | 0.000 | None | 两个以上 >1500 K 的根会按卡第 2 节第 3 条停止该窗 |

## 5. 结论要点与边界

- **W2**：`L_wsr` 从最小到最大，X_C 中位变化 **-3.67 pp**（范围 L_wsr 0.2575→6，9 窗中位；范围 -4.748…-0.6394）。
- **W3**：X_C 随 L_wsr 增大而下降 9 窗 / 上升 0 窗——**方向一致且朝装置值**：`L_wsr` 增大使 X_C 下降、CO2 上升，9 个窗口无一例外。
- **W4**：到达（|偏差| ≤ 1 pp）判定为 **存在**；但只有 `L_wsr` = 4 m 这一档有 1/9 窗落在 ±1 pp 内，其余档 0/9。中位偏差随 `L_wsr` 单调下降（0.2575 m 的 +7.32 pp → 6 m 的 +2.95 pp），6 m 时最小窗已越过装置值（−1.19 pp）。

### W5 的通过只发生在扫描上界，必须按边界值读

- 比值随 `L_wsr` **单调增**：0.2575 m 时 1.09，4 m 时 2.81，6 m 时 **3.62**（阈值 3）。
- 即「本征动力学恢复杠杆」**只在 6 m 成立**，而 6 m 是卡面允许的**唯一追加档、且是扫描上界**。4 m 及以下均不满足。
- 这与 P1.8-A R3 的教训同形：判定阈值在扫描边界上被跨过，意味着真实最优点可能在此之外，或该指标本身已接近失效。**不得**把「6 m 通过」写成「加入全混流区后动力学恢复杠杆」，正确表述是「在本扫描的 6 个档位上，唯一达到阈值的是上界档 6 m」。

### 流程偏差（须记录）

卡面第 4 节要求：*若 W4 最优点落在 4 m 边界，只追加一档 `L_wsr` = 6 m*。本轮实现**在 W4 判定之前**就追加了 6 m（原代码条件只要求 4 m 已被准入）。事后核实：4 m 确实是 W4 的有效最优点（1/9 窗到达、中位偏差最小），**故卡面的追加条件成立**，6 m 属卡面允许的额外一档，判读阈值不受影响。但执行顺序与卡面不符，已在本轮记录；复现时脚本已改为先判 W4 再追加。

## 6. 锚点（只记录，不判通过）

| case_name | L_wsr | N_cells | execution_status | T_wsr_K | T_out_K | X_C_pct | CO2_vol_pct |
|---|---|---|---|---|---|---|---|
| texaco i-1 | 0.020 | 320.000 | completed | 2520.569 | 1310.013 | 96.349 | 4.343 |
| texaco i-1 | 0.100 | 312.000 | completed | 2507.628 | 1307.470 | 96.005 | 4.462 |
| texaco i-1 | 0.250 | 298.000 | completed | 2484.689 | 1303.448 | 95.373 | 4.678 |
| texaco i-1 | 0.500 | 273.000 | completed | 2449.747 | 1298.544 | 94.361 | 5.018 |
| texaco i-1 | 0.750 | 249.000 | completed | 2418.211 | 1295.493 | 93.392 | 5.334 |
| texaco i-2 | 0.020 | 320.000 | completed | 2214.566 | 1243.499 | 85.791 | 6.473 |
| texaco i-2 | 0.100 | 312.000 | completed | 2110.378 | 1240.713 | 88.367 | 6.640 |
| texaco i-2 | 0.250 | 298.000 | completed | 2098.446 | 1240.977 | 87.959 | 6.772 |
| texaco i-2 | 0.500 | 273.000 | completed | 2079.656 | 1241.994 | 87.299 | 6.983 |
| texaco i-2 | 0.750 | 249.000 | completed | 2062.084 | 1243.746 | 86.655 | 7.183 |
| texaco i-5c | 0.020 | 320.000 | completed | 2347.582 | 1315.404 | 99.709 | 7.256 |
| texaco i-5c | 0.100 | 312.000 | completed | 2050.272 | 1314.755 | 99.682 | 7.557 |
| texaco i-5c | 0.250 | 298.000 | completed | 2040.343 | 1311.028 | 99.485 | 7.621 |
| texaco i-5c | 0.500 | 273.000 | completed | 2024.585 | 1306.727 | 99.087 | 7.737 |
| texaco i-5c | 0.750 | 249.000 | completed | 2009.717 | 1304.338 | 98.625 | 7.858 |
| texaco i-10 | 0.020 | 320.000 | completed | 2421.527 | 1299.418 | 95.059 | 4.897 |
| texaco i-10 | 0.100 | 312.000 | completed | 2387.637 | 1298.903 | 95.674 | 5.028 |
| texaco i-10 | 0.250 | 298.000 | completed | 2368.455 | 1296.098 | 95.077 | 5.211 |
| texaco i-10 | 0.500 | 273.000 | completed | 2338.929 | 1292.751 | 94.113 | 5.499 |
| texaco i-10 | 0.750 | 249.000 | completed | 2311.988 | 1290.838 | 93.184 | 5.770 |
| texaco exxon | 0.020 | 320.000 | completed | 2211.403 | 1272.036 | 98.018 | 8.315 |
| texaco exxon | 0.100 | 312.000 | completed | 1881.571 | 1272.852 | 98.053 | 8.964 |
| texaco exxon | 0.250 | 298.000 | completed | 1774.862 | 1273.893 | 97.953 | 9.291 |
| texaco exxon | 0.500 | 273.000 | completed | 1771.425 | 1274.804 | 97.529 | 9.298 |
| texaco exxon | 0.750 | 249.000 | completed | 1764.715 | 1276.550 | 97.126 | 9.417 |

## 7. 复现边界

- 本轮只改网格结构（加入固定物理长度的全混流区），不调动力学常数、不采纳任何 `L_wsr` 值、不改合同。
- W1–W6 阈值在运行前写入脚本常量，运行后未改。
- 下游段是**串联 CSTR**（tanks-in-series），加密使其逼近平推流；W1 检验的是这一近似是否已收敛。
- 锚点用模式 F（`hhv_percent`）+ 同一网格族，只记录。

