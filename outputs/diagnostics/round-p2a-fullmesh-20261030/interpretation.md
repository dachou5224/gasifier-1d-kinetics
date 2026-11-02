# P2-A 全长网格收敛与颗粒尺寸模型对照 —— interpretation

行数：54（预期 54）；配对窗（全部网格都完成）：2/9；no_root 行：14。

## 1. N_star

`N_star = 160`，判据状态 `not_converged_within_scan`（判据：相邻网格 |Δdry_flow_ratio| ≤ 0.01、|ΔX_C| ≤ 0.01、|ΔH2| ≤ 0.2、|ΔCO| ≤ 0.2，且对所有更细网格继续满足）。固定样本（2 个配对窗）给出 N_star = 40。

两个口径不一致的原因：每个 N 上完成的窗口集合不同（2/9 → 6/9 → 8/9 → 7/9 → 8/9），中位数不是同一样本；40→80 的 H2 中位数移动 0.22，恰好越过 0.2 的容差，而这 0.22 主要来自由 8 窗变成 7 窗的构成变化，不是离散误差。因此严格判据记 `not_converged_within_scan`，工作点取最细网格 N=160。

主判据表按卡片口径：每个 N 上全部完成窗口的中位数（样本数随 N 变化，见第 3 列）。

| N | 完成窗数 | X_C | 干气流量比 | CO vol% | H2 vol% |
|---:|---:|---:|---:|---:|---:|
| 10 | 2/9 | 0.9048 | 0.9740 | 68.95 | 23.65 |
| 20 | 6/9 | 0.9376 | 0.9874 | 70.58 | 23.40 |
| 40 | 8/9 | 0.9587 | 0.9923 | 71.79 | 23.05 |
| 80 | 7/9 | 0.9596 | 0.9961 | 71.83 | 23.28 |
| 160 | 8/9 | 0.9630 | 0.9949 | 71.96 | 23.08 |

配对子集（2 窗 × 全部网格，样本固定）中位数：
N=10: X_C 0.9048/干气比 0.9740、N=20: X_C 0.9157/干气比 0.9857、N=40: X_C 0.9211/干气比 0.9917、N=80: X_C 0.9240/干气比 0.9948、N=160: X_C 0.9256/干气比 0.9964。

表观阶与 Richardson 外推（固定配对子集，最细三个网格 40/80/160）：

- `X_C_implied`：表观阶 p = 0.82，Richardson 外推极限 0.9277
- `dry_flow_ratio`：表观阶 p = 0.98，Richardson 外推极限 0.9980
- `CO_vol_pct`：表观阶 p = 0.87，Richardson 外推极限 69.9462
- `H2_vol_pct`：表观阶 p = 4.74，Richardson 外推极限 23.8281

`p` 接近 1 与均匀网格 CSTR 串联的一阶格式一致；H2 的 p 值偏大是因为增量已接近数值噪声。

## 2. N_star 下的出口量

- `X_C_implied`：fortran_constant 0.9630，shrinking_legacy 0.9548
- `dry_flow_ratio`：fortran_constant 0.9949，shrinking_legacy 0.9884
- `CO_vol_pct`：fortran_constant 71.9594，shrinking_legacy 71.7829
- `CO2_vol_pct`：fortran_constant 5.0307，shrinking_legacy 5.5777
- `H2_vol_pct`：fortran_constant 23.0765，shrinking_legacy 22.9883
- 装置中位数：CO 63.27、H2 26.65、CO2 9.27 vol%，干气 134221 Nm³/h
华谊运行数据反算的装置碳转化率本轮未提供，故未列出。

## 3. shrinking_legacy 相对 fortran_constant 的差

- `X_C_implied`：-0.0082（shrinking − fortran）
- `dry_flow_ratio`：-0.0065（shrinking − fortran）
- `CO_vol_pct`：-0.1766（shrinking − fortran）
- `CO2_vol_pct`：+0.5471（shrinking − fortran）
- `H2_vol_pct`：-0.0882（shrinking − fortran）

## 4. 燃烧区末端位置随 N

N=10: 2.060 m、N=20: 1.030 m、N=40: 0.515 m、N=80: 0.258 m、N=160: 0.129 m。

## 5. 网格无关极限下限幅激活的单元长度占比

- `C_O2_LimitedByO2`：0.012
- `C_H2O_LimitedByH2O`：0.006
- `C_CO2_LimitedByCO2`：0.006
- `C_H2_LimitedByH2`：0.006
- `WGS_NegativeTruncated`：0.006
- `MSR_Limited`：0.012
- `H2Ox_AtCap099`：0.006

## 6. cell0_n_distinct_roots > 1 的行数

0 / 45 行。

## 7. 墙钟时间随 N 的标度

N=10: 0.79 s、N=20: 0.97 s、N=40: 1.09 s、N=80: 1.50 s、N=160: 2.17 s。

## 8. 闸门

网格无关极限下（全部完成窗口中位数）X_C = 0.9630、干气流量比 = 0.9949。**触发**「X_C ≥ 0.95 且干气流量比 ≥ 0.97」：反演动力学以补转化亏欠不成立，主线应改为以出口组成为靶子、碳转化率改作约束。

稳健性：固定配对子集给出 X_C = 0.9256（< 0.95），因此 X_C 的结论对窗口子集敏感（0.92–0.96）；干气流量比在两种口径下都 ≥ 0.97。无论如何，出口组成偏差（CO 高于装置约 8.5 个百分点、H2 低约 3.6、CO2 低约 4.1）都不依赖这个边界判断。

## 9. 偏离记录

按用户 2026-10-31 决定，本卡在 P1.5-C 模式 F 未达锚定判据的情况下继续执行；no_root 行（粗网格燃烧区解析不足、以及动力学 X≥0.999 截断造成的 1e-4 残差地板）如实记录，未放宽阈值、未改方程。
