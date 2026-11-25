# 06A 预期产物与放行判定（rev 2）

## 判定：`STOP_TO_07`

依据：在**已逐位复现**的 P2.2-A 历史 C0 底座上（最大 \|dCO\| = 0.000e+00），
组成与响应信号在三档预设分辨率下均未超过 2σ 判据；S2 的边界偏差与物理候选之间
的响应差也未超过 2σ_r。

## 关键数字（rev 2）

| 阶段 | 组成水平杠杆 | 响应杠杆 | 组成相对 2σ_y | 响应相对 2σ_r |
|---|---:|---:|---:|---:|
| P1 | 0.0089 pp | 0.00207 pp/5% | 22.4366× | 166.9807× |
| P2 | 0.0040 pp | 0.00415 pp/5% | 49.6648× | 83.6155× |
| P3 | 0.0035 pp | 0.00317 pp/5% | 57.3066× | 109.6478× |
| P4 | 0.1649 pp | 0.20084 pp/5% | 1.2127× | 1.7203× |

「相对 2σ」列 = 预设误差 / 分离所需误差；> 1 表示不可分辨。

## S1 消融（范围受限）

只报事实：

- P1：C 的 argmin θ=2.0，R 的 argmin θ=2.0，同端点=True，全部候选通过门槛=True，R_dev 在本网格内单调=False
- P2：C 的 argmin θ=2.0，R 的 argmin θ=2.0，同端点=True，全部候选通过门槛=True，R_dev 在本网格内单调=False
- P3：C 的 argmin θ=2.0，R 的 argmin θ=2.0，同端点=True，全部候选通过门槛=True，R_dev 在本网格内单调=False
- P4：C 的 argmin θ=2.0，R 的 argmin θ=2.0，同端点=True，全部候选通过门槛=True，R_dev 在本网格内单调=True

**不代表**「单标量族必然退化」或「任何门槛都不可能改变动作」。真值取网格内 θ=2.0、
无噪声、同一求解器，因此该候选的 C_dev=R_dev=0 是按构造成立。

## S2 边界偏差（rev 2 实测）

注入：载气流量 ×(1+3%)，反应族保持 C0。逐阶段取与候选 C 最接近者：

| 阶段 | 最接近候选 θ | C 差 | 组成近似 | 响应差 | 响应可分 | 判定 |
|---|---:|---:|---|---:|---|---|
| P1 | 2.0 | 0.088462 | True | 0.010175123 | False | NOT_SEPARABLE_mimicry_indistinguishable |
| P2 | 2.0 | 0.089283 | True | 0.008604988 | False | NOT_SEPARABLE_mimicry_indistinguishable |
| P3 | 2.0 | 0.097454 | True | 0.003961724 | False | NOT_SEPARABLE_mimicry_indistinguishable |
| P4 | 2.0 | 0.085024 | True | 0.029080801 | False | NOT_SEPARABLE_mimicry_indistinguishable |

## 若 RUN_ONCE 应交付什么

`generator_manifest.json`、`scenario_manifest.csv`、`truth_lock.json`、`selection_trace.csv`、
`predictions.csv`、`controlled_scores.csv`、`decision_table.csv`、`interpretation.md`。
本轮**不生成**。

## 给 07 的边界

可写：历史 C0 可逐位复现；两个未匹配配置的调用不等价（差异原因未分离）；所列预设下的信号上界（条件性）。
不可写：响应判据有效/无效；工业响应已识别；WGS 已近平衡；任何通用精度的分析仪结论。
详细清单见 `limitations.md`。
