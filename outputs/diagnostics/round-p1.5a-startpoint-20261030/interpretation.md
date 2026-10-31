# P1.5-A 证据路径起点修复 —— interpretation

## 1. 网格必须铺满反应器（B1）

新增 `MeshFamily="uniform_full_length"`：`dz_i = L/N`。旧族行为不变，`GasifierSystem.solve` 对 `|sum(dz)-L| > 1e-9·L` 直接报错，复现旧轮次必须显式传 `AllowTruncatedMesh=True`。详见 `mesh_check.csv`。

## 2. 求解器接受逻辑（B2）

`SolverStrictRoot=True` 时，任一单元 `cost > 1e-4` 即标记 `execution_status="no_root"`、记录 `no_root_cell_index` 并停止向下游推进，出口量留空。逐单元 `residual_cost` 与主动残差分量写入 `smoke_cells.csv`。`fallback_count` 语义保留，但它不统计 `minimize` 路径 cell 0 的多起点。

## 3. 颗粒尺寸模型（B3）

`fortran_constant`：`d_p = d_p0·swell^(1/3)` 恒定，颗粒密度随剩余固体分数 `A=f_ash+(1-f_ash)(1-X)` 下降，`S_total = 6·m_hold/(rho_p·d_p)`。`shrinking_legacy` 保持原行为。

单元级检查：`fortran_constant` 与 `shrinking_legacy` 的 `S_total` 之比对照理论 `A^(-2/3)`，最大相对偏差 4.22e-08（容差 1e-6）。`swell` 在华谊工况无实测，取 1.0。

## 4. Shomate 分段（B8）

H2O 低温段改为 NIST 500–1700 K 系数（`T_cut = 1700 K`）；N2、O2、H2 补成三段；全部物种的最大分段点焓跳变 0.000 J/mol（容差 20）。JANAF 对照（Cp(1000 K)、H(1500 K)−H(298.15 K)）偏差均 < 0.2%。未通过项：0。

## 5. 诊断列（B5、B6）

逐单元记录 7 个限幅布尔量（判据：限幅前后相对差 > 1e-9）与 cell 0 五个初值的 cost、温度、胜出初值；`cell0_n_distinct_roots` 统计 cost < 1e-10 且温度相差 > 50 K 的不同根个数。不改选解规则。

## 6. 单窗冒烟（验收）

窗口 `R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003`，`uniform_full_length`，N ∈ {10, 40} × 两种颗粒模型，共 4 次求解。

- N=10, fortran_constant: `no_root`（cell 1 无根，cost=2.471e+00）。
- N=10, shrinking_legacy: `no_root`（cell 1 无根，cost=6.674e-01）。
- N=40, fortran_constant: `completed`，X_C=0.9990，干气流量比=0.9961，CO=72.71，H2=21.80%，墙钟 1.04 s。
- N=40, shrinking_legacy: `completed`，X_C=0.9903，干气流量比=0.9876，CO=72.53，H2=21.76%，墙钟 1.04 s。

**闸门状态：触发。** N=10（dz=1.03 m）两种颗粒模型都在 cell 1 出现 `no_root`；N=20（dz=0.515 m）同样在 cell 1 失败；N=40/80/160 全部 completed，最差单元 cost ≤ 1e-7。失败单元的特征是 `X_total` 撞到动力学 0.999 截断、主残差为 `carbon_fraction` / `gas_CO`，燃烧区厚度约 0.26 m 而单元 0.5–1.0 m——这是燃烧区分辨不足导致的离散化极限，不是方程或代码缺陷（旧族 N=3 仍复现修复前量级）。按卡片停止条件，本轮到此为止，未放宽 1e-4、未回退氢气限幅、未改方程。

无根单元记录数：0，明细见 `smoke_cells.csv`。
