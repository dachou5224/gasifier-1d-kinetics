# 1D Gasifier Cell 物理与化学计算公式手册

本文档详细罗列了模型中每个单元（Cell）所执行的核心物理与化学计算公式，特别是您关注的**传质过程**与**停留时间**计算。

---

## 1. 核心守恒方程 (Residual Equations)

每个 Cell 的状态通过求解下列残差方程 $F(x) = 0$ 获得：

### 1.1 组分质量平衡 (Species Balance)
对于气体组分 $i$ (O2, CH4, CO, CO2, H2S, H2, N2, H2O)：
$$n_{i, out} = n_{i, in} + \sum \text{Production}_{i, reactions} - \sum \text{Consumption}_{i, reactions} + \text{Src}_{i, instant}$$
- **Src_instant**：仅在 Cell 0 中发生，计入瞬时热解释放的挥发分（Rajan 经验模型）。
- 单位：mol/s

### 1.2 固体质量平衡 (Solid Mass Balance)
$$\dot{W}_{s, out} = \dot{W}_{s, in} - \dot{W}_{carbon, gasified}$$
- **Char Basis**：在入口处已扣除挥发分质量。
- 单位：kg/s

### 1.3 能量平衡 (Energy Balance)
$$H_{total, out} = H_{total, in} - Q_{loss} + Q_{source}$$
- $H_{total} = \sum (n_i \cdot h_i(T)) + \dot{W}_s \cdot h_s(T)$
- $Q_{loss} = \frac{dz}{L_{reactor}} \cdot (\text{Loss\%} \cdot \dot{W}_{coal} \cdot HHV)$
- 单位：Watts (J/s)

### 1.4 原子守恒约束 (Atomic Conservation Constraints)
为了确保数值稳定性并杜绝“物质凭空产生（Mass Spawning）”，模型在组分平衡外引入了显式的原子流量平衡：
$$\sum \text{Atoms}_{in, i} = \sum \text{Atoms}_{out, i}$$
针对 C, H, O, N 四大原子进行约束，公式定义如下：
- **C 守恒**：$n_{solid} \cdot X_c/12 + n_{CO} + n_{CO2} + n_{CH4} = \text{Const}$
- **H 守恒**：$4n_{CH4} + 2n_{H2O} + 2n_{H2} + 2n_{H2S} = \text{Const}$
- **O 守恒**：$2n_{O2} + n_{CO} + 2n_{CO2} + n_{H2O} = \text{Const}$

### 1.5 动力学消减与安全性 (Kinetics Damping & Safety)
在求解非线性刚性方程时，为了防止数值震荡导致的组分超量消耗，实施了“动力学消减”策略：
- **消费限额**：单步内任何组分的消耗量（Consumption）不得超过其可用总量（Inlet + Source）的 99%。
- **边界约束**：气相流量严格限制在 [0, 3000] mol/s 范围内，防止求解器落入非物理搜索空间。

---

## 2. 停留时间与传质过程 (Mass Transfer & Residence Time)

这是模型中最关键的物理环节，决定了气固反应的接触强度。

### 2.1 气体速度 ($v_g$) 计算
$$v_g = \frac{\dot{m}_{gas}}{\rho_{gas} \cdot A}$$
- $\dot{m}_{gas}$：局部气体质量流量 (kg/s)
- $\rho_{gas} = \frac{P \cdot M_{avg}}{R \cdot T}$：局部气体密度
- $A$：反应器横截面积 ($m^2$)

### 2.2 焦炭颗粒停留时间 ($\tau$)
基于单相流假设（无滑移速度 $v_s \approx v_g$）：
$$\tau_{cell} = \frac{dz}{v_g}$$
- $dz$：单元步长 (m)

### 2.3 焦炭颗粒粒径 ($d_p$) - 变粒径缩核模型
根据总转化率 $X$ 动态计算有效粒径：
$$d_p = d_{p0} \cdot (f_{ash} + (1.0 - f_{ash}) \cdot (1.0 - X))^{1/3}$$
- $X = 1 - (C_{solid, local} / C_{coal, total})$：基于原始煤总碳量的总转化率。
- $f_{ash}$：基于原始煤定义的灰分/（碳+灰分）比例。
- **物理意义**：颗粒随反应缩小（包括热解释放挥发分导致的初始缩小），直至剩余灰骨架。

### 2.4 总反应表面积 ($S_{total}$)
由于气化过程是连续稳态过程，单元内的瞬时反应表面积取决于单元内“留存”了多少焦炭：
$$m_{hold} = \dot{W}_{s, inlet} \cdot \tau_{cell}$$
$$S_{total} = \frac{6 \cdot m_{hold}}{\rho_p \cdot d_p}$$
- $\rho_p$：局部颗粒密度（随转化率 $X$ 动态减小）
- $d_p$：颗粒直径（模型设定或缩核模型计算）
- **关键点**：该表面积直接进入多相反应速率计算。

---

## 3. 反应动力学 (Reaction Kinetics)

### 3.1 异相反应速率 (Heterogeneous - UCSM 模型)
针对 C+O2, C+H2O, C+CO2, C+H2，单颗粒速率 $r$ 为：
$$r = \frac{C_i}{ \frac{1}{k_d} + \frac{1-Y}{k_{ash} Y} + \frac{1}{k_s Y^2} } \quad (\text{kmol/m}^2\cdot\text{s})$$
其中：
- **气膜扩散 $k_d$**：$k_d = \frac{2 \cdot D_i}{d_p}$ (Sh = 2)
- **灰层扩散 $k_{ash}$**：$k_{ash} = k_d \cdot \epsilon^{2.5}$
- **表面反应 $k_s$**：$k_s = A \cdot \exp(-E / RT_p)$
- **缩核因子 $Y$**：$Y = (1-X)^{1/3}$

**Cell 总速率**：$R_{het} = r \cdot S_{total} \cdot 1000 \quad (\text{mol/s})$

### 3.2 同相反应速率 (Homogeneous)
基于 Arrhenius 形式：
$$R_{homo} = k \cdot C_a \cdot C_b \cdot V_{cell}$$
- $k = A \cdot \exp(-E / RT)$
- $C_a, C_b$：组分浓度 ($\text{kmol/m}^3$)

---

## 4. 传热过程说明

- **对流/辐射损失**：文献取为**入炉煤 HHV 的 2%**；通过 `Q_loss` 按 dz 比例摊分到每个 cell（`HeatLossPercent`，默认 2%）。
- **蒸发吸热**：
  - 默认实现中，煤水分+浆液水在 **Cell 0 一次性** 以液态焓源项加入（`EvaporationSource`），等价于全部蒸发吸热集中在首格。
  - 若浆液量大（如 62% 浓度），Cell 0 吸热可达数十 MW，**可能高估首格吸热**，导致该格或整体预测温度偏低。
  - 可通过 `op_conds['L_evap_m']`（单位 m，如 0.5）将蒸发按轴向长度分摊到前若干格，减轻 Cell 0 吸热；默认 0 表示全部在 Cell 0。
- **反应热**：隐含在组分焓变中。$H = \int C_p dT + \Delta H_f$。

---

## 总结：您关注的停留时间
在 Cell 中，停留时间计算链条如下：
**[Gas Moles & T]** $\rightarrow$ **[Gas Density]** $\rightarrow$ **[Gas Velocity $v_g$]** $\rightarrow$ **[Residence Time $\tau = dz/v_g$]** $\rightarrow$ **[Solid Holdup $m_{hold}$]** $\rightarrow$ **[Active Surface Area $S_{total}$]** $\rightarrow$ **[Heterogeneous Rates]**。

这种计算方式确保了 1D 模型的 axial profile 能够真实反映体积反应强度的变化。
