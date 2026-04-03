# 1D Kinetic Gasifier: 物理模型与数值算法说明

本模块详细描述了 `gasifier-1d-kinetic` 中实现的物理逻辑、热化学基准以及数值求解策略。

## 1. 能量平衡与煤生成焓 (Hf_coal)

为了确保能量平衡（Energy Balance）与煤炭的热值定义（HHV/LHV）完全对齐，模型采用了基于热值的生成焓反算法。

### 1.1 热化学基准
*   **输入基准**：干燥基高位热值 ($HHV_d$) 和元素分析 ($C, H, O, N, S, Ash$)。
*   **LHV 计算**：
    $$LHV_d = HHV_d - 9 \cdot H \cdot \frac{\Delta H_{vap,H2O}}{M_{w,H2O}}$$
    其中 $\Delta H_{vap,H2O}$ 取 25°C 下的汽化潜热（~2442 kJ/kg）。
*   **生成焓 ($H_{f,coal}$)**：
    根据燃烧放热定义：$Q_{comb} (LHV) = \sum H_{f,products} - H_{f,coal}$
    $$H_{f,coal} = \left( \frac{C}{12.01} \cdot H_{f,CO2} + \frac{H}{2.016} \cdot H_{f,H2O(g)} + \frac{S}{32.06} \cdot H_{f,SO2} \right) + LHV_d$$
    *注：计算过程中必须严格执行元素组分归一化，否则会引入系统性能量偏差。*

## 2. 热损失模型 (Heat Loss)

### 2.1 辐射敏感分摊逻辑 ($T^4$ Bias)
在干粉气化炉（Dry-fed）中，燃烧区（Cell 0-5）温度可达 2200K+，其辐射散热强度远高于下游气化区。模型不再采用均匀分摊（Uniform Distribution），而是引入了基于局部温度四次方的分配权重：

$$Q_{loss,i} = \frac{Loss\% \cdot P_{total}}{100} \cdot \frac{dz_i}{L_{total}} \cdot \left( \frac{T_i}{T_{ref}} \right)^4$$

*   **$T_{ref}$**：基准工艺温度（默认 1800K）。
*   **物理效应**：自动在最高温区施加剧烈散热，模拟真实炉膛的水冷壁/耐火砖传热特性，有效压低了干粉炉出口的异常高温。

## 3. 投料物理模型判定 (Feedstock Awareness)

系统根据工况定义自动切换物理子模型：
*   **Slurry-fed (水煤浆)**：Cell 0 包含巨大的液态水汽化潜热（Evaporation Sink），这会显著降低起燃温度。
*   **Dry-fed (干粉)**：蒸汽（Steam）以气态直接进入，不含相变潜热，但会参与均相/异相吸热反应。

## 4. 数值求解器 (JAX Pure Solver)

*   **Newton-Raphson 核心**：在 CPU 上运行的 NumPy 阻尼 Newton 法。
*   **Jacobian 近似**：采用前向差分（Forward Difference），针对气化反应的病态 Jacobian 进行了 `lstsq` 奇异值截断处理。
*   **多初值策略**：对 Cell 0 尝试 3000K, 2000K, 1500K 等多组初值，确保起燃（Ignition）逻辑的稳健性。
