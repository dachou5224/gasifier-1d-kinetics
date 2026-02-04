# 1 维气化反应动力学算法说明书 (1-D Gasification Algorithm Manual)

> **适用对象**: 气流床气化炉（Texaco / Shell 类）
> **模型类型**: 1-D Plug Flow + Cell Model + Kinetics
> **核心特征**: 气-固强耦合、轴向稳态、非均匀网格解析

---

## 1. 模型总体结构 (Model Structure)

### 1.1 空间离散 (Spatial Discretization)

将细长的气化炉轴向划分为 **N 个小室 (Cells)**。
由于气流床反应前段（点火区）变化极其剧烈，采用 **几何级数非均匀网格 (Geometric Grid)**：

*   总高度 $H$，网格数 $N$
*   网格增长比 $r \approx 1.1 \sim 1.2$
*   第 $i$ 个小室高度 $\Delta z_i = \Delta z_0 \cdot r^i$

**假设**:
*   轴向推流 (Plug Flow)，无轴向返混。
*   径向完全均匀 (1D)。
*   稳态系统 (Steady State, $\partial/\partial t = 0$)。

### 1.2 求解目标 (State Variables)

对于第 $i$ 个小室，模型联立求解以下 **11 个状态变量** ($\mathbf{X}_i$)：

| 变量索引 | 符号 | 物理意义 | 单位 |
| :--- | :--- | :--- | :--- |
| 0-7 | $F_{j}$ | 8种气体组分摩尔流量 ($O_2, CH_4, CO, CO_2, H_2S, H_2, N_2, H_2O$) | mol/s |
| 8 | $W_s$ | 固体（煤/焦/渣）质量流量 | kg/s |
| 9 | $X_c$ | 碳转化率 (Carbon Conversion) | - (0~1) |
| 10 | $T$ | 气相与固相温度 (假设热平衡 $T_g \approx T_p$) | K |

> **注**: 当前版本引入了 Ross 模型计算颗粒温度 $T_p$，但在求解微元平衡时，主要求解统一场温度 $T$，也就是出口温度。

---

## 2. 计算流程总览 (Main Algorithm)

```text
输入：煤质分析(Coal Props)、操作条件(P, T_in, Flows)、几何尺寸(L, D)
网格生成：计算非均匀网格体积 V_cells

初始化：Current_Inlet = Core_Inlet_Conditions

FOR i = 0 → N-1 (遍历每个小室)
    1. 设定初值 X_guess (继承上游出口，首室特殊处理)
    2. 调用非线性方程组求解器 (Scipy Trust Region Reflective)
        目标：Residuals(X) = 0
        约束：F >= 0, 0 <= Xc <= 1, T >= 300
        
        IN LOOP (Residual Calculation):
            a. 物理参数更新 (Cp, Enthalpy, Viscosity, Diffusivity)
            b. 异相反应速率计算 (UCSM Model -> k_diff, k_ash, k_s)
            c. 均相反应处理 (WGS Equilibrium)
            d. 建立守恒方程 (Species, Atom, Solid, Energy)
            
    3. 检查收敛性 (Residual < tol, Cost < tol)
    4. 记录结果，Current_Inlet = Current_Outlet
    5. 进入下一小室
END FOR

输出：沿程分布数据 (Profiles of T, Xc, Composition)
```

---

## 3. 反应动力学算法 (Kinetics)

### 3.1 反应体系

本模型包含 4 个主要的 **异相反应 (Heterogeneous)** 和 1 个关键的 **均相反应 (Homogeneous)**。

#### A. 异相反应 (Surface Reactions)
采用 **未反应缩核模型 (Unreacted Core Shrinking Model, UCSM)**。

1.  $C + \phi O_2 \rightarrow 2(1-\frac{1}{\phi})CO + (\frac{2}{\phi}-1)CO_2$
    *   **机理因子 $\phi$**: 随温度变化，低温生成 $CO_2$，高温生成 $CO$。
    *   $\phi = \frac{2p+2}{p+2}, \quad p = 2500 \exp(-6249/T)$
2.  $C + H_2O \rightarrow CO + H_2$ (水煤气反应)
3.  $C + CO_2 \rightarrow 2CO$ (Boudouard 反应)
4.  $C + 2H_2 \rightarrow CH_4$ (加氢气化，通常较慢)

#### B. 均相反应 (Gas Phase)
*   **水煤气变换 (WGS)**: $CO + H_2O \leftrightarrow CO_2 + H_2$
*   **处理方式**: 假设气相反应极快，在每个小室出口达到 **化学平衡**。
*   $K_{eq}(T) = \exp(4578/T - 4.33)$
*   通过非线性方程 `res[7]` 强制满足平衡约束：$F_{CO2}F_{H2} - K_{eq}F_{CO}F_{H2O} = 0$

---

### 3.2 异相速率计算核心 (UCSM Implementation)

对于每个异相反应 $j$，总反应速率 $R_j$ (kmol/m²·s) 由串联阻力决定：

1.  **气膜扩散阻力 ($1/k_{diff}$)**:
    $$k_{diff} = \frac{Sh \cdot D_{AB}}{d_p}, \quad Sh=2.0$$
2.  **灰层扩散阻力 ($1/k_{ash}$)**:
    随着碳转化率 $X_c$ 增加，灰层变厚，阻力增大。
    缩核因子 $Y = (\frac{1-X_c}{1-f})^{1/3}$
    $$R_{ash} = \frac{1-Y}{k_{ash} \cdot Y}, \quad k_{ash} = k_{diff} \cdot \varepsilon^{2.5}$$
3.  **化学反应阻力 ($1/k_s$)**:
    $$k_s = A_j \exp(\frac{-E_j}{R T_p})$$
### 3.1 均相反应 (Homogeneous Reactions)
模型考虑了 6 个均相反应，采用 Arrhenius 速率公式：$r = k \cdot C_a \cdot C_b$，单位 $kmol/(m^3 \cdot s)$。

| 反应名称 | 反应方程式 | 指前因子 $A$ | 活化能 $E$ (J/kmol) |
| :--- | :--- | :--- | :--- |
| CO 燃烧 | $CO + 0.5 O_2 \to CO_2$ | $1.3 \times 10^{11}$ | $1.25 \times 10^8$ |
| H2 燃烧 | $H_2 + 0.5 O_2 \to H_2O$ | $1.0 \times 10^{11}$ | $8.37 \times 10^7$ |
| WGS | $CO + H_2O \to CO_2 + H_2$ | $2.78 \times 10^3$ | $1.25 \times 10^7$ |
| RWGS | $CO_2 + H_2 \to CO + H_2O$ | $1.0 \times 10^5$ | $6.27 \times 10^7$ |
| CH4 燃烧 | $CH_4 + 2 O_2 \to CO_2 + 2 H_2O$ | $1.6 \times 10^{10}$ | $1.25 \times 10^8$ |
| MSR | $CH_4 + H_2O \to CO + 3 H_2$ | $4.4 \times 10^{15}$ | $2.49 \times 10^8$ |

> **注**: WGS 活化能已根据物理合理性校正（文献原值 1.25e4 疑似为 kJ/kmol 或打印错误）。

### 3.2 异相反应 (Heterogeneous Reactions)
采用 **未反应缩核模型 (UCSM)**。

#### 反应体系与参数 (Table 2-7)
| 反应 | 方程式 | 指前因子 $A$ | 活化能 $E$ (J/kmol) |
| :--- | :--- | :--- | :--- |
| 燃烧 | $C + O_2 \to CO/CO_2$ | $2.3 \times 10^2$ | $1.1 \times 10^8$ |
| 水蒸气气化 | $C + H_2O \to CO + H_2$ | $2.4 \times 10^4$ | $1.43 \times 10^8$ |
| 二氧化碳气化 | $C + CO_2 \to 2CO$ | $2.4 \times 10^4$ | $1.43 \times 10^8$ |
| 加氢气化 | $C + 2H_2 \to CH_4$ | $6.4$ | $1.3 \times 10^8$ |

#### 阻力组合 (Resistances)
总反应速率 $R_{tot}$ (kmol/m²·s) 考虑三项阻力的串联：
1. **气膜扩散**: $k_d = 2 D / d_p$
2. **灰层扩散**: $k_{ash} = k_d \cdot \varepsilon^{2.5}$, 其中 $\varepsilon=0.75$
3. **表面化学反应**: $k_s = A \exp(-E/RT_p)$

$$R_{total} = \frac{C_{gas}}{\frac{1}{k_d} + \frac{1-Y}{k_{ash} Y} + \frac{1}{k_s Y^2}}$$
其中 $Y = (1-X)^{1/3}$ 为缩核因子，$X$ 为局部碳转化率。

---

## 4. 热解过程算法 (Pyrolysis)

模型采用 **Rajan 热解模型**，这是一套基于大量实验数据拟合的经验相关式。假定在进入首个单元格（Cell 0）时瞬时完成。

### 4.1 产物关联式 (kg/kg_daf)
基于煤的 DAF 基组分（C, H, O, N, S）及挥发分含量 $V_{daf}$（或干基挥发分 $V_d$）：
- $Y_{CH4} = -0.16 + 0.05 V_{daf} + 0.03 H_{daf}$
- $Y_{CO2} = 0.02 + 0.03 O_{daf}$
- $Y_{CO} = -0.05 + 0.01 V_{daf} + 0.05 O_{daf}$
- $Y_{H2O} = 0.03 + 0.03 O_{daf}$
- $Y_{H2S} = S_{daf} / 100$
- $Y_{N2} = N_{daf} / 100$
- $Y_{H2}$ 通过质量差值补全。

### 4.2 归一化处理 (Normalization)
由于经验公式计算得到的各项产物质量总和可能不等于 $V_d$，模型会通过比例缩放修正各组分比例，强制满足总质量平衡。

### 4.3 物理意义
Rajan 模型旨在通过煤种的宏观指标快速估算热解初产物，为后续的多步动力学反应提供反应物起始浓度。

---

## 5. 守恒方程与数值求解

### 5.1 能量平衡 (Energy Balance)
模型通过维持入口总焓（含煤、氧气、蒸汽）与出口总焓（气体组分、固体灰分/残碳、潜热、热损失）相等来求解温度：
$$H_{in} - Q_{loss} + Q_{pilot} - H_{out} = 0$$

### 5.2 求解器配置
- **算法**: `scipy.optimize.least_squares` (TRF method)。
- **初始猜测**: 点火区强制设为 2000K。
- **网格**: 50-100 网格单元。

---

## 5. 工程实现与目录结构 (Project Structure)

### 5.1 目录树
```text
gasifier-1d-kinetic/
├── src/
│   ├── model/
│   │   ├── gasifier_system.py # 系统编排器 (Orchestrator)
│   │   ├── cell.py           # 微元守恒残差计算
│   │   ├── state.py          # 状态向量数据结构
│   │   ├── material.py       # 物性计算服务
│   │   ├── kinetics_service.py # 反应速率计算服务（含均相、异相多步反应）
│   │   ├── pyrolysis_service.py # 热解产率计算服务
│   │   ├── kinetics.py       # 动力学底层物理模型 (UCSM)
│   │   ├── physics.py        # 基础物理关联式 (Cp, Enthalpy)
│   │   └── chemistry.py      # 化学组分定义与数据库
│   └── main_ui.py            # Streamlit 交互界面
├── tests/
│   ├── verify_cases.py       # 多工况验证脚本
│   └── diagnose_detailed.py  # 详细诊断工具
├── docs/
│   ├── 1D_Gasifier_Model_Manual_cn.md  # 本文档
│   └── grid_strategy_cn.md            # 网格优化策略
└── plots/                    # 结果可视化输出
```

### 5.2 核心算法逻辑
模型采用解耦架构：
1. **GasifierSystem** 负责网格划分、初值管理及逐小室顺序求解。
2. **Cell** 负责构建基于 `scipy.optimize.least_squares` 的非线性方程组。
3. **KineticsService** 提供精细的反应动力学数据（不再局限于单一平衡假设）。

### 5.3 数值技巧 (Numerical Tricks)
1.  **非负约束**: 使用 `least_squares` 的 `bounds` 参数强制 $F_i \ge 0, T \ge 300$，避免非物理根求解失败。
2.  **初始猜测 ($x_0$)**:
    *   通常使用**上游小室出口**作为初值。
    *   **点火强化**: 在入口首个小室 ($i=0$)，如果 $T_{in}$ 过低，强制设 $T_{guess}=1500K$ 以诱导求解器找到点火解（针对多稳态问题）。
3.  **WGS 鲁棒性**: 将化学平衡方程写为乘积形式 $K P_{CO} P_{H2O} - P_{CO2} P_{H2}$ 而非比值形式，避免分母为零。

---

## 6. 后续扩展方向

*   ✅ **Python 代码化**: 已完成。
*   ✅ **工业级验证**: 已完成基于德士古 (Texaco) 工业气化炉数据的对比验证。
*   🚧 **辐射热损**: 当前模型为绝热/简化散热，可进一步精化 $Q_{loss}$ 模型。
*   🚧 **熔渣流动**: 当前仅计算固体质量，可扩展熔渣粘度模型预测排渣特性。
