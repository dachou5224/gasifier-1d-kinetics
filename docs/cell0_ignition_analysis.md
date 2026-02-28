# Cell 0 挥发分燃烧放热与网格尺寸分析

## 现象总结

| 炉型 | 进煤量 | Cell 0 推荐 dz | 原因 |
|------|--------|----------------|------|
| 小型试验炉 (Original Paper) | ~77 g/s | **较小** (0.05–0.1 m) | 气体流速低，τ 自然较长 |
| 大型工业炉 (Paper_Case_6) | ~11575 g/s | **较大** (0.2–0.3 m) | 气体流速高，需更大 dz 才能保证 τ |

---

## 1. 流体力学与停留时间

### 1.1 基本关系

$$\tau = \frac{dz}{v_g} = \frac{V_{cell}}{Q_{gas}} = \frac{A \cdot dz}{Q_{gas}}$$

其中：
- $dz$：Cell 0 轴向长度 (m)
- $v_g$：气体表观流速 (m/s)
- $A$：反应器横截面积 (m²)
- $Q_{gas}$：气体体积流量 (m³/s)

### 1.2 气体流量与进煤量的关系

$$Q_{gas} \approx \frac{\dot{n}_{gas} \cdot R \cdot T}{P} \propto \dot{n}_{gas}$$

在固定 O2/Coal、蒸汽比下，$\dot{n}_{gas}$ 与进煤量 $\dot{m}_{coal}$ 近似成正比：
$$\dot{n}_{gas} \propto \dot{m}_{coal}$$

故：
$$v_g = \frac{Q_{gas}}{A \cdot \varepsilon} \propto \frac{\dot{m}_{coal}}{A}$$

**结论**：进煤量越大，气体流速越高；相同 $dz$ 下，停留时间 $\tau$ 越短。

### 1.3 典型数值对比

| 工况 | 进煤 (g/s) | A (m²) | $v_g$ 量级 | dz=0.05 m 时 τ | dz=0.20 m 时 τ |
|------|------------|--------|------------|----------------|----------------|
| Texaco_I-1 (pilot) | 76.66 | 1.82 | ~0.05 m/s | ~1.0 s | ~4 s |
| Paper_Case_6 (industrial) | 11575 | 3.14 | ~1–4 m/s | ~0.01–0.05 s | ~0.05–0.2 s |

---

## 2. 挥发分燃烧放热与 τ 的关系

### 2.1 燃烧速率（动力学控制）

挥发分燃烧为均相反应，近似二级动力学：
$$r = k(T) \cdot C_{fuel} \cdot C_{O_2} \cdot V$$

其中 $k(T) = A \exp(-E/RT)$。在 Cell 0 内，若近似为 CSTR：
- 转化率 $\eta \approx \frac{k \tau}{1 + k \tau}$
- 当 $k\tau \gg 1$ 时，$\eta \to 1$（接近完全燃烧）
- 当 $k\tau \ll 1$ 时，$\eta \approx k\tau$（转化率与 τ 成正比）

**放热量**：
$$Q = \eta \cdot n_{vol} \cdot \Delta H_{comb}$$

若 τ 过短，$\eta$ 小，放热不足，温度难以升高。

### 2.2 τ 的临界值

要使挥发分基本燃尽（如 $\eta > 0.95$），需 $k\tau \gtrsim 20$，即：
$$\tau \gtrsim \frac{20}{k(T)}$$

高温下 $k$ 大，所需 τ 小；但若初猜温度偏低，$k$ 小，则需要更长 τ 才能起燃。因此：
- **τ 过短**：挥发分未燃尽即离开 Cell 0，放热不足，温度偏低，形成“冷态解”
- **τ 足够**：挥发分充分燃烧，放热充分，温度升高，进入“热态解”

---

## 3. 稀释效应（dz 过大的问题）

### 3.1 浓度与体积

$$C_i = \frac{n_i}{V} = \frac{\dot{n}_i \cdot \tau}{V} = \frac{\dot{n}_i}{Q_{gas}}$$

在固定进料下，$C_i$ 与 $V$ 无关，只与 $Q_{gas}$ 有关。因此**稳态时浓度不直接受 dz 影响**。

### 3.2 但反应速率与 τ 有关

对于 CSTR 近似，出口转化率：
$$\eta = 1 - \frac{1}{1 + k\tau}$$

$k$ 与温度强相关。若 dz 过大：
- τ 长，转化率高
- 但若 Cell 0 体积过大，**轴向梯度被抹平**，可能掩盖局部高温区
- 对 1D 模型，主要影响是：dz 过大时，一个 Cell 代表的空间范围大，等效于“平均”了更长的反应区，可能弱化起燃区的峰值温度

对于**小试炉**，$v_g$ 低，即使 dz=0.05 m，τ 已有 ~1 s，挥发分燃烧已较充分。再增大 dz 会稀释反应强度（单位长度放热降低），对起燃未必有利。

---

## 4. 综合结论与 dz 选取建议

### 4.1 物理图像

| 因素 | dz 小 | dz 大 |
|------|-------|-------|
| 停留时间 τ | 短 | 长 |
| 挥发分燃尽度 | 可能不足（高流量时） | 更充分 |
| 单位长度放热 | 集中 | 分散 |
| 适用场景 | 低流量 pilot | 高流量 industrial |

### 4.2 建议的 dz 标度关系

为使 Cell 0 内挥发分燃烧程度相近，可令 τ 接近常数：
$$\tau \approx \frac{dz \cdot A}{Q_{gas}} \approx \text{const}$$

即：
$$dz \propto \frac{Q_{gas}}{A} \propto \dot{m}_{coal}$$

**经验公式**（需结合具体工况标定）：
$$dz_{cell0} \approx d_0 \cdot \left(\frac{\dot{m}_{coal}}{\dot{m}_{ref}}\right)^\alpha$$

其中 $\dot{m}_{ref}$ 为参考进煤量（如 77 g/s），$\alpha \in [0.3, 0.5]$ 为经验指数。$\alpha=0.5$ 对应 $\tau \propto \sqrt{\dot{m}}$ 的折中。

### 4.3 实现（已集成）

`gasifier_system.py` 支持三种方式：

1. **固定值**：`FirstCellLength=0.2`（默认）
2. **手动覆盖**：`op_conds['FirstCellLength'] = 0.3`
3. **自适应**：`op_conds['AdaptiveFirstCellLength'] = True`
   - 公式：`dz_cell0 = dz_base * (m_coal / 77)^0.4`，限制在 [0.03, 0.5] m
   - 77 g/s 为 Texaco pilot 参考进煤量

---

## 5. 参考文献

- `cell.py`：`tau = dz / v_g`
- `docs/cell_physics_manual.md`：停留时间与传质
- `docs/temperature_diagnosis.md`：Cell 0 体积与起燃
