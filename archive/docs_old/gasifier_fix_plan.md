# 气化炉模型修改方案（出口温度偏低问题）
> 所有修改均有 Fortran 源码（Source1_副本.for）直接证据支撑，按优先级排序。

---

## 修改 1 【P0-必改】删除 pyrolysis_service.py 中的能量归一化逻辑（Step 9）

**文件**：`src/model/pyrolysis_service.py`

**问题**：Step 9 是 Python 层自行添加的约束，在 Fortran `pyroly` 子程序（L1032-1090）中**完全没有对应逻辑**。Fortran 挥发分产率由元素守恒决定，不存在 HHV 截断。

当前代码将 `HHV_d`（单位 kJ/kg）误乘以 `1e6`（当作 MJ/kg），使截断条件永远不触发，掩盖了这个逻辑错误。一旦单位修正后，Step 9 会将挥发分产率削减至约 46%，减少约 28 MW 燃烧热，使出口温度更低。

**修改方案**：

第一步，修复 HHV 单位（1 行）：
```python
# 修改前
hhv_coal = coal_props.get('HHV_d', 30.0) * 1e6

# 修改后（kJ/kg → J/kg，正确换算）
hhv_coal = coal_props.get('HHV_d', 30.0) * 1e3
```

第二步，删除 Step 9 的整个能量归一化块。找到类似以下内容的代码段并**整段删除**：
```python
# 删除以下整段（Step 9 / energy normalization）
# ------- 删除起始 -------
lhv_basis = (Cd / 100.0) * (LHV_C / 0.012011)
target_lhv = hhv_coal * 0.95
target_excess = target_lhv - lhv_basis
current_excess = (
    n_CH4 * d_lhv[1] + n_CO * d_lhv[2] +
    n_H2 * d_lhv[5] + n_H2S * d_lhv[4]
)
if current_excess > target_excess > 0:
    scale = target_excess / current_excess
    n_CH4 *= scale
    n_CO  *= scale
    n_H2  *= scale
    n_H2S *= scale
# ------- 删除终止 -------
```

> **Fortran 证据**：`pyroly` 子程序（L1032-1090）无任何 HHV 截断。`rheat1`（挥发分燃烧热）是纯计算结果，不受 HHV 约束。

---

## 修改 2 【P0-必改】固相焓计算简化

**文件**：`src/model/material.py`，函数 `get_solid_enthalpy`

**问题**：Fortran `enthal` 子程序（L1131-1145）固相焓仅为：
```fortran
enths = cps * fcoal * (1.0 + fash - wl/100.) * (Ts - 298.)
```
即纯粹的显热，**不区分煤和焦炭，不使用 `Hf_char`**。当前 Python 用经验公式 `(1 - VM×0.7) × Hf_coal` 估算 `Hf_char`，引入约 6 MW 的系统性误差，且在 `is_char` 判断切换时产生不连续跳变。

**修改方案**：

```python
def get_solid_enthalpy(T_s: float, fcoal: float, fash: float,
                       wl: float, coal_props: dict) -> float:
    """
    固相焓（对齐 Fortran enthal 子程序 L1142）。
    纯显热，不区分 coal/char，基准温度 298.15 K。
    """
    cp_s = coal_props.get('cp_char', 1300.0)   # J/kg/K，对应 Fortran cps=0.45 cal/g/K
    # Fortran: cps * fcoal * (1 + fash - wl/100) * (Ts - 298)
    # Python 等价（SI 单位）：
    solid_mass_fraction = 1.0 + fash - wl / 100.0
    h_sensible = cp_s * fcoal * solid_mass_fraction * (T_s - 298.15)

    # 加上 Hf_coal（生成焓，维持 Shomate 全焓框架的基准一致性）
    # 注意：不再区分 Hf_char，统一用 Hf_coal
    hf = coal_props.get('Hf_coal', -3.0e6)     # J/kg
    h_formation = hf * fcoal * solid_mass_fraction

    return h_sensible + h_formation
```

> **Fortran 证据**：`enthal` L1142 直接证据，固相焓仅含 `cps*(T-298)` 项，无 `Hf_char`。

---

## 修改 3 【P1-必改】确认并修正异相动力学活化能

**文件**：`src/model/kinetics.py`，`HeterogeneousKinetics.__init__`

**问题**：需确认当前 C+H2O 和 C+CO2 的活化能是否已是正确值。Fortran 原始参数为：

| 反应 | Fortran 行 | Fortran 写法 | 正确 E (J/mol) |
|------|-----------|-------------|----------------|
| C+O2  | L1173 combus | `ats = -17967./ts` | **149,400** |
| C+H2O | L1215 cbstm  | `ats = -21060./ts` | **175,100** |
| C+CO2 | L1246 cbco2  | `bts = -21060./ts` | **175,100** |
| C+H2  | L1331 cbhym  | `bts = -17921./ts` | **149,000** |

换算公式：`E(J/mol) = E_over_R(K) × 1.987(cal/mol/K) × 4.184(J/cal)`

**修改方案**：

```python
# src/model/kinetics.py
# Fortran 原始参数，直接对应 Source1_副本.for 各子程序

R_CAL = 1.987   # cal/mol/K
CAL2J = 4.184   # J/cal

self.params = {
    # C+O2：combus L1173: ks = 8710*exp(-17967/ts)
    'C+O2': {
        'A': 8710.0,
        'E': 17967.0 * R_CAL * CAL2J,   # = 149,400 J/mol
    },
    # C+H2O：cbstm L1215,1218: ks = 247*exp(-21060/ts)
    'C+H2O': {
        'A': 247.0,
        'E': 21060.0 * R_CAL * CAL2J,   # = 175,100 J/mol
    },
    # C+CO2：cbco2 L1246,1249: ks = 247*exp(-21060/ts)
    'C+CO2': {
        'A': 247.0,
        'E': 21060.0 * R_CAL * CAL2J,   # = 175,100 J/mol
    },
    # C+H2：cbhym L1331,1333: ks = 0.12*exp(-17921/ts)
    'C+H2': {
        'A': 0.12,
        'E': 17921.0 * R_CAL * CAL2J,   # = 149,000 J/mol
    },
}
```

> **注意**：另一 AI 建议 C+O2=110 kJ/mol、C+H2=130 kJ/mol **是错误的**，请勿采用。

---

## 修改 4 【P1-必改】确认并修正均相动力学活化能

**文件**：`src/model/kinetics_service.py`，`KineticsService.__init__`

**问题**：Fortran 均相反应写法为 `exp(-E_cal / (1.987 × ts))`，分子直接是 cal/mol，换算只需 `× 4.184`。

| 反应 | Fortran 行 | Fortran 写法 | 正确 E (J/mol) |
|------|-----------|-------------|----------------|
| WGS | L1271 wgshift | `exp(-27760./(1.987*ts))` | **116,100** |
| MSR | L1355 ch4ref  | `exp(-30000./(1.987*ts))` | **125,520** |

**修改方案**：

```python
# src/model/kinetics_service.py
CAL2J = 4.184  # J/cal

# WGS（水煤气变换）：wgshift L1271
# Fortran: ek = exp(-27760./(1.987*ts))
# 27760 单位是 cal/mol（R 已在分母中），E = 27760 × 4.184
E_WGS = 27760.0 * CAL2J    # = 116,113 J/mol

# WGS 前因子：wgshift L1281
# Fortran: rate4 = f*(2.877e5)*ek*(pexc/pt)*pf*rat
# 其中 f=0.2（催化因子，Fortran L1267）
A_WGS = 2.877e5             # Fortran 原值

# MSR（甲烷重整）：ch4ref L1355
# Fortran: ek = 312.*exp(-30000./(1.987*ts))
# 30000 单位是 cal/mol，E = 30000 × 4.184
E_MSR = 30000.0 * CAL2J    # = 125,520 J/mol
A_MSR = 312.0               # Fortran 原值

# 温度阈值（严格遵守 Fortran）：
# WGS：ts <= 1000K 时 rate=0（wgshift L1268）
# MSR：ts <= 1000K 时 rate=0（ch4ref L1354）
T_THRESHOLD_WGS = 1000.0    # K
T_THRESHOLD_MSR = 1000.0    # K
```

> **注意**：MSR 活化能从 249 kJ/mol 降到 125.5 kJ/mol 会使甲烷重整**加快**（吸热增加），这是还原正确参数，不是为了提高温度，需与其他修改配合才能看到净效果。

---

## 修改 5 【P2-建议】能量残差自适应缩放

**文件**：`src/model/cell.py`，`residuals()` 函数

**问题**：当前固定缩放系数 `5e5` 与实际能量残差量级（~10⁷ W）不匹配，导致归一化后能量方程权重远大于质量方程（缩放后约 225 vs 1），Newton 迭代容易收敛到冷态解。

**修改方案**：

```python
# src/model/cell.py，residuals() 中能量残差归一化处

# 修改前
res_E_sc = res_E / 5.0e5

# 修改后：自适应缩放，使能量残差与质量残差量级一致
# ref_flow 是格内总气相摩尔流量 [mol/s]，Cp_ref 约 35 J/mol/K
ref_energy = max(self.ref_flow * 35.0 * 200.0, 5.0e5)  # 最低保留 5e5 兜底
res_E_sc = res_E / ref_energy
```

---

## 修改 6 【P2-需确认】入口浆液水的焓基准

**文件**：`src/model/gasifier_system.py` 或 `src/model/source_terms.py`

**问题**：Fortran 将 `fsteam`（水煤浆水分）以**气态蒸汽**直接输入（L138），无额外蒸发处理。Python 的 `EvaporationSource` 使用液态水生成焓（`H_LIQUID = -285830 J/mol`）在 Shomate 全焓框架下热力学自洽，**但前提是入口浆液水必须被设为液态水焓**，否则形成双重计算（多出 ~17 MW 热沉）。

**需要确认**：检查入口 `State` 初始化时，`H2O` 的焓值是用 Shomate 气态焓（`H_gas(T_inlet)`）还是液态水焓（`H_LIQUID`）设置的。

- 若用 `H_gas(T_inlet)` 初始化入口 H2O → `EvaporationSource` 构成双重计算 → 应**删除** `EvaporationSource`，改为在入口直接设液态水焓
- 若用 `H_LIQUID` 初始化入口 H2O → `EvaporationSource` 正确 → 无需修改此处

---

## 修改优先级汇总

| 优先级 | 文件 | 修改内容 | 预期效果 |
|--------|------|---------|---------|
| **P0** | `pyrolysis_service.py` | 修复 `*1e6` → `*1e3`，删除 Step 9 | 恢复完整挥发分，+28 MW |
| **P0** | `material.py` | 固相焓去掉 `Hf_char` 经验公式 | 消除 ~6 MW 误差和跳变 |
| **P1** | `kinetics.py` | 确认 C+H2O/CO2 E=175 kJ/mol | 还原 Fortran 气化动力学 |
| **P1** | `kinetics_service.py` | WGS E=116 kJ/mol，MSR E=125.5 kJ/mol | 还原 Fortran 均相动力学 |
| **P2** | `cell.py` | 能量残差自适应缩放 | 改善 Newton 收敛稳定性 |
| **P2** | `gasifier_system.py` | 确认入口 H2O 焓基准无双重计算 | 消除潜在 17 MW 虚假热沉 |

---

## 重要提示

1. **P0 两项先做**，做完后单独跑 Base Case 验证，再做 P1。
2. **不要采用**另一 AI 建议的 C+O2=110 kJ/mol 和 C+H2=130 kJ/mol，这两个值没有 Fortran 依据（正确值分别是 149.4 和 149.0 kJ/mol）。
3. **MSR 降低活化能会使吸热加快**，这是还原参数而非调优，需要配合 P0 的能量框架修正才能看到净效果。
