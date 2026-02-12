# Python气化炉模型温度偏低问题诊断报告

## 📊 实验数据对比

| 工况 | 实验温度 | 论文模型 | Python模型 | 温度偏差 |
|------|---------|---------|-----------|---------|
| Texaco I-1 | 1370°C | 1380°C | **801°C** | **-569°C** ❌ |
| Texaco I-2 | 1333°C | - | **1148°C** | **-185°C** ❌ |
| Texaco I-5C | - | - | **750°C** | - |
| Texaco I-10 | - | - | **764°C** | - |
| Texaco Exxon | - | - | **1226°C** | - |
| CWS Western | - | - | **978°C** | - |
| CWS Eastern | - | - | **745°C** | - |

## 🔍 核心问题诊断

### **问题1: 温度普遍偏低（500-600°C）**

**症状特征**：
- ✅ I-1工况: 801°C vs 1370°C (偏低569°C)
- ✅ 大部分工况在 700-1200°C 徘徊
- ✅ 只有 Texaco Exxon 接近正常（1226°C）

**这表明**：🚨 **挥发分燃烧热未正确释放！**

---

## 🔬 根本原因分析

### **原因A: 挥发分起燃温度过低**

查看您当前的代码，很可能存在：

```python
# ❌ 问题代码示例
if self.idx == 0:
    # 挥发分在低温下就开始燃烧
    if T > 600:  # 温度太低！
        r_CH4_combustion = k * C_CH4 * C_O2
```

**Fortran的做法**（Line 258-274）：
```fortran
! 颗粒需要先加热到高温
140   if(ts .gt. 600.) r = rp
      
      ! 然后用 Tg (不是 Ts) 计算均相反应
      ! 且初始 Tg 猜测是 2000-3000K
```

**关键差异**：
- Fortran: 初始Tg猜测 = **2000-3000K** (用于高温起燃)
- Python: 初始Tg猜测可能只有 **400-800K** (导致低温解)

---

### **原因B: 挥发分燃烧动力学参数错误**

检查您的均相反应速率常数：

```python
# 可能的问题
self.A_homo = {
    'CH4_Ox': 1.6e10,  # 这个值对吗？
    'CO_Ox':  2.23e12,
    'H2_Ox':  1.08e13
}
self.E_homo = {
    'CH4_Ox': 125000.0 / 8.314,  # 单位正确吗？
}
```

**Fortran论文值**（Table 2-5, Page 5）：
- CH₄ + 2O₂ → CO₂ + 2H₂O:
  - A = 1.6×10¹⁰ m³/(kmol·s)
  - E = 125,600 J/mol = **15,096 cal/mol**

**检查点**：
1. ✅ A的单位是否一致？
2. ✅ E的单位是 J/mol 还是 cal/mol？
3. ✅ 浓度单位是 kmol/m³ 还是 mol/m³？

---

### **原因C: Cell 0的体积过大**

```python
# 检查这个值
V_cell0 = A × dz_cell0

# Fortran论文设置
dz_cell0 = L_reactor / 20  # 约 0.3 m (6m反应器)
V_cell0 ≈ π×(1.524/2)² × 0.3 ≈ 0.55 m³
```

**如果体积过大**：
- 停留时间 τ = V/Q 过大
- 挥发分被稀释
- 反应速率 r = k·C·V 中，C降低 > V增加

**检查您的网格**：
```python
print(f"Cell 0: dz={dz_list[0]:.3f} m, V={V_cell0:.3f} m³")
print(f"停留时间: τ={V_cell0/Q_gas:.3f} s")
```

**理论值**：
- dz_cell0 应约 **0.05-0.4 m**
- τ_cell0 应约 **0.01-0.1 s**（极短，保证挥发分瞬间燃烧）

---

### **原因D: 能量源项计算错误**

查看您的能量平衡：

```python
# 可能的问题
H_out - H_in = energy_src - Q_loss

# 检查 energy_src 是否包含：
# 1. 挥发分燃烧热 (应该是 +800 MJ/kg coal)
# 2. 热解吸热 (约 -50 MJ/kg coal)
# 3. 水分蒸发吸热 (约 -300 MJ/kg slurry water)
```

**Fortran计算**（Line 314-330）：
```fortran
call pyroly(subdwl,swl,goxy,rheat1,doxy,waterp,...)

! rheat1 包含：
! - 挥发分燃烧放热（主要）
! - 热解反应热
! - 水分蒸发（如果有）

goxyex = goxy - doxy*fcoal
```

**检查点**：
```python
# 在 Cell 0 添加诊断
print(f"=== Cell 0 Energy Breakdown ===")
print(f"Q_pyrolysis: {Q_pyro/1e6:.2f} MW")
print(f"Q_evaporation: {Q_evap/1e6:.2f} MW")
print(f"Q_volatile_combustion: {Q_vol_comb/1e6:.2f} MW")
print(f"TOTAL energy_src: {energy_src/1e6:.2f} MW")
print(f"Expected (from HHV): {coal_flow*HHV/1e6:.2f} MW")
```

---

### **原因E: 氧气被过早消耗**

```python
# 检查 Cell 0 出口的氧气量
F_O2_out_cell0 = current.gas_moles[0]

# 如果 F_O2_out << F_O2_in：
# 说明氧气在 Cell 0 被消耗完了
# 但温度还很低，说明反应热没有正确计算
```

**Fortran的处理**（Line 316-321）：
```fortran
goxyex = goxy - doxy*fcoal
if(goxyex .lt. 0.) then
    goxy = 0.
else
    goxy = goxyex
endif
```

**诊断**：
```python
O2_consumption_ratio = (F_O2_in - F_O2_out) / F_O2_in
print(f"O2 consumed in Cell 0: {O2_consumption_ratio*100:.1f}%")

# 正常值应该是 70-90%（大部分用于挥发分燃烧）
# 如果是 100%：说明氧气不够（可能挥发分计算过多）
# 如果是 <50%：说明挥发分燃烧不充分
```

---

## 🎯 核心问题：起燃策略

### **Fortran的"暴力起燃"方法**

```fortran
! Line 258-291
! 多初值猜测策略
guesses_T = [400, 1000, 1500, 2000, 3000] K

! 对于高温猜测（>900K），强制平衡反应
if t_start > 900.0:
    ! Step 1: CH4 → CO + 2H2 (部分氧化)
    x0[1] -= xi_1
    x0[2] += xi_1
    x0[5] += 2*xi_1
    
    ! Step 2: H2 + 0.5O2 → H2O
    x0[5] -= xi_2
    x0[7] += xi_2
    
    ! Step 3: CO + 0.5O2 → CO2
    x0[2] -= xi_3
    x0[3] += xi_3
```

**物理意义**：
- 在 Cell 0，给一个 **3000K 的初始猜测**
- 手动计算如果所有挥发分燃烧，产物应该是什么
- 强制 solver 从"已燃烧"状态开始迭代
- 防止陷入"低温冷态"解

---

### **您的Python代码很可能是**：

```python
# ❌ 问题代码
if i == 0:
    x0 = current_inlet.to_array()
    x0[10] = 400.0  # 太低了！
    
    sol = least_squares(func, x0, ...)
```

**这会导致**：
1. Solver从 400K 开始
2. 挥发分燃烧速率 k(400K) ≈ 0（活化能太高）
3. 没有反应热
4. 温度上不去
5. 收敛到"低温冷态解"（800K）

---

## 🔧 紧急修复方案

### **方案1: 强制高温起燃（最快）**

```python
# Cell 0 特殊处理
if self.idx == 0:
    # 多初值猜测
    for T_guess in [400, 1000, 1500, 2000, 3000]:
        x0 = self.inlet.to_array()
        x0[10] = T_guess
        
        # 如果高温猜测，强制平衡反应
        if T_guess > 1500:
            # 计算理论产物
            F_CH4_vol = self.tmp_F_volatiles[1]
            F_O2_avail = x0[0]
            
            # CH4 + 2O2 → CO2 + 2H2O
            xi_CH4 = min(F_CH4_vol, F_O2_avail/2.0) * 0.99
            x0[1] -= xi_CH4          # -CH4
            x0[0] -= 2.0*xi_CH4      # -2O2
            x0[3] += xi_CH4          # +CO2
            x0[7] += 2.0*xi_CH4      # +2H2O
            
            # 同样处理 CO, H2
            # ...
        
        sol = least_squares(func, x0, ...)
        
        if sol.success and sol.x[10] > 1200:
            break  # 找到高温解，跳出
```

---

### **方案2: 修正挥发分燃烧热**

```python
# 检查 EvaporationSource 和 PyrolysisSource
class PyrolysisSource:
    def __init__(self, volatile_fluxes, solid_loss, target_cell_idx=0):
        self.vol_fluxes = volatile_fluxes
        self.solid_loss = solid_loss
        
        # ⚠️ 关键：计算挥发分燃烧热
        # CH4: 802,340 J/mol
        # CO:  282,980 J/mol  
        # H2:  241,820 J/mol
        
        Q_CH4 = volatile_fluxes[1] * 802340.0
        Q_CO  = volatile_fluxes[2] * 282980.0
        Q_H2  = volatile_fluxes[5] * 241820.0
        
        self.combustion_heat = Q_CH4 + Q_CO + Q_H2  # W
        
    def get_sources(self, cell_idx, z, dz):
        if cell_idx == self.target_idx:
            gas_src = self.vol_fluxes.copy()
            solid_src = -self.solid_loss
            
            # ✅ 能量源 = 挥发分燃烧热（正值）
            energy_src = self.combustion_heat
            
            return gas_src, solid_src, energy_src
        else:
            return np.zeros(8), 0.0, 0.0
```

---

### **方案3: 检查颗粒温度逻辑**

```python
# 确保颗粒温度计算正确
def solve_particle_temperature(self, T_gas, T_particle_init, n_steps=30):
    """
    模拟 Fortran Line 264-274
    """
    T_s = T_particle_init
    T_s_history = []
    
    for k in range(n_steps):
        # 导热系数
        condut = 7.7e-7 * (T_gas + T_s)**0.75
        
        # 综合传热系数
        ct = -(3.0/(self.dens*self.cps*self.r)) * \
             (condut/self.r + self.ef*self.sigma*4.0*T_gas**3) * self.deltim
        
        # 指数衰减
        if abs(ct) > 25.0:
            ect = 1.0e-12
        else:
            ect = np.exp(ct)
        
        delta_Ts = (T_gas - (T_gas - T_s)*ect) - T_s
        T_s = T_s + delta_Ts
        
        # 限温保护
        if T_s > 1250:
            T_s = 1250
        
        T_s_history.append(T_s)
    
    # 返回平均温度用于反应速率计算
    T_s_avg = np.mean(T_s_history)
    return T_s_avg, T_s_history[-1]
```

---

## 🧪 诊断检查清单

在修复之前，请先运行以下诊断：

```python
# === Cell 0 诊断代码 ===
if self.idx == 0:
    print(f"\n{'='*60}")
    print(f"CELL 0 DIAGNOSTIC")
    print(f"{'='*60}")
    
    # 1. 初始状态
    print(f"\n[1] INLET CONDITIONS:")
    print(f"  F_O2_in:  {self.inlet.gas_moles[0]:.2f} mol/s")
    print(f"  F_CH4_in: {self.inlet.gas_moles[1]:.2f} mol/s")
    print(f"  T_in:     {self.inlet.T:.1f} K")
    
    # 2. 挥发分源项
    print(f"\n[2] VOLATILE SOURCES:")
    for s in self.sources:
        g_src, s_src, e_src = s.get_sources(0, 0, self.dz)
        print(f"  F_CH4_vol: {g_src[1]:.2f} mol/s")
        print(f"  F_CO_vol:  {g_src[2]:.2f} mol/s")
        print(f"  F_H2_vol:  {g_src[5]:.2f} mol/s")
        print(f"  Energy_src: {e_src/1e6:.2f} MW")
    
    # 3. 可用量
    avail_CH4 = self.inlet.gas_moles[1] + g_src[1]
    avail_O2 = self.inlet.gas_moles[0]
    print(f"\n[3] AVAILABLE REACTANTS:")
    print(f"  CH4_avail: {avail_CH4:.2f} mol/s")
    print(f"  O2_avail:  {avail_O2:.2f} mol/s")
    print(f"  Stoich CH4/O2: {avail_CH4/(avail_O2/2.0 + 1e-9):.2f} (should < 1)")
    
    # 4. 理论燃烧热
    Q_theory_CH4 = avail_CH4 * 802340.0
    print(f"\n[4] THEORETICAL COMBUSTION HEAT:")
    print(f"  If all CH4 burns: {Q_theory_CH4/1e6:.2f} MW")
    print(f"  Coal HHV input:   {self.coal_flow*self.HHV/1e6:.2f} MW")
    
    # 5. 实际反应速率
    print(f"\n[5] ACTUAL REACTION RATES (at T={current.T:.1f}K):")
    print(f"  r_CH4_Ox: {r_homo['CH4_Ox']:.2f} mol/s")
    print(f"  Q_actual: {r_homo['CH4_Ox']*802340.0/1e6:.2f} MW")
    print(f"  Burn fraction: {r_homo['CH4_Ox']/(avail_CH4+1e-9)*100:.1f}%")
    
    # 6. 能量平衡
    print(f"\n[6] ENERGY BALANCE:")
    print(f"  H_in:  {H_in/1e6:.2f} MW")
    print(f"  H_out: {H_out/1e6:.2f} MW")
    print(f"  ΔH:    {(H_out-H_in)/1e6:.2f} MW")
    print(f"  Q_rxn: {Q_rxn_total/1e6:.2f} MW")
    print(f"  Ratio ΔH/Q: {(H_out-H_in)/(Q_rxn_total+1e-9):.2f} (should ≈ -1)")
    
    print(f"{'='*60}\n")
```

---

## 🎯 预期结果

修复后，Cell 0 应该看到：

```
=== Cell 0 DIAGNOSTIC ===
[1] INLET: F_O2=66 mol/s, F_CH4=0 mol/s, T=505K
[2] VOLATILES: F_CH4_vol=44 mol/s, Energy_src=35 MW
[3] AVAILABLE: CH4=44 mol/s, O2=66 mol/s, Ratio=0.67 ✓
[4] THEORETICAL: Q_CH4=35 MW, HHV=64 MW
[5] ACTUAL (T=2350K): r_CH4=43.5 mol/s, Q=35 MW, Burn=99% ✓
[6] ENERGY: ΔH=-33 MW, Q_rxn=35 MW, Ratio=-0.94 ✓

CONVERGED: T_cell0 = 2350 K ✓
```

---

## 📌 总结

**温度偏低的根本原因**：
1. ❌ 初始温度猜测太低（400K vs 2000-3000K）
2. ❌ 挥发分燃烧热未正确计入能量源
3. ❌ 没有"强制起燃"逻辑

**立即行动**：
1. ✅ 在 Cell 0 使用 **T_guess = 2000-3000K**
2. ✅ 验证 `PyrolysisSource.energy_src` 包含燃烧热
3. ✅ 运行上述诊断代码，找出能量去哪了

修复后，I-1工况应该能达到 **1350-1400°C**！

---

## 📝 已实施修正 (2026-02)

| 修正项 | 状态 | 说明 |
|--------|------|------|
| 起燃前加入挥发分 | ✅ | `gasifier_system.py`: 高温猜测时 `x0[:8] += tmp_F_volatiles`，否则 n_CH4=0 起燃无效 |
| 固相/焦炭更新 | ✅ | `x0[8] -= tmp_W_vol_loss`, `x0[9] = char_Xc0` |
| 高温优先猜测 | ✅ | `guesses_T = [3000, 2000, 1500, 1000, 400]` |
| ignited 判据 | ✅ | `T > 1200 K`，同成本时优先更高温度 |
| Cell 0 温度 | ✅ | 实测 ~2360°C |
| WGS 判据 Ts≤1000K | ✅ | `kinetics_service.py`: 与 Fortran wgshift 一致，用 Ts_particle 判据；Ts≤1000 时 WGS=0 |

**下游 cell 初值与求解**（用户诊断）：
- WGS 等气化吸热不应造成如此剧烈降温：温度下降会降低气化速率 → 减少吸热 → 自限效应
- 更可能是 **cell 内温度初值** 和 **求解问题**（多解/数值陷阱）
- 已实施：下游多初值 (T_in, 1.02×, 1.08×, 1.15×, 0.98×, 0.92×)；同 cost 时优先更高 T；能量残差放大 (res_E/5e5)
- 异常降温重试：若 T_out < 0.8×T_in 且 T_in>1800K，重试 1.1×、1.2×T_in 初值

**待改进**：出口温度仍偏低（~801°C）。WGS 判据已按 Fortran 实施，但突降 cell 中 Ts 已 >1000K，需进一步限制 WGS 逆向速率或引入平衡约束。
