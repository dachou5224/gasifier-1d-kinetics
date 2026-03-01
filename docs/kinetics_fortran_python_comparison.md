# 异相与均相反应动力学参数 — Fortran vs Python 核对

> 核对依据：`reference_fortran/Source1_副本.for`  
> 换算：Fortran `exp(-E_over_R/ts)` → E(J/mol) = E_over_R × 1.987 × 4.184  
> Fortran `exp(-E_cal/(1.987*ts))` → E(J/mol) = E_cal × 4.184  

---

## 1. 异相反应（Heterogeneous）

| 反应 | Fortran 子程序 | Fortran 行 | Fortran 公式 | A (Fortran) | E_over_R (K) | E (J/mol) | Python A | Python E (J/mol) | 一致 |
|------|----------------|------------|--------------|-------------|--------------|-----------|----------|------------------|------|
| C+O2 | combus | L1173,1179 | `ats=-17967/ts`<br>`ks=8710*exp(ats)` | 8710 | 17967 | 149,400 | 8710 | 149,400 | ✅ |
| C+H2O | cbstm | L1215,1218 | `ats=-21060/ts`<br>`ks=247*exp(ats)` | 247 | 21060 | 175,100 | 247 | 175,100 | ✅ |
| C+CO2 | cbco2 | L1246,1249 | `bts=-21060/ts`<br>`ks=247*exp(bts)` | 247 | 21060 | 175,100 | 247 | 175,100 | ✅ |
| C+H2 | cbhym | L1331,1333 | `bts=-17921/ts`<br>`ks=0.12*exp(bts)` | 0.12 | 17921 | 149,000 | 0.12 | 149,000 | ✅ |

**换算公式**：`E(J/mol) = E_over_R × 1.987 × 4.184`

- C+O2: 17967 × 1.987 × 4.184 = **149,400 J/mol**
- C+H2O/CO2: 21060 × 1.987 × 4.184 = **175,100 J/mol**
- C+H2: 17921 × 1.987 × 4.184 = **149,000 J/mol**

---

## 2. 均相反应（Homogeneous）

### 2.1 WGS 与 MSR（Fortran 有直接对应）

| 反应 | Fortran 子程序 | Fortran 行 | Fortran 公式 | A (Fortran) | E_cal (cal/mol) | E (J/mol) | Python A | Python E (J/mol) | 一致 |
|------|----------------|------------|--------------|-------------|-----------------|-----------|----------|------------------|------|
| WGS | wgshift | L1271,1281 | `ek=exp(-27760/(1.987*ts))`<br>`rate4=f*(2.877e5)*ek*...` | 2.877e5 | 27760 | 116,100 | 2.877e5 | 116,100 | ✅ |
| MSR | ch4ref | L1355 | `ek=312*exp(-30000/(1.987*ts))` | 312 | 30000 | 125,500 | 312 | 125,500 | ✅ |

**换算公式**：`E(J/mol) = E_cal × 4.184`

- WGS: 27760 × 4.184 = **116,100 J/mol**
- MSR: 30000 × 4.184 = **125,500 J/mol**

**注意**：WGS 有效指前因子 = `f × 2.877e5`，其中 `f = 0.2`（催化因子，L1267）。Python 在 `calc_homogeneous_rates` 中单独乘 `WGS_CATALYTIC_FACTOR = 0.2`。

### 2.2 CO/H2/CH4 氧化（Fortran 无动力学）

| 反应 | Fortran 处理 | Python A | Python E (J/mol) | 说明 |
|------|---------------|----------|------------------|------|
| CO_Ox | **瞬时燃烧**（pyroly L1057-1088）<br>无 Arrhenius 动力学 | 2.23e12 | 125,000 | 来自 Wen & Chaung Table 2-5 |
| H2_Ox | **瞬时燃烧** | 1.08e13 | 83,700 | 来自 Wen & Chaung Table 2-5 |
| CH4_Ox | **瞬时燃烧** | 1.6e10 | 125,600 | 来自 Wen & Chaung Table 2-5 |

**Fortran 机制**：挥发分（H2、CO、CH4）在燃烧区（pO2 > 0.05 atm）按化学计量**瞬时完全燃烧**，仅用燃烧焓计算放热，**无速率方程**。见 `fortran_combustion_mechanism.md`。

**Python 机制**：使用 Arrhenius 动力学 `r = k·C_A·C_B`，A/E 来自 Wen & Chaung (1979) 论文 Table 2-5。与 Fortran 实现方式不同，属设计选择。

---

## 3. 汇总

| 类别 | 反应数 | 与 Fortran 一致 | 说明 |
|------|--------|-----------------|------|
| 异相 | 4 | 4/4 | 全部对齐 Fortran |
| 均相（WGS/MSR） | 2 | 2/2 | 全部对齐 Fortran |
| 均相（氧化） | 3 | 0/3 | Fortran 用瞬时燃烧，无 A/E |

---

## 4. Fortran 源码摘录

### C+O2 (combus L1173-1179)
```fortran
      ats = -17967./ts
      eats = exp(ats)
      ks = 8710.*eats
```

### C+H2O (cbstm L1215-1218)
```fortran
      ats = -21060./ts
      eats = exp(ats)
      ks = 247.*eats
```

### C+CO2 (cbco2 L1246-1249)
```fortran
      bts = -21060./ts
      ebts = exp(bts)
      ks = 247.*ebts
```

### C+H2 (cbhym L1331-1333)
```fortran
      bts = -17921./ts
      ks = 0.12*exp(bts)
```

### WGS (wgshift L1267-1281)
```fortran
      f = 0.2
      if(ts .le. 1000.) goto 10
      ek = exp(-27760./(1.987*ts))
      ...
      rate4 = f*(2.877e5)*ek*(pexc/pt)*pf*rat
```

### MSR (ch4ref L1355)
```fortran
      ek = 312.*exp(-30000./(1.987*ts))
```
