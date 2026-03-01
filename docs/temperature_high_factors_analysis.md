# 温度偏高影响因素分析

> 除 P0/P1 修改外，其他可能导致温度偏高的因素

---

## 1. 热损失（Heat Loss）— 与 Fortran 显著不同

### 1.1 Fortran 热损计算（Source1_副本.for）

| 区域 | 条件 | 公式 | 说明 |
|------|------|------|------|
| **燃烧区** | `poxyin > 0.05 atm` | `hloss = fcoal*abs(rheat)*hl/100.` | **30% 的格内反应热** |
| **气化区** | `poxyin ≤ 0.05 atm` | `hloss = u0*3.14*dt*nc*delh*(tg-tw)` | 对流散热 |

- `hl = 30`（L30）：燃烧热传给壁面的比例
- `rheat`：格内总反应热（挥发分燃烧 + 焦炭燃烧 + 气化 + WGS 等）
- 燃烧区：**hloss = 30% × 格内 rheat**，与局部放热成正比

### 1.2 Python 热损计算（cell.py L366-377）

```python
Q_loss_W = (loss_pct / 100.0) * coal_flow * HHV * (self.dz / L_total)
```

- **基准**：入炉煤 HHV 的 `loss_pct`%（通常 2–3%）
- **分布**：按 `dz/L_total` 均匀摊分到各格
- 总热损 = `loss_pct% × 煤流量 × HHV`，与局部反应热无关

### 1.3 差异与对温度的影响

| 项目 | Fortran | Python | 影响 |
|------|---------|--------|------|
| 基准 | 30% × 格内 rheat | 2–3% × 煤 HHV | 燃烧区 Fortran 热损更大 |
| 空间分布 | 与 rheat 成正比（燃烧区热损大） | 按 dz 均匀 | Fortran 在燃烧区散热更多 |
| 量级估算 | 燃烧区 rheat~50–100 MW → 15–30 MW 热损 | 3%×96 MW ≈ 2.9 MW 总热损 | **Python 热损明显偏小** |

**结论**：Python 热损远小于 Fortran 燃烧区热损，是温度偏高的**重要因素**。若将热损改为与 Fortran 类似的“反应热比例”形式，或提高 `HeatLossPercent`，可显著降温。

---

## 2. WGS 平衡常数 — 与 Fortran 不一致

### 2.1 公式对比

| 来源 | 公式 | 等价形式 |
|------|------|----------|
| Fortran wgshift L1270 | `ckwg = exp(-3.6893 + 7234/(1.8*tm))` | `exp(4019/T - 3.69)` |
| Python kinetics.py | `K = exp(4578/T - 4.33)` | 文献 Table 2-2 |

- `tm = (tg+ts)/2`（气相与颗粒平均温度）

### 2.2 数值差异（T=1500 K）

| 来源 | K |
|------|---|
| Fortran | exp(2.68 - 3.69) ≈ 0.36 |
| Python | exp(3.05 - 4.33) ≈ 0.28 |

Python K 更小 → 平衡更偏向 CO+H2O（反应物）→ WGS 正向净速率略小 → 略少放热。因此该差异会**略降**温度，不是升温原因。

---

## 3. WGS 动力学与 rat 因子

### 3.1 rat 因子

| 来源 | 公式 | Python 默认 |
|------|------|-------------|
| Fortran L1274 | `rat = exp(-8.91 + 5553/ts)` | **关闭**（`WGS_RatFactor=False`） |

- rat 在高温下约 0.01–0.1，会压低 WGS 速率
- Python 默认不启用 → WGS 速率偏大 → CO→CO2 更多，WGS 正向略多放热
- 影响量级有限，但方向为**略升**温

### 3.2 催化因子 f=0.2

- Fortran 与 Python 均为 0.2，一致。

---

## 4. 其他可能因素

| 因素 | 说明 | 对温度影响 |
|------|------|------------|
| **气化区热损** | Fortran 用对流 `u0*π*dt*nc*delh*(tg-tw)`，Python 仍用 HHV% | 气化区 Python 热损可能偏小 |
| **蒸发热沉** | EvaporationSource 用 H_LIQUID，与 Fortran 浆液处理方式不同 | 已确认无双重计算 |
| **固相焓** | P0-2 简化后实际降温 | 非升温原因 |
| **挥发分产率** | 元素守恒产率可能偏大 | 燃烧放热偏大，可能升温 |

---

## 5. 建议修改（优先热损）

### 5.1 热损对齐 Fortran（已回滚）

曾尝试在燃烧区采用 Fortran 式热损（30%/15% × 格内反应热），温度显著下降，但**合成气组分（CO、H2、CO2）几乎不变**。温度变化对气体组分无影响，说明组分偏差来自 WGS/氧化动力学，非热损。已回滚至原始 HHV% 热损方式。

### 5.2 WGS 平衡常数对齐 Fortran

```python
# kinetics.py
def calculate_wgs_equilibrium(T):
    # Fortran: ckwg = exp(-3.6893 + 7234/(1.8*tm)), tm≈T
    # 等价: exp(4019/T - 3.69)
    return np.exp(4019.0 / T - 3.6893)
```

### 5.3 启用 WGS rat 因子

在 `op_conds` 中设置 `WGS_RatFactor: True`，与 Fortran 一致。

---

## 6. 汇总

| 因素 | Fortran | Python | 对温度影响 | 优先级 |
|------|---------|--------|------------|--------|
| **热损** | 燃烧区 30%×rheat | 2–3%×HHV 均匀 | **Python 热损过小 → 升温** | 高 |
| WGS 平衡 K | exp(4019/T-3.69) | exp(4578/T-4.33) | Python K 小 → 略降温 | 低 |
| WGS rat | 启用 | 默认关闭 | 关闭 → 略升温 | 中 |

**结论**：热损计算方式与 Fortran 不一致，是温度偏高的主要潜在原因，建议优先调整热损逻辑或参数。
