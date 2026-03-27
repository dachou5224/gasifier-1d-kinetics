# JAX 求解器升级方案
## 气化炉一维模型加速路线图

---

## 一、当前性能基准（实测）

| 指标 | 数值 |
|---|---|
| scipy `least_squares` (TRF) 单 cell | **23 ms/cell** |
| JAX `jacfwd` + JIT 单 Newton 步 | **0.37 ms/step** |
| 20 cells 完整运行（scipy） | **~460 ms/run** |
| scipy 平均 nfev（每 cell）| ~43 次函数评估 |

**加速潜力估算**：JAX Newton 30 步 × 0.37 ms = **11 ms/cell** → 预期 **2–3x 加速**  
Cell 0（多初值猜测 × 5）目前最慢，有额外加速空间。

---

## 二、升级策略：三个独立阶段（可逐步落地）

### 阶段 0：现状（基线）

```
gasifier_system.py
  └── solve()
        └── for cell in cells:
              └── scipy.least_squares(cell.residuals, x0, method='trf')
                    └── cell.residuals(x) → numpy 有限差分 Jacobian
```

**瓶颈**：每次 `least_squares` 调用需要 12×（nfev≈43 × 11变量扰动）次函数评估。

---

### 阶段 1：JAX 替换 Jacobian 计算（最小改动，无风险）

**原理**：保留 scipy `least_squares` 框架，只把 Jacobian 来源换成 JAX 自动微分。  
scipy 的 `jac` 参数支持传入精确 Jacobian 函数。

**改动范围**：仅新增 `jax_kinetics.py` 和修改 `solver.py`，不碰 `cell.py`。

```
# solver.py 修改（约 20 行）
from scipy.optimize import least_squares
import jax, jax.numpy as jnp
from jax import jit, jacfwd

class HybridSolver:
    """scipy TRF + JAX 精确 Jacobian"""
    
    def __init__(self, residuals_fn):
        # residuals_fn: 原始 numpy 函数
        # 将其包装为 JAX 版本
        self._res_jax = lambda x: jnp.array(residuals_fn(np.array(x)))
        self._jac_jit = jit(jacfwd(self._res_jax))
    
    def solve(self, x0, bounds):
        def jac_fn(x):
            return np.array(self._jac_jit(jnp.array(x)))
        
        return least_squares(
            residuals_fn, x0, jac=jac_fn,  # ← 关键：注入精确 Jacobian
            bounds=bounds, method='trf',
            xtol=1e-10, ftol=1e-10, max_nfev=500
        )
```

**预期收益**：  
- Jacobian 精度从有限差分（O(ε)）提升到机器精度  
- 减少因 Jacobian 不精确导致的额外 Newton 步  
- 估计加速 **1.5–2x**，且收敛更稳定（对 Cell 0 高温起燃尤为重要）

**实施工作量**：1–2 天

---

### 阶段 2：纯 JAX Newton 求解器（替换 scipy）

**原理**：用 JAX JIT 编译完整的 Newton-Raphson 循环，消除 Python overhead。

```python
# src/model/jax_solver.py（新文件，约 80 行）

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, lax
from functools import partial

@partial(jit, static_argnums=(2,))
def newton_solve_cell(x0, inlet_arr, n_iter=40):
    """
    JIT编译的单 cell Newton-Raphson 求解器
    
    参数:
        x0: 初始猜测 (11,) float32
        inlet_arr: 入口状态 (11,) 打包数组
        n_iter: 固定迭代次数（JAX lax.scan 要求静态）
    返回:
        x_final: 收敛解 (11,)
        converged: bool
        max_res: 最终残差范数
    """
    lower = jnp.array([0.]*8 + [0., 0., 300.])
    upper = jnp.array([1e5]*8 + [100., 1.0, 4000.])
    
    def body_fn(carry, _):
        x, prev_res = carry
        F = _residuals_jax(x, inlet_arr)        # 残差
        J = jacfwd(_residuals_jax)(x, inlet_arr)  # 精确 Jacobian
        
        # 阻尼 Newton 步（damping=0.8）
        delta = jnp.linalg.solve(J + 1e-10*jnp.eye(11), -F)
        x_new = jnp.clip(x + 0.8*delta, lower, upper)
        
        max_res = jnp.max(jnp.abs(F))
        return (x_new, max_res), max_res
    
    (x_final, final_res), res_history = lax.scan(
        body_fn, (x0, 1e10), None, length=n_iter
    )
    converged = final_res < 1e-8
    return x_final, converged, final_res


# 多初值版本（用于 Cell 0 起燃）
@partial(jit, static_argnums=(2,3))
def newton_solve_multistart(x0_list, inlet_arr, n_iter=40, n_starts=5):
    """
    vmap 并行尝试多个初始猜测（Cell 0 起燃策略）
    替代现有 gasifier_system.py 中的 for t_start in guesses_T 循环
    """
    # vmap 对 n_starts 个初值并行求解
    solve_batch = jax.vmap(
        lambda x0: newton_solve_cell(x0, inlet_arr, n_iter)
    )
    x_sols, converged, res_norms = solve_batch(x0_list)  # (n_starts, 11)
    
    # 选择：已收敛且温度最高的解
    T_vals = x_sols[:, 10]
    ignited = (T_vals > 1200.0) & converged
    
    # 优先选已起燃解，其次选残差最小解
    score = jnp.where(ignited, -T_vals, res_norms * 1e6)
    best_idx = jnp.argmin(score)
    return x_sols[best_idx], converged[best_idx], res_norms[best_idx]
```

**关键设计决策**：

| 问题 | JAX 方案 |
|---|---|
| `lax.scan` 要求固定迭代次数 | 设 n_iter=40，内部通过 `converged` flag 提前收敛时 delta≈0 |
| Cell 0 多初值猜测 | `vmap` 并行化，不再 for 循环 |
| bounds clip | `jnp.clip` 代替 scipy bounds 参数 |
| 奇异 Jacobian | `J + 1e-10*I` 正则化，等价于 Tikhonov |

**预期收益**：  
- 消除 scipy/Python 的每次调用 overhead  
- Cell 0 的 5 个初值从串行变并行（vmap）  
- 估计加速 **3–5x**

**实施工作量**：3–5 天（含调试）

---

### 阶段 3：残差函数迁移到纯 JAX（完整加速）

**原理**：把 `cell.py` 中的 `residuals()` 完全用 JAX 重写，消除 numpy ↔ jnp 转换开销。

**最小可行重写结构**：

```python
# src/model/jax_cell.py（新文件）

def make_cell_residuals_jax(inlet_state, coal_props, op_conds, dz, A):
    """
    工厂函数：返回一个 JIT 可编译的 residuals 函数（闭包）
    
    设计原则：
    - 所有"编译时常量"（coal_props, geometry）作为闭包捕获
    - 只有 x（状态向量）是运行时变量
    - 这样 JIT 只需编译一次，后续每次调用极快
    """
    # 预计算静态参数
    Cd = coal_props['Cd'] / 100.0
    HHV = coal_props.get('HHV_d', 30.0)
    L = op_conds['L_reactor']
    
    inlet_arr = jnp.array(inlet_state.to_array())
    
    @jit
    def residuals(x):
        T = jnp.clip(x[10], 300., 4000.)
        Ws = jnp.clip(x[8], 0., 100.)
        
        # 动力学（Arrhenius，全 jnp 运算）
        rates = _calc_all_rates_jax(x, T, dz * A, coal_props)
        
        # 质量守恒残差（8 gas + Ws + Xc + T）
        res_mass = _mass_balance_jax(x, inlet_arr, rates)
        
        # 能量守恒残差
        H_in  = _enthalpy_jax(inlet_arr, coal_props)
        H_out = _enthalpy_jax(x, coal_props)
        Q_rxn = _reaction_heat_jax(rates)
        res_E = (H_out - H_in - Q_rxn + Q_loss) / 5e5  # 归一化
        
        return jnp.concatenate([res_mass, jnp.array([res_E])])
    
    return residuals


# 关键子函数（均为纯 jnp）
def _calc_all_rates_jax(x, T, V, coal_props):
    """均相+异相反应速率，纯 JAX"""
    R = 8.314
    # Shomate 热力学 → 平衡常数（jnp.exp）
    # Arrhenius 动力学 → 反应速率
    # UCSM 串联阻力 → 异相速率
    ...
    return {'CO_Ox': r1, 'H2_Ox': r2, 'WGS': r3, 'C+O2': r4, ...}
```

**迁移优先级（按对速度的影响排序）**：

1. `calc_homogeneous_rates` → 纯代数运算，迁移最简单
2. `calc_heterogeneous_rates` → UCSM 串联阻力，无隐式方程，可迁移
3. `_calc_energy_balance` → Shomate 积分，迁移后可消除 `calculate_enthalpy` 的 Python 循环
4. `_calc_particle_temperature` → 包含 `lax.scan` 的子循环，最复杂

**实施工作量**：1–2 周

---

## 三、完整架构图（升级后）

```
gasifier_system.py
  └── solve()
        ├── Cell 0: newton_solve_multistart(5 x0s, inlet, n_iter=40)  ← vmap并行
        │           ↓ ~4ms (vs 现在 ~115ms = 23ms × 5次尝试)
        └── Cell 1–N: newton_solve_cell(x0, inlet, n_iter=40)         ← JIT顺序
                    ↓ ~8ms/cell (vs 现在 ~23ms)
                    
总计 20 cells: ~170ms (vs 现在 ~460ms) → 约 2.7x 加速
```

---

## 四、实施路线图

```
周 1: 阶段1（最小风险）
  ├── 新建 src/model/jax_adapter.py
  │     └── wrap_residuals_for_jax(): numpy fn → JAX JIT Jacobian
  └── 修改 solver.py: NewtonSolver 支持 jac= 参数注入
  
周 2: 阶段2
  ├── 新建 src/model/jax_solver.py
  │     ├── newton_solve_cell()
  │     └── newton_solve_multistart()
  └── 修改 gasifier_system.py: 将 least_squares 替换为 jax_solver
  
周 3–4: 阶段3（选做，视阶段2效果决定）
  ├── 新建 src/model/jax_cell.py
  └── 逐函数迁移 kinetics_service.py → jax_kinetics.py
```

---

## 五、风险点与对策

| 风险 | 严重度 | 对策 |
|---|---|---|
| `lax.scan` 固定迭代数导致未收敛就返回 | 🟡 中 | 设 n_iter=50，返回 `converged` flag，未收敛时 fallback 到 scipy |
| JAX `jnp.linalg.solve` 奇异矩阵（Cell 0 起燃） | 🟡 中 | 加 `1e-10*I` 正则化 + 条件数检查 |
| `jit` 第一次调用编译耗时（~2s） | 🟢 低 | 在 `GasifierSystem.__init__` 中预热 |
| 现有 `cell.py` 中 `logger.info` 等副作用无法 JIT | 🟢 低 | 保留 numpy fallback 路径，JIT版本去除日志 |
| `_calc_particle_temperature` 中的条件分支 | 🔴 高 | 用 `jnp.where` 替代 Python if/else；`lax.cond` 处理复杂分支 |

---

## 六、验证方案

升级后必须与原始结果对比：

```python
# test_jax_solver.py
def test_equivalence():
    case = VALIDATION_CASES['Paper_Case_6']
    
    # 原始 scipy 结果
    sys_scipy = GasifierSystem(..., solver='scipy')
    res_scipy, _ = sys_scipy.solve(N_cells=20)
    
    # JAX 结果  
    sys_jax = GasifierSystem(..., solver='jax')
    res_jax, _ = sys_jax.solve(N_cells=20)
    
    # 温度偏差 < 5K，组分偏差 < 0.1 mol/s
    assert np.max(np.abs(res_scipy[:, 10] - res_jax[:, 10])) < 5.0
    assert np.max(np.abs(res_scipy[:, :8] - res_jax[:, :8])) < 0.1
```
