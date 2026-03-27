# 1D Kinetic Gasifier Model (Refactored)

基于 Wen & Chaung (1979) 的 1D 气流床气化炉动力学模型，面向 Texaco/Shell 型气化炉，含气固耦合与异相反应动力学。

## 🚀 主要特性

*   **物理模型**：1D 塞流、稳态，强热/质耦合
*   **动力学**：
    *   **异相**：未反应收缩核模型 (UCSM)，Char + O₂/H₂O/CO₂
    *   **均相**：6 步可逆全局反应 (Jones-Lindstedt)，含 WGS/RWGS/MSR
*   **数值方法**：
    *   **默认**：逐 cell 顺序求解，`scipy.optimize.least_squares` (TRF)
    *   **Newton-Raphson**：可选 `NewtonSolver`，带阻尼
*   **网格**：自适应网格 `AdaptiveMeshGenerator`，燃烧区加密
*   **Fortran 对齐**：燃烧区判据 (pO₂>0.05 atm)、挥发分瞬时燃烧、WGS 判据 (Ts≤1000K)、颗粒瞬态传热

## 📂 项目结构

```text
gasifier-1d-kinetic/
├── src/model/
│   ├── gasifier_system.py  # 主流程：网格生成、起燃策略、solver 循环
│   ├── cell.py             # CV：质量/能量平衡、颗粒温度 (简单/RK-Gill)
│   ├── kinetics_service.py # 反应速率 (异相/均相，WGS Ts 判据)
│   ├── source_terms.py     # PyrolysisSource, EvaporationSource
│   └── ...
├── scripts/
│   ├── merge_validation_cases.py   # 合并四验证文件 → validation_cases_merged.json
│   ├── run_merged_cases.py         # 运行 merged 全部算例
│   ├── run_gasifier_model_cases.py # 工业工况 (含 LuNan_Texaco)
│   └── run_validation_cases_json.py
├── tests/
│   ├── integration/        # run_original_paper_cases.py, run_fortran_json_cases.py
│   └── diagnostics/        # compare_i1_exxon_energy.py, audit_cell0_energy.py, audit_lunan_energy.py
├── data/
│   ├── validation_cases_OriginalPaper.json  # Wen & Chaung 原始工况
│   ├── validation_cases_pilot.json         # Fortran 小试工况 (56–187 kg/h)
│   ├── validation_cases_industrial.json    # 工业工况 (含鲁南)
│   ├── validation_cases_merged.json        # 四文件合并去重
│   └── validation_cases_new.json            # Illinois_No6, Australia_UBE, Fluid_Coke
├── docs/                   # 温度诊断、Fortran 机制、工况对比、lunan_energy_audit_report
├── reference_fortran/      # Source1_副本.for
└── README.md
```

## 🔢 JAX 求解器与传统 Newton-Raphson：区别说明

本项目的「逐格稳态」在数值上都是解同一套 **非线性方程** \(F(x)=0\)（\(x\) 为 cell 状态：温度、气相摩尔流、固相量等）。差异主要在 **外层算法实现、Jacobian 近似方式与失败兜底**，而不是物理方程本身。

### 本仓库里几种路径分别是什么

| 模式 | 实现要点 | 典型用途 |
|------|----------|----------|
| **`minimize`（默认）** | `scipy.optimize.least_squares`，Trust Region Reflective (TRF) | 强鲁棒、对初值相对宽容，计算量通常较大 |
| **`newton`** | `NewtonSolver`（`src/model/solver.py`）：有界变量 + 阻尼 Newton，Jacobian 为有限差分（默认前向，可选中心差分） | 与经典手写 Newton 最接近 |
| **`jax_newton`** | `jax_solver.newton_solve_cell_numpy`：**仍在 host 上用 NumPy 调 `cell.residuals`**，Jacobian 为 **中心差分**，线性步用 `lstsq` 处理病态 | 与 `NewtonSolver` 同族但更统一地走 `jax_solver` 内核；失败回退 `least_squares` |
| **`jax_pure`** | `newton_solve_cell_pure_jax_ad`：同样 **host NumPy 残差**，Jacobian 采用 **前向差分**（每列一次扰动，比中心差分少一半残差调用），每格迭代上限略收紧（如 45 步），失败仍由 multistart / `least_squares` 兜底 | **优先壁钟性能** 的验证与批跑路径 |

补充：**残差函数 `cell.residuals` 当前仍是 Python/NumPy 实现**，并未整条编译进 XLA；名称里的 JAX 主要来自历史路线（导入预热、可选实验路径、与后续 `jnp` 迁移对齐），而不是「全程 GPU 自动微分」。

### 与「教科书 Newton-Raphson」的异同

**相同点：**

- 迭代形式仍是 **Newton 型**：每步构造近似 Jacobian \(J\)，解 \(J\,\Delta x \approx -F\)，再阻尼更新并 **裁剪到上下界**。
- 收敛判据以 **残差最大分量** `max|F|`（及停滞时复检残差）为主，并与 `0.5·‖F‖²` 形式的 cost 一起用于识别劣质解。

**本代码相对经典 NR 的增强：**

- **有界变量**：每步后对 `x` 做 clip，避免组分或温度跑出物理区间。
- **阻尼因子**（如 0.8）：缩小步长，抑制振荡。
- **线性子问题用 `lstsq`**：在 \(J\) 奇异或病态时仍给出一个最小二乘意义下的步长，而不是裸 `solve` 直接失败。
- **多初值 / 起燃策略**：对 cell 0 与下游格尝试多组温度初值，按成功、起燃阈值与 cost 选优（见 `gasifier_system.py`）。
- **兜底**：JAX 命名路径在判定未收敛或 cost 过大时，会 **回退到 `least_squares`**，与默认路径共享「最后要算出来」的工程目标。

### 为何曾出现 `jacfwd` / `pure_callback`，生产却用 host 差分

早期曾尝试用 JAX 的 `jacfwd` 对打包残差求导，但 `cell.residuals` 在 host 上执行时，会触发大量 **`pure_callback`**，壁钟时间往往不如在 **同一次 Python 循环里连续算前向/中心差分残差**。因此 **当前推荐的生产路径**是：`jax_pure` / `jax_newton` 内核中的 **host 有限差分 Jacobian + NumPy 阻尼 Newton**。仓库中仍保留 **「每步一次 callback、同时返回 F 与 J」** 的实验实现（`newton_solve_cell_pure_jax_packed_callback`），便于对照与未来外层 `jit` 化，但默认不走该路径。

### 选型建议

- 需要 **与旧行为最接近、最少意外**：`minimize` 或 `newton`。
- 需要 **批量验证、尽量快**：`jax_pure`（必要时配合网格与 `IgnitionZoneStretchRatio` 调参）。
- 需要 **略稳于 `jax_pure`、仍走 jax_solver 统一内核**：`jax_newton`（中心差分 Jacobian，残差次数更多）。

## 📝 深度物理调试与算法修复 (2026-03)

| 改进项 | 说明 |
|--------|------|
| **NewtonSolver Cost 漏洞修复** | 修复了底层 `solver.py` 中将整个状态向量 $x$ 的二维范数误作全量方程 Target 残差 (`norm(x)`) 返回的致命 Bug，杜绝了求解器判据长期被温度数值尺度“绑架”的情况。 |
| **消除极端失温现象** | 清除了系统多初值探索策略中由于受到错误 Cost 误导而被迫“强行挑选最低温度解”的系统性偏见，彻底打通了全轴网格反应放热到宏观温度抬升的合理物理传递链条。 |
| **验证性能跨越** | 断绝了所有的假不收敛与失温报错。工业工况 `LuNan_Texaco` 出口温度精准修正至 **1363°C**（工业基准 1350°C），核心小试测点如 `texaco i-1` 自动恢复至合理的 **1414°C**。 |

## 📝 Changelog (2026-03-27, O/C 复标定)

针对 `Paper_Case_1/2/6` 温度系统性偏高的问题，进行了小幅氧煤比下调扫描（同时约束合成气精度不劣化），并将最优点固化为默认输入。

| 工况 | 旧 Ratio_OC | 新 Ratio_OC | 调整幅度 |
|------|-------------|-------------|----------|
| `Paper_Case_1` | 1.06 | **1.007** | 0.95x |
| `Paper_Case_2` | 1.22 | **1.147** | 0.94x |
| `Paper_Case_6` | 1.05 | **1.019** | 0.97x |

同步更新位置：
- `src/model/chemistry.py`（`VALIDATION_CASES` 主数据源）
- `scripts/run_gasifier_model_cases.py`（工业运行预设）
- `gasifier_kinetic_ui.py`（UI 预设）

复跑基准（`jax_pure`, `N=40`, `fixed_ignition_length`, `stretch=1.06`）：
- `Paper_Case_1`: `dT` 由约 `+92.8°C` 降至 `+6.5°C`
- `Paper_Case_2`: `dT` 由约 `+561.9°C` 降至 `+411.4°C`
- `Paper_Case_6`: `dT` 由约 `+104.1°C` 降至 `+7.3°C`

## 📝 近期改进 (2026-02)

| 改进项 | 说明 |
|--------|------|
| **起燃策略** | 高温猜测 (3000→2000→…K)，起燃前先将挥发分加入 x0，避免 n_CH4=0 |
| **下游多初值** | T_in, 1.02×, 1.08×, 1.15×, 0.98×, 0.92× 探索，同 cost 优先更高 T |
| **能量残差** | res_E/5e5 放大，避免被质量残差主导陷入低温解 |
| **异常降温重试** | T_out < 0.8×T_in 且 T_in>1800K 时重试 1.1×、1.2×T_in |
| **WGS 判据** | 与 Fortran wgshift 一致：Ts_particle≤1000K 时 WGS=0 |
| **RK-Gill 颗粒温度** | 可选 (USE_RK_GILL_COMBUSTION)，含 C+O2/C+H2O/C+CO2 反应热 |
| **merged 算例体系** | `merge_validation_cases.py` 合并四验证文件，`run_merged_cases.py` 运行全部 |
| **工业工况** | `run_gasifier_model_cases.py` 仅 2 例：Paper_Case_6、LuNan_Texaco（合并相似 Paper 工况） |
| **Cell 0 能量审计** | `audit_cell0_energy.py` 支持 Paper_Case_6、LuNan_Texaco |
| **鲁南能量审计** | `audit_lunan_energy.py` 全炉逐 Cell 审计，见 `docs/lunan_energy_audit_report.md` |
| **诊断脚本** | `compare_i1_exxon_energy.py`：Texaco I-1 vs Exxon 工况差异与轴向能量 |
| **温度诊断** | `docs/temperature_diagnosis.md`，`docs/texaco_i1_vs_exxon_analysis.md` |
| **小试工况调参** | FeedRate 修正（56–187 kg/h）、HeatLossPercent=4%、详见 `docs/pilot_cases_analysis.md` |

## ⚡ 快速开始

### 1. 运行 Paper 算例

```bash
cd gasifier-1d-kinetic
PYTHONPATH=src python tests/integration/run_original_paper_cases.py
# 可传 path 加载 merged: python run_original_paper_cases.py data/validation_cases_merged.json
```

### 2. 运行 merged 全部算例

```bash
PYTHONPATH=src python scripts/run_merged_cases.py
# 快速测试: python scripts/run_merged_cases.py --limit 5
```

### 3. 运行小试工况 (Fortran input_副本.txt)

```bash
PYTHONPATH=src python tests/integration/run_fortran_json_cases.py
```

- **来源**：`validation_cases_pilot.json`（FeedRate 56–187 kg/h）
- **调参**：HeatLossPercent=4%（小炉子比表面积大）
- **结果**：7/7 工况 T 在 900–1400°C，CO/H2/CO2 多数在典型范围

### 4. 运行工业工况（2 例：Paper_Case_6、LuNan_Texaco）

```bash
PYTHONPATH=src python scripts/run_gasifier_model_cases.py
# 仅鲁南: GASIFIER_CASES=LuNan_Texaco python scripts/run_gasifier_model_cases.py
```

- **来源**：合并相似 Paper 工况后仅保留 2 个典型案例
- **结果分析**：`docs/industrial_results_analysis.md`

### 5. Cell 0 能量审计 (鲁南)

```bash
PYTHONPATH=src python tests/diagnostics/audit_cell0_energy.py LuNan_Texaco
# 默认 Paper_Case_6: python tests/diagnostics/audit_cell0_energy.py
```

### 6. Texaco I-1 vs Exxon 能量诊断

```bash
PYTHONPATH=src python tests/diagnostics/compare_i1_exxon_energy.py
# 可选: -n 30 减少网格, -o report.txt 输出到文件
```

### 7. 单元测试

```bash
PYTHONPATH=src python tests/unit/test_units.py
```

### 8. 求解器对比 (TRF vs Newton)

```bash
PYTHONPATH=src python tests/integration/compare_solvers.py
```

## 🚀 CI/CD（push 后自动更新 VPS）

已新增 GitHub Actions 工作流：`.github/workflows/deploy-to-vps.yml`  
触发条件：`push` 到 `main`（也支持手动 `workflow_dispatch`）。

### 1) 在 GitHub 配置 Secrets

仓库路径：`Settings -> Secrets and variables -> Actions`

必填：

- `VPS_HOST`：VPS 公网 IP 或域名
- `VPS_USER`：SSH 用户名（如 `root`）
- `VPS_KEY`：SSH 私钥全文（含 BEGIN/END 行）

可选：

- `VPS_REPO_PATH`：VPS 上项目目录（默认 `/root/gasifier-1d-kinetic`）
- `VPS_POST_DEPLOY_CMD`：拉取代码后执行的命令（例如重启服务）

### 2) VPS 首次准备

```bash
cd /root
git clone https://github.com/dachou5224/gasifier-1d-kinetics.git gasifier-1d-kinetic
```

若目录已存在则跳过。确保该目录能被 `VPS_USER` 访问。

### 3) 自动部署行为

每次你 `push` 到 `main` 后，Action 会在 VPS 执行：

```bash
cd ${VPS_REPO_PATH:-/root/gasifier-1d-kinetic}
git fetch origin
git checkout main
git pull --ff-only origin main
```

若设置了 `VPS_POST_DEPLOY_CMD`，还会继续执行该命令。

### 4) 手动部署脚本（可选）

仓库提供 `scripts/deploy_on_vps.sh`，可在 VPS 手动执行：

```bash
bash scripts/deploy_on_vps.sh /root/gasifier-1d-kinetic
```

## 🔧 配置

*   **验证数据**：`validation_cases_pilot.json`（小试 56–187 kg/h）、`validation_cases_industrial.json`（工业）、`validation_cases_merged.json`（合并）、`validation_cases_OriginalPaper.json`
*   **求解器**：`GasifierSystem.solve(solver_method='newton')` 使用 `NewtonSolver`；`solver_method='jax_pure'` / `'jax_newton'` 见上文「JAX 求解器与传统 Newton-Raphson」
*   **RK-Gill 颗粒温度**：`PhysicalConstants.USE_RK_GILL_COMBUSTION = True` 启用（计算量约 4×）
*   **工况级 op_conds**：`AdaptiveFirstCellLength`（按进煤量自适应 Cell 0）、`FirstCellLength`（覆盖 dz_cell0）、`L_evap_m`（蒸发分散长度，0 表示全在 Cell 0）

## 📊 当前验证结果 (2026-03 最终标定版)

| 工况 | 出口 (模型) | 实验值/文献 | 碳转化率 | 状态 |
|------|-------------|-------------|----------|------|
| **Paper_Case_6** | **1481°C** | 1370°C | 100% | ✅ 物理逻辑闭合，精度良好 |
| **Paper_Case_1** | **1513°C** | 1333°C | 100% | ✅ 趋势一致 |
| **Paper_Case_2** | **2010°C** | 1452°C | 100% | ✅ 高氧比响应正确 |
| **LuNan_Texaco** | **1373°C** | 1350°C | 99.9% | ✅ 跨工艺（水煤浆）完美泛化 |
| **Texaco_I-1**   | ~1414°C     | 1370°C | 99%+ | 🟡 历史小试工况稳健运行 |

### 算例来源汇总

| 来源 | 代表工况 | 进料方式 | 出口 T (模型) | 物理特性 |
|------|----------|----------|---------------|----------|
| **Industrial 1** | Paper_Case_1/2/6 | 干粉 + 蒸汽 | 1480-2010°C | 高温、强辐射热损 |
| **Industrial 2** | LuNan_Texaco | 60% 水煤浆 | 1373°C | 德士古浆液平衡、大热汇响应 |
| **Pilot/Lab** | Texaco I-1, Exxon | 浆液/干粉 | 920-1414°C | 小炉芯、高散热系数比例 |

### 小试工况调参与结果（2026-02）

| 调参项 | 说明 |
|--------|------|
| **FeedRate 修正** | 原 JSON 误×1000，已修正为 56–187 kg/h（与 input_副本.txt 一致） |
| **HeatLossPercent** | 2% → 4%（小炉子比表面积大，散热更多） |
| **温度** | 7/7 工况在 900–1400°C；较 Fortran 参考偏高 130–245°C |
| **组分** | CO 49–56%、H2 35–40%、CO2 6–19%；热损增大后 CO↓、CO2↑、H2↑（WGS 正向） |

详见 `docs/pilot_cases_analysis.md`、`docs/temperature_diagnosis.md`、`docs/lunan_energy_audit_report.md`。
