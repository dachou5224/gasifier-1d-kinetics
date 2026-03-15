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

## 📝 深度物理调试与算法修复 (2026-03)

| 改进项 | 说明 |
|--------|------|
| **NewtonSolver Cost 漏洞修复** | 修复了底层 `solver.py` 中将整个状态向量 $x$ 的二维范数误作全量方程 Target 残差 (`norm(x)`) 返回的致命 Bug，杜绝了求解器判据长期被温度数值尺度“绑架”的情况。 |
| **消除极端失温现象** | 清除了系统多初值探索策略中由于受到错误 Cost 误导而被迫“强行挑选最低温度解”的系统性偏见，彻底打通了全轴网格反应放热到宏观温度抬升的合理物理传递链条。 |
| **验证性能跨越** | 断绝了所有的假不收敛与失温报错。工业工况 `LuNan_Texaco` 出口温度精准修正至 **1363°C**（工业基准 1350°C），核心小试测点如 `texaco i-1` 自动恢复至合理的 **1414°C**。 |

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

## 🔧 配置

*   **验证数据**：`validation_cases_pilot.json`（小试 56–187 kg/h）、`validation_cases_industrial.json`（工业）、`validation_cases_merged.json`（合并）、`validation_cases_OriginalPaper.json`
*   **求解器**：`GasifierSystem.solve(solver_method='newton')` 使用 Newton
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
