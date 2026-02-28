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
│   ├── validation_cases_fortran.json
│   ├── validation_cases_merged.json          # 四文件合并去重
│   ├── validation_cases_industrial.json     # 工业工况 (含鲁南)
│   └── validation_cases_new.json            # Illinois_No6, Australia_UBE, Fluid_Coke
├── docs/                   # 温度诊断、Fortran 机制、工况对比、lunan_energy_audit_report
├── reference_fortran/      # Source1_副本.for
└── README.md
```

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
| **工业工况** | `run_gasifier_model_cases.py` 含 LuNan_Texaco（鲁南化肥厂 Texaco 气化炉） |
| **Cell 0 能量审计** | `audit_cell0_energy.py` 支持 Paper_Case_6、LuNan_Texaco |
| **鲁南能量审计** | `audit_lunan_energy.py` 全炉逐 Cell 审计，见 `docs/lunan_energy_audit_report.md` |
| **诊断脚本** | `compare_i1_exxon_energy.py`：Texaco I-1 vs Exxon 工况差异与轴向能量 |
| **温度诊断** | `docs/temperature_diagnosis.md`，`docs/texaco_i1_vs_exxon_analysis.md` |

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

### 3. 运行工业工况 (含鲁南 Texaco)

```bash
PYTHONPATH=src python scripts/run_gasifier_model_cases.py
# 仅鲁南: GASIFIER_CASES=LuNan_Texaco python scripts/run_gasifier_model_cases.py
```

### 4. Cell 0 能量审计 (鲁南)

```bash
PYTHONPATH=src python tests/diagnostics/audit_cell0_energy.py LuNan_Texaco
# 默认 Paper_Case_6: python tests/diagnostics/audit_cell0_energy.py
```

### 5. Texaco I-1 vs Exxon 能量诊断

```bash
PYTHONPATH=src python tests/diagnostics/compare_i1_exxon_energy.py
# 可选: -n 30 减少网格, -o report.txt 输出到文件
```

### 6. 单元测试

```bash
PYTHONPATH=src python tests/unit/test_units.py
```

### 7. 求解器对比 (TRF vs Newton)

```bash
PYTHONPATH=src python tests/integration/compare_solvers.py
```

## 🔧 配置

*   **验证数据**：`validation_cases_OriginalPaper.json`、`validation_cases_fortran.json`、`validation_cases_merged.json`（四文件合并）、`validation_cases_industrial.json`（工业工况）
*   **求解器**：`GasifierSystem.solve(solver_method='newton')` 使用 Newton
*   **RK-Gill 颗粒温度**：`PhysicalConstants.USE_RK_GILL_COMBUSTION = True` 启用（计算量约 4×）
*   **工况级 op_conds**：`AdaptiveFirstCellLength`（按进煤量自适应 Cell 0）、`FirstCellLength`（覆盖 dz_cell0）、`L_evap_m`（蒸发分散长度，0 表示全在 Cell 0）

## 📊 当前验证结果

| 工况 | 出口 (模型) | 实验 | 状态 |
|------|-------------|------|------|
| Texaco_I-1 | ~804°C | 1370°C | 偏低 |
| Texaco_Exxon | ~1226°C | - | 较合理 |
| Texaco_I-2 | ~1149°C | 1333°C | 偏低 |
| Illinois_No6 | ~1212°C | - | 较合理 |
| LuNan_Texaco | ~864°C | 1350°C | 偏低（已用 HeatLoss 1.5%、CharCombustionRateFactor 0.2 缓解） |
| Australia_UBE | ~1248°C | - | 较合理 |
| Fluid_Coke | ~1937°C | - | 较合理 |

### 算例来源汇总

| 来源 | 代表工况 | 几何 | 出口 T (模型) |
|------|----------|------|---------------|
| OriginalPaper | Texaco_I-1, Texaco_Exxon, Texaco_I-2 | L=6.096, D=1.524 | 见 temperature_diagnosis |
| merged | Illinois_No6, Australia_UBE, Fluid_Coke, Paper_Case_6 | 按 oc 或默认 | Illinois ~1212°C |
| industrial | LuNan_Texaco, Paper_Case_6 系列 | LuNan: L=6.87, D=1.68 | LuNan ~829°C (期望 1350°C) |

详见 `docs/temperature_diagnosis.md`、`docs/texaco_i1_vs_exxon_analysis.md`、`docs/lunan_energy_audit_report.md`。
