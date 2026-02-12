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
├── tests/
│   ├── integration/        # run_original_paper_cases.py, run_fortran_json_cases.py
│   └── diagnostics/        # compare_i1_exxon_energy.py, audit_reaction_heat_texaco.py
├── data/
│   ├── validation_cases_OriginalPaper.json  # Wen & Chaung 原始工况
│   └── validation_cases_fortran.json
├── docs/                   # 温度诊断、Fortran 机制、工况对比
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
| **诊断脚本** | `compare_i1_exxon_energy.py`：Texaco I-1 vs Exxon 工况差异与轴向能量 |
| **温度诊断** | `docs/temperature_diagnosis.md`，`docs/texaco_i1_vs_exxon_analysis.md` |

## ⚡ 快速开始

### 1. 运行 Paper 算例

```bash
cd gasifier-1d-kinetic
PYTHONPATH=src python tests/integration/run_original_paper_cases.py
```

### 2. 运行 Texaco I-1 vs Exxon 能量诊断

```bash
PYTHONPATH=src python tests/diagnostics/compare_i1_exxon_energy.py
# 可选: -n 30 减少网格, -o report.txt 输出到文件
```

### 3. 单元测试

```bash
PYTHONPATH=src python tests/unit/test_units.py
```

### 4. 求解器对比 (TRF vs Newton)

```bash
PYTHONPATH=src python tests/integration/compare_solvers.py
```

## 🔧 配置

*   **验证数据**：`data/validation_cases_OriginalPaper.json`，`data/validation_cases_fortran.json`
*   **求解器**：`GasifierSystem.solve(solver_method='newton')` 使用 Newton
*   **RK-Gill 颗粒温度**：`PhysicalConstants.USE_RK_GILL_COMBUSTION = True` 启用（计算量约 4×）

## 📊 当前验证结果

| 工况 | 出口 (模型) | 实验 | 状态 |
|------|-------------|------|------|
| Texaco_I-1 | ~804°C | 1370°C | 偏低 |
| Texaco_Exxon | ~1226°C | - | 较合理 |
| Texaco_I-2 | ~1149°C | 1333°C | 偏低 |

详见 `docs/temperature_diagnosis.md`、`docs/texaco_i1_vs_exxon_analysis.md`。
