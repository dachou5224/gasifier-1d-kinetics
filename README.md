# 1D Kinetic Gasifier Model (一维动力学气化炉模型)

A robust 1-D plug flow model for entrained-flow gasifiers (Texaco/Shell type) featuring strong gas-solid coupling and heterogeneous reaction kinetics.

## 🚀 Key Features
*   **Physics**: 1-D Plug Flow, Steady State.
*   **Kinetics**: Unreacted Core Shrinking Model (UCSM) for Char + $O_2/H_2O/CO_2$.
*   **Algorithm**: Non-uniform Geometric Grid for capturing rapid ignition.
*   **Validation**: Verified against Texaco Pilot Plant and Industrial data.

## 📂 Project Structure
```text
gasifier-1d-kinetic/
├── src/
│   ├── model/          # Physics & Solver Core
│   └── main_ui.py      # Streamlit GUI
├── tests/              # Verification Scripts
├── docs/               # Manuals & Reports
└── README.md
```

## 📖 Documentation
*   [**Algorithm Manual (算法说明书)**](docs/1D_Gasifier_Model_Manual_cn.md): Detailed mathematical formulation.
*   [**Validation Report (验证报告)**](docs/validation_report_cn.md): Performance benchmarks against experimental data.
*   [**Grid Strategy (网格策略)**](docs/grid_strategy_cn.md): Meshing recommendations for different scales.

## ⚡ Quick Start

### 1. Run the UI
```bash
streamlit run src/main_ui.py
```

### 2. Run Verification Suite
```bash
# Validate against Pilot/Industrial Cases
python3 tests/verify_cases.py

# Check Grid Convergence
python3 tests/verify_grid.py
```

## 🛠 Requirements
*   Python 3.8+
*   numpy, scipy, pandas, streamlit, plotly
