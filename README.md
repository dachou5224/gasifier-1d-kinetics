# 1D Kinetic Gasifier Model (Refactored)

A robust, modular 1-D plug flow model for entrained-flow gasifiers (Texaco/Shell type) featuring strong gas-solid coupling and heterogeneous reaction kinetics.

## 🚀 Key Features
*   **Physics**: 1-D Plug Flow, Steady State with strong Heat/Mass coupling.
*   **Kinetics**:
    *   **Heterogeneous**: Unreacted Core Shrinking Model (UCSM) for Char + $O_2/H_2O/CO_2$.
    *   **Homogeneous**: 6-step reversible global reaction mechanism (Jones-Lindstedt).
*   **Numerical Methods**:
    *   **Default**: Sequential cell-by-cell solver using `scipy.optimize.least_squares` (TRF method).
    *   **Newton-Raphson**: Manual implementation with numerical Jacobian and damping (`NewtonSolver`).
*   **Grid Management**: Self-adaptive mesh via `AdaptiveMeshGenerator` for fine resolution in high-gradient zones.
*   **Architecture**: Service-oriented architecture (SOA) decoupling Physics, Kinetics, and Grid logic.

## 📂 Project Structure
```text
gasifier-1d-kinetic/
├── src/model/
│   ├── gasifier_system.py  # Orchestrator: Grid generation & solver loop
│   ├── cell.py             # Control Volume: Balance equations & audit logs
│   ├── solver.py           # Custom Newton-Raphson implementation
│   ├── kinetics_service.py # Reaction Rate Service (Homo/Het)
│   ├── physics.py          # Thermodynamics (Shomate) & Transport
│   └── ...                 # Modular services for grid, material, etc.
├── tests/
│   ├── unit/               # Core component tests
│   ├── integration/        # Full model validation & solver benchmarks
│   └── diagnostics/        # Audit, debugging & visualization scripts
├── data/                   # Shared configuration & validation case data
├── logs/                   # Centralized simulation diagnostic logs
├── scripts/                # Utility scripts
└── README.md
```

## 📝 Recent Improvements
*   **Directory Reorganization**: Restructured project into a professional layout (`src/`, `tests/`, `data/`, `logs/`).
*   **Parallel Solver**: Implemented `NewtonSolver` as an alternative to Scipy TRF for stability comparison.
*   **Energy Balance Fixes**: Corrected Liquid Water formation enthalpy sink (~115 MW).
*   **Kinetics Audit**: Implemented axial reaction rate profiling confirming self-quenching effects.

## ⚡ Quick Start

### 1. Run Unit Tests
```bash
python3 tests/unit/test_units.py
```

### 2. Run Validation Benchmark (Case 6)
```bash
python3 tests/integration/test_validation_config.py
```

### 3. Compare Solvers (TRF vs Newton)
```bash
python3 tests/integration/compare_solvers.py
```

## 🔧 Configuration
*   **Validation Data**: Located in `data/validation_cases.json`.
*   **Solver Selection**: Toggle `solver_method='newton'` in `GasifierSystem.solve()`.
