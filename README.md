# 1D Kinetic Gasifier Model (Refactored)

A robust, modular 1-D plug flow model for entrained-flow gasifiers (Texaco/Shell type) featuring strong gas-solid coupling and heterogeneous reaction kinetics. This codebase has been significantly refactored for maintainability, performance, and robustness.

## 🚀 Key Features
*   **Physics**: 1-D Plug Flow, Steady State with strong Heat/Mass coupling.
*   **Kinetics**:
    *   **Heterogeneous**: Unreacted Core Shrinking Model (UCSM) for Char + $O_2/H_2O/CO_2$.
    *   **Homogeneous**: 6-step reversible global reaction mechanism (Jones-Lindstedt).
*   **Grid Generation**: Self-adaptive mesh via `AdaptiveMeshGenerator` (Fine resolution in flame zone).
*   **Algorithm**: Sequential cell-by-cell solver using `scipy.optimize.least_squares`.
*   **Architecture**: Service-oriented architecture (SOA) decoupling Physics, Kinetics, and Grid logic.

## 🛠 Refactoring Improvements
This project has undergone a comprehensive 13-step optimization process:
1.  **Constants**: Centralized physics constants in `constants.py`.
2.  **Input Validation**: Strict type and range checking for simulation inputs.
3.  **Error Handling**: Robust zero-division protection and convergence checks.
4.  **Modularization**: Split monolithic solver method into `_calc_rates` and `_calc_balances`.
5.  **Internationalization**: Unified all code comments to English.
6.  **Unit Testing**: Added dedicated unit tests (`tests/test_units.py`).
7.  **Performance**: Vectorized molar mass and property calculations.
8.  **Documentation**: Added comprehensive docstrings with physics context.
9.  **Grid Service**: Extracted mesh generation into `AdaptiveMeshGenerator`.
10. **Logging**: Replaced `print` with structured `logging` (Info/Debug/Warning).
11. **Type Hints**: Full Python typing support for core services.
12. **De-duplication**: Shared concentration calculation logic.
13. **Configuration**: Externalized validation cases to `tests/validation_cases.json`.

## 📂 Project Structure
```text
gasifier-1d-kinetic/
├── src/model/
│   ├── gasifier_system.py  # Orchestrator: Grid generation & cell-by-cell loop
│   ├── cell.py             # Control Volume: Mass & Energy balance equations
│   ├── kinetics_service.py # Homogeneous & Heterogeneous Rate Service
│   ├── kinetics.py         # Physics: UCSM & Equilibrium functions
│   ├── pyrolysis_service.py# Coal Devolatilization (with Energy Normalization)
│   ├── grid_service.py     # Adaptive Meshing (refined near inlet)
│   ├── source_terms.py     # Source term models (Moisture, Pyrolysis)
│   ├── material.py         # Pure service for thermophysical properties
│   ├── state.py            # StateVector Immutable Data Structure
│   ├── physics.py          # Thermodynamic properties (Shomate) & Transport
│   └── constants.py        # Centralized physical & grid constants
├── tests/
│   ├── test_validation_config.py # Main validation runner (Case 6)
│   ├── audit_full_heat_balance.py# Diagnostic: Full reactor energy profile
│   ├── audit_resistances.py      # Diagnostic: Kinetic/Diffusion resistance audit
│   ├── audit_pyrolysis_energy.py # Diagnostic: Pyrolysis energy conservation audit
│   ├── test_units.py             # Unit tests for core physics
│   └── validation_cases.json     # Experimental benchmark data
└── README.md
```

## 📝 Modification Records (Recent)
### 1. Physics & Mass Balance Fixes
*   **Mass Conservation**: Fixed a critical bug in `cell.py` by adding `solid_src` to the solid mass balance, resolving ignition failure.
*   **Fluid Dynamics**: Re-implemented gas velocity $u_g = F_{total}RT / (P A \epsilon)$ based on local atomic flux; removed average velocity approximation.
*   **Residence Time**: Calculated locally as $\tau = dz / u_g$, significantly improving heat release accuracy.

### 2. Energy & Pyrolysis
*   **Energy Conservation**: Implemented HHV-based normalization in `PyrolysisService` to prevent "energy creation" artifacts in volatiles.
*   **Enthalpy Basis**: Verified and unified Shomate-based formation enthalpies for all species and coal.

### 3. Grid & Numerical Stability
*   **Adaptive Grid Refinement**: Implemented finer discretization near the inlet (0.4m $\rightarrow$ 0.05m) to dampen initial heat release and improve convergence.
*   **Ignition Strategy**: Improved multi-guess strategy in `GasifierSystem` with balanced initial atom distribution.
*   **Parameter Rollback**: Restored $E_{CH4\_Ox}$ to standard JL values (125.6 kJ/mol) to isolate kinetic unit issues.

### 4. Diagnostics & Auditing
*   **Heat Balance Audit**: Created detailed per-cell energy profiling tools identifying 40 MW endothermic sinks.
*   **Kinetic Resistance Audit**: Implemented UCSM resistance breakdown ($R_{diff}$ vs $R_{kin}$) to verify mass transfer limits.

## ⚡ Quick Start

### 1. Run Unit Tests (Verify Logic)
```bash
python3 tests/test_units.py
```

### 2. Run Integration Tests (Verify Solver)
```bash
python3 tests/test_model.py
```

### 3. Run Validation Case (Paper Case 6)
```bash
python3 tests/test_validation_config.py
```
*Note: Currently produces warnings and low temperature output. Requires **Kinetics Calibration** (Next Step).*

## 🔧 Parameters & Configuration
*   **Validation Cases**: Modify `tests/validation_cases.json` to add new experimental benchmarks.
*   **Kinetics Tuning**: Adjust pre-exponential factors in `src/model/kinetics_service.py`.

## 📖 Key Modules
*   **GasifierSystem**: The orchestrator. Initializes the grid using `AdaptiveMeshGenerator` and solves cells sequentially.
*   **StateVector**: Immutable data structure holding ($T, P, F_i, W_s, X_c$) at any grid point.
*   **KineticsService**: Pure functional service calculating rates for specific states.
