import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.cell import Cell
from model.state import StateVector
from model.kinetics_service import KineticsService
from model.pyrolysis_service import PyrolysisService
from model.chemistry import COAL_DATABASE

def check_cell_residuals():
    print("[Residual Audit] Checking single CV balance consistency...")
    
    # 1. Setup a standard high-temp cell environment
    coal_props = COAL_DATABASE['Paper_Base_Coal']
    # Add Hf_coal if missing (consistent with GasifierSystem)
    coal_props['Hf_coal'] = -3.37e6 # J/kg
    
    op_conds = {
        'coal_flow': 41670.0/3600.0,
        'o2_flow': (41670.0 * 1.05)/3600.0,
        'steam_flow': 0.0,
        'T_in': 1500.0,
        'P': 4.0e6,
        'HeatLossPercent': 3.0,
        'L_reactor': 8.0,
        'Q_total_MW': 350.0,
        'particle_diameter': 100e-6
    }
    
    # Inlet State (partially reacted for testing)
    inlet_gas = np.array([0.1, 0.5, 100.0, 50.0, 0.1, 80.0, 100.0, 20.0])
    inlet = StateVector(gas_moles=inlet_gas, solid_mass=5.0, carbon_fraction=0.4, T=1500.0, P=4e6, z=1.0)
    
    dz = 0.16
    V = 0.8 * dz
    A = 0.8
    
    kin = KineticsService()
    pyro = PyrolysisService()
    
    cell = Cell(cell_index=5, z=1.0, dz=dz, V=V, A=A, inlet_state=inlet, 
                kinetics=kin, pyrolysis=pyro, coal_props=coal_props, op_conds=op_conds)
    
    # Target State (perturbed inlet)
    target_arr = inlet.to_array()
    target_arr[10] += 5.0 # T + 5K
    target_arr[2] += 2.0  # CO + 2 mol/s
    
    res = cell.residuals(target_arr)
    print(f"Residual Array Shape: {res.shape}")
    print(f"Residuals (Scaled): {res}")
    
    # Detailed Conservation Check (Unscaled)
    # Atoms: C, H, O
    # We expect Carbon in = Carbon out IF residuals are 0.
    
    print("\n--- Conservation Rules Audit ---")
    print("1. Carbon Balance: Solid Carbon + Gas Carbon")
    print("2. Hydrogen Balance: Solid H (Volatiles) + Gas H")
    print("3. Oxygen Balance: Gas O + Solid O")
    print("4. Energy Balance: H_total_in - Q_loss = H_total_out")
    
if __name__ == "__main__":
    check_cell_residuals()
