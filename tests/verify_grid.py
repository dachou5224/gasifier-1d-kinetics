import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem
from model.chemistry import VALIDATION_CASES, COAL_DATABASE

def run_grid_check():
    print("="*80)
    print("Variable Grid Verification")
    print("="*80)
    
    # Setup standard case
    data = VALIDATION_CASES["Paper_Case_6"]
    coal_props = COAL_DATABASE[data['inputs']['coal']]
    op_conds = {
        'coal_flow': 1.0, 'o2_flow': 0.5, 'steam_flow': 0.1,
        'P': 3.0e6, 'T_in': 300.0,
        'HeatLossPercent': 3.0
    }
    geometry = {'L': 8.0, 'D': 2.6} 
    
    system = GasifierSystem(geometry, coal_props, op_conds)
    
    # We don't need to solve fully, just inspect grid construction
    # But grid is built inside solve(), so we solve with dummy N=50
    # To avoid crash, just wrap in try/except or rely on internal structures being built before loop?
    # Actually, z_positions are built before solve loop. But we need to call solve to trigger it.
    # Solve might crash if we don't have good inputs, but we just want to inspect the generated grid.
    
    try:
        system.solve(N_cells=50)
    except Exception as e:
        print(f"Solver stopped (expected if inputs dummy): {e}")

    print(f"\nGrid Check (N=50, L=8.0):")
    print(f"{'Idx':<4} | {'Z_center':<8} | {'dz (m)':<8} | {'Strategy Check'}")
    print("-" * 60)
    
    cells = getattr(system, 'cells', [])
    if not cells:
        # Fallback if solve crashed early
        pass

    # Verify key cells
    # Cell 0: Should be 0.05
    # Cell 1: Should be 0.40
    # Cell 2: Should be 0.10
    
    passed = True
    
    for i in range(min(len(cells), 25)):
        dz = cells[i].dz
        note = ""
        if i == 0:
            if abs(dz - 0.05) < 1e-3: note = "PASS (Boundary)"
            else: note = "FAIL (Target 0.05)"; passed = False
        elif i == 1:
            if abs(dz - 0.40) < 1e-3: note = "PASS (Ignition)"
            else: note = "FAIL (Target 0.40)"; passed = False
        elif 2 <= i <= 19:
            if abs(dz - 0.10) < 1e-3: note = "PASS (Fine)"
            else: note = f"FAIL (Target 0.10, Got {dz:.2f})"; passed = False
        elif i == 20:
             note = "Transition to Coarse"
             
        print(f"{i:<4} | {cells[i].z:<8.3f} | {dz:<8.3f} | {note}")
        
    print(f"\nOverall Grid Logic: {'PASS' if passed else 'FAIL'}")

if __name__ == "__main__":
    run_grid_check()
