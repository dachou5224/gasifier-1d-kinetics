
import sys
import os
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE

def scan_lunan():
    # LuNan setup
    coal_props = COAL_DATABASE["LuNan_Coal"]
    geometry = {'L': 6.87, 'D': 2.8}
    base_flow = 17917.0 / 3600.0
    
    print(f"Scanning O/C ratio impact for LuNan (Conc=66.0%)...")
    print(f"{'O2/Coal':<10} | {'T_exit (C)':<10} | {'CO (dry%)':<10} | {'Status'}")
    print("-" * 50)
    
    oc_range = np.linspace(0.85, 1.20, 10)
    
    for oc in oc_range:
        op_conds = {
            'coal_flow': base_flow,
            'o2_flow': base_flow * oc,
            'steam_flow': 0.0,
            'P': 4.0e6,
            'T_in': 300.0,
            'HeatLossPercent': 1.2,
            'SlurryConcentration': 66.0,
            'AdaptiveFirstCellLength': True
        }
        
        try:
            system = GasifierSystem(geometry, coal_props, op_conds)
            arr, z = system.solve(N_cells=20, solver_method='newton_fd', jacobian_mode='centered_fd')
            last = arr[-1]
            T_out = last[10] - 273.15
            
            gas = last[:8]
            co_pct = (gas[2] / (np.sum(gas[:7]) + 1e-12)) * 100
            
            status = "Target OK" if T_out > 1300 else "Too Cold"
            print(f"{oc:<10.3f} | {T_out:<10.1f} | {co_pct:<10.1f} | {status}")
        except:
            print(f"{oc:<10.3f} | Failed")

if __name__ == "__main__":
    scan_lunan()
