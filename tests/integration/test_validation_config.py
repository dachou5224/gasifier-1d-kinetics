import unittest
import json
import os
import sys
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.gasifier_system import GasifierSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from model.chemistry import COAL_DATABASE

class TestValidationCases(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load JSON
        json_path = os.path.join(os.path.dirname(__file__), '../../data/validation_cases.json')
        with open(json_path, 'r') as f:
            cls.config = json.load(f)
            
        # Use Centralized Coal Database
        cls.coals = COAL_DATABASE


    def test_cases_from_json(self):
        cases = self.config.get('cases', {})
        
        for case_name, data in cases.items():
            with self.subTest(case=case_name):
                logger.info(f"Running Validation Case: {case_name}")
                
                # 1. Prepare Inputs
                inputs = data['inputs']
                coal_key = data['coal_type']
                coal_props = self.coals.get(coal_key).copy()
                
                # Convert Units
                coal_flow_kg_s = inputs['FeedRate_kg_h'] / 3600.0
                o2_flow_kg_s = coal_flow_kg_s * inputs['Ratio_OC']
                steam_flow_kg_s = coal_flow_kg_s * inputs.get('SteamRatio_SC', 0.0)
                
                op_conds = {
                    'coal_flow': coal_flow_kg_s,
                    'o2_flow': o2_flow_kg_s,
                    'steam_flow': steam_flow_kg_s,
                    'P': inputs['P_Pa'],
                    'T_in': inputs['T_in_K'],
                    'SlurryConcentration': inputs.get('SlurryConcentration', 100.0)
                }

                
                geometry = {
                    'L': inputs.get('L_reactor', 6.0),
                    'D': inputs.get('D_reactor', 2.0)
                }
                
                # 2. Run Model
                system = GasifierSystem(geometry, coal_props, op_conds)
                results, z = system.solve(N_cells=25)
                
                # 3. Check Results
                # Exit Temperature (Last Cell)
                # results is (N_cells, 11) array. 
                # [O2, CH4, CO, CO2, H2S, H2, N2, H2O, Ws, Xc, T]
                res_out = results[-1]
                T_out = res_out[10]
                
                # Exit Composition (CO mole fraction)
                # Gas Moles: Indices 0-7
                gas_moles_vec = res_out[0:8]
                total_gas = np.sum(gas_moles_vec)
                co_moles = res_out[2]
                if total_gas > 1e-9:
                    y_co = (co_moles / total_gas) * 100.0
                else:
                    y_co = 0.0
                
                expected = data['expected']
                
                logger.info(f"  Result T_out: {T_out:.1f} K (Exp: ~{expected.get('TOUT_K_approx')})")
                logger.info(f"  Result Y_CO:  {y_co:.1f} % (Exp: ~{expected.get('YCO_pct_approx')})")
                
                print("\n=== Axial Temperature Profile ===")
                print(f"{'Z (m)':<10} | {'T (K)':<10}")
                print("-" * 25)
                for i in range(len(z)):
                    print(f"{z[i]:<10.3f} | {results[i][10]:<10.1f}")
                print("=================================\n")
                
                print("\n=== Axial Het Reaction Rate Profile (mol/s) ===")
                print(f"{'Z (m)':<10} | {'T (K)':<8} | {'C+O2 (Comb)':<12} | {'C+H2O (Gasif)':<12}")
                print("-" * 55)
                # Need to run a pass to get rates, or extracting from stored result if possible.
                # Actually, 'results' doesn't store rates. We need to recalculate or instrument.
                # Simplest way: Recalculate rates using system.kinetics
                
                for i in range(len(z)):
                    # Reconstruct state
                    res_i = results[i]
                    # [O2, CH4, CO, CO2, H2S, H2, N2, H2O, Ws, Xc, T]
                    # To array x
                    x_Arr = res_i
                    
                    # We need Cell object or similar context.
                    # Or just call kinetics directly.
                    # Creating a dummy state vector
                    from model.state import StateVector
                    sv = StateVector.from_array(x_Arr, P=op_conds['P'])
                    
                    # We need partial pressures.
                    # Heterogeneous rates need T, P, Xc, Ws
                    # Accessing system.cells[i]
                    cell_i = system.cells[i]
                    
                    # 1. Calc Physics
                    # Need solid ref flux. Approximate with coal_flow * 0.6
                    ref_flux = cell_i.coal_flow_dry * 0.6
                    phys = cell_i._calc_physics_props(sv, ref_flux)
                    
                    # 2. Calc Rates
                    # We pass zero source terms for this diagnostic visualization
                    # This might under-estimate availability (clipping), but for major components (C, O2, H2O)
                    # availability comes from Inlet (Solid) or Gas (if abundant).
                    # Actually for Gasification, H2O comes from Inlet+Source.
                    # This diagnostic will show rates as seen by the cell logic
                    dummy_src = np.zeros(8)
                    r_het, r_homo, _ = cell_i._calc_rates(sv, phys, dummy_src)
                    
                    print(f"{z[i]:<10.3f} | {sv.T:<8.1f} | {r_het['C+O2']:<12.2f} | {r_het['C+H2O']:<12.2f}")
                    
                print("=================================\n")


                # Just checking it runs and produces physical values
                self.assertGreater(T_out, 1000.0) 
                self.assertLess(T_out, 3000.0)
                self.assertGreater(y_co, 10.0)
                
                # Note: Exact match requires calibration step
                
if __name__ == '__main__':
    unittest.main()
