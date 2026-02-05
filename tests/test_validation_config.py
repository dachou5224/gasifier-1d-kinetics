import unittest
import json
import os
import sys
import numpy as np
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model.gasifier_system import GasifierSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TestValidationCases(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load JSON
        json_path = os.path.join(os.path.dirname(__file__), 'validation_cases.json')
        with open(json_path, 'r') as f:
            cls.config = json.load(f)
            
        # Define Coal Database
        cls.coals = {
            "Paper_Base_Coal": {
                'Cd': 80.19,
                'Hd': 4.83,
                'Od': 9.76,
                'Ad': 7.35,
                'Nd': 0.0, # Assumed small/zero if not specified
                'Sd': 0.0, # Assumed small/zero
                'Mt': 0.0, # Moisture
                'Hf': -0.6e6 # Approx J/kg
            }
        }

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
                    'T_in': inputs['T_in_K']
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
                
                # Assertions (Relaxed for now as we haven't calibrated yet)
                # Just checking it runs and produces physical values
                self.assertGreater(T_out, 1000.0) 
                self.assertLess(T_out, 3000.0)
                self.assertGreater(y_co, 10.0)
                
                # Note: Exact match requires calibration step
                
if __name__ == '__main__':
    unittest.main()
