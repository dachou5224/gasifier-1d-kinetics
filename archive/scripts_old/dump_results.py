import os, sys
sys.path.insert(0, os.path.abspath('src'))
from model.gasifier_system import GasifierSystem
from model.chemistry import COAL_DATABASE, VALIDATION_CASES
import logging
logging.disable(logging.CRITICAL)

try:
    case_data = VALIDATION_CASES['Paper_Case_6']
    model = GasifierSystem(case_data['coal_props'], case_data['op_conds'])
    model.solve(N_cells=10)
    df = model.get_results_dataframe()
    df.to_csv('1d_kinetic_results.csv', index=False)
    print("CSV saved successfully.")
    print(df[['z', 'T', 'CH4', 'O2', 'CO2', 'CO', 'H2O', 'H2']].head(5))
except Exception as e:
    import traceback
    traceback.print_exc()
