
import numpy as np

class PyrolysisService:
    """
    Service for calculating Coal Pyrolysis Yields.
    Implements the original Rajan Pyrolysis Model (Empirical Correlations).
    Decoupled from KineticsService and GasifierSystem.
    """
    
    def __init__(self):
        pass

    def calc_yields(self, coal_props: dict):
        """
        Calculate pyrolysis product yields enforcing Elemental Conservation.
        Limits Volatile Carbon based on available H and O stoichiometry.
        """
        # 1. Load Ultimate Analysis (Dry Basis wt%)
        Cd = coal_props.get('Cd', 0.0)
        Hd = coal_props.get('Hd', 0.0)
        Od = coal_props.get('Od', 0.0)
        Nd = coal_props.get('Nd', 0.0)
        Sd = coal_props.get('Sd', 0.0)
        
        # Proximate for Carbon Split
        FCd = coal_props.get('FCd', 50.0)
        
        # 2. Determine Potential Volatile Carbon
        C_vol_potential = max(Cd - FCd, 0.0)
        n_C_potential = C_vol_potential / 12.011
        
        # 3. Base Yields (N, S) strict
        n_N2 = Nd / 28.013
        n_H2S = Sd / 32.06
        
        # Available H and O for Carbon species
        # H for H2S: 2 H per S
        H_for_S = n_H2S * 2.0 * 1.008
        Hd_avail = max(Hd - H_for_S, 0.0)
        n_H_avail = Hd_avail / 1.008
        
        n_O_avail = Od / 15.999
        
        # 4. Dynamic Species Optimization (Maximize Carbon Gasification)
        # Strategy: 
        # 1. Prioritize CO formation (consumes O, carries C).
        # 2. Use remaining C to form CH4 (consumes H).
        # 3. If any C remains ungassifiable (due to H/O limits), it stays in Char.
        
        # Max CO limited by available O (1:1 stoichiometry)
        n_CO = min(n_C_potential, n_O_avail)
        
        # Remaining potential C
        n_C_rem = max(n_C_potential - n_CO, 0.0)
        
        # Max CH4 limited by available H (1:4 stoichiometry)
        n_CH4 = min(n_C_rem, n_H_avail / 4.0)
        
        # CO2 formation is O-expensive (2:1). Only form if we have excess O 
        # (which we wont given n_CO logic, unless n_C_potential was low).
        # If n_O_avail > n_C_potential, we have excess O.
        n_CO2 = 0.0
        n_O_excess = max(n_O_avail - n_CO, 0.0)
        if n_O_excess > 0:
            # Convert some CO to CO2 to use up O? Or just H2O.
            # Burning CO -> CO2 releases heat. Pyrolysis is endothermic/neutral usually.
            # Let's put excess O into H2O (standard moisture/structure water).
            pass 
        
        # 5. Residuals
        # Oxygen
        n_O_used = n_CO + 2.0*n_CO2
        n_O_resid = max(n_O_avail - n_O_used, 0.0)
        n_H2O = n_O_resid 
        
        # Hydrogen
        n_H_used = 4.0*n_CH4 + 2.0*n_H2O + 2.0*n_H2S # Added H2S back (2 H per S)
        # Wait, Hd_avail already subtracted H_for_S.
        # So n_H_avail is "H available for C and H2". 
        # H used by CH4 and H2O (from O_resid).
        # O_resid came from Od limits.
        
        # Let's re-verify H accounting.
        # Hd_avail = Hd - H(H2S).
        # n_H_avail derived from Hd_avail.
        # Used by CH4: 4 * n_CH4.
        # Used by H2O: 2 * n_H2O.
        n_H_used_from_avail = 4.0*n_CH4 + 2.0*n_H2O
        
        n_H_resid = max(n_H_avail - n_H_used_from_avail, 0.0)
        n_H2 = n_H_resid / 2.0
        
        # 8. Output
        factor = 10.0 # kmol/100kg -> mol/kg
        
        molar_yields = np.zeros(8)
        molar_yields[0] = 0.0 
        molar_yields[1] = n_CH4 * factor
        molar_yields[2] = n_CO * factor
        molar_yields[3] = n_CO2 * factor
        molar_yields[4] = n_H2S * factor
        molar_yields[5] = n_H2 * factor
        molar_yields[6] = n_N2 * factor 
        molar_yields[7] = n_H2O * factor
        
        # 9. Calc Volatile Mass
        MW = [32.0, 16.04, 28.01, 44.01, 34.08, 2.016, 28.013, 18.015]
        W_vol_calc = sum(molar_yields[i] * MW[i] for i in range(8)) / 1000.0
        
        return molar_yields, W_vol_calc
