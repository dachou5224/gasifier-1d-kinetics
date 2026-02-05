import numpy as np
from typing import Dict, List, Optional
from model.state import StateVector
from model.kinetics import HeterogeneousKinetics
from model.constants import PhysicalConstants
from model.physics import R_CONST
from model.material import SPECIES_NAMES

class KineticsService:
    """
    Service for calculating all reaction rates (Pyrolysis, Heterogeneous, Homogeneous).
    Decouples logic from the Solver/Cell.
    """
    def __init__(self):
        # Underlying Physics Model
        self.het_model = HeterogeneousKinetics()
        
        # Homogeneous Parameters (Table 2-5, converted to J/mol)
        self.A_homo = {
            'CO_Ox':  2.23e12,
            'H2_Ox':  1.08e13,
            'WGS':    2.78e3,
            'RWGS':   1.0e5,
            'CH4_Ox': 1.6e10,
            'MSR':    4.4e11 
        }
        self.E_homo = {
            'CO_Ox':  1.25e8 / 1000.0,
            'H2_Ox':  8.37e7 / 1000.0,
            'WGS':    1.25e7 / 1000.0,
            'RWGS':   6.27e7 / 1000.0,
            'CH4_Ox': 1.256e8 / 1000.0, 
            'MSR':    30000.0 * R_CONST 
        }
        
    def calc_heterogeneous_rates(self, state: StateVector, particle_diameter: float, 
                                 surface_area: float, X_total: float = 0.0, 
                                 Re: float = 0.0, Sc: float = 1.0) -> Dict[str, float]:
        """
        Calculate Surface Reaction Rates (mol/s).
        Returns: Dict {'C+O2': rate, ...}
        """
        r = {}
        F_total = state.total_gas_moles
        if F_total < PhysicalConstants.TOLERANCE_SMALL:
             # This is expected for initial cells with no gas yet, debug level only
             # logging.debug(f"LogicWarning: F_total ~ 0 at T={state.T:.1f}")
             F_total = PhysicalConstants.TOLERANCE_SMALL
        
        # 1. Particle Temperature Model (Ross)
        P, T = state.P, state.T
        F_O2 = state.gas_moles[0]
        # P_eff / RT for O2 = C_O2 [kmol/m3]
        C_O2_kmol = state.get_concentration(0)
        T_p = T + 6.6e4 * C_O2_kmol
        T_p = min(T_p, 4000.0)
        
        # 2. Mechanism Factor phi (Determines CO/CO2 ratio)
        p_fac = 2500.0 * np.exp(-5.19e4 / (R_CONST * T_p))
        phi = (2*p_fac + 2) / (p_fac + 2)
        r['phi'] = phi 
        
        # 3. Y (Shrinking Core Factor)
        # Y = Rc / Rp = (1-X)**(1/3) for sphere
        Y = (1.0 - X_total)**(1.0/3.0)
        Y = max(Y, 1e-3)
        
        mapping = {'C+O2': 'O2', 'C+H2O': 'H2O', 'C+CO2': 'CO2', 'C+H2': 'H2'}
        
        for rxn, sp in mapping.items():
            idx = SPECIES_NAMES.index(sp)
            F_i = max(state.gas_moles[idx], 0.0)
            P_i = (F_i / F_total) * P # Pa
            
            # 4. Driving Force Correction (P_eq)
            # Use concentration-derived partial pressures if possible, but here we need P_i
            # P_i = C_i * R * T (Pa)
            # But let's keep P_i = y_i * P for simplicity as it handles F_total ~ 0 better in some checks
            
            P_eq = 0.0
            if rxn == 'C+CO2':
                # C + CO2 <-> 2CO. Kp = P_CO^2 / P_CO2 (atm)
                from model.kinetics import calculate_boudouard_equilibrium
                Kp_atm = calculate_boudouard_equilibrium(T)
                P_CO = (state.gas_moles[2] / F_total) * P
                P_eq = (P_CO**2 / (Kp_atm * 1.01325e5 + 1e-9))
            elif rxn == 'C+H2O':
                # C + H2O <-> CO + H2. Kp = P_CO * P_H2 / P_H2O
                from model.kinetics import calculate_wgs_equilibrium
                Kp_wgs_homo = calculate_wgs_equilibrium(T)
                from model.kinetics import calculate_boudouard_equilibrium
                # Roughly Kp_C+H2O = Kp_Boudouard / Kp_WGS
                Kp_het_atm = calculate_boudouard_equilibrium(T) / (Kp_wgs_homo + 1e-9)
                P_CO = (state.gas_moles[2] / F_total) * P
                P_H2 = (state.gas_moles[5] / F_total) * P
                P_eq = (P_CO * P_H2) / (Kp_het_atm * 1.01325e5 + 1e-9)
            
            # 5. Stoichiometric factor nu (Mols Carbon per mol oxidant)
            nu = phi if rxn == 'C+O2' else 1.0
            
            # UCSM Rate Evaluation (kmol/m2.s)
            P_eff = max(P_i - P_eq, 0.0)
            rate_kmols = self.het_model.calculate_total_rate(
                rxn, T, P, P_eff, particle_diameter, Y, nu=nu, Re=Re, Sc=Sc, T_p=T_p
            )
            
            # Convert kmol/m2.s -> mol/m2.s
            rate_flux = rate_kmols * 1000.0
            
            # Convert kmol/m2.s -> mol/m2.s
            rate_flux = rate_kmols * 1000.0
            
            r[rxn] = rate_flux * surface_area 
            
        return r

    def calc_homogeneous_rates(self, state: StateVector, volume: float) -> Dict[str, float]:
        """
        Calculate Homogeneous Rates (mol/s) based on Table 2-5.
        """
        rates = {}
        P, T = state.P, state.T
        F_total = state.total_gas_moles
        if F_total < PhysicalConstants.TOLERANCE_SMALL:
             F_total = PhysicalConstants.TOLERANCE_SMALL
        
        # Concentrations (kmol/m3) - Table 2-5 refers to Ca, Cb in kmol/m3
        # Concentrations (kmol/m3) - Table 2-5 refers to Ca, Cb in kmol/m3
        C = {}
        for i, sp in enumerate(SPECIES_NAMES):
            C[sp] = state.get_concentration(i)

        # 1-5: Second Order (r = k Ca Cb) [kmol/m3.s]
        # (1) CO Combustion
        k = self.A_homo['CO_Ox'] * np.exp(-self.E_homo['CO_Ox'] / (R_CONST * T))
        r1 = k * C['CO'] * C['O2']
        rates['CO_Ox'] = r1 * volume * 1000.0 # mol/s
        
        # (2) H2 Combustion
        k = self.A_homo['H2_Ox'] * np.exp(-self.E_homo['H2_Ox'] / (R_CONST * T))
        r2 = k * C['H2'] * C['O2']
        rates['H2_Ox'] = r2 * volume * 1000.0
        
        # (3) WGS
        k = self.A_homo['WGS'] * np.exp(-self.E_homo['WGS'] / (R_CONST * T))
        r3 = k * C['CO'] * C['H2O']
        rates['WGS'] = r3 * volume * 1000.0
        
        # (4) RWGS
        k = self.A_homo['RWGS'] * np.exp(-self.E_homo['RWGS'] / (R_CONST * T))
        r4 = k * C['CO2'] * C['H2']
        rates['RWGS'] = r4 * volume * 1000.0
        
        # (5) CH4 Combustion
        k = self.A_homo['CH4_Ox'] * np.exp(-self.E_homo['CH4_Ox'] / (R_CONST * T))
        r5 = k * C['CH4'] * C['O2']
        rates['CH4_Ox'] = r5 * volume * 1000.0
        
        # (6) MSR
        k = self.A_homo['MSR'] * np.exp(-self.E_homo['MSR'] / (R_CONST * T))
        r6 = k * C['CH4']
        rates['MSR'] = r6 * volume * 1000.0
        return rates
