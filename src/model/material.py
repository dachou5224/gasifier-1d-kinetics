import numpy as np
from model.state import StateVector
from model.physics import calculate_enthalpy, calculate_cp, calculate_gas_density
from model.physics import MOLAR_MASS
from model.constants import PhysicalConstants

# Global Species Order Definition
# MUST MATCH StateVector comments and Solver logic
SPECIES_NAMES = ['O2', 'CH4', 'CO', 'CO2', 'H2S', 'H2', 'N2', 'H2O']

class MaterialService:
    """
    Pure Functional Service for Material Properties.
    Calculates thermodynamic properties based on State.
    """
    
    @staticmethod
    def get_gas_enthalpy(state: StateVector) -> float:
        """Calculate Total Heat Flow of Gas Phase (J/s)"""
        H_total = 0.0
        for i, sp in enumerate(SPECIES_NAMES):
            moles = state.gas_moles[i]
            if moles > 0:
                h_i = calculate_enthalpy(sp, state.T) # J/mol
                H_total += moles * h_i
        return H_total


    @staticmethod
    def get_solid_enthalpy(state: StateVector, coal_props: dict) -> float:
        """
        Calculate Total Heat Flow of Solid Phase (J/s).
        
        Distinguishes between raw coal and char:
        - Raw coal (before pyrolysis): Uses Hf_coal and cp_coal
        - Char (after pyrolysis): Uses Hf_char and cp_char
        
        The carbon fraction (Xc) is used as a proxy:
        - Initial Xc (Cd_total) indicates raw coal
        - Higher Xc indicates char (volatiles have left, concentrating carbon)
        """
        # Get carbon fractions
        Cd_raw = coal_props.get('Cd', 60.0) / 100.0  # Raw coal carbon fraction
        Xc_current = state.carbon_fraction  # Current carbon fraction
        
        # Determine if solid is predominantly char or coal
        # After pyrolysis, Xc increases (char is ~80-90% C, coal is ~60-70% C)
        # Use a threshold: if Xc > Cd_raw * 1.1, it's char
        is_char = Xc_current > Cd_raw * 1.05  # 5% tolerance
        
        if is_char:
            # Char properties
            cp_s = coal_props.get('cp_char', 1300.0)  # J/kgK (char ~1100-1500)
            # Char formation enthalpy: Much lower than coal (volatiles gone)
            # Estimate: Hf_char ≈ Hf_coal + (energy carried by volatiles)
            # Simplified: Hf_char ≈ 0 for pure carbon (graphite reference)
            # For practical char: use a fraction of coal Hf
            hf_coal = coal_props.get('Hf_coal', -3e6)
            vol_frac = coal_props.get('VM', 30.0) / 100.0  # Volatile matter fraction
            # Char Hf is roughly the non-volatile fraction's contribution
            # Simplified estimate: Hf_char = (1 - vol_frac) * Hf_coal
            hf_solid = coal_props.get('Hf_char', (1.0 - vol_frac * 0.7) * hf_coal)
        else:
            # Raw coal properties
            cp_s = coal_props.get('cp_coal', 1500.0)  # J/kgK (coal ~1300-1700)
            hf_solid = coal_props.get('Hf_coal', -3e6)
        
        h_s = hf_solid + cp_s * (state.T - 298.15)
        return state.solid_mass * h_s


    @staticmethod
    def get_total_enthalpy(state: StateVector, coal_props: dict) -> float:
        return MaterialService.get_gas_enthalpy(state) + MaterialService.get_solid_enthalpy(state, coal_props)

    @staticmethod
    def get_gas_mix_property(state: StateVector, prop='density') -> float:
        # Calculate mole fractions
        y = state.gas_fractions
        composition = {sp: y[i] for i, sp in enumerate(SPECIES_NAMES)}
        
        if prop == 'density':
            return calculate_gas_density(state.P, state.T, composition)
        elif prop == 'viscosity':
            # Simplified: Use N2 viscosity or mix
            from model.physics import calculate_gas_viscosity
            return calculate_gas_viscosity(state.T)
        return 0.0
