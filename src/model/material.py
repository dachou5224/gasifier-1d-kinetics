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
        Uses simple capacity model + Formation Enthalpy of Coal.
        """
        cp_s = PhysicalConstants.HEAT_CAPACITY_SOLID # J/kgK
        
        # Consistent Hf Calculation (if not provided)
        # Hf = -LHV + sum(n_i * Hf_prod_i)
        if 'Hf_coal' in coal_props:
            hf_coal = coal_props['Hf_coal']
        else:
            # Estimate Hf from LHV (MJ/kg)
            # Standard approach: hf = -lhv_d * 1e6 + (mass_frac_C/MC * Hf_CO2 + mass_frac_H/2MH * Hf_H2O...)
            # Since this is complex, we assume GasifierSystem already injected a consistent Hf_coal.
            # If not, we use a fallback to ensure we don't return 0. (Already handled in solve)
            hf_coal = coal_props.get('Hf_coal', -3e6) 

        h_s = hf_coal + cp_s * (state.T - 298.15)
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
