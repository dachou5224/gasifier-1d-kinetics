import numpy as np
from typing import Optional
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
    def get_solid_enthalpy(state: StateVector, coal_props: dict, T_solid_override: Optional[float] = None) -> float:
        """
        固相焓（对齐 Fortran enthal L1142）。
        保持生成焓基准的一致性，避免热解阶段出现能量跳变。
        """
        T_s = T_solid_override if T_solid_override is not None else state.T
        cp_s = coal_props.get('cp_char', 1300.0)  # J/kg/K
        hf_coal = coal_props.get('Hf_coal', 0.0)  # J/kg
        
        h_sensible = cp_s * (T_s - 298.15)
        h_s = h_sensible + hf_coal
        return state.solid_mass * h_s



    @staticmethod
    def get_total_enthalpy(state: StateVector, coal_props: dict, T_solid_override: Optional[float] = None) -> float:
        return MaterialService.get_gas_enthalpy(state) + MaterialService.get_solid_enthalpy(state, coal_props, T_solid_override)

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
