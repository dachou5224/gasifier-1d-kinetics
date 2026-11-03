import numpy as np
from typing import Optional
from model.state import StateVector
from model.physics import calculate_enthalpy, calculate_cp, calculate_gas_density
from model.physics import MOLAR_MASS
from model.constants import PhysicalConstants
from model.species_bridge import MAINLINE_SPECIES_NAMES

# 全局物种顺序：与 species_bridge.MAINLINE_SPECIES_NAMES / StateVector 一致
SPECIES_NAMES = list(MAINLINE_SPECIES_NAMES)

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
    def get_solid_enthalpy(state: StateVector, coal_props: dict, T_solid_override: Optional[float] = None,
                           include_formation_enthalpy: bool = True) -> float:
        """
        固相焓（对齐 Fortran enthal L1142）。

        ``include_formation_enthalpy=True``：按每 kg 固体的 ``Hf_coal``（LHV 口径公式）计生成焓，
        用于**入炉原煤**，保持煤的生成焓基准。
        ``include_formation_enthalpy=False``：只计显热。P1.6-A（审计 B9）起，热解之后的焦炭
        按组成计焓——碳的生成焓取 0（石墨）、灰为惰性只计显热，因此生成焓项不再出现。
        """
        T_s = T_solid_override if T_solid_override is not None else state.T
        cp_s = coal_props.get('cp_char', 1300.0)  # J/kg/K
        h_sensible = cp_s * (T_s - 298.15)
        if include_formation_enthalpy:
            return state.solid_mass * (h_sensible + coal_props.get('Hf_coal', 0.0))
        return state.solid_mass * h_sensible



    @staticmethod
    def get_total_enthalpy(state: StateVector, coal_props: dict, T_solid_override: Optional[float] = None,
                           include_formation_enthalpy: bool = True) -> float:
        return (
            MaterialService.get_gas_enthalpy(state)
            + MaterialService.get_solid_enthalpy(
                state, coal_props, T_solid_override,
                include_formation_enthalpy=include_formation_enthalpy,
            )
        )

    @staticmethod
    def get_char_formation_enthalpy_per_kg(coal_props: dict) -> float:
        """P1.6-A（B9）：焦炭的生成焓——碳取石墨 0、灰惰性，故恒为 0 J/kg。"""
        return 0.0

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
