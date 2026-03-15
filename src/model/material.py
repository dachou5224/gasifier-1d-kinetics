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
        Fortran: enths = cps*fcoal*(1+fash-wl/100)*(ts-298)，纯显热，不区分 coal/char。
        为维持 Shomate 全焓框架基准一致性，加上 Hf_coal。
        [BUGFIX] 煤脱除挥发分变成焦炭后，剩下的本质是单质碳，单质碳(石墨)标准生成焓定义为 0！
        所以，固体的生成焓不应该永远乘以 Hf_coal，否则挥发分离去后，这部分生成焓凭空消失变成了“黑洞”。
        通过比较入口态的 char_Xc0（纯焦炭碳含量）与当前 Xc，我们可以插值：
        如果尚未脱挥发分（Xc == 原煤 Cd），生成焓是 Hf_coal
        如果完全脱挥发分（Xc == 纯焦炭，接近 1），生成焓是 0。
        这里做一个简化物理假设：假若模型区分原煤与焦炭，此处只需以 Xc 对比即可。
        为兼容 1D 动力模型的“瞬间挥发分脱除”假设：
        第一格的入口 state.carbon_fraction 其实还是原煤的 Cd。
        """
        T_s = T_solid_override if T_solid_override is not None else state.T
        cp_s = coal_props.get('cp_char', 1300.0)  # J/kg/K
        hf_coal = coal_props.get('Hf_coal', -3e6)  # J/kg 原煤生成焓
        
        # 煤的原基准碳分
        Cd_raw = coal_props.get('Cd', 60.0) / 100.0
        
        # 估算是否为原煤：如果当前 Xc 接近于原煤 Cd_raw，则携带全部 hf_coal
        # 在瞬间脱挥发分后，Xc 会骤升，此时挥发分已作为气体进入系统，焦炭的 Hf 应当视为 0。
        # 简单阈值：当碳含量升高偏离原煤时，生成焓线性衰减至 0。
        if state.carbon_fraction <= Cd_raw + 0.05:
            h_formation = hf_coal
        else:
            h_formation = 0.0 # 焦炭或灰的生成焓基准定义为 0

        h_sensible = cp_s * (T_s - 298.15)
        h_s = h_sensible + h_formation
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
