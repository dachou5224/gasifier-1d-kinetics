"""
异相反应动力学 (UCSM)
Fortran 原版参数为压力驱动: A 单位 g_C/(cm²·s·atm)，rate = k_s * P_atm。
Python 使用浓度驱动: k_d/k_s 单位 m/s，rate = k_s * C。
转换: k_s [m/s] = k_s_fortran [g/(cm²·s·atm)] * (R*T) * 1e4 / (101325 * M_C)
"""
import numpy as np
from .physics import (
    R_CONST,
    calculate_cp,
    calculate_enthalpy,
    calculate_gas_viscosity,
    calculate_diffusion_coefficient,
    calculate_kdiff_fortran,
)

# 压力驱动 → 浓度驱动 转换因子
# k_s_conc [m/s] = k_s_fortran [g/(cm²·s·atm)] * PRESSURE_TO_CONC_FACTOR(T)
# PRESSURE_TO_CONC_FACTOR = (R*T) * 1e4 / (101325 * 12)
P_ATM = 101325.0  # Pa
M_C_G_MOL = 12.0  # g/mol


def _pressure_to_conc_factor(T: float) -> float:
    """将 Fortran 压力驱动 k_s 转为浓度驱动 (m/s) 的乘数。"""
    return (R_CONST * T) * 1e4 / (P_ATM * M_C_G_MOL)


class PyrolysisModel:
    """Coal pyrolysis (Devolatilization) model"""
    @staticmethod
    def calculate_yield(T, P, V_initial, dt):
        """
        Calculate pyrolysis volatile yield
        T: Temperature (K)
        P: Pressure (Pa)
        V_initial: Initial volatile content (kg/kg_coal)
        dt: Time step (s)
        """
        # Pressure correction (Ref: Algorithms Doc)
        P_ref = 1.01325e5
        V_star = V_initial * (1 - 0.066 * np.log(max(P / P_ref, 1.0)))
        
        # First-order reaction kinetics
        k = 2.0e5 * np.exp(-74000 / (R_CONST * T))
        V_released = V_star * (1 - np.exp(-k * dt))
        return V_released

class HeterogeneousKinetics:
    """
    Heterogeneous Reaction Kinetics (Unreacted Core Shrinking Model - UCSM)
    参数对齐 Fortran: A 单位 g_C/(cm²·s·atm)，E 单位 J/mol。
    Fortran 写法 exp(-E_over_R/ts)，E_over_R 单位 K，换算 E(J/mol)=E_over_R*R_cal*CAL2J。
    """
    R_CAL = 1.987   # cal/(mol·K)
    CAL2J = 4.184   # J/cal

    def __init__(self):
        # Fortran 压力驱动参数 (Source1: combus L1173, cbstm L1215, cbco2 L1246, cbhym L1331)
        # exp(-E_over_R/ts) → E(J/mol) = E_over_R * R_cal * CAL2J
        self.params = {
            'C+O2':   {'A': 8710.0, 'E': 17967.0 * self.R_CAL * self.CAL2J},   # 149.4 kJ/mol
            'C+H2O':  {'A': 247.0,  'E': 21060.0 * self.R_CAL * self.CAL2J},   # 175.1 kJ/mol
            'C+CO2':  {'A': 247.0,  'E': 21060.0 * self.R_CAL * self.CAL2J},   # 175.1 kJ/mol
            'C+H2':   {'A': 0.12,   'E': 17921.0 * self.R_CAL * self.CAL2J}    # 149.0 kJ/mol
        }
        self.Xc0 = 0.8  # Default initial carbon fraction

    def calculate_total_rate(self, reaction, T, P, partial_pressure, d_p, Y, nu=1.0, Re=0.0, Sc=None, porosity=0.75, T_p=None, use_fortran_diffusion=False, phi=1.0):
        """
        Calculate heterogeneous flux (kmol Carbon / (m^2·s)) per unit EXTERNAL surface area.
        UCSM 分母含 1/(k_s*Y^2)，故返回的 flux 已折算到颗粒外表面积。
        调用方乘以的 surface_area 必须为几何外表面积 N*pi*d_p^2，不得含孔隙率/BET/Y^2。
        Driving force: P_eff = P_i - P_eq
        
        Args:
            reaction (str): Reaction type ('C+O2', 'C+H2O', etc.)
            T (float): Gas temperature [K]
            P (float): Total pressure [Pa]
            partial_pressure (float): Effective driving pressure [Pa]
            d_p (float): Particle diameter [m]
            Y (float): Shrinking core factor (Rc/Rp)
            nu (float): Stoichiometric coefficient [mol_C/mol_oxidant]
            Re (float): Reynolds number (Optional)
            Sc (float): Schmidt number (Optional)
            porosity (float): Particle porosity (Default 0.75)
            T_p (float): Particle temperature (Optional, defaults to T)
            use_fortran_diffusion (bool): 若 True，用 Fortran 经验扩散公式替代 Fuller
            phi (float): C+O2 机制因子，仅 use_fortran_diffusion 且 reaction=='C+O2' 时使用

        Returns:
            float: Flux [kmol/(m^2·s)] per unit external surface area
        """
        if reaction not in self.params: return 0.0
        if T_p is None: T_p = T
        T_mean = (T + T_p) / 2.0
        
        A_p = self.params[reaction]['A']
        E_p = self.params[reaction]['E']
        
        # 1. Film diffusion rate k_d [m/s]
        if use_fortran_diffusion:
            k_d = calculate_kdiff_fortran(reaction, T, T_mean, P, d_p, phi=phi)
        else:
            species_map = {'C+O2': 'O2', 'C+H2O': 'H2O', 'C+CO2': 'CO2', 'C+H2': 'H2'}
            gas_i = species_map.get(reaction, 'O2')
            D_i = calculate_diffusion_coefficient(T, P, gas_i)
            if Sc is None: Sc = 1.0
            Sh = 2.0 + 0.6 * (Re**0.5) * (Sc**(1/3.0))
            k_d = (Sh * D_i) / d_p
        
        # 2. Ash layer diffusion rate k_ash [m/s]
        # Formula: k_ash = k_d * eps^2.5 (Normalized to external surface)
        # Note: In standard UCSM, D_e = D_bulk * eps / tau. 
        # Here we follow the literature correlation provided.
        k_ash = k_d * (porosity ** 2.5)
        
        # 3. Surface chemical reaction rate k_s
        # Fortran A 为压力驱动 [g/(cm²·s·atm)]，需乘 (R*T) 转为浓度驱动 [m/s]
        # k_s [m/s] = A_p * exp(-E/RT) * (R*T) * 1e4 / (101325 * 12)
        k_s_pressure = A_p * np.exp(-E_p / (R_CONST * T_p))
        k_s = k_s_pressure * _pressure_to_conc_factor(T_p)
        
        # 4. Equilibrium Driving Force (P_eff = P_i - P_eq)
        # For C+O2, P_eq = 0.
        P_eq = 0.0
        # Placeholder for P_eq logic (will be calculated in KineticsService)
        # But we can calculate simple ones here if needed.
        
        # 5. Resistance Assembly
        # R_tot = 1/(nu*k_d) + (1-Y)/(nu*k_ash*Y) + 1/(k_s*Y^2)
        # This gives Carbon consumption rate per unit external area.
        if Y < 1e-4: return 0.0
        
        denom = (1.0/(nu * k_d) + (1.0-Y)/(nu * k_ash * Y) + 1.0/(k_s * Y**2))
        
        # Effective Driving Force Conc: kmol/m3
        P_eff = max(partial_pressure - P_eq, 0.0)
        conc_eff_kmol = P_eff / (R_CONST * T) / 1000.0
        
        rate = conc_eff_kmol / denom
        return rate

def calculate_wgs_equilibrium(T):
    """Water Gas Shift (WGS) Equilibrium Constant (Table 2-2)"""
    # K = exp(4578/T - 4.33)；Fortran exp(4019/T-3.69) 使 CO 更差，暂保留原式
    return np.exp(4578.0 / T - 4.33)

def calculate_methanation_equilibrium(T):
    """Methanation Equilibrium (Table 2-3)"""
    # K = exp(21832/T - 21.03) 
    return np.exp(21832.0 / T - 21.03)

def calculate_boudouard_equilibrium(T):
    """Boudouard Reaction Equilibrium (Table 2-4)"""
    # K = exp(-20573/T + 20.32)
    return np.exp(-20573.0 / T + 20.32)


def calculate_cstm_equilibrium(T: float) -> float:
    """
    C+H2O ⇌ CO+H2 平衡常数 (Fortran cbstm Line 1200-1202)
    cseqk = exp(17.644 - 30260/(ts*1.8)) = exp(17.644 - 16811/T)
    K = P_CO * P_H2 / P_H2O → P_H2O,eq = P_H2 * P_CO / K
    """
    return np.exp(17.644 - 16811.0 / T)
