import numpy as np
from .physics import R_CONST, calculate_cp, calculate_enthalpy, calculate_gas_viscosity, calculate_diffusion_coefficient

class PyrolysisModel:
    """煤的热解（脱挥发分）模型"""
    @staticmethod
    def calculate_yield(T, P, V_initial, dt):
        """
        计算热解挥发分产率
        T: 温度 (K)
        P: 压力 (Pa)
        V_initial: 初始挥发分含量 (kg/kg_coal)
        dt: 时间步长 (s)
        """
        # 压力修正 (参考算法文档)
        P_ref = 1.01325e5
        V_star = V_initial * (1 - 0.066 * np.log(max(P / P_ref, 1.0)))
        
        # 一阶反应动力学
        k = 2.0e5 * np.exp(-74000 / (R_CONST * T))
        V_released = V_star * (1 - np.exp(-k * dt))
        return V_released

class HeterogeneousKinetics:
    """气固异相反应动力学 (未反应缩核模型 UCSM)"""
    def __init__(self):
        # Table 2-7 Parameters (E converted to J/mol)
        self.params = {
            'C+O2':   {'A': 2.3e2,  'E': 1.1e8 / 1000.0},
            'C+H2O':  {'A': 2.4e4,  'E': 1.43e8 / 1000.0},
            'C+CO2':  {'A': 2.4e4,  'E': 1.43e8 / 1000.0},
            'C+H2':   {'A': 6.4,    'E': 1.3e8 / 1000.0}
        }
        self.Xc0 = 0.8 # Default initial carbon fraction

    def calculate_total_rate(self, reaction, T, P, partial_pressure, d_p, Y, nu=1.0, Re=0.0, Sc=None, porosity=0.75, T_p=None):
        """
        计算异相反应的总速率 (kmol Carbon / (m^2·s))
        Based on Equation 2-12 and Table 2-7.
        Driving force: P_eff = P_i - P_eq
        
        nu: Stoichiometric factor (mols Carbon per mol oxidant). 
            For C+O2, nu = phi. For gasification, nu = 1.0 usually.
        """
        if reaction not in self.params: return 0.0
        if T_p is None: T_p = T
        
        A_p = self.params[reaction]['A']
        E_p = self.params[reaction]['E']
        
        # 1. 气膜扩散速率 k_d (Formula: Sh * D / dp) [m/s]
        species_map = {'C+O2': 'O2', 'C+H2O': 'H2O', 'C+CO2': 'CO2', 'C+H2': 'H2'}
        gas_i = species_map.get(reaction, 'O2')
        D_i = calculate_diffusion_coefficient(T, P, gas_i)
        
        # Sherwood Correlation: Sh = 2 + 0.6 * Re^0.5 * Sc^0.33
        if Sc is None: Sc = 1.0 # Sc approx 1 for simple gas
        Sh = 2.0 + 0.6 * (Re**0.5) * (Sc**(1/3.0))
        k_d = (Sh * D_i) / d_p 
        
        # 2. 灰层扩散速率 k_ash [m/s]
        # Formula: k_ash = k_d * eps^2.5 (Normalized to external surface)
        # Note: In standard UCSM, D_e = D_bulk * eps / tau. 
        # Here we follow the literature correlation provided.
        k_ash = k_d * (porosity ** 2.5)
        
        # 3. 表面化学反应速率 k_s (m/s)
        k_s = A_p * np.exp(-E_p / (R_CONST * T_p))
        
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
    """水煤气变换反应 (WGS) 平衡常数 (Table 2-2)"""
    # K = exp(4578/T - 4.33)
    return np.exp(4578.0 / T - 4.33)

def calculate_methanation_equilibrium(T):
    """甲烷化反应平衡 (Table 2-3)"""
    # K = exp(21832/T - 21.03) 
    return np.exp(21832.0 / T - 21.03)

def calculate_boudouard_equilibrium(T):
    """Boudouard反应平衡 (Table 2-4)"""
    # K = exp(-20573/T + 20.32)
    return np.exp(-20573.0 / T + 20.32)
