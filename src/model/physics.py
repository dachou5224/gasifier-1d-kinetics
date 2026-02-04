import numpy as np

# 通用气体常数 R (J/(mol*K))
R_CONST = 8.3144626

# 分子量 (g/mol 或 kg/kmol)
MOLAR_MASS = {
    'CO': 28.01,
    'H2': 2.016,
    'CO2': 44.01,
    'CH4': 16.04,
    'H2O': 18.015,
    'O2': 31.998,
    'N2': 28.013,
    'H2S': 34.08,
    'COS': 60.07
}

# =============================================================================
# NIST Shomate Equation Coefficients
# 数据来源: NIST Chemistry WebBook
# 结构: { 'Species': { 'Low': [coeffs...], 'High': [coeffs...], 'T_cut': 1000.0 } }
# Coeffs format: [A, B, C, D, E, F, G, H]
# =============================================================================

SHOMATE_DB = {
    'CO': {
        'T_cut': 1300.0,
        'Low': [25.56759, 6.096130, 4.054656, -2.671301, 0.131021, -118.0089, 227.3665, -110.5271],
        'High': [35.15070, 1.300095, -0.205921, 0.013550, -3.282780, -127.8375, 231.7120, -110.5271]
    },
    'CO2': {
        'T_cut': 1200.0,
        'Low': [24.99735, 55.18696, -33.69137, 7.948387, -0.136638, -403.6075, 228.2431, -393.5224],
        'High': [58.16639, 2.720074, -0.492289, 0.038844, -6.447293, -425.9186, 263.6125, -393.5224]
    },
    'H2': {
        'T_cut': 1000.0,
        'Low': [33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974, 0.0],
        # NIST 1000-2500K
        'High': [18.563083, 12.257357, -2.859786, 0.268238, 1.977990, -1.147438, 156.288133, 0.0]
    },
    'H2O': {
        'T_cut': 1000.0,
        'Low': [30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967, -241.8264],
        'High': [41.96426, 8.622053, -1.499780, 0.098119, -11.15764, -272.1797, 219.7809, -241.8264]
    },
    'CH4': {
        'T_cut': 1300.0,
        'Low': [-0.703029, 108.4773, -42.52157, 5.862788, 0.678565, -76.84376, 158.7163, -74.87310],
        'High': [85.81217, 11.26467, -2.114146, 0.138190, -26.42221, -153.5327, 224.4143, -74.87310]
    },
    'N2': {
        'T_cut': 1000.0,
        'Low': [28.98641, 1.853978, -9.647459, 16.63537, 0.000117, -8.671914, 212.0238, 0.0],
        'High': [19.50583, 19.88705, -8.598535, 1.369784, 0.527601, -4.935202, 212.3900, 0.0]
    },
    'O2': {
        'T_cut': 700.0,
        'Low': [31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471, 246.7945, 0.0],
        'High': [30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 236.1663, 0.0]
    },
    'H2S': {
        'T_cut': 1400.0, 
        'Low': [26.88412, 18.67809, 3.434203, -3.378702, 0.135882, -28.91211, 233.3747, -20.50202],
        'High': [51.22136, 4.147486, -0.643566, 0.041621, -10.46385, -55.87606, 243.6900, -20.50202]
    },
    'COS': {
        'T_cut': 6000.0,
        'Low': [43.66691, 19.99841, -18.73031, 8.524458, -0.490011, -153.2847, 260.6784, -142.1648],
        'High': [43.66691, 19.99841, -18.73031, 8.524458, -0.490011, -153.2847, 260.6784, -142.1648]
    }
}

def _get_coeffs(species, T):
    """根据温度选择合适的 Shomate 系数"""
    if species not in SHOMATE_DB:
        return None
    
    data = SHOMATE_DB[species]
    if T < data['T_cut']:
        return data['Low']
    else:
        return data['High']

def _calculate_shomate(species, T, prop_type):
    """
    内部辅助函数：根据 Shomate 方程计算热力学性质
    T: 温度 (K)
    prop_type: 'H' (Enthalpy), 'S' (Entropy), 'Cp' (Heat Capacity)
    """
    coeffs = _get_coeffs(species, T)
    if coeffs is None:
        return 0.0
    
    t = T / 1000.0
    A, B, C, D, E, F, G, H_const = coeffs

    if prop_type == 'Cp':
        # Cp = A + B*t + C*t^2 + D*t^3 + E/t^2 (J/mol*K)
        Cp = A + B*t + C*t**2 + D*t**3 + E/(t**2)
        return Cp 

    elif prop_type == 'H':
        # H = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H_ref (kJ/mol)
        # NIST 计算出的是 H(T) - H(298.15) + Delta_f H(298.15)
        # 注意：公式里的 H 项 (coeffs[7]) 对应的是 Delta_f H(298.15) 吗？
        # NIST 定义: H - H(298) = ... + F - H. 
        # 这里的 F 包含了生成焓信息. NIST 网站结果 H = standard enthalpy.
        H_val = A*t + B*(t**2)/2 + C*(t**3)/3 + D*(t**4)/4 - E/t + F
        return H_val * 1000.0  # kJ/mol -> J/mol

    elif prop_type == 'S':
        # S = A*ln(t) + B*t + C*t^2/2 + D*t^3/3 - E/(2*t^2) + G (J/mol*K)
        S_val = A*np.log(t) + B*t + C*(t**2)/2 + D*(t**3)/3 - E/(2*t**2) + G
        return S_val 

def get_enthalpy_molar(species, T):
    """计算摩尔焓 (J/mol)"""
    return _calculate_shomate(species, T, 'H')

def get_entropy_molar(species, T):
    """计算摩尔熵 (J/(mol K))"""
    return _calculate_shomate(species, T, 'S')

def get_gibbs_free_energy(species, T):
    """
    计算吉布斯自由能 G = H - TS (J/mol)
    """
    H = get_enthalpy_molar(species, T)
    S = get_entropy_molar(species, T)
    return H - T * S

def get_equilibrium_constants(T):
    """
    计算平衡常数 K (仅用于显示，核心计算已转为 Gibbs 最小化)
    R1: CO + H2O <-> CO2 + H2
    R2: CO + 3H2 <-> CH4 + H2O
    """
    G_CO  = get_gibbs_free_energy('CO', T)
    G_H2O = get_gibbs_free_energy('H2O', T)
    G_CO2 = get_gibbs_free_energy('CO2', T)
    G_H2  = get_gibbs_free_energy('H2', T)
    G_CH4 = get_gibbs_free_energy('CH4', T)
    
    delta_G1 = (G_CO2 + G_H2) - (G_CO + G_H2O)
    delta_G2 = (G_CH4 + G_H2O) - (G_CO + 3*G_H2)
    
    K1 = np.exp(-delta_G1 / (R_CONST * T))
    K2 = np.exp(-delta_G2 / (R_CONST * T))
    
    return K1, K2

# Aliases for compatibility
calculate_enthalpy = get_enthalpy_molar

def calculate_cp(species, T):
    return _calculate_shomate(species, T, 'Cp')

# Missing Physics Functions (Restored)
# Fuller Diff Volumes
FULLER_VOLUMES = {
    'CO': 18.0, 'H2': 6.0, 'CO2': 26.9, 'CH4': 25.14,
    'H2O': 13.1, 'O2': 16.3, 'N2': 18.5, 'C': 15.9,
}

def calculate_gas_viscosity(T, species='N2'):
    mu_ref = 1.781e-5
    T_ref = 273.15
    S = 111.0
    return mu_ref * (T/T_ref)**1.5 * (T_ref + S) / (T + S)

def calculate_diffusion_coefficient(T, P, species_A, species_B='N2'):
    P_bar = P / 1e5
    M_A = MOLAR_MASS.get(species_A, 28.0)
    M_B = MOLAR_MASS.get(species_B, 28.0)
    M_AB = 2 / (1/M_A + 1/M_B)
    
    v_A = FULLER_VOLUMES.get(species_A, 18.0)
    v_B = FULLER_VOLUMES.get(species_B, 18.0)
    
    D_cm2s = (0.00143 * T**1.75) / (P_bar * np.sqrt(M_AB) * (v_A**(1/3) + v_B**(1/3))**2)
    return D_cm2s * 1e-4

def calculate_gas_density(P, T, composition):
    M_avg = sum(MOLAR_MASS.get(s, 28.0) * y for s, y in composition.items())
    return (P * M_avg * 1e-3) / (R_CONST * T)

def get_lhv(species):
    lhv_data = {
        'H2': 241820, 'CO': 282980, 'CH4': 802340, 
        'H2S': 518800, 'O2': 0, 'CO2': 0, 'H2O': 0, 'N2': 0, 'C': 393510
    }
    return lhv_data.get(species, 0.0)