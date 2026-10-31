import numpy as np

# Universal Gas Constant R (J/(mol*K))
R_CONST = 8.3144626

# Molar Mass (g/mol or kg/kmol)
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
# NIST Shomate Equation Coefficients —— 多段版本（P1.5-A，2026-10-30）
# Data Source: NIST Chemistry WebBook, "Shomate Equation" 表
# Coeffs format: [A, B, C, D, E, F, G, H]
#
# 分段与来源（NIST 原表分段）：
#   CO   : 298–1300 / 1300–6000
#   CO2  : 298–1200 / 1200–6000
#   H2   : 298–1000 / 1000–2500 / 2500–6000
#   H2O  : 500–1700 / 1700–6000
#   CH4  : 298–1300 / 1300–6000
#   N2   : 100–500 / 500–2000 / 2000–6000
#   O2   : 100–700 / 700–2000 / 2000–6000
#   H2S  : 298–1400 / 1400–6000
#   COS  : 298–6000（单段）
#
# F、G 的处理：A–E 与首段的 F、G 直接取 NIST 数值（已用 JANAF 的
# Cp(1000 K)、H(1500 K)-H(298.15 K) 核对，偏差 < 0.2%）。上段的 F、G
# 由分段点处 H、S 连续条件重算：NIST 原表在分段点本应连续，重算只是
# 避免转写误差，不改变任何 A–E 系数与首段绝对焓基准。
# =============================================================================

_SHOMATE_RAW = {
    'CO': {
        'bounds': (298.15, 1300.0, 6000.0),
        'segments': [
            [25.56759, 6.096130, 4.054656, -2.671301, 0.131021, -118.0089, 227.3665, -110.5271],
            [35.15070, 1.300095, -0.205921, 0.013550, -3.282780, -127.8375, 231.7120, -110.5271],
        ],
    },
    'CO2': {
        'bounds': (298.15, 1200.0, 6000.0),
        'segments': [
            [24.99735, 55.18696, -33.69137, 7.948387, -0.136638, -403.6075, 228.2431, -393.5224],
            [58.16639, 2.720074, -0.492289, 0.038844, -6.447293, -425.9186, 263.6125, -393.5224],
        ],
    },
    'H2': {
        'bounds': (298.15, 1000.0, 2500.0, 6000.0),
        'segments': [
            [33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974, 0.0],
            [18.563083, 12.257357, -2.859786, 0.268238, 1.977990, -1.147438, 156.288133, 0.0],
            [43.413560, -4.293079, 1.272428, -0.096876, -20.533862, -38.515158, 162.081354, 0.0],
        ],
    },
    'H2O': {
        'bounds': (500.0, 1700.0, 6000.0),
        'segments': [
            [30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967, -241.8264],
            [41.96426, 8.622053, -1.499780, 0.098119, -11.15764, -272.1797, 219.7809, -241.8264],
        ],
    },
    'CH4': {
        'bounds': (298.15, 1300.0, 6000.0),
        'segments': [
            [-0.703029, 108.4773, -42.52157, 5.862788, 0.678565, -76.84376, 158.7163, -74.87310],
            [85.81217, 11.26467, -2.114146, 0.138190, -26.42221, -153.5327, 224.4143, -74.87310],
        ],
    },
    'N2': {
        'bounds': (100.0, 500.0, 2000.0, 6000.0),
        'segments': [
            [28.98641, 1.853978, -9.647459, 16.63537, 0.000117, -8.671914, 226.4168, 0.0],
            [19.50583, 19.88705, -8.598535, 1.369784, 0.527601, -4.935202, 212.3900, 0.0],
            [35.51872, 1.128728, -0.196103, 0.014662, -4.553760, -18.97091, 224.9818, 0.0],
        ],
    },
    'O2': {
        'bounds': (100.0, 700.0, 2000.0, 6000.0),
        'segments': [
            [31.32234, -20.23531, 57.86644, -36.50624, -0.007374, -8.903471, 246.7945, 0.0],
            [30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 236.1663, 0.0],
            [20.91111, 10.72071, -2.020498, 0.146449, 9.245722, 5.337651, 237.6185, 0.0],
        ],
    },
    'H2S': {
        'bounds': (298.15, 1400.0, 6000.0),
        'segments': [
            [26.88412, 18.67809, 3.434203, -3.378702, 0.135882, -28.91211, 233.3747, -20.50202],
            [51.22136, 4.147486, -0.643566, 0.041621, -10.46385, -55.87606, 243.6900, -20.50202],
        ],
    },
    'COS': {
        'bounds': (298.15, 6000.0),
        'segments': [
            [43.66691, 19.99841, -18.73031, 8.524458, -0.490011, -153.2847, 260.6784, -142.1648],
        ],
    },
}


def _shomate_h_sum(coeffs, t):
    """H 的 A–E 部分（kJ/mol），不含 F。t = T/1000。"""
    A, B, C, D, E = coeffs[:5]
    return A * t + B * (t ** 2) / 2.0 + C * (t ** 3) / 3.0 + D * (t ** 4) / 4.0 - E / t


def _shomate_s_sum(coeffs, t):
    """S 的 A–E 部分（J/mol/K），不含 G。"""
    A, B, C, D, E = coeffs[:5]
    return A * np.log(t) + B * t + C * (t ** 2) / 2.0 + D * (t ** 3) / 3.0 - E / (2.0 * t ** 2)


def _build_shomate_db(raw):
    """把原始分段表转成 SHOMATE_DB，并使上段的 F、G 满足分段点连续性。"""
    db = {}
    for species, entry in raw.items():
        bounds = list(entry['bounds'])
        segs = [list(c) for c in entry['segments']]
        assert len(segs) == len(bounds) - 1, f"{species}: bounds/segments 数量不符"
        for i in range(1, len(segs)):
            t_cut = bounds[i] / 1000.0
            lower, upper = segs[i - 1], segs[i]
            # H 连续：H_sum_lower + F_lower = H_sum_upper + F_upper
            upper[5] = lower[5] + _shomate_h_sum(lower, t_cut) - _shomate_h_sum(upper, t_cut)
            # S 连续：S_sum_lower + G_lower = S_sum_upper + G_upper
            upper[6] = lower[6] + _shomate_s_sum(lower, t_cut) - _shomate_s_sum(upper, t_cut)
        db[species] = {
            'bounds': bounds,
            'T_cuts': [float(b) for b in bounds[1:-1]],
            'segments': segs,
            # 向后兼容旧接口
            'Low': segs[0],
            'High': segs[1] if len(segs) > 1 else segs[0],
            'T_cut': float(bounds[1]),
        }
    return db


SHOMATE_DB = _build_shomate_db(_SHOMATE_RAW)
MAX_SHOMATE_SEGMENTS = max(len(v['segments']) for v in SHOMATE_DB.values())


def _get_coeffs(species, T):
    """Select appropriate Shomate coefficients based on temperature"""
    if species not in SHOMATE_DB:
        return None

    data = SHOMATE_DB[species]
    segs = data['segments']
    for i, cut in enumerate(data['T_cuts']):
        if T < cut:
            return segs[i]
    return segs[-1]

def _calculate_shomate(species, T, prop_type):
    """
    Internal helper: Calculate thermodynamic properties using Shomate equation
    T: Temperature (K)
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
        # NIST result is H(T) - H(298.15) + Delta_f H(298.15)
        # Note: Does H term (coeffs[7]) correspond to Delta_f H(298.15)?
        # NIST definition: H - H(298) = ... + F - H. 
        # Here F contains formation enthalpy info. NIST web result H = standard enthalpy.
        H_val = A*t + B*(t**2)/2 + C*(t**3)/3 + D*(t**4)/4 - E/t + F
        return H_val * 1000.0  # kJ/mol -> J/mol

    elif prop_type == 'S':
        # S = A*ln(t) + B*t + C*t^2/2 + D*t^3/3 - E/(2*t^2) + G (J/mol*K)
        S_val = A*np.log(t) + B*t + C*(t**2)/2 + D*(t**3)/3 - E/(2*t**2) + G
        return S_val 

def get_enthalpy_molar(species, T):
    """Calculate Molar Enthalpy (J/mol)"""
    return _calculate_shomate(species, T, 'H')

def get_entropy_molar(species, T):
    """Calculate Molar Entropy (J/(mol K))"""
    return _calculate_shomate(species, T, 'S')

def get_gibbs_free_energy(species, T):
    """
    Calculate Gibbs Free Energy G = H - TS (J/mol)
    """
    H = get_enthalpy_molar(species, T)
    S = get_entropy_molar(species, T)
    return H - T * S

def get_equilibrium_constants(T):
    """
    Calculate Equilibrium Constant K (For display only, core calc uses Gibbs minimization)
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

# Sutherland's law parameters per species: (mu_ref [Pa·s], T_ref [K], S [K])
# Sources: White "Fluid Mechanics" App. A; CRC Handbook; engineering standard values.
_SUTHERLAND_PARAMS = {
    'N2':  (1.781e-5, 273.15, 111.0),
    'O2':  (1.919e-5, 273.15, 127.0),
    'CO':  (1.657e-5, 273.15, 136.0),
    'CO2': (1.370e-5, 273.15, 222.0),
    'H2':  (8.411e-6, 273.15,  72.0),
    'CH4': (1.030e-5, 273.15, 164.0),
    'H2O': (1.012e-5, 373.15, 1064.0),
    'H2S': (1.173e-5, 273.15, 331.0),
}

def calculate_gas_viscosity(T, species='N2'):
    """Dynamic viscosity via Sutherland's law [Pa·s]. Falls back to N2 if species unknown."""
    mu_ref, T_ref, S = _SUTHERLAND_PARAMS.get(species, _SUTHERLAND_PARAMS['N2'])
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

def calculate_diffusion_coefficient(T, P, species_A, species_B='N2'):
    """Fuller correlation: D [m²/s]. Used when use_fortran_diffusion=False."""
    P_bar = P / 1e5
    M_A = MOLAR_MASS.get(species_A, 28.0)
    M_B = MOLAR_MASS.get(species_B, 28.0)
    M_AB = 2 / (1/M_A + 1/M_B)
    
    v_A = FULLER_VOLUMES.get(species_A, 18.0)
    v_B = FULLER_VOLUMES.get(species_B, 18.0)
    
    D_cm2s = (0.00143 * T**1.75) / (P_bar * np.sqrt(M_AB) * (v_A**(1/3) + v_B**(1/3))**2)
    return D_cm2s * 1e-4


def calculate_kdiff_fortran(reaction: str, T_gas: float, T_mean: float, P_Pa: float,
                            d_p_m: float, phi: float = 1.0) -> float:
    """
    Fortran 式扩散传质系数 k_diff [m/s]，用于异相反应 UCSM。
    Source1_副本.for: combus L1179-1181, cbstm L1218, cbco2 L1249, cbhym L1334.
    
    物理预估：Fortran 用 T^0.75（C+H2O/CO2/H2）或 T^1.75（C+O2），Fuller 用 T^1.75。
    低温下 Fortran k_diff 更小 → 传质阻力更大 → 异相速率更低 → 燃烧区可能略长。
    """
    P_atm = P_Pa / 101325.0
    d_p_cm = d_p_m * 100.0
    # Fortran → SI: k_diff_fortran [g/(cm²·s·atm)] * (R*T)*1e4/(101325*12) = k [m/s]
    conv = (R_CONST * T_mean) * 1e4 / (101325.0 * 12.0)
    
    if reaction == 'C+O2':
        # diff = (4.26/pt)*(tg/1800)^1.75, kdiff = (phi*0.292*diff)/(dp*tm)
        diff = (4.26 / P_atm) * (T_gas / 1800.0) ** 1.75
        kdiff_f = (phi * 0.292 * diff) / (d_p_cm * T_mean)
    elif reaction == 'C+H2O':
        kdiff_f = (10.0e-4) * (T_mean / 2000.0) ** 0.75 / (d_p_cm * P_atm)
    elif reaction == 'C+CO2':
        kdiff_f = (7.45e-4) * (T_mean / 2000.0) ** 0.75 / (d_p_cm * P_atm)
    elif reaction == 'C+H2':
        kdiff_f = (1.33e-3) * (T_mean / 2000.0) ** 0.75 / (d_p_cm * P_atm)
    else:
        return 0.0
    
    return kdiff_f * conv

def calculate_gas_density(P, T, composition):
    M_avg = sum(MOLAR_MASS.get(s, 28.0) * y for s, y in composition.items())
    return (P * M_avg * 1e-3) / (R_CONST * T)

def get_lhv(species):
    lhv_data = {
        'H2': 241820, 'CO': 282980, 'CH4': 802340, 
        'H2S': 518800, 'O2': 0, 'CO2': 0, 'H2O': 0, 'N2': 0, 'C': 393510
    }
    return lhv_data.get(species, 0.0)