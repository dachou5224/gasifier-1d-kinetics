"""
JAX 版本的物理性质与热力学计算 (统一 9 组分顺序版)。
组分顺序: [O2, CH4, CO, CO2, H2O, H2, N2, H2S, COS]
"""
import jax.numpy as jnp
from jax import jit
from functools import partial
from .physics import SHOMATE_DB, R_CONST, FULLER_VOLUMES, MOLAR_MASS

# 唯一的、全局统一的顺序
SPECIES_ORDER = ['O2', 'CH4', 'CO', 'CO2', 'H2O', 'H2', 'N2', 'H2S', 'COS']

def _get_shomate_coeffs_jax():
    coeffs_all = []
    t_cuts = []
    for sp in SPECIES_ORDER:
        data = SHOMATE_DB[sp]
        coeffs_all.append([data['Low'], data['High']])
        t_cuts.append(data['T_cut'])
    return jnp.array(coeffs_all), jnp.array(t_cuts)

COEFFS_MATRIX, T_CUTS_MATRIX = _get_shomate_coeffs_jax()

@partial(jit, static_argnums=(2,))
def calculate_shomate_jax(species_indices: jnp.ndarray, T: jnp.ndarray, prop_type: str = 'H'):
    t = T / 1000.0
    t_cuts = T_CUTS_MATRIX[species_indices]
    range_idx = jnp.where(T < t_cuts, 0, 1)
    coeffs = COEFFS_MATRIX[species_indices, range_idx]
    
    A, B, C, D, E, F, G = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3], coeffs[:, 4], coeffs[:, 5], coeffs[:, 6]
    
    if prop_type == 'Cp':
        return A + B*t + C*t**2 + D*t**3 + E/(t**2)
    elif prop_type == 'H':
        return (A*t + B*(t**2)/2 + C*(t**3)/3 + D*(t**4)/4 - E/t + F) * 1000.0
    return jnp.zeros_like(T)

@jit
def calculate_enthalpy_mixture_jax(moles_9: jnp.ndarray, T: float):
    """moles_9: (9,) [mol/s]"""
    indices = jnp.arange(9)
    h_molar = calculate_shomate_jax(indices, jnp.full((9,), T), 'H')
    return jnp.sum(moles_9 * h_molar)

@jit
def calculate_kdiff_fortran_jax(reaction_idx: int, T_gas: float, T_mean: float, P_Pa: float,
                                d_p_m: float, phi: float = 1.0) -> float:
    P_atm = P_Pa / 101325.0; d_p_cm = d_p_m * 100.0; conv = (R_CONST * T_mean) * 1e4 / (101325.0 * 12.0)
    diff_o2 = (4.26 / P_atm) * (T_gas / 1800.0) ** 1.75
    k_o2 = (phi * 0.292 * diff_o2) / (d_p_cm * T_mean)
    k_h2o = (10.0e-4) * (T_mean / 2000.0) ** 0.75 / (d_p_cm * P_atm)
    k_co2 = (7.45e-4) * (T_mean / 2000.0) ** 0.75 / (d_p_cm * P_atm)
    k_h2 = (1.33e-3) * (T_mean / 2000.0) ** 0.75 / (d_p_cm * P_atm)
    return jnp.array([k_o2, k_h2o, k_co2, k_h2])[reaction_idx] * conv

@jit
def calculate_gas_viscosity_jax(T):
    mu_ref, T_ref, S = 1.781e-5, 273.15, 111.0
    return mu_ref * (T/T_ref)**1.5 * (T_ref + S) / (T + S)

@jit
def calculate_diffusion_coefficient_jax(T, P, species_idx_A, species_idx_B=6):
    P_bar = P / 1e5
    molar_masses = jnp.array([MOLAR_MASS[sp] for sp in SPECIES_ORDER])
    fuller_vols_dict = FULLER_VOLUMES.copy()
    if 'H2S' not in fuller_vols_dict: fuller_vols_dict['H2S'] = 20.96
    if 'COS' not in fuller_vols_dict: fuller_vols_dict['COS'] = 35.3
    fuller_vols = jnp.array([fuller_vols_dict[sp] for sp in SPECIES_ORDER])
    
    M_A, M_B = molar_masses[species_idx_A], molar_masses[species_idx_B]
    M_AB = 2 / (1/M_A + 1/M_B)
    v_A, v_B = fuller_vols[species_idx_A], fuller_vols[species_idx_B]
    
    D_cm2s = (0.00143 * T**1.75) / (P_bar * jnp.sqrt(M_AB) * (v_A**(1/3) + v_B**(1/3))**2)
    return D_cm2s * 1e-4
