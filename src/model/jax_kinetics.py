"""
均相与异相反应动力学 (统一 9 组分顺序版)。
组分顺序: [O2, CH4, CO, CO2, H2O, H2, N2, H2S, COS]
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import jit, vmap
from .physics import R_CONST
from .jax_physics import calculate_kdiff_fortran_jax, calculate_diffusion_coefficient_jax

CAL2J = 4.184

@jit
def wgs_equilibrium_k_jax(T):
    return jnp.exp(4578.0 / T - 4.33)


# 兼容旧测试/脚本导入名
wgs_equilibrium_k = wgs_equilibrium_k_jax

@jit
def boudouard_equilibrium_k_jax(T):
    return jnp.exp(-20573.0 / T + 20.32)

@jit
def cstm_equilibrium_k_jax(T):
    return jnp.exp(17.644 - 16811.0 / T)

@jit
def heterogeneous_rates_jax(
    gas_moles_9: jnp.ndarray,  # (9,) [mol/s]
    P: float,
    T: float,
    d_p: float,
    surface_area: float,
    X_total: float,
    T_p: float,
    *,
    Re: float = 0.0,
    Sc: float = 1.0,
    porosity: float = 0.75,
    char_combustion_factor: float = 1.0,
    phi_manual: float = None
) -> jnp.ndarray:
    """
    计算四种异相反应速率 (mol/s): [C+O2, C+H2O, C+CO2, C+H2]
    """
    F_total = jnp.maximum(jnp.sum(gas_moles_9), 1e-12)
    
    # Mechanism Factor phi
    p_fac = 2500.0 * jnp.exp(-5.19e4 / (R_CONST * T_p))
    phi = (2*p_fac + 2) / (p_fac + 2)
    if phi_manual is not None: phi = phi_manual
    
    Y = jnp.maximum((1.0 - X_total)**(1.0/3.0), 1e-3)
    
    # 动力学参数 (A, E)
    R_CAL = 1.987
    params_A = jnp.array([8710.0, 247.0, 247.0, 0.12])
    params_E = jnp.array([17967.0, 21060.0, 21060.0, 17921.0]) * R_CAL * CAL2J
    nus = jnp.array([phi, 1.0, 1.0, 1.0])
    
    # 氧化剂索引 (在 ORDER 中的位置): O2:0, H2O:4, CO2:3, H2:5
    ox_indices = jnp.array([0, 4, 3, 5])
    
    def calc_single_rxn(i):
        rxn_idx = i
        sp_idx = ox_indices[i]
        
        P_i = (gas_moles_9[sp_idx] / F_total) * P
        P_CO = (gas_moles_9[2] / F_total) * P
        P_H2 = (gas_moles_9[5] / F_total) * P
        
        # P_eq
        P_eq = jnp.zeros(4).at[1].set((P_H2 * P_CO) / (cstm_equilibrium_k_jax(T_p) * 101325.0 + 1e-12)) \
                          .at[2].set((P_CO**2) / (boudouard_equilibrium_k_jax(T_p) * 101325.0 + 1e-12))[rxn_idx]
        
        P_eff = jnp.maximum(P_i - P_eq, 0.0)
        
        # 传质
        D_i = calculate_diffusion_coefficient_jax(T, P, sp_idx)
        Sh = 2.0 + 0.6 * (Re ** 0.5) * (Sc ** (1.0/3.0))
        k_d = (Sh * D_i) / d_p
        k_ash = k_d * (porosity ** 2.5)
        
        # 化学动力学
        k_s_pressure = params_A[rxn_idx] * jnp.exp(-params_E[rxn_idx] / (R_CONST * T_p))
        k_s = k_s_pressure * (R_CONST * T_p) * 1e4 / (101325.0 * 12.0)
        
        # UCSM assembly
        nu = nus[rxn_idx]
        denom = (1.0/(nu * k_d) + (1.0-Y)/(nu * k_ash * Y) + 1.0/(k_s * Y**2))
        
        rate_kmol = (P_eff / (R_CONST * T) / 1000.0) / denom
        rate_mol = rate_kmol * 1000.0 * surface_area
        
        # Guards
        is_active = jnp.array([True, P_i >= 1e-6, T_p > 850.0, T_p > 1200.0])[rxn_idx]
        rate = jnp.where(is_active & (X_total < 0.999) & (Y >= 1e-4), rate_mol, 0.0)
        
        return jnp.where(rxn_idx == 0, rate * char_combustion_factor, rate)

    return vmap(calc_single_rxn)(jnp.arange(4))

@jit
def homogeneous_rates_jax(
    gas_moles_9: jnp.ndarray, 
    T: float,
    volume_m3: float,
    P: float,
    *,
    inlet_moles_9: jnp.ndarray = None, 
    wgs_catalytic: float = 0.2,
    T_wgs_min: float = 1000.0,
    T_msr_min: float = 1000.0,
    Ts_particle: float = -1.0,
) -> jnp.ndarray:
    """
    均相速率: [CO_Ox, H2_Ox, WGS, CH4_Ox, MSR]

    与 KineticsService.calc_homogeneous_rates 对齐：WGS/MSR 是否启用由 **颗粒温度** 判定
    （Ts_particle > 1000 K）；Arrhenius 仍用气相 T。若 Ts_particle<0 则回退为 T（兼容旧调用）。
    """
    avg_moles = (gas_moles_9 + inlet_moles_9) / 2.0 if inlet_moles_9 is not None else gas_moles_9
    F_total = jnp.maximum(jnp.sum(avg_moles), 1e-12)
    C = (avg_moles / F_total * P / (R_CONST * T)) / 1000.0 # kmol/m3
    vol_scale = volume_m3 * 1000.0
    
    A_homo = jnp.array([2.23e12, 1.08e13, 2.877e5, 1.6e10, 312.0])
    E_homo = jnp.array([1.25e8, 8.37e7, 27760.0*CAL2J, 1.256e8, 30000.0*CAL2J])
    ks = A_homo * jnp.exp(-E_homo / (R_CONST * T))
    
    r_co = ks[0] * C[2] * C[0]
    r_h2 = ks[1] * C[5] * C[0]
    K_eq = wgs_equilibrium_k_jax(T)
    r_wgs = ks[2] * wgs_catalytic * (C[2] * C[4] - C[3] * C[5] / (K_eq + 1e-12))
    T_gate = jnp.where(Ts_particle > 0.0, Ts_particle, T)
    r_wgs = jnp.where(T_gate > T_wgs_min, r_wgs, 0.0)
    r_ch4 = ks[3] * C[1] * C[0]
    r_msr = ks[4] * C[1]
    r_msr = jnp.where(T_gate > T_msr_min, r_msr, 0.0)
    
    return jnp.array([r_co, r_h2, r_wgs, r_ch4, r_msr]) * vol_scale
