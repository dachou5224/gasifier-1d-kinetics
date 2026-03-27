"""
均相反应等核心代数的 JAX 版本（可 JIT），供后续 jax_cell 纯残差迁移。

当前 KineticsService 仍以 NumPy 为主；此处提供与 kinetics.py / kinetics_service 对齐的 jnp 基元。
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import jit

from .physics import R_CONST

CAL2J = 4.184


@jit
def wgs_equilibrium_k(T):
    """WGS 平衡常数，与 kinetics.calculate_wgs_equilibrium 一致。"""
    return jnp.exp(4578.0 / T - 4.33)


@jit
def arrhenius_rate(A: float, E_J_mol: float, T: jnp.ndarray) -> jnp.ndarray:
    return A * jnp.exp(-E_J_mol / (R_CONST * T))


@jit
def homogeneous_rates_outlet_jax(
    C_kmol_m3: jnp.ndarray,
    T: jnp.ndarray,
    volume_m3: jnp.ndarray,
    *,
    A_co_ox: float = 2.23e12,
    E_co_ox: float = 1.25e8,
    A_h2_ox: float = 1.08e13,
    E_h2_ox: float = 8.37e7,
    A_wgs: float = 2.877e5,
    E_wgs: float = 27760.0 * CAL2J,
    A_ch4_ox: float = 1.6e10,
    E_ch4_ox: float = 1.256e8,
    A_msr: float = 312.0,
    E_msr: float = 30000.0 * CAL2J,
    wgs_catalytic: float = 0.2,
    T_wgs_min: float = 1000.0,
    T_msr_min: float = 1000.0,
) -> dict:
    """
    出口浓度近似下的均相速率 (mol/s)，与 KineticsService.calc_homogeneous_rates 的 fallback 分支结构一致。

    C_kmol_m3: 长度 8，顺序与 SPECIES_NAMES 一致。
    """
    # 二阶反应 r = k * Ca * Cb [kmol/m3/s] → mol/s = r * 1000 * V
    vol_scale = volume_m3 * 1000.0
    k_co = A_co_ox * jnp.exp(-E_co_ox / (R_CONST * T))
    r_co = k_co * C_kmol_m3[2] * C_kmol_m3[0]
    k_h2 = A_h2_ox * jnp.exp(-E_h2_ox / (R_CONST * T))
    r_h2 = k_h2 * C_kmol_m3[5] * C_kmol_m3[0]
    k_ch4 = A_ch4_ox * jnp.exp(-E_ch4_ox / (R_CONST * T))
    r_ch4 = k_ch4 * C_kmol_m3[1] * C_kmol_m3[0]
    K_eq = wgs_equilibrium_k(T)
    k_wgs = A_wgs * jnp.exp(-E_wgs / (R_CONST * T))
    k_eff = k_wgs * wgs_catalytic
    r_wgs_net = jnp.where(
        T > T_wgs_min,
        k_eff * (C_kmol_m3[2] * C_kmol_m3[7] - C_kmol_m3[3] * C_kmol_m3[5] / (K_eq + 1e-12)),
        0.0,
    )
    k_msr = A_msr * jnp.exp(-E_msr / (R_CONST * T))
    r_msr = jnp.where(T > T_msr_min, k_msr * C_kmol_m3[1], 0.0)
    return {
        'CO_Ox': r_co * vol_scale,
        'H2_Ox': r_h2 * vol_scale,
        'WGS': r_wgs_net * vol_scale,
        'RWGS': jnp.array(0.0),
        'CH4_Ox': r_ch4 * vol_scale,
        'MSR': r_msr * vol_scale,
    }
