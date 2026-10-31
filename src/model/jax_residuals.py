"""
JAX 版本的残差方程实现 (统一 9 组分 + 能量/硫平衡最终修正版)。

硫释放 ``S_release`` 由 ``f_s_coal`` 控制；默认在 ``input_contract.resolve_f_s_coal`` 中为 0，
与当前主线 ``Cell``（无显式表面硫释放项）短期等价。显式建模需设置 ``op_conds['f_s_coal']`` 等。

颗粒温度：``_calc_particle_temperature_jax`` 为指数型瞬态模型；主线燃烧区可选用 RK-Gill（见 ``Cell``），
二者不完全一致，以主线为参考时请关注该差异。
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from .physics import R_CONST
from .jax_physics import (
    calculate_gas_viscosity_jax, 
    calculate_diffusion_coefficient_jax,
    calculate_enthalpy_mixture_jax
)
from .jax_kinetics import (
    heterogeneous_rates_jax,
    homogeneous_rates_jax
)

# Constants
PARTICLE_DENSITY = 1400.0
MIN_SLIP_VELOCITY = 0.1
STEFAN_BOLTZMANN = 5.67e-8
EF_PARTICLE_TRANSIENT = 0.9
CONDUT_COEFF = 3.22168e-4
TS_TRANSIENT_NC = 30

@jit
def _calc_physics_props_jax(x_9, inlet_9, C_fed_total, d_p0, epsilon, P, dz, A, f_ash,
                            particle_model_fortran=0.0):
    """与 Cell._calc_physics_props 同一套颗粒尺寸模型（P1.5-B）。

    particle_model_fortran=0 → shrinking_legacy；1 → fortran_constant
    （d_p 恒定、密度随剩余固体分数下降）。
    """
    Ws, Xc, Tg = x_9[9], x_9[10], x_9[11]
    X_total = jnp.clip(1.0 - (Ws * Xc / (C_fed_total + 1e-12)), 0.0, 1.0)
    solids_fraction = jnp.maximum(f_ash + (1.0 - f_ash) * (1.0 - X_total), 1e-9)
    is_fortran = particle_model_fortran > 0.5
    d_p = jnp.where(is_fortran, d_p0, d_p0 * solids_fraction ** (1.0 / 3.0))
    d_p = jnp.maximum(d_p, 1e-6)
    rho_p = jnp.where(is_fortran, PARTICLE_DENSITY * solids_fraction, PARTICLE_DENSITY)
    rho_p = jnp.maximum(rho_p, 1e-6)
    F_total = jnp.maximum(jnp.sum(x_9[:9]), 1e-9)
    v_g = (F_total * R_CONST * Tg) / (P * A * epsilon + 1e-9)
    tau = dz / jnp.maximum(v_g, MIN_SLIP_VELOCITY)
    S_total = (6.0 * Ws * tau) / (rho_p * d_p + 1e-9)
    molar_masses = jnp.array([31.998, 16.04, 28.01, 44.01, 18.015, 2.016, 28.013, 34.08, 60.07])
    M_avg = jnp.sum(x_9[:9] * molar_masses) / F_total
    rho_g = (P * M_avg * 1e-3) / (R_CONST * Tg)
    mu_g = calculate_gas_viscosity_jax(Tg)
    Re = (rho_g * MIN_SLIP_VELOCITY * d_p) / (mu_g + 1e-9)
    D_ref = calculate_diffusion_coefficient_jax(Tg, P, 0)
    Sc = mu_g / (jnp.maximum(rho_g, 1e-3) * D_ref + 1e-12)
    return X_total, d_p, S_total, Re, Sc, v_g, tau, rho_p

@partial(jit, static_argnums=(9,))
def _calc_particle_temperature_jax(Tg, Ts_in, tau, d_p, dens, cps, ef, sigma, condut_coeff, nc=30):
    deltim = tau / float(nc)
    def body_fn(ts, _):
        condut = condut_coeff * ((Tg + ts) ** 0.75)
        rad_term = ef * sigma * 4.0 * (Tg ** 3)
        ct = -(3.0 / (dens * cps * (d_p/2.0) + 1e-12)) * (condut/(d_p/2.0) + rad_term) * deltim
        delts = (Tg - (Tg - ts) * jnp.exp(jnp.clip(ct, -25.0, 25.0))) - ts
        return ts + delts, ts + delts/2.0
    Ts_out, Ts_history = lax.scan(body_fn, Ts_in, None, length=nc)
    return jnp.mean(Ts_history), Ts_out

@jit
def cell_residuals_jax_flat(
    x_9, inlet_9, g_src_9, s_src, e_src,
    dz, A, P, C_fed_total, Ts_in, coal_flow_kg_s, d_p0, epsilon,
    HeatLossPct, L_reactor, CharCombFac, WGS_CatFac,
    Comb_CO2_Frac, P_O2_Comb_atm, L_heatloss_norm,
    HeatLossRefTemp,
    Hf_coal, cp_char, HHV_mj, f_ash,
    ref_flow, ref_energy, char_Xc0_global,
    f_s_coal,
    particle_model_fortran=0.0,
):
    Tg = x_9[11]
    X_total, d_p, S_tot, Re, Sc, v_g, tau, rho_p = _calc_physics_props_jax(
        x_9, inlet_9, C_fed_total, d_p0, epsilon, P, dz, A, f_ash, particle_model_fortran
    )
    Ts_avg, Ts_out = _calc_particle_temperature_jax(Tg, Ts_in, tau, d_p, rho_p, cp_char, EF_PARTICLE_TRANSIENT, STEFAN_BOLTZMANN, CONDUT_COEFF, nc=TS_TRANSIENT_NC)
    
    r_het = heterogeneous_rates_jax(x_9[:9], P, Tg, d_p, S_tot, X_total, Ts_avg, Re=Re, Sc=Sc, char_combustion_factor=CharCombFac)
    r_homo = homogeneous_rates_jax(
        x_9[:9],
        Tg,
        dz * A,
        P,
        inlet_moles_9=inlet_9[:9] + g_src_9,
        wgs_catalytic=WGS_CatFac,
        Ts_particle=Ts_avg,
    )
    
    avail = jnp.maximum(inlet_9[:9] + g_src_9, 0.0)
    pO2_Pa = P * (avail[0] / jnp.maximum(jnp.sum(avail), 1e-9))
    is_comb = pO2_Pa > (P_O2_Comb_atm * 101325.0)
    
    phi_fac = 2500.0 * jnp.exp(-5.19e4 / (R_CONST * Ts_avg))
    phi = (2*phi_fac + 2) / (phi_fac + 2)
    
    f = Comb_CO2_Frac
    r_CH4_ox = jnp.minimum(avail[1]*0.99, avail[0]/(1.5+0.5*f))
    rem_O2 = jnp.maximum(avail[0] - r_CH4_ox*(1.5+0.5*f), 0.0)
    r_CO_ox = jnp.minimum(avail[2]*0.99, rem_O2/(0.5*f+1e-9))
    rem_O2 = jnp.maximum(rem_O2 - r_CO_ox*0.5*f, 0.0)
    r_H2_ox = jnp.minimum(avail[5]*0.99, rem_O2/0.5)
    # 与主线 Cell._calc_rates 保持一致：
    # 瞬时挥发分燃烧在“可燃气是否能吃到 O2”这一层使用 f_co2 相关需氧量，
    # 但在给 char 氧化分配“剩余 O2”时，主线固定按完全氧化的 O2 账本扣减：
    #   O2_after_vol = O2 - (2*CH4_Ox + 0.5*CO_Ox + 0.5*H2_Ox)
    # 若继续沿用上一段的 reduced demand，会系统性高估 char C+O2，导致 O2/CO/Ws/Xc 偏差。
    rem_O2_for_char = jnp.maximum(avail[0] - (2.0 * r_CH4_ox + 0.5 * r_CO_ox + 0.5 * r_H2_ox), 0.0)
    
    r_ho = jnp.where(is_comb, jnp.array([r_CO_ox, r_H2_ox, r_homo[2], r_CH4_ox, r_homo[4]]), jnp.array([0.0, 0.0, r_homo[2], 0.0, r_homo[4]]))
    rCOmb = jnp.minimum(r_het[0], rem_O2_for_char * phi)
    
    r1, r2, r3, r5, r6 = r_ho[0], r_ho[1], r_ho[2], r_ho[3], r_ho[4]
    rH2Og = jnp.minimum(r_het[1], avail[4] * 0.99)
    rCO2g = jnp.minimum(r_het[2], avail[3] * 0.99)
    # 燃烧区 C+CO2（P1.5-B B4）：Fortran 的 cbco2 在燃烧区照常调用（源程序 L457/L485，
    # 只按 ts<=850 K、pco2<=0、wl>=99.9 判据屏蔽），poxyin>0.05 只用于选择热损闭合
    # （L348-352）。因此这里不再置零，与 NumPy Cell 一致。
    # 与 Cell.limit_h2_consumers_after_oxidation 相同：剩余氢气从 avail[5] - r_H2_ox 起，
    # 先扣 C+H2，再只截断负向净 WGS。不新增 RWGS，不改 S_release。
    rem_h2 = jnp.maximum(avail[5] - r2, 0.0)
    rH2g = jnp.minimum(r_het[3], rem_h2 * 0.5 * 0.99)
    rem_h2 = jnp.maximum(rem_h2 - 2.0 * rH2g, 0.0)
    r3 = jnp.where(r3 < 0.0, jnp.maximum(r3, -rem_h2 * 0.99), r3)

    # 对齐 Cell._calc_rates：MSR 受“剩余 CH4 与剩余 H2O”双约束
    # rem_CH4 = max(avail['CH4'] - r_homo['CH4_Ox'], 0)
    # rem_H2O = max(avail['H2O']  - r_het['C+H2O'], 0)
    rem_CH4 = jnp.maximum(avail[1] - r5, 0.0)
    rem_H2O = jnp.maximum(avail[4] - rH2Og, 0.0)
    r6_cap = jnp.minimum(rem_CH4 * 0.99, rem_H2O * 0.99)
    r6 = jnp.minimum(r6, r6_cap)
    
    W_surf = (rCOmb + rH2Og + rCO2g + rH2g) * 0.012011
    # 硫只释放一次（P1.5-B A2）：PyrolysisService 已把煤中全部硫作为 H2S 写入气相源项，
    # 这里的表面释放项是重复释放，已删除。f_s_coal 由 input_contract.resolve_f_s_coal
    # 在非零时直接报错，避免静默双计。
    dO2 = -(rCOmb/phi + 0.5*f*r1 + 0.5*r2 + (1.5+0.5*f)*r5)
    dCH4 = rH2g - (r5 + r6)
    dCO = (2.0-2.0/phi)*rCOmb + rH2Og + 2*rCO2g + r6 + (1.0-f)*r5 - (f*r1 + r3)
    dCO2 = (2.0/phi-1.0)*rCOmb + f*r1 + r3 + f*r5 - rCO2g
    dH2O = r2 + 2*r5 - (rH2Og + r3 + r6)
    dH2 = rH2Og + r3 + 3*r6 - (2*rH2g + r2)

    deltas = jnp.array([dO2, dCH4, dCO, dCO2, dH2O, dH2, 0.0, 0.0, 0.0])
    res_gas = x_9[:9] - (inlet_9[:9] + g_src_9 + deltas)
    
    res_Ws = x_9[9] - (inlet_9[9] + s_src - W_surf)
    C_pyro = (g_src_9[1] + g_src_9[2] + g_src_9[3]) * 0.012011
    res_Xc = (x_9[9] * x_9[10]) - (inlet_9[9] * inlet_9[10] - C_pyro - W_surf)
    
    H_g_in, H_g_out = calculate_enthalpy_mixture_jax(inlet_9[:9], inlet_9[11]), calculate_enthalpy_mixture_jax(x_9[:9], Tg)
    H_s_in, H_s_out = inlet_9[9]*(cp_char*(Ts_in-298.15)+Hf_coal), x_9[9]*(cp_char*(Ts_out-298.15)+Hf_coal)
    # Q_loss：与 Cell._calc_energy_balance 一致 — 湿煤流量×HHV、dz/L_total、T^4 权重（相对 HeatLossRefTemp）
    t_weight = jnp.clip((Tg / (HeatLossRefTemp + 1e-9)) ** 4.0, 0.1, 5.0)
    Q_loss = (
        (HeatLossPct / 100.0)
        * (coal_flow_kg_s * HHV_mj * 1.0e6)
        * (dz / (L_heatloss_norm + 1e-9))
        * t_weight
    )
    res_E = (H_g_out + H_s_out) - (H_g_in + H_s_in + e_src - Q_loss)

    # 与 Cell.residuals 一致：多数气相用 ref_flow；N2（索引 6）用 ref_N2；固相用 max(Ws_in,1e-3)；
    # Xc 残差除以 max(char_Xc0, 0.1)（无量纲），勿乘入口质量。
    ref_N2 = jnp.maximum(jnp.abs(inlet_9[6] + g_src_9[6]), 1e-3)
    scales = jnp.full((9,), ref_flow).at[6].set(ref_N2)
    ref_solid = jnp.maximum(inlet_9[9], 1e-3)
    res_gas_sc = res_gas / scales
    res_Ws_sc = res_Ws / ref_solid
    res_Xc_sc = res_Xc / jnp.maximum(char_Xc0_global, 0.1)
    res_E_sc = res_E / ref_energy

    return jnp.concatenate([res_gas_sc, jnp.array([res_Ws_sc, res_Xc_sc, res_E_sc])])
