import numpy as np
import jax.numpy as jnp

from model.state import StateVector
from model.material import MaterialService, SPECIES_NAMES
from model.physics import calculate_enthalpy
from model.kinetics_service import KineticsService
from model.jax_kinetics import homogeneous_rates_outlet_jax


def _sample_state() -> StateVector:
    # 典型高温合成气状态（避免低浓度极限）
    gas = np.array([12.0, 8.0, 420.0, 110.0, 1.2, 240.0, 5.0, 160.0], dtype=float)
    return StateVector(
        gas_moles=gas,
        solid_mass=0.85,
        carbon_fraction=0.78,
        T=1650.0,
        P=4.0e6,
        z=0.5,
    )


def test_material_enthalpy_consistency():
    state = _sample_state()
    coal_props = {"cp_char": 1300.0, "Hf_coal": -3.0e6, "Cd": 60.0}

    h_gas_ref = MaterialService.get_gas_enthalpy(state)
    h_gas_manual = 0.0
    for i, sp in enumerate(SPECIES_NAMES):
        if state.gas_moles[i] > 0.0:
            h_gas_manual += state.gas_moles[i] * calculate_enthalpy(sp, state.T)

    assert np.isfinite(h_gas_ref)
    assert np.isfinite(h_gas_manual)
    assert np.isclose(h_gas_ref, h_gas_manual, rtol=1e-10, atol=1e-6)

    h_total = MaterialService.get_total_enthalpy(state, coal_props)
    h_solid = MaterialService.get_solid_enthalpy(state, coal_props)
    assert np.isclose(h_total, h_gas_ref + h_solid, rtol=1e-12, atol=1e-8)


def test_homogeneous_rates_jax_vs_numpy_fallback():
    state = _sample_state()
    volume = 0.35
    kin = KineticsService()

    rates_np = kin.calc_homogeneous_rates(
        state,
        volume,
        inlet_state=None,
        gas_src=None,
        Ts_particle=None,
        wgs_rat_factor=False,
        msr_tmin_k=1000.0,
        wgs_catalytic_factor=0.2,
        wgs_k_factor=1.0,
    )

    c_arr = np.array([state.get_concentration(i) for i in range(8)], dtype=float)
    rates_jax = homogeneous_rates_outlet_jax(
        jnp.asarray(c_arr),
        jnp.asarray(state.T),
        jnp.asarray(volume),
        wgs_catalytic=0.2,
        T_wgs_min=1000.0,
        T_msr_min=1000.0,
    )

    keys = ["CO_Ox", "H2_Ox", "WGS", "CH4_Ox", "MSR"]
    for k in keys:
        v_np = float(rates_np[k])
        v_j = float(rates_jax[k])
        # WGS 在高温下可能数值较小，给一个温和的绝对误差
        assert np.isclose(v_np, v_j, rtol=2e-6, atol=1e-2), f"{k} mismatch: np={v_np}, jax={v_j}"

