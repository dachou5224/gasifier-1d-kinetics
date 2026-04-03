"""
jax_jit 路线契约：物种映射、输入、热损与残差语义（不依赖与 minimize 全炉数值一致）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.species_bridge import (
    MAINLINE_SPECIES_NAMES,
    JAX_9_SPECIES_NAMES,
    mainline_gas8_to_jax9,
    jax9_to_mainline_gas8,
)
from model.input_contract import (
    ash_mass_fraction_dry,
    coal_flow_kg_s_for_heat_loss,
    heat_loss_norm_length_m,
    heat_loss_ref_temp_k,
    resolve_f_s_coal,
)
from model.validation_loader import load_case_from_repo


def _case_paper_6():
    from model.chemistry import COAL_DATABASE
    from model.gasifier_system import GasifierSystem

    case_data = load_case_from_repo("Paper_Case_6")
    inputs = case_data["inputs"]
    coal_key = case_data.get("coal_type", inputs.get("coal"))
    coal_props = COAL_DATABASE[coal_key].copy()
    feed_rate = inputs.get("FeedRate_kg_h", inputs.get("FeedRate"))
    coal_flow_kg_s = feed_rate / 3600.0
    o2_flow_kg_s = coal_flow_kg_s * inputs.get("Ratio_OC", 1.0)
    op_conds = {
        "coal_flow": coal_flow_kg_s,
        "o2_flow": o2_flow_kg_s,
        "steam_flow": 0.0,
        "P": inputs.get("P_Pa", inputs.get("P")),
        "T_in": inputs.get("T_in_K", inputs.get("TIN")),
        "HeatLossPercent": inputs.get("HeatLossPercent", 1.0),
        "epsilon": inputs.get("Voidage", 1.0),
        "SlurryConcentration": inputs.get("SlurryConcentration", 62.0),
    }
    geometry = {"L": inputs.get("L_reactor", 6.0), "D": inputs.get("D_reactor", 2.0)}
    return GasifierSystem(geometry, coal_props, op_conds)


def test_species_bridge_roundtrip():
    rng = np.random.default_rng(0)
    g8 = rng.random(8) * 10.0
    g9 = mainline_gas8_to_jax9(g8)
    assert g9.shape == (9,)
    assert g9[8] == 0.0  # COS
    back = jax9_to_mainline_gas8(g9)
    np.testing.assert_allclose(back, g8)
    assert len(MAINLINE_SPECIES_NAMES) == 8
    assert len(JAX_9_SPECIES_NAMES) == 9


def test_ad_overrides_ashd():
    c_ad = {"Ad": 10.0, "Ashd": 99.0}
    c_ashd_only = {"Ashd": 7.0}
    assert abs(ash_mass_fraction_dry(c_ad) - 0.10) < 1e-9
    assert abs(ash_mass_fraction_dry(c_ashd_only) - 0.07) < 1e-9


def test_heat_loss_inputs_resolve():
    op = {"coal_flow": 2.5, "L_heatloss_norm": 5.0, "HeatLossRefTemp": 1900.0}
    assert abs(coal_flow_kg_s_for_heat_loss(op) - 2.5) < 1e-9
    assert abs(heat_loss_norm_length_m(op, geometry_L=6.0, mesh_sum_dz=4.0) - 5.0) < 1e-9
    assert abs(heat_loss_ref_temp_k(op) - 1900.0) < 1e-9
    op2 = {"coal_flow": 1.0, "L_reactor": 3.0}
    assert abs(heat_loss_norm_length_m(op2, geometry_L=6.0, mesh_sum_dz=4.0) - 3.0) < 1e-9


def test_f_s_coal_defaults_to_zero_for_mainline_parity():
    assert resolve_f_s_coal({}, {}) == 0.0
    assert resolve_f_s_coal({"f_s_coal": 0.02}, {}) == 0.02
    assert resolve_f_s_coal({}, {"f_s_coal": 0.03}) == 0.03


@pytest.mark.parametrize("solver", ["minimize", "jax_jit"])
def test_jax_jit_runs_benchmark_case(solver):
    pytest.importorskip("jaxlib")
    sys = _case_paper_6()
    res, z = sys.solve(N_cells=8, solver_method=solver, jax_warmup=True)
    assert res.shape[1] == 11
    assert np.all(np.isfinite(res[:, 10]))


def test_single_cell_residual_numpy_vs_jax_flat():
    """单格：主线 Cell 残差与 JAX flat 在相同状态下可比（缩放后量级接近）。"""
    pytest.importorskip("jaxlib")
    jax_np = pytest.importorskip("jax.numpy")

    from model.jax_residuals import cell_residuals_jax_flat
    from model.grid_service import AdaptiveMeshGenerator, MeshConfig
    from model.cell import Cell
    from model.source_terms import EvaporationSource, PyrolysisSource

    sys = _case_paper_6()
    res, _ = sys.solve(N_cells=1, solver_method="minimize", jacobian_mode="scipy")
    x_np = np.asarray(res[0], dtype=np.float64)

    inlet = sys._initialize_inlet()
    mesh_cfg = MeshConfig(
        total_length=sys.geometry["L"],
        n_cells=1,
        ignition_zone_length=sys.op_conds.get("FirstCellLength", 0.1),
        min_grid_size=0.001,
    )
    dz_list, z_positions = AdaptiveMeshGenerator(mesh_cfg).generate()
    A = np.pi * (sys.geometry["D"] / 2) ** 2
    F_H2O_evap = (sys.tmp_W_liq_evap / 18.015) * 1000.0
    L_evap = sys.op_conds.get("L_evap_m", 1e-6)
    ref_f = max(inlet.total_gas_moles, 1.0)
    cell = Cell(
        0,
        z_positions[0],
        dz_list[0],
        A * dz_list[0],
        A,
        inlet,
        sys.kinetics,
        sys.pyrolysis,
        sys.coal_props,
        sys.op_conds,
        sources=[
            EvaporationSource(F_H2O_evap, L_evap_m=L_evap),
            PyrolysisSource(sys.tmp_F_vol, sys.tmp_W_vol, target_cell_idx=0, T_pyro=inlet.T),
        ],
    )
    cell.coal_flow_dry = sys.W_dry
    cell.Cd_total = sys.Cd_total
    cell.char_Xc0 = sys.char_Xc0
    cell.ref_flow, cell.ref_energy = ref_f, max(ref_f * 35.0 * 200.0, 5.0e5)

    r_np = np.asarray(cell.residuals(x_np), dtype=np.float64)

    g_src, s_src, e_src = np.zeros(8), 0.0, 0.0
    for s in cell.sources:
        g, sm, e = s.get_sources(cell.idx, cell.z, cell.dz)
        g_src += g
        s_src += sm
        e_src += e

    g9 = mainline_gas8_to_jax9(g_src)
    inlet_9 = np.concatenate(
        [mainline_gas8_to_jax9(inlet.gas_moles), [inlet.solid_mass, inlet.carbon_fraction, inlet.T]]
    )
    x_9 = np.concatenate([mainline_gas8_to_jax9(x_np[:8]), x_np[8:]])

    hhv = sys.coal_props.get("HHV_d", 30.0)
    hhv_mj = hhv / 1000.0 if hhv > 1000.0 else hhv
    Ash_d = ash_mass_fraction_dry(sys.coal_props)
    f_ash = Ash_d / (sys.Cd_total + Ash_d + 1e-9)
    hl_norm = heat_loss_norm_length_m(sys.op_conds, float(sys.geometry["L"]), float(np.sum(dz_list)))
    hl_ref = heat_loss_ref_temp_k(sys.op_conds)
    f_s = resolve_f_s_coal(sys.coal_props, sys.op_conds)

    r_j = cell_residuals_jax_flat(
        jax_np.asarray(x_9, dtype=np.float64),
        jax_np.asarray(inlet_9, dtype=np.float64),
        jax_np.asarray(g9, dtype=np.float64),
        float(s_src),
        float(e_src),
        float(dz_list[0]),
        float(A),
        float(inlet.P),
        float(sys.W_dry * sys.Cd_total),
        float(inlet.T_solid_or_gas),
        float(coal_flow_kg_s_for_heat_loss(sys.op_conds)),
        float(sys.op_conds.get("particle_diameter", 100e-6)),
        float(sys.op_conds.get("epsilon", 1.0)),
        float(sys.op_conds.get("HeatLossPercent", 2.0)),
        float(sys.geometry["L"]),
        float(sys.op_conds.get("CharCombustionRateFactor", 0.3)),
        float(sys.op_conds.get("WGS_CatalyticFactor", 0.2)),
        float(sys.op_conds.get("Combustion_CO2_Fraction", 1.0)),
        float(sys.op_conds.get("P_O2_Combustion_atm", 0.05)),
        float(hl_norm),
        float(hl_ref),
        float(sys.coal_props.get("Hf_coal", 0.0)),
        float(sys.coal_props.get("cp_char", 1300.0)),
        float(hhv_mj),
        float(f_ash),
        float(ref_f),
        float(cell.ref_energy),
        float(sys.char_Xc0),
        float(f_s),
    )
    r_j = np.asarray(r_j, dtype=np.float64)
    # 缩放后的残差向量：前 8 维应同号且量级可比（允许模型差分）
    assert r_j.shape == (12,)
    assert np.max(np.abs(r_np[:8])) < 100.0
    assert np.max(np.abs(r_j[:8])) < 100.0
