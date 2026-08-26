from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from model.gasifier_system import GasifierSystem
from model.model_case_adapter import (
    TRANSPORTED_COAL_MOISTURE_PERCENT,
    build_gasifier_inputs_from_model_case,
    normal_m3_h_to_kg_s,
)


def _minimal_system(**op_overrides):
    geometry = {"L": 1.0, "D": 0.5}
    coal_props = {
        "Cd": 70.0,
        "Hd": 4.0,
        "Od": 10.0,
        "Nd": 1.0,
        "Sd": 0.5,
        "Ad": 10.0,
        "Vd": 30.0,
        "FCd": 60.0,
        "Mt": 0.0,
        "HHV_d": 28000.0,
    }
    op_conds = {"coal_flow": 1.0, "o2_flow": 2.0, "steam_flow": 0.0, "P": 3.8e6, "T_in": 433.15}
    op_conds.update(op_overrides)
    return GasifierSystem(geometry, coal_props, op_conds)


def _preview_row():
    return {
        "case_id_value": "sy3_trial_0928_feed_product__modelcase_001",
        "segment_id": "sy3_trial_0928_feed_product__candidate_001",
        "model_case_ready": True,
        "coal_mass_flow_value": 17.782341312164125,
        "coal_weighfeeder_ar_flow_value": 76.00193366234221,
        "monitor_pulverized_coal_line_1_kg_s_value": 5.919811896051892,
        "monitor_pulverized_coal_line_2_kg_s_value": 5.9546222874754555,
        "monitor_pulverized_coal_line_3_kg_s_value": 5.846806251199157,
        "monitor_pulverized_coal_line_4_kg_s_value": 5.72493268769986,
        "o2_volumetric_flow_value": 37560.362208064514,
        "co2_carrier_volumetric_flow_value": 9655.458800645161,
        "gasifier_pressure_value": 3.791407616875,
        "o2_inlet_temperature_c_value": 160.0,
        "co2_carrier_inlet_temperature_c_value": 80.0,
        "reactor_bed_diameter_m_value": 3.8,
        "reactor_bed_height_m_value": 10.3,
        "coal_carbon_mass_fraction_adb_value": 0.7301,
        "coal_hydrogen_mass_fraction_adb_value": 0.04324,
        "coal_oxygen_mass_fraction_adb_value": 0.10855,
        "coal_nitrogen_mass_fraction_adb_value": 0.0094,
        "coal_sulfur_mass_fraction_adb_value": 0.00371,
        "coal_ash_proximate_d_value": 0.105,
        "coal_volatile_matter_d_value": 0.32773,
        "coal_fixed_carbon_d_value": 0.56727,
        "coal_heating_value_dry_kj_kg_value": 27720,
        "coal_moisture_ar_value": 0.1577,
        "syngas_dry_volumetric_flow_value": 124871.32042042461,
        "outlet_co_dry_vol_pct_value": 61.709213714796626,
        "outlet_h2_dry_vol_pct_value": 28.220983716249997,
        "outlet_co2_dry_vol_pct_value": 9.381570928078542,
        "outlet_ch4_dry_vol_pct_value": 0.0579205672622742,
        "outlet_n2_dry_vol_pct_value": 0.5512591533625001,
        "outlet_h2s_dry_vol_pct_value": 0.11739087300124999,
    }


def test_default_inlet_has_no_carrier_backcompat():
    system = _minimal_system()
    inlet = system._initialize_inlet()
    assert inlet.gas_moles[3] == 0.0
    assert inlet.gas_moles[6] == 0.0


def test_legacy_n2_flow_still_maps_to_n2_slot():
    system = _minimal_system(n2_flow=1.0)
    inlet = system._initialize_inlet()
    np.testing.assert_allclose(inlet.gas_moles[6], 1.0 / 28.013 * 1000.0)
    assert inlet.gas_moles[3] == 0.0


def test_carrier_gas_type_maps_to_mainline_species_slots():
    n2_system = _minimal_system(carrier_gas_type="N2", carrier_gas_flow=1.0)
    n2_inlet = n2_system._initialize_inlet()
    np.testing.assert_allclose(n2_inlet.gas_moles[6], 1.0 / 28.013 * 1000.0)
    assert n2_inlet.gas_moles[3] == 0.0

    co2_system = _minimal_system(carrier_gas_type="CO2", carrier_gas_flow=1.0)
    co2_inlet = co2_system._initialize_inlet()
    np.testing.assert_allclose(co2_inlet.gas_moles[3], 1.0 / 44.01 * 1000.0)
    assert co2_inlet.gas_moles[6] == 0.0


def test_invalid_carrier_type_is_rejected():
    with pytest.raises(ValueError, match="carrier_gas_type"):
        _minimal_system(carrier_gas_type="AR", carrier_gas_flow=1.0)


def test_model_case_adapter_builds_inputs_and_uses_transport_moisture():
    built = build_gasifier_inputs_from_model_case(_preview_row())
    monitoring_sum = sum(
        _preview_row()[f"monitor_pulverized_coal_line_{idx}_kg_s_value"]
        for idx in range(1, 5)
    )
    assert built["geometry"] == {"D": 3.8, "L": 10.3}
    assert built["coal_props"]["Mt"] == TRANSPORTED_COAL_MOISTURE_PERCENT
    assert built["coal_props"]["Mt"] != pytest.approx(15.77)
    assert built["op_conds"]["coal_flow"] == pytest.approx(_preview_row()["coal_mass_flow_value"])
    assert built["op_conds"]["coal_flow"] != pytest.approx(monitoring_sum)
    assert "weighfeeder" in built["assumptions"]["coal_flow_basis"]
    assert "monitor_pulverized_coal_line" in built["assumptions"]["forbidden_coal_flow_aliases"]
    assert built["op_conds"]["carrier_gas_type"] == "CO2"
    np.testing.assert_allclose(
        built["op_conds"]["o2_flow"],
        normal_m3_h_to_kg_s(_preview_row()["o2_volumetric_flow_value"], "O2"),
    )
    np.testing.assert_allclose(
        built["op_conds"]["carrier_gas_flow"],
        normal_m3_h_to_kg_s(_preview_row()["co2_carrier_volumetric_flow_value"], "CO2"),
    )

    system = GasifierSystem(built["geometry"], built["coal_props"], built["op_conds"])
    inlet = system._initialize_inlet()
    np.testing.assert_allclose(
        inlet.solid_mass,
        built["op_conds"]["coal_flow"] * (1.0 - TRANSPORTED_COAL_MOISTURE_PERCENT / 100.0),
    )
    assert inlet.solid_mass != pytest.approx(built["op_conds"]["coal_flow"] * (1.0 - 0.1577))
    assert inlet.gas_moles[3] > 0.0


def test_model_case_preview_parquet_schema_builds_if_available():
    pd = pytest.importorskip("pandas")
    preview = Path(
        "/Users/liuzhen/AI-projects/gasifier-data-prep/outputs/model_cases/"
        "sy3_trial_0928_feed_product.model_cases_preview.parquet"
    )
    if not preview.exists():
        pytest.skip("dataprep model_cases preview is not available")
    row = pd.read_parquet(preview).iloc[0]
    built = build_gasifier_inputs_from_model_case(row)
    monitoring_sum = sum(
        float(row[f"monitor_pulverized_coal_line_{idx}_kg_s_value"])
        for idx in range(1, 5)
    )
    assert built["case_id"] == "sy3_trial_0928_feed_product__modelcase_001"
    assert built["op_conds"]["P"] > 3.0e6
    assert built["op_conds"]["carrier_gas_type"] == "CO2"
    assert built["targets"]["outlet_co_dry_vol_pct"] > 0.0
    assert built["op_conds"]["coal_flow"] == pytest.approx(float(row["coal_mass_flow_value"]))
    assert built["op_conds"]["coal_flow"] != pytest.approx(monitoring_sum)
