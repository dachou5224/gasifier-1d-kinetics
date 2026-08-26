"""
Adapter from dataprep ModelCase preview rows to ``GasifierSystem`` inputs.

The adapter is intentionally small: it performs unit conversion and records
assumptions, but does not run calibration or alter solver behavior.
"""
from __future__ import annotations

from typing import Any, Mapping

from .physics import MOLAR_MASS

NORMAL_M3_PER_KMOL = 22.414
TRANSPORTED_COAL_MOISTURE_PERCENT = 2.0


def normal_m3_h_to_kg_s(value_nm3_h: float, species: str) -> float:
    """Convert normal cubic meters per hour to species mass flow [kg/s]."""
    return float(value_nm3_h) / NORMAL_M3_PER_KMOL / 3600.0 * MOLAR_MASS[species]


def normal_m3_h_to_kmol_s(value_nm3_h: float) -> float:
    """Convert normal cubic meters per hour to kmol/s at the dataprep STP basis."""
    return float(value_nm3_h) / NORMAL_M3_PER_KMOL / 3600.0


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def build_gasifier_inputs_from_model_case(row: Mapping[str, Any]) -> dict:
    """
    Build ``geometry``, ``coal_props``, ``op_conds`` and target metadata.

    Dataprep preview units:
    - pressure: MPa -> Pa
    - O2 / CO2 carrier / dry syngas: Nm3/h
    - inlet temperatures: degC -> K

    Carrier gas enters through the explicit ``carrier_gas_type`` /
    ``carrier_gas_flow`` op_conds contract as a mass flow [kg/s].
    """
    if not bool(_get(row, "model_case_ready", False)):
        reason = _get(row, "not_model_ready_reason", "model_case_ready is false")
        raise ValueError(f"ModelCase row is not ready: {reason}")

    geometry = {
        "D": float(_get(row, "reactor_bed_diameter_m_value")),
        "L": float(_get(row, "reactor_bed_height_m_value")),
    }
    coal_props = {
        "Cd": float(_get(row, "coal_carbon_mass_fraction_adb_value")) * 100.0,
        "Hd": float(_get(row, "coal_hydrogen_mass_fraction_adb_value")) * 100.0,
        "Od": float(_get(row, "coal_oxygen_mass_fraction_adb_value")) * 100.0,
        "Nd": float(_get(row, "coal_nitrogen_mass_fraction_adb_value")) * 100.0,
        "Sd": float(_get(row, "coal_sulfur_mass_fraction_adb_value")) * 100.0,
        "Ad": float(_get(row, "coal_ash_proximate_d_value")) * 100.0,
        "Vd": float(_get(row, "coal_volatile_matter_d_value")) * 100.0,
        "FCd": float(_get(row, "coal_fixed_carbon_d_value")) * 100.0,
        "Mt": TRANSPORTED_COAL_MOISTURE_PERCENT,
        "HHV_d": float(_get(row, "coal_heating_value_dry_kj_kg_value")),
    }
    op_conds = {
        "coal_flow": float(_get(row, "coal_mass_flow_value")),
        "o2_flow": normal_m3_h_to_kg_s(float(_get(row, "o2_volumetric_flow_value")), "O2"),
        "carrier_gas_type": "CO2",
        "carrier_gas_flow": normal_m3_h_to_kg_s(
            float(_get(row, "co2_carrier_volumetric_flow_value")), "CO2"
        ),
        "steam_flow": 0.0,
        "P": float(_get(row, "gasifier_pressure_value")) * 1.0e6,
        "T_in": float(_get(row, "o2_inlet_temperature_c_value")) + 273.15,
    }
    targets = {
        "syngas_dry_flow_kmol_s": normal_m3_h_to_kmol_s(
            float(_get(row, "syngas_dry_volumetric_flow_value"))
        ),
        "outlet_co_dry_vol_pct": float(_get(row, "outlet_co_dry_vol_pct_value")),
        "outlet_h2_dry_vol_pct": float(_get(row, "outlet_h2_dry_vol_pct_value")),
        "outlet_co2_dry_vol_pct": float(_get(row, "outlet_co2_dry_vol_pct_value")),
        "outlet_ch4_dry_vol_pct": float(_get(row, "outlet_ch4_dry_vol_pct_value", 0.0)),
        "outlet_n2_dry_vol_pct": float(_get(row, "outlet_n2_dry_vol_pct_value", 0.0)),
        "outlet_h2s_dry_vol_pct": float(_get(row, "outlet_h2s_dry_vol_pct_value", 0.0)),
    }
    assumptions = {
        "transported_coal_moisture_percent": TRANSPORTED_COAL_MOISTURE_PERCENT,
        "coal_flow_basis": "B-column weighfeeder raw coal converted with M_ar; C-F pulverized-coal lines are monitoring-only aliases",
        "forbidden_coal_flow_aliases": "monitor_pulverized_coal_line_1..4_kg_s_value",
        "steam_flow": "not_applicable_zero_feed",
        "carrier_gas_type": "CO2",
        "normal_volume_basis": "Nm3/h converted with 22.414 Nm3/kmol",
        "co2_carrier_inlet_temperature_K": float(_get(row, "co2_carrier_inlet_temperature_c_value")) + 273.15,
    }
    return {
        "case_id": _get(row, "case_id_value", _get(row, "case_id")),
        "segment_id": _get(row, "segment_id"),
        "geometry": geometry,
        "coal_props": coal_props,
        "op_conds": op_conds,
        "targets": targets,
        "assumptions": assumptions,
    }


__all__ = [
    "NORMAL_M3_PER_KMOL",
    "TRANSPORTED_COAL_MOISTURE_PERCENT",
    "build_gasifier_inputs_from_model_case",
    "normal_m3_h_to_kg_s",
    "normal_m3_h_to_kmol_s",
]
