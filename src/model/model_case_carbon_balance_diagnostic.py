"""
Available-field carbon diagnostics for dataprep ModelCase preview rows.

This module computes an apparent dry-gas carbon recovery from fields already
present in the ModelCase preview. It is intentionally not a full carbon-balance
closure because unconverted solid carbon, slag/fly-ash/fines carbon, tars, and
other unmeasured carbon outlets are not included.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .model_case_adapter import NORMAL_M3_PER_KMOL, normal_m3_h_to_kmol_s

CARBON_ATOMIC_MASS_KG_PER_KMOL = 12.011

REQUIRED_CARBON_BALANCE_COLUMNS: tuple[str, ...] = (
    "case_id",
    "source_candidate_id",
    "independence_group",
    "coal_mass_flow_kg_s",
    "coal_weighfeeder_ar_flow_t_h",
    "coal_moisture_ar_fraction",
    "coal_carbon_mass_fraction_adb",
    "coal_C_kmol_s",
    "monitor_pulverized_coal_c_f_sum_kg_s",
    "monitor_to_primary_coal_flow_ratio",
    "co2_carrier_volumetric_flow_Nm3_h",
    "carrier_CO2_C_kmol_s",
    "total_available_C_in_kmol_s",
    "target_dry_syngas_Nm3_h",
    "target_dry_syngas_kmol_s",
    "target_CO_pct",
    "target_CO2_pct",
    "target_CH4_pct",
    "target_dry_gas_C_out_kmol_s",
    "apparent_gas_phase_C_recovery",
    "apparent_C_gap",
    "coal_only_apparent_gas_C_yield",
    "coal_only_apparent_C_gap",
    "required_dry_syngas_Nm3_h_for_98pct_gross_C_recovery",
    "required_to_observed_dry_syngas_factor_gross_98pct",
    "required_dry_syngas_Nm3_h_for_98pct_coal_C_yield",
    "required_to_observed_dry_syngas_factor_coal_98pct",
    "gross_recovery_if_c_f_monitoring_sum_used",
    "coal_only_yield_if_c_f_monitoring_sum_used",
    "c_f_monitoring_alias_warning",
    "interpretation_status",
    "missing_fields",
)

SANITY_CARBON_CONVERSION_FRACTION = 0.98


def _get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _numeric(row: Mapping[str, Any], key: str, missing_fields: list[str], default: float | None = None) -> float:
    value = _get(row, key)
    if _is_missing(value):
        missing_fields.append(key)
        if default is None:
            raise ValueError(f"Missing required ModelCase carbon diagnostic field: {key}")
        return float(default)
    return float(value)


def build_carbon_balance_row(model_case_row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one apparent gas-phase carbon recovery row."""
    missing_fields: list[str] = []

    coal_mass_flow = _numeric(model_case_row, "coal_mass_flow_value", missing_fields)
    coal_weighfeeder_ar_flow = _numeric(
        model_case_row,
        "coal_weighfeeder_ar_flow_value",
        missing_fields,
        default=math.nan,
    )
    coal_moisture_ar = _numeric(
        model_case_row,
        "coal_moisture_ar_value",
        missing_fields,
        default=math.nan,
    )
    coal_carbon_fraction = _numeric(model_case_row, "coal_carbon_mass_fraction_adb_value", missing_fields)
    co2_carrier_nm3_h = _numeric(model_case_row, "co2_carrier_volumetric_flow_value", missing_fields)
    target_dry_syngas_nm3_h = _numeric(model_case_row, "syngas_dry_volumetric_flow_value", missing_fields)
    target_co_pct = _numeric(model_case_row, "outlet_co_dry_vol_pct_value", missing_fields)
    target_co2_pct = _numeric(model_case_row, "outlet_co2_dry_vol_pct_value", missing_fields)
    target_ch4_pct = _numeric(
        model_case_row,
        "outlet_ch4_dry_vol_pct_value",
        missing_fields,
        default=0.0,
    )

    coal_c_kmol_s = coal_mass_flow * coal_carbon_fraction / CARBON_ATOMIC_MASS_KG_PER_KMOL
    monitoring_line_keys = (
        "monitor_pulverized_coal_line_1_kg_s_value",
        "monitor_pulverized_coal_line_2_kg_s_value",
        "monitor_pulverized_coal_line_3_kg_s_value",
        "monitor_pulverized_coal_line_4_kg_s_value",
    )
    monitoring_values = [_get(model_case_row, key) for key in monitoring_line_keys]
    if all(not _is_missing(value) for value in monitoring_values):
        monitor_c_f_sum = float(sum(float(value) for value in monitoring_values))
    else:
        monitor_c_f_sum = math.nan
    carrier_co2_c_kmol_s = normal_m3_h_to_kmol_s(co2_carrier_nm3_h)
    total_available_c_in = coal_c_kmol_s + carrier_co2_c_kmol_s
    target_dry_syngas_kmol_s = normal_m3_h_to_kmol_s(target_dry_syngas_nm3_h)
    target_dry_gas_c_out = target_dry_syngas_kmol_s * (
        (target_co_pct + target_co2_pct + target_ch4_pct) / 100.0
    )
    recovery = target_dry_gas_c_out / total_available_c_in
    gap = 1.0 - recovery
    carbon_vol_fraction = (target_co_pct + target_co2_pct + target_ch4_pct) / 100.0
    coal_only_yield = (target_dry_gas_c_out - carrier_co2_c_kmol_s) / coal_c_kmol_s
    coal_only_gap = 1.0 - coal_only_yield
    required_gross_c_out = SANITY_CARBON_CONVERSION_FRACTION * total_available_c_in
    required_coal_basis_c_out = (
        SANITY_CARBON_CONVERSION_FRACTION * coal_c_kmol_s + carrier_co2_c_kmol_s
    )
    required_gross_dry_kmol_s = required_gross_c_out / carbon_vol_fraction
    required_coal_basis_dry_kmol_s = required_coal_basis_c_out / carbon_vol_fraction
    required_gross_dry_nm3_h = required_gross_dry_kmol_s * NORMAL_M3_PER_KMOL * 3600.0
    required_coal_basis_dry_nm3_h = (
        required_coal_basis_dry_kmol_s * NORMAL_M3_PER_KMOL * 3600.0
    )
    if math.isfinite(monitor_c_f_sum) and monitor_c_f_sum > 0.0:
        cf_coal_c_kmol_s = monitor_c_f_sum * coal_carbon_fraction / CARBON_ATOMIC_MASS_KG_PER_KMOL
        cf_total_available_c = cf_coal_c_kmol_s + carrier_co2_c_kmol_s
        cf_gross_recovery = target_dry_gas_c_out / cf_total_available_c
        cf_coal_only_yield = (target_dry_gas_c_out - carrier_co2_c_kmol_s) / cf_coal_c_kmol_s
        monitor_ratio = monitor_c_f_sum / coal_mass_flow
        cf_warning = "do_not_use_c_f_monitoring_sum_as_primary_coal_feed"
    else:
        cf_gross_recovery = math.nan
        cf_coal_only_yield = math.nan
        monitor_ratio = math.nan
        cf_warning = "c_f_monitoring_fields_not_available"

    row = {
        "case_id": _get(model_case_row, "case_id_value", _get(model_case_row, "case_id")),
        "source_candidate_id": _get(
            model_case_row,
            "segment_id_value",
            _get(model_case_row, "segment_id"),
        ),
        "independence_group": _get(
            model_case_row,
            "independence_group_id_value",
            _get(model_case_row, "independence_group_id"),
        ),
        "coal_mass_flow_kg_s": coal_mass_flow,
        "coal_weighfeeder_ar_flow_t_h": coal_weighfeeder_ar_flow,
        "coal_moisture_ar_fraction": coal_moisture_ar,
        "coal_carbon_mass_fraction_adb": coal_carbon_fraction,
        "coal_C_kmol_s": coal_c_kmol_s,
        "monitor_pulverized_coal_c_f_sum_kg_s": monitor_c_f_sum,
        "monitor_to_primary_coal_flow_ratio": monitor_ratio,
        "co2_carrier_volumetric_flow_Nm3_h": co2_carrier_nm3_h,
        "carrier_CO2_C_kmol_s": carrier_co2_c_kmol_s,
        "total_available_C_in_kmol_s": total_available_c_in,
        "target_dry_syngas_Nm3_h": target_dry_syngas_nm3_h,
        "target_dry_syngas_kmol_s": target_dry_syngas_kmol_s,
        "target_CO_pct": target_co_pct,
        "target_CO2_pct": target_co2_pct,
        "target_CH4_pct": target_ch4_pct,
        "target_dry_gas_C_out_kmol_s": target_dry_gas_c_out,
        "apparent_gas_phase_C_recovery": recovery,
        "apparent_C_gap": gap,
        "coal_only_apparent_gas_C_yield": coal_only_yield,
        "coal_only_apparent_C_gap": coal_only_gap,
        "required_dry_syngas_Nm3_h_for_98pct_gross_C_recovery": required_gross_dry_nm3_h,
        "required_to_observed_dry_syngas_factor_gross_98pct": required_gross_dry_nm3_h
        / target_dry_syngas_nm3_h,
        "required_dry_syngas_Nm3_h_for_98pct_coal_C_yield": required_coal_basis_dry_nm3_h,
        "required_to_observed_dry_syngas_factor_coal_98pct": required_coal_basis_dry_nm3_h
        / target_dry_syngas_nm3_h,
        "gross_recovery_if_c_f_monitoring_sum_used": cf_gross_recovery,
        "coal_only_yield_if_c_f_monitoring_sum_used": cf_coal_only_yield,
        "c_f_monitoring_alias_warning": cf_warning,
        "interpretation_status": "corrected_primary_coal_flow_available_field_gas_carbon_check_not_closure",
        "missing_fields": ";".join(missing_fields),
    }
    return {column: row.get(column) for column in REQUIRED_CARBON_BALANCE_COLUMNS}


def build_carbon_balance_rows(model_case_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [build_carbon_balance_row(row) for row in model_case_rows]


def markdown_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    recoveries = [float(row["apparent_gas_phase_C_recovery"]) for row in rows]
    coal_only_yields = [float(row["coal_only_apparent_gas_C_yield"]) for row in rows]
    gaps = [float(row["apparent_C_gap"]) for row in rows]
    gross_factors = [
        float(row["required_to_observed_dry_syngas_factor_gross_98pct"]) for row in rows
    ]
    cf_recoveries = [
        float(row["gross_recovery_if_c_f_monitoring_sum_used"])
        for row in rows
        if not math.isnan(float(row["gross_recovery_if_c_f_monitoring_sum_used"]))
    ]
    lines = [
        "# ModelCase apparent gas-phase carbon recovery diagnostic",
        "",
        "This is an available-field dry-gas carbon check for pilot ModelCase rows, not a full carbon-balance closure.",
        "The corrected primary coal boundary uses weighfeeder raw coal minus M_ar moisture; C-F pulverized-coal lines are monitoring-only aliases.",
        "",
        f"- Rows: {len(rows)}",
        f"- Normal-volume basis: {NORMAL_M3_PER_KMOL} Nm3/kmol",
        f"- Coal carbon basis: coal_mass_flow_value * coal_carbon_mass_fraction_adb_value / {CARBON_ATOMIC_MASS_KG_PER_KMOL} kg/kmol",
        "- Carrier CO2 carbon basis: co2_carrier_volumetric_flow_value / 22.414 / 3600",
        "- Dry-gas carbon basis: dry syngas kmol/s * (CO + CO2 + CH4 dry vol%) / 100",
        f"- Gross apparent gas-phase C recovery range: {min(recoveries):.6f} to {max(recoveries):.6f}" if recoveries else "- Gross apparent gas-phase C recovery range: n/a",
        f"- Coal-only apparent gas C yield range: {min(coal_only_yields):.6f} to {max(coal_only_yields):.6f}" if coal_only_yields else "- Coal-only apparent gas C yield range: n/a",
        f"- If C-F monitoring sum is misused as primary coal feed, gross recovery would be: {min(cf_recoveries):.6f} to {max(cf_recoveries):.6f}" if cf_recoveries else "- If C-F monitoring sum is misused as primary coal feed: n/a",
        f"- Apparent gap range: {min(gaps):.6f} to {max(gaps):.6f}" if gaps else "- Apparent gap range: n/a",
        f"- Dry syngas factor needed for 98% gross C recovery: {min(gross_factors):.3f}x to {max(gross_factors):.3f}x observed" if gross_factors else "- Dry syngas factor needed for 98% gross C recovery: n/a",
        "",
        "| case_id | IG | primary_coal_kg_s | C-F_monitor_kg_s | gross_recovery | coal_only_yield | C-F_misuse_gross_recovery | dry_flow_factor_for_98pct | missing_fields |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {ig} | {coal_flow:.6f} | {cf_sum:.6f} | {recovery:.6f} | {coal_yield:.6f} | {cf_recovery:.6f} | {factor:.3f} | {missing} |".format(
                case_id=row["case_id"],
                ig=row["independence_group"],
                coal_flow=float(row["coal_mass_flow_kg_s"]),
                cf_sum=float(row["monitor_pulverized_coal_c_f_sum_kg_s"]),
                recovery=float(row["apparent_gas_phase_C_recovery"]),
                coal_yield=float(row["coal_only_apparent_gas_C_yield"]),
                cf_recovery=float(row["gross_recovery_if_c_f_monitoring_sum_used"]),
                factor=float(row["required_to_observed_dry_syngas_factor_gross_98pct"]),
                missing=row["missing_fields"] or "none",
            )
        )
    lines.extend(
        [
            "",
            "Evidence boundary: this apparent gas-phase carbon recovery uses only available ModelCase fields "
            "for dry syngas and CO2 carrier carbon. The apparent gap is not a complete carbon imbalance: "
            "it excludes unconverted solid carbon, slag carbon, fly-ash/fines carbon, tar, entrained carbon, "
            "and other unmeasured carbon-bearing outlets. The old round-29/30 low recovery values are "
            "superseded when C-F monitoring lines are not used as the primary coal-feed boundary.",
        ]
    )
    return "\n".join(lines) + "\n"


def audit_markdown_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the root-cause audit for the low apparent recovery values."""
    gross = [float(row["apparent_gas_phase_C_recovery"]) for row in rows]
    coal_only = [float(row["coal_only_apparent_gas_C_yield"]) for row in rows]
    gross_factor = [
        float(row["required_to_observed_dry_syngas_factor_gross_98pct"]) for row in rows
    ]
    coal_factor = [
        float(row["required_to_observed_dry_syngas_factor_coal_98pct"]) for row in rows
    ]
    cf_recoveries = [
        float(row["gross_recovery_if_c_f_monitoring_sum_used"])
        for row in rows
        if not math.isnan(float(row["gross_recovery_if_c_f_monitoring_sum_used"]))
    ]
    lines = [
        "# Audit: corrected ModelCase apparent gas-phase carbon recovery",
        "",
        "Conclusion: the round-29/30 low apparent recovery values are superseded for the current ModelCase preview. "
        "The root cause was the coal-feed boundary: the corrected primary boundary is B-column weighfeeder raw coal "
        "converted to dry coal with M_ar, while C-F pulverized-coal lines are monitoring-only aliases.",
        "",
        "## Checked basis",
        "",
        "- `syngas_dry_volumetric_flow_value`: dataprep contract labels this as total dry syngas `Nm3/h`; the current preview marks it measured from the performance-sheet dry syngas formula/column rather than the older U/V/W Psat-derived path.",
        "- `coal_mass_flow_value`: dataprep contract labels this as B-column weighfeeder raw coal converted to dry coal with `M_ar`, in `kg/s`. This is the ModelCase primary coal boundary.",
        "- `monitor_pulverized_coal_line_1..4_kg_s_value`: C-F line values are retained only as monitoring/forbidden aliases. They must not be summed into the primary model coal feed.",
        f"- Normal-volume basis: model-side diagnostic and dataprep feature definitions both use `{NORMAL_M3_PER_KMOL}` Nm3/kmol.",
        "- CO2 carrier: included in the gross carbon input because the dataprep contract marks CO2 carrier as carbon-containing feed. A supplemental coal-only apparent gas C yield subtracts carrier CO2 carbon from dry-gas carbon out before dividing by coal carbon; this is only a rough interface check, not a rigorous conversion definition.",
        "",
        "## Recomputed indicators",
        "",
        f"- Gross apparent gas-phase C recovery: {min(gross):.6f} to {max(gross):.6f}.",
        f"- Coal-only apparent gas C yield: {min(coal_only):.6f} to {max(coal_only):.6f}.",
        f"- Gross recovery if C-F monitoring sum were misused: {min(cf_recoveries):.6f} to {max(cf_recoveries):.6f}." if cf_recoveries else "- Gross recovery if C-F monitoring sum were misused: n/a.",
        f"- Dry syngas flow required for 98% gross apparent C recovery: {min(gross_factor):.3f}x to {max(gross_factor):.3f}x the current target dry flow.",
        f"- Dry syngas flow required for 98% coal-only apparent C yield: {min(coal_factor):.3f}x to {max(coal_factor):.3f}x the current target dry flow.",
        "",
        "| case_id | primary_coal_kg_s | C-F_monitor_kg_s | observed_dry_Nm3_h | gross_recovery | C-F_misuse_gross | required_dry_Nm3_h_gross98 | gross_factor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {primary:.6f} | {cf_sum:.6f} | {observed:.1f} | {gross:.6f} | {cf_gross:.6f} | {gross_req:.1f} | {gross_factor:.3f} |".format(
                case_id=row["case_id"],
                primary=float(row["coal_mass_flow_kg_s"]),
                cf_sum=float(row["monitor_pulverized_coal_c_f_sum_kg_s"]),
                observed=float(row["target_dry_syngas_Nm3_h"]),
                gross=float(row["apparent_gas_phase_C_recovery"]),
                cf_gross=float(row["gross_recovery_if_c_f_monitoring_sum_used"]),
                gross_req=float(row["required_dry_syngas_Nm3_h_for_98pct_gross_C_recovery"]),
                gross_factor=float(row["required_to_observed_dry_syngas_factor_gross_98pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Root-cause interpretation",
            "",
            "The old 0.69-level recovery is reproduced when the C-F monitoring sum is treated as the primary coal feed. With the corrected weighfeeder/M_ar coal boundary, the same available-field check moves to about 0.92 gross apparent gas-phase C recovery.",
            "",
            "At the current CO+CO2+CH4 dry composition, a 98% carbon-conversion-style sanity target would require only about 1.07x larger dry syngas flow than the current `syngas_dry_volumetric_flow_value`. The remaining gap is not evidence of a failed full carbon balance because dry gas alone excludes solid/slag/fly-ash/tar/entrained carbon outlets.",
            "",
            "This audit does not convert apparent gas-carbon recovery into industrial carbon conversion. It only verifies that the model-data interface is now using the corrected coal boundary and that the previous C-F alias misuse would reproduce the anomalously low recovery.",
            "",
            "Evidence boundary: the corrected artifact can supersede round-29/30 for ModelCase interface diagnostics, but it remains an available-field gas-carbon check, not complete carbon-balance closure or calibrated model accuracy evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "CARBON_ATOMIC_MASS_KG_PER_KMOL",
    "REQUIRED_CARBON_BALANCE_COLUMNS",
    "SANITY_CARBON_CONVERSION_FRACTION",
    "audit_markdown_summary",
    "build_carbon_balance_row",
    "build_carbon_balance_rows",
    "markdown_summary",
]
