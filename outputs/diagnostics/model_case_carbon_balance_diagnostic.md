# ModelCase apparent gas-phase carbon recovery diagnostic

This is an available-field dry-gas carbon check for pilot ModelCase rows, not a full carbon-balance closure.
The corrected primary coal boundary uses weighfeeder raw coal minus M_ar moisture; C-F pulverized-coal lines are monitoring-only aliases.

- Rows: 1
- Normal-volume basis: 22.414 Nm3/kmol
- Coal carbon basis: coal_mass_flow_value * coal_carbon_mass_fraction_adb_value / 12.011 kg/kmol
- Carrier CO2 carbon basis: co2_carrier_volumetric_flow_value / 22.414 / 3600
- Dry-gas carbon basis: dry syngas kmol/s * (CO + CO2 + CH4 dry vol%) / 100
- Gross apparent gas-phase C recovery range: 0.917102 to 0.917102
- Coal-only apparent gas C yield range: 0.907925 to 0.907925
- If C-F monitoring sum is misused as primary coal feed, gross recovery would be: 0.712720 to 0.712720
- Apparent gap range: 0.082898 to 0.082898
- Dry syngas factor needed for 98% gross C recovery: 1.069x to 1.069x observed

| case_id | IG | primary_coal_kg_s | C-F_monitor_kg_s | gross_recovery | coal_only_yield | C-F_misuse_gross_recovery | dry_flow_factor_for_98pct | missing_fields |
|---|---|---:|---:|---:|---:|---:|---:|---|
| sy3_trial_0928_feed_product__modelcase_001 | IG001 | 17.782341 | 23.446173 | 0.917102 | 0.907925 | 0.712720 | 1.069 | none |

Evidence boundary: this apparent gas-phase carbon recovery uses only available ModelCase fields for dry syngas and CO2 carrier carbon. The apparent gap is not a complete carbon imbalance: it excludes unconverted solid carbon, slag carbon, fly-ash/fines carbon, tar, entrained carbon, and other unmeasured carbon-bearing outlets. The old round-29/30 low recovery values are superseded when C-F monitoring lines are not used as the primary coal-feed boundary.
