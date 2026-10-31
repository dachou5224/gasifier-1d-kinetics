# Audit: corrected ModelCase apparent gas-phase carbon recovery

Conclusion: the round-29/30 low apparent recovery values are superseded for the current ModelCase preview. The root cause was the coal-feed boundary: the corrected primary boundary is B-column weighfeeder raw coal converted to dry coal with M_ar, while C-F pulverized-coal lines are monitoring-only aliases.

## Checked basis

- `syngas_dry_volumetric_flow_value`: dataprep contract labels this as total dry syngas `Nm3/h`; the current preview marks it measured from the performance-sheet dry syngas formula/column rather than the older U/V/W Psat-derived path.
- `coal_mass_flow_value`: dataprep contract labels this as B-column weighfeeder raw coal converted to dry coal with `M_ar`, in `kg/s`. This is the ModelCase primary coal boundary.
- `monitor_pulverized_coal_line_1..4_kg_s_value`: C-F line values are retained only as monitoring/forbidden aliases. They must not be summed into the primary model coal feed.
- Normal-volume basis: model-side diagnostic and dataprep feature definitions both use `22.414` Nm3/kmol.
- CO2 carrier: included in the gross carbon input because the dataprep contract marks CO2 carrier as carbon-containing feed. A supplemental coal-only apparent gas C yield subtracts carrier CO2 carbon from dry-gas carbon out before dividing by coal carbon; this is only a rough interface check, not a rigorous conversion definition.

## Recomputed indicators

- Gross apparent gas-phase C recovery: 0.917102 to 0.917102.
- Coal-only apparent gas C yield: 0.907925 to 0.907925.
- Gross recovery if C-F monitoring sum were misused: 0.712720 to 0.712720.
- Dry syngas flow required for 98% gross apparent C recovery: 1.069x to 1.069x the current target dry flow.
- Dry syngas flow required for 98% coal-only apparent C yield: 1.071x to 1.071x the current target dry flow.

| case_id | primary_coal_kg_s | C-F_monitor_kg_s | observed_dry_Nm3_h | gross_recovery | C-F_misuse_gross | required_dry_Nm3_h_gross98 | gross_factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| sy3_trial_0928_feed_product__modelcase_001 | 17.782341 | 23.446173 | 124871.3 | 0.917102 | 0.712720 | 133435.4 | 1.069 |

## Root-cause interpretation

The old 0.69-level recovery is reproduced when the C-F monitoring sum is treated as the primary coal feed. With the corrected weighfeeder/M_ar coal boundary, the same available-field check moves to about 0.92 gross apparent gas-phase C recovery.

At the current CO+CO2+CH4 dry composition, a 98% carbon-conversion-style sanity target would require only about 1.07x larger dry syngas flow than the current `syngas_dry_volumetric_flow_value`. The remaining gap is not evidence of a failed full carbon balance because dry gas alone excludes solid/slag/fly-ash/tar/entrained carbon outlets.

This audit does not convert apparent gas-carbon recovery into industrial carbon conversion. It only verifies that the model-data interface is now using the corrected coal boundary and that the previous C-F alias misuse would reproduce the anomalously low recovery.

Evidence boundary: the corrected artifact can supersede round-29/30 for ModelCase interface diagnostics, but it remains an available-field gas-carbon check, not complete carbon-balance closure or calibrated model accuracy evidence.
