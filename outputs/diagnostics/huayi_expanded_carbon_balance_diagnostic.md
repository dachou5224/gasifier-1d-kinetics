# ModelCase apparent gas-phase carbon recovery diagnostic

This is an available-field dry-gas carbon check for pilot ModelCase rows, not a full carbon-balance closure.
The corrected primary coal boundary uses weighfeeder raw coal minus M_ar moisture; C-F pulverized-coal lines are monitoring-only aliases.

- Rows: 22
- Normal-volume basis: 22.414 Nm3/kmol
- Coal carbon basis: coal_mass_flow_value * coal_carbon_mass_fraction_adb_value / 12.011 kg/kmol
- Carrier CO2 carbon basis: co2_carrier_volumetric_flow_value / 22.414 / 3600
- Dry-gas carbon basis: dry syngas kmol/s * (CO + CO2 + CH4 dry vol%) / 100
- Gross apparent gas-phase C recovery range: 0.833162 to 1.045911
- Coal-only apparent gas C yield range: 0.816100 to 1.045971
- If C-F monitoring sum is misused as primary coal feed, gross recovery would be: 0.696049 to 0.801708
- Apparent gap range: -0.045911 to 0.166838
- Dry syngas factor needed for 98% gross C recovery: 0.937x to 1.176x observed

| case_id | IG | primary_coal_kg_s | C-F_monitor_kg_s | gross_recovery | coal_only_yield | C-F_misuse_gross_recovery | dry_flow_factor_for_98pct | missing_fields |
|---|---|---:|---:|---:|---:|---:|---:|---|
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-28 | IG001 | 18.244940 | 25.003809 | 0.937009 | 0.929752 | 0.703367 | 1.046 | none |
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-29 | IG003 | 17.720530 | 23.280544 | 0.925977 | 0.918040 | 0.721517 | 1.058 | none |
| huayi_sy3_phase1_wgs1__shift_evening_handover_2022-09-28 | IG002 | 17.664877 | 23.378549 | 0.924258 | 0.915955 | 0.715650 | 1.060 | none |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29D_010 | IG004 | 18.228359 | 22.512640 | 0.898347 | 0.887747 | 0.740696 | 1.091 | none |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29D_011 | IG005 | 18.271646 | 23.164013 | 0.887072 | 0.875027 | 0.714258 | 1.105 | none |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29N_012 | IG006 | 19.331652 | 24.478200 | 0.874307 | 0.861333 | 0.704338 | 1.121 | none |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29N_014 | IG007 | 19.256084 | 23.168238 | 0.833162 | 0.816100 | 0.703497 | 1.176 | none |
| huayi_sy3_phase1_wgs1__plateau_2022-09-30D_017 | IG008 | 18.959112 | 25.467627 | 0.910971 | 0.901019 | 0.696049 | 1.076 | none |
| huayi_sy3sy2_phase2_wgs1__shift_morning_handover_2022-10-02 | IG001 | 19.936579 | 25.027324 | 0.897575 | 0.886931 | 0.728960 | 1.092 | none |
| huayi_sy3sy2_phase2_wgs1__shift_morning_handover_2022-10-04 | IG004 | 19.449828 | 25.687922 | 1.045911 | 1.045971 | 0.792169 | 0.937 | none |
| huayi_sy3sy2_phase2_wgs1__plateau_2022-10-02N_010 | IG002 | 20.188092 | 25.246370 | 0.902441 | 0.892303 | 0.735503 | 1.086 | none |
| huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03D_012 | IG003 | 20.017498 | 25.203546 | 1.009148 | 1.009160 | 0.801708 | 0.971 | none |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-06 | IG001 | 19.783256 | 24.941167 | 0.935185 | 0.928361 | 0.756693 | 1.048 | none |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-07 | IG002 | 20.848414 | 25.687996 | 0.883180 | 0.871753 | 0.729029 | 1.110 | none |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-09 | IG005 | 20.636478 | 24.758783 | 0.879313 | 0.866975 | 0.744404 | 1.115 | none |
| huayi_sy3sy2_phase3_wgs1__shift_evening_handover_2022-10-07 | IG003 | 20.928566 | 25.613059 | 0.877270 | 0.865221 | 0.728738 | 1.117 | none |
| huayi_sy3sy2_phase3_wgs1__plateau_2022-10-07N_012 | IG004 | 21.025546 | 25.357044 | 0.867929 | 0.854995 | 0.730804 | 1.129 | none |
| huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_015 | IG006 | 17.554707 | 25.392227 | 1.035610 | 1.040010 | 0.741114 | 0.946 | none |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-10 | IG001 | 20.090268 | 25.334856 | 0.917542 | 0.908952 | 0.742094 | 1.068 | none |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-11 | IG004 | 21.091893 | 25.221213 | 0.886653 | 0.875324 | 0.752685 | 1.105 | none |
| huayi_sy3sy2_phase4_wgs1__plateau_2022-10-10N_006 | IG002 | 20.127816 | 25.061088 | 0.903204 | 0.893287 | 0.738925 | 1.085 | none |
| huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11D_007 | IG003 | 20.200510 | 25.204168 | 0.922525 | 0.914450 | 0.753501 | 1.062 | none |

Evidence boundary: this apparent gas-phase carbon recovery uses only available ModelCase fields for dry syngas and CO2 carrier carbon. The apparent gap is not a complete carbon imbalance: it excludes unconverted solid carbon, slag carbon, fly-ash/fines carbon, tar, entrained carbon, and other unmeasured carbon-bearing outlets. The old round-29/30 low recovery values are superseded when C-F monitoring lines are not used as the primary coal-feed boundary.
