# Pilot fixed-physics diagnostic baseline

This is a diagnostic smoke/baseline artifact, not calibrated accuracy evidence.

- Rows: 44
- N_cells coverage: 3, 5
- Solver path: GasifierSystem.solve(solver_method='minimize', jacobian_mode='scipy')
- Poor convergence threshold recorded, not relaxed: cost > 1e-4

| case_id | IG | N | runtime_s | poor_cells | T_out_C | CO | H2 | CO2 | dry_flow | dCO | dH2 | dCO2 | d_flow |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-28 | IG001 | 3 | 0.960 | 0 | 1356.41 | 66.94 | 21.50 | 10.46 | 1.312 | 5.32 | -6.56 | 0.87 | -0.314 |
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-29 | IG003 | 3 | 1.087 | 0 | 1315.98 | 65.72 | 22.90 | 10.23 | 1.236 | 3.71 | -5.53 | 1.42 | -0.322 |
| huayi_sy3_phase1_wgs1__shift_evening_handover_2022-09-28 | IG002 | 3 | 1.000 | 0 | 1315.07 | 65.79 | 22.75 | 10.32 | 1.236 | 4.07 | -5.44 | 0.96 | -0.312 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29D_010 | IG004 | 3 | 0.956 | 0 | 1293.74 | 64.99 | 23.59 | 10.23 | 1.242 | 3.29 | -4.49 | 0.66 | -0.299 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29D_011 | IG005 | 3 | 0.985 | 0 | 1289.25 | 64.93 | 23.52 | 10.35 | 1.243 | 3.55 | -4.15 | 0.83 | -0.294 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29N_012 | IG006 | 3 | 1.306 | 0 | 1286.11 | 64.66 | 23.85 | 10.27 | 1.300 | 3.12 | -4.65 | 1.01 | -0.300 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29N_014 | IG007 | 3 | 1.300 | 0 | 1249.84 | 63.70 | 24.67 | 10.34 | 1.245 | 2.05 | -3.62 | 0.99 | -0.267 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-30D_017 | IG008 | 3 | 0.984 | 0 | 1310.36 | 65.63 | 22.72 | 10.50 | 1.319 | 3.13 | -5.35 | 1.78 | -0.318 |
| huayi_sy3sy2_phase2_wgs1__shift_morning_handover_2022-10-02 | IG001 | 3 | 1.026 | 0 | 1297.24 | 63.57 | 24.87 | 10.22 | 1.343 | 0.77 | -2.79 | 1.39 | -0.344 |
| huayi_sy3sy2_phase2_wgs1__shift_morning_handover_2022-10-04 | IG004 | 3 | 1.348 | 0 | 1377.31 | 64.01 | 29.04 | 5.83 | 1.265 | 0.60 | 1.78 | -2.78 | -0.465 |
| huayi_sy3sy2_phase2_wgs1__plateau_2022-10-02N_010 | IG002 | 3 | 0.966 | 0 | 1309.10 | 63.82 | 24.62 | 10.23 | 1.366 | 1.14 | -2.77 | 1.01 | -0.346 |
| huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03D_012 | IG003 | 3 | 1.140 | 0 | 1350.09 | 62.89 | 29.92 | 6.04 | 1.276 | -0.21 | 2.91 | -3.14 | -0.436 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-06 | IG001 | 3 | 1.193 | 0 | 1332.25 | 64.17 | 24.67 | 9.85 | 1.363 | 1.18 | -2.14 | 0.34 | -0.344 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-07 | IG002 | 3 | 1.162 | 0 | 1280.39 | 62.38 | 26.42 | 9.81 | 1.362 | -0.90 | -0.50 | 0.72 | -0.328 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-09 | IG005 | 3 | 0.991 | 0 | 1271.70 | 62.36 | 26.25 | 9.98 | 1.347 | -1.30 | -0.09 | 0.78 | -0.314 |
| huayi_sy3sy2_phase3_wgs1__shift_evening_handover_2022-10-07 | IG003 | 3 | 1.038 | 0 | 1278.88 | 62.35 | 26.41 | 9.83 | 1.362 | -0.81 | -0.25 | 0.37 | -0.318 |
| huayi_sy3sy2_phase3_wgs1__plateau_2022-10-07N_012 | IG004 | 3 | 1.296 | 0 | 1275.31 | 62.26 | 26.50 | 9.83 | 1.363 | -0.84 | -0.19 | 0.33 | -0.308 |
| huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_015 | IG006 | 3 | 1.114 | 0 | 1418.37 | 67.77 | 21.16 | 9.86 | 1.287 | 4.51 | -5.47 | 0.50 | -0.414 |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-10 | IG001 | 3 | 1.403 | 0 | 1257.97 | 62.57 | 25.31 | 10.71 | 1.301 | -1.45 | -1.31 | 2.12 | -0.421 |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-11 | IG004 | 3 | 1.470 | 0 | 1233.39 | 61.92 | 26.10 | 10.51 | 1.313 | -2.25 | -0.18 | 1.67 | -0.419 |
| huayi_sy3sy2_phase4_wgs1__plateau_2022-10-10N_006 | IG002 | 3 | 1.451 | 0 | 1256.06 | 62.48 | 25.49 | 10.63 | 1.299 | -1.31 | -0.95 | 1.59 | -0.393 |
| huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11D_007 | IG003 | 3 | 1.485 | 0 | 1263.38 | 62.68 | 25.21 | 10.72 | 1.315 | -1.27 | -1.23 | 1.85 | -0.421 |
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-28 | IG001 | 5 | 1.294 | 0 | 1246.25 | 67.90 | 22.70 | 8.31 | 1.394 | 6.28 | -5.36 | -1.28 | -0.232 |
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-29 | IG003 | 5 | 1.120 | 0 | 1209.85 | 66.61 | 24.08 | 8.17 | 1.314 | 4.60 | -4.34 | -0.65 | -0.243 |
| huayi_sy3_phase1_wgs1__shift_evening_handover_2022-09-28 | IG002 | 5 | 1.159 | 0 | 1209.13 | 66.68 | 23.93 | 8.26 | 1.314 | 4.96 | -4.26 | -1.11 | -0.234 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29D_010 | IG004 | 5 | 1.070 | 0 | 1189.66 | 65.78 | 24.81 | 8.23 | 1.323 | 4.08 | -3.27 | -1.34 | -0.219 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29D_011 | IG005 | 5 | 1.142 | 0 | 1185.51 | 65.70 | 24.75 | 8.36 | 1.323 | 4.32 | -2.93 | -1.16 | -0.213 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29N_012 | IG006 | 5 | 1.363 | 0 | 1182.57 | 65.39 | 25.10 | 8.30 | 1.385 | 3.85 | -3.40 | -0.96 | -0.214 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29N_014 | IG007 | 5 | 1.463 | 0 | 1151.68 | 64.08 | 26.04 | 8.60 | 1.329 | 2.43 | -2.24 | -0.75 | -0.183 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-30D_017 | IG008 | 5 | 1.009 | 0 | 1204.18 | 66.52 | 23.91 | 8.42 | 1.403 | 4.02 | -4.16 | -0.29 | -0.235 |
| huayi_sy3sy2_phase2_wgs1__shift_morning_handover_2022-10-02 | IG001 | 5 | 1.075 | 0 | 1187.47 | 64.61 | 25.92 | 8.14 | 1.429 | 1.81 | -1.73 | -0.69 | -0.259 |
| huayi_sy3sy2_phase2_wgs1__shift_morning_handover_2022-10-04 | IG004 | 5 | 1.294 | 0 | 1280.89 | 64.94 | 30.21 | 3.80 | 1.382 | 1.53 | 2.95 | -4.81 | -0.348 |
| huayi_sy3sy2_phase2_wgs1__plateau_2022-10-02N_010 | IG002 | 5 | 1.083 | 0 | 1197.24 | 64.91 | 25.68 | 8.10 | 1.454 | 2.22 | -1.71 | -1.12 | -0.258 |
| huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03D_012 | IG003 | 5 | 1.149 | 0 | 1256.35 | 64.07 | 30.95 | 3.89 | 1.393 | 0.97 | 3.94 | -5.29 | -0.320 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-06 | IG001 | 5 | 1.114 | 0 | 1216.22 | 65.43 | 25.67 | 7.61 | 1.451 | 2.44 | -1.14 | -1.89 | -0.256 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-07 | IG002 | 5 | 1.183 | 0 | 1172.97 | 63.38 | 27.45 | 7.80 | 1.451 | 0.10 | 0.53 | -1.29 | -0.240 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-09 | IG005 | 5 | 1.208 | 0 | 1165.85 | 63.28 | 27.30 | 8.03 | 1.434 | -0.39 | 0.96 | -1.17 | -0.227 |
| huayi_sy3sy2_phase3_wgs1__shift_evening_handover_2022-10-07 | IG003 | 5 | 1.285 | 0 | 1171.73 | 63.30 | 27.47 | 7.85 | 1.451 | 0.14 | 0.81 | -1.61 | -0.229 |
| huayi_sy3sy2_phase3_wgs1__plateau_2022-10-07N_012 | IG004 | 5 | 1.112 | 0 | 1168.87 | 63.16 | 27.57 | 7.87 | 1.452 | 0.07 | 0.88 | -1.62 | -0.218 |
| huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_015 | IG006 | 5 | 1.148 | 0 | 1297.47 | 68.24 | 22.64 | 7.94 | 1.376 | 4.98 | -3.99 | -1.42 | -0.325 |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-10 | IG001 | 5 | 1.870 | 0 | 1154.85 | 62.76 | 26.74 | 9.12 | 1.390 | -1.27 | 0.12 | 0.53 | -0.333 |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-11 | IG004 | 5 | 1.618 | 0 | 1134.40 | 61.75 | 27.67 | 9.13 | 1.406 | -2.43 | 1.40 | 0.30 | -0.326 |
| huayi_sy3sy2_phase4_wgs1__plateau_2022-10-10N_006 | IG002 | 5 | 1.694 | 0 | 1153.36 | 62.63 | 26.93 | 9.06 | 1.388 | -1.16 | 0.48 | 0.02 | -0.304 |
| huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11D_007 | IG003 | 5 | 1.285 | 0 | 1159.42 | 62.92 | 26.62 | 9.09 | 1.405 | -1.03 | 0.18 | 0.22 | -0.332 |

Evidence boundary: reproducible pilot fixed-physics diagnostic baseline. Do not interpret as calibrated validation or industrial generalization.
