ROUND55_FIXED_BASELINE_DONE
Decision: FIXED_BASELINE_READY
Frozen split consumed: True; R52-C-75 row IDs and roles preserved
Rows attempted/completed: 142/142
Train/validation/test completed: 81/44/17
Runtime summary: total_wall_time_s=494.297; median_row_time_s=3.300; row_timeout_s=15.0
Metric summary: test:CO2_dry_vol_pct:rmse=4.54069,n=17/17; test:CO_dry_vol_pct:rmse=18.7319,n=17/17; test:H2_CO_ratio:rmse=0.410997,n=17/17; test:H2_dry_vol_pct:rmse=26.3702,n=17/17; test:dry_syngas_flow:rmse=0,n=17/17; train:CO2_dry_vol_pct:rmse=6.45109,n=81/81; train:CO_dry_vol_pct:rmse=19.5943,n=81/81; train:H2_CO_ratio:rmse=0.442005,n=81/81; train:H2_dry_vol_pct:rmse=27.6551,n=81/81; train:dry_syngas_flow:rmse=0,n=81/81; validation:CO2_dry_vol_pct:rmse=1.13657,n=44/44; validation:CO_dry_vol_pct:rmse=22.9144,n=44/44; validation:H2_CO_ratio:rmse=0.421353,n=44/44; validation:H2_dry_vol_pct:rmse=26.6863,n=44/44; validation:dry_syngas_flow:rmse=0,n=44/44
Failures/timeouts: timeout=0; error=0
Cell-0/fallback audit: fallback_count_sum=0; poor_convergence_count_sum=142; cell0_cost_min=0.005218903479905665; cell0_cost_max=0.015841905685236544
WGS comparator readiness: ready: full fixed-physics baseline completed; WGS comparators can be dispatched using the same timeout/cache harness
Created files: huayi_round55_fixed_physics_predictions.csv, huayi_round55_fixed_physics_metric_matrix.csv, huayi_round55_runtime_and_convergence.csv, huayi_round55_failure_audit.md, huayi_round55_wgs_comparator_readiness.md, huayi_round55_claim_boundary.md, huayi_round55_result.md
Tests/checks run: frozen split row-ID intake check; full 142-row fixed-physics timeout/cache run; row-level cache write/read; split-target metric matrix generated; CSV schema/readback checks; git diff --check scoped to Round55 files
Known limitations: fixed-physics uses Round54 N_cells=1 policy and converter CO2 carrier engineering ratio; no WGS/direct residual/adaptive comparators were run.
Allowed next task: Dispatch Round56 global bounded WGS scalar and low-capacity WGS comparator using the frozen split and unchanged metrics.
Forbidden claims: no WGS scalar fitting; no low-capacity alpha_wgs(x_t); no direct residual fitting; no learned multi-alpha; no z_d; no Arrhenius tuning; no window, split, target, or metric changes; no predictive-improvement claim
