# M10 Direct residual comparator on R52-C-75

Locked post-outlet residual patch on the same 81/44/17 protocol as Table 4.
This is a black-box comparator / alarm, not a physical update and not a reactor source term.

- Dataset: `R52-C-75`
- Split counts: {'train': 81, 'validation': 44, 'test': 17}
- N_cells: 1 (copied from Table 4 fixed physics; not the 22-case N=3 path)
- Primary method: `residual_constant_train_mean`
- Exploratory method: `residual_linear_operating` (not used to choose the primary form)
- Fit split: train only; validation/test excluded from the residual mean and from linear scaling
- Residual definition: `y_residual = y_fixed - mean_train(y_fixed - y_plant)`
- Source commit: `3c1d1a2cf8268de5ca94e6dd878a2834205bee7e`

## Train-only constant corrections

- `CO_dry_vol_pct`: 19.4595
- `CO2_dry_vol_pct`: 5.74455
- `H2_dry_vol_pct`: -27.6505
- `H2_CO_ratio`: -0.441803
- `dry_syngas_flow`: -72924.2

## Test n=17 vs Table 4 fixed physics

| target | fixed RMSE | residual RMSE | ΔRMSE | % | n_completed |
|---|---:|---:|---:|---:|---:|
| CO_dry_vol_pct | 18.7319 | 0.856361 | -17.8756 | -95.43 | 17 |
| CO2_dry_vol_pct | 4.54069 | 1.25775 | -3.28293 | -72.3 | 17 |
| H2_dry_vol_pct | 26.3702 | 1.2892 | -25.081 | -95.11 | 17 |
| H2_CO_ratio | 0.410997 | 0.0310018 | -0.379995 | -92.46 | 17 |

## Paper action

Residual RMSE falls for both test CO and CO2. Keep the method **locked** (comparator/alarm). Do not promote to audit-supported: there is still no reactor route. The apparent test-set collapse of RMSE is a large additive bias correction on the N=1 Huayi path. Validation CO2 RMSE moves from 1.137 to 4.681 under the same constant patch, so the offset does not travel uniformly. The exploratory linear patch was not used for selection; on test it changes CO2 RMSE by 0.505. First-cell residual_quality_status is copied from fixed-physics rows and is not hidden.

## Boundary

locked post-outlet residual comparator on R52-C-75; not a physical update, not a reactor source term, not eligible for monitored evaluation, not learned WGS kinetics

Do not read this package as learned kinetics, WGS identification, N=5 IFT, oxidation-channel updating, or production control.
