# Huayi global WGS alpha baseline

This is a non-learned global scalar comparator. Alpha is selected using train split only.

- Selected alpha: 0.3
- Bounds: [0.02, 2.0]
- Hit lower bound: False
- Hit upper bound: False
- Training objective initial alpha=1.0: 2.23998
- Training objective selected: 1.00036
- Grid points: 13
- Validation excluded from selection: True
- Test excluded from selection: True

## Grid

| alpha | train_objective | status |
|---:|---:|---|
| 0.02 | 3.39211 | ok |
| 0.05 | 2.40602 | ok |
| 0.1 | 1.6122 | ok |
| 0.15 | 1.23957 | ok |
| 0.2 | 1.06899 | ok |
| 0.3 | 1.00036 | ok |
| 0.4 | 1.10315 | ok |
| 0.5 | 1.27236 | ok |
| 0.75 | 1.76749 | ok |
| 1 | 2.23998 | ok |
| 1.25 | 2.65137 | ok |
| 1.5 | 3.00241 | ok |
| 2 | 3.55791 | ok |

## Metrics

# Global WGS alpha split metrics

| model | split | N | target | n | RMSE | MAE | bias | fallback | poor |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| global_wgs_alpha | test | 3 | CO2_volpct_dry | 4 | 2.39247 | 2.38541 | 2.38541 | 0 | 4 |
| global_wgs_alpha | test | 3 | CO_volpct_dry | 4 | 2.93574 | 2.8896 | -2.8896 | 0 | 4 |
| global_wgs_alpha | test | 3 | H2_CO_ratio | 4 | 0.0207142 | 0.0169389 | 0.0169389 | 0 | 4 |
| global_wgs_alpha | test | 3 | H2_volpct_dry | 4 | 0.543219 | 0.513536 | -0.163165 | 0 | 4 |
| global_wgs_alpha | train | 3 | CO2_volpct_dry | 12 | 1.72133 | 1.45905 | 0.3312 | 0 | 12 |
| global_wgs_alpha | train | 3 | CO_volpct_dry | 12 | 2.95452 | 2.53721 | 2.53721 | 0 | 12 |
| global_wgs_alpha | train | 3 | H2_CO_ratio | 12 | 0.083934 | 0.0765852 | -0.0680037 | 0 | 12 |
| global_wgs_alpha | train | 3 | H2_volpct_dry | 12 | 4.23019 | 3.97161 | -3.28927 | 0 | 12 |
| global_wgs_alpha | validation | 3 | CO2_volpct_dry | 6 | 0.581149 | 0.493767 | 0.407489 | 0 | 6 |
| global_wgs_alpha | validation | 3 | CO_volpct_dry | 6 | 2.68439 | 2.13679 | 0.33975 | 0 | 6 |
| global_wgs_alpha | validation | 3 | H2_CO_ratio | 6 | 0.0537128 | 0.0342871 | -0.0216673 | 0 | 6 |
| global_wgs_alpha | validation | 3 | H2_volpct_dry | 6 | 2.61016 | 1.47633 | -1.36158 | 0 | 6 |

Evidence boundary: non-learned global bounded WGS scalar comparator selected on train split only; not alpha_j(x_t), trained hybrid model, learned multiplier interpretability, production control, final temporal generalization, or Yulin/slurry generalization.

Evidence boundary: non-learned global bounded WGS scalar comparator selected on train split only; not alpha_j(x_t), trained hybrid model, learned multiplier interpretability, production control, final temporal generalization, or Yulin/slurry generalization.
