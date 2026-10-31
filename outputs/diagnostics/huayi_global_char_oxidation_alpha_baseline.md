# Huayi global char-oxidation alpha baseline

This is a non-learned global scalar comparator. Alpha is selected using train split only.

## Parameter-path audit

- Mapped parameter: `op_conds.CharCombustionRateFactor`
- Mainline path: Cell._calc_rates -> KineticsService.calc_heterogeneous_rates(char_combustion_factor=...)
- JAX contract path: GasifierSystem passes CharCombustionRateFactor to reactor_solve_v4 hot path
- Per-run injection without public API change: True
- JAX/parity caveat: round43 selection/evaluation uses existing NumPy/minimize diagnostic path; JAX parity is parameter-wired but not rerun as a char-alpha training baseline here

## Selection

- Selected alpha: 0.1
- Bounds: [0.05, 1.0]
- Hit lower bound: False
- Hit upper bound: False
- Training objective initial alpha=1.0: 1.15924
- Training objective selected: 1.01118
- Training objective delta: -0.148061
- Validation excluded from selection: True
- Test excluded from selection: True

## Grid

| alpha | train_objective | fallback_sum | poor_sum | cell0_min | cell0_max | status |
|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 1.01348 | 0 | 12 | 0.0056383 | 0.0190228 | ok |
| 0.1 | 1.01118 | 0 | 12 | 0.00564357 | 0.0181519 | ok |
| 0.2 | 1.03902 | 0 | 12 | 0.00561705 | 0.0173226 | ok |
| 0.35 | 1.07826 | 0 | 12 | 0.00557483 | 0.0168866 | ok |
| 0.5 | 1.10215 | 0 | 12 | 0.00554252 | 0.01668 | ok |
| 0.7 | 1.12882 | 0 | 12 | 0.0055112 | 0.0165178 | ok |
| 0.85 | 1.14607 | 0 | 12 | 0.00549359 | 0.0164382 | ok |
| 1 | 1.15924 | 0 | 12 | 0.00547943 | 0.0163789 | ok |

## Metrics

| model | split | N | target | n | RMSE | MAE | bias | rmse_delta_vs_fixed | fallback | poor |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| global_char_oxidation_alpha | test | 3 | CO2_volpct_dry | 4 | 2.50462 | 2.49453 | 2.49453 | 0.686127 | 0 | 4 |
| global_char_oxidation_alpha | test | 3 | CO_volpct_dry | 4 | 2.65223 | 2.62896 | -2.62896 | 1.03045 | 0 | 4 |
| global_char_oxidation_alpha | test | 3 | H2_CO_ratio | 4 | 0.0127988 | 0.00853647 | 0.00853647 | 0.00228956 | 0 | 4 |
| global_char_oxidation_alpha | test | 3 | H2_volpct_dry | 4 | 0.723025 | 0.654688 | -0.564358 | -0.29748 | 0 | 4 |
| global_char_oxidation_alpha | train | 3 | CO2_volpct_dry | 12 | 1.80058 | 1.67044 | 0.669646 | 0.205823 | 0 | 12 |
| global_char_oxidation_alpha | train | 3 | CO_volpct_dry | 12 | 2.70005 | 2.25917 | 2.24366 | -0.303153 | 0 | 12 |
| global_char_oxidation_alpha | train | 3 | H2_CO_ratio | 12 | 0.0846667 | 0.0786444 | -0.0671201 | -0.00255716 | 0 | 12 |
| global_char_oxidation_alpha | train | 3 | H2_volpct_dry | 12 | 4.34753 | 4.12497 | -3.34524 | -0.0407822 | 0 | 12 |
| global_char_oxidation_alpha | validation | 3 | CO2_volpct_dry | 6 | 0.897971 | 0.856526 | 0.856526 | 0.359638 | 0 | 6 |
| global_char_oxidation_alpha | validation | 3 | CO_volpct_dry | 6 | 2.46273 | 2.06151 | 0.0125588 | 0.398334 | 0 | 6 |
| global_char_oxidation_alpha | validation | 3 | H2_CO_ratio | 6 | 0.0527208 | 0.0325963 | -0.0223376 | 0.00516875 | 0 | 6 |
| global_char_oxidation_alpha | validation | 3 | H2_volpct_dry | 6 | 2.66798 | 1.52194 | -1.52194 | 0.258932 | 0 | 6 |

## Boundary

Evidence boundary: non-learned global char-oxidation scalar comparator selected on train split only through op_conds.CharCombustionRateFactor; not trained hybrid, alpha_j(x_t), learned multiplier interpretability, weak-label proof, production control, final temporal generalization, or Yulin/slurry generalization.

This comparator can indicate whether the existing char-oxidation severity knob is worth staged learned-alpha exploration. It does not train alpha_j(x_t), identify learned mechanisms, prove temporal generalization, or support production control.
