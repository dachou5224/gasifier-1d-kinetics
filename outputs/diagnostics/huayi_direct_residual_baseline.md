# Huayi direct outlet residual baseline

This artifact fits black-box outlet residual corrections on the frozen Huayi chronological train split only.
Validation and test rows are transform-only; they are not used for fitting or feature scaling.

- Split counts: {'train': 12, 'validation': 6, 'test': 4}
- Duplicate case IDs: False
- Expected 12/6/4 counts match: True
- Validation excluded from fit/preprocessing: True
- Test excluded from fit/preprocessing: True
- Variants: residual_constant_train_mean, residual_linear_operating
- Operating features for linear variant: o_c_ratio, coal_flow_kg_s, o2_flow_kg_s, carrier_gas_flow_kg_s, carrier_to_coal_ratio, pressure_pa, T_in_K

## Model audit

| variant | N | target | train_n | fit_split | preprocessing | status |
|---|---:|---|---:|---|---|---|
| residual_constant_train_mean | 3 | CO2_volpct_dry | 12 | train | train | ok |
| residual_constant_train_mean | 3 | CO_volpct_dry | 12 | train | train | ok |
| residual_constant_train_mean | 3 | H2_CO_ratio | 12 | train | train | ok |
| residual_constant_train_mean | 3 | H2_volpct_dry | 12 | train | train | ok |
| residual_constant_train_mean | 5 | CO2_volpct_dry | 12 | train | train | ok |
| residual_constant_train_mean | 5 | CO_volpct_dry | 12 | train | train | ok |
| residual_constant_train_mean | 5 | H2_CO_ratio | 12 | train | train | ok |
| residual_constant_train_mean | 5 | H2_volpct_dry | 12 | train | train | ok |
| residual_linear_operating | 3 | CO2_volpct_dry | 12 | train | train | exploratory_fit_ok |
| residual_linear_operating | 3 | CO_volpct_dry | 12 | train | train | exploratory_fit_ok |
| residual_linear_operating | 3 | H2_CO_ratio | 12 | train | train | exploratory_fit_ok |
| residual_linear_operating | 3 | H2_volpct_dry | 12 | train | train | exploratory_fit_ok |
| residual_linear_operating | 5 | CO2_volpct_dry | 12 | train | train | exploratory_fit_ok |
| residual_linear_operating | 5 | CO_volpct_dry | 12 | train | train | exploratory_fit_ok |
| residual_linear_operating | 5 | H2_CO_ratio | 12 | train | train | exploratory_fit_ok |
| residual_linear_operating | 5 | H2_volpct_dry | 12 | train | train | exploratory_fit_ok |

## Metrics

| variant | split | N | target | n | fixed RMSE | corrected RMSE | delta | fixed MAE | corrected MAE |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| residual_constant_train_mean | test | 3 | CO2_volpct_dry | 4 | 1.81849 | 1.40451 | -0.413981 | 1.80728 | 1.38996 |
| residual_constant_train_mean | test | 3 | CO_volpct_dry | 4 | 1.62178 | 4.13748 | 2.5157 | 1.57202 | 4.11823 |
| residual_constant_train_mean | test | 3 | H2_CO_ratio | 4 | 0.0105093 | 0.0659299 | 0.0554207 | 0.0102801 | 0.0652256 |
| residual_constant_train_mean | test | 3 | H2_volpct_dry | 4 | 1.02051 | 2.51166 | 1.49115 | 0.916973 | 2.4714 |
| residual_constant_train_mean | train | 3 | CO2_volpct_dry | 12 | 1.59476 | 1.53919 | -0.05557 | 1.40338 | 1.12517 |
| residual_constant_train_mean | train | 3 | CO_volpct_dry | 12 | 3.00321 | 1.5925 | -1.41071 | 2.58151 | 1.39792 |
| residual_constant_train_mean | train | 3 | H2_CO_ratio | 12 | 0.0872238 | 0.0527352 | -0.0344887 | 0.0814051 | 0.0415058 |
| residual_constant_train_mean | train | 3 | H2_volpct_dry | 12 | 4.38831 | 2.78859 | -1.59973 | 4.17063 | 2.11439 |
| residual_constant_train_mean | validation | 3 | CO2_volpct_dry | 6 | 0.538332 | 0.201542 | -0.33679 | 0.507212 | 0.15887 |
| residual_constant_train_mean | validation | 3 | CO_volpct_dry | 6 | 2.0644 | 3.02997 | 0.965579 | 1.58907 | 2.89286 |
| residual_constant_train_mean | validation | 3 | H2_CO_ratio | 6 | 0.047552 | 0.0619544 | 0.0144024 | 0.0271746 | 0.0591567 |
| residual_constant_train_mean | validation | 3 | H2_volpct_dry | 6 | 2.40905 | 2.74535 | 0.336303 | 1.43839 | 2.64331 |
| residual_constant_train_mean | test | 5 | CO2_volpct_dry | 4 | 0.323173 | 1.89654 | 1.57337 | 0.267195 | 1.88781 |
| residual_constant_train_mean | test | 5 | CO_volpct_dry | 4 | 1.57297 | 4.92603 | 3.35306 | 1.4708 | 4.89436 |
| residual_constant_train_mean | test | 5 | H2_CO_ratio | 4 | 0.022008 | 0.0760962 | 0.0540882 | 0.0185229 | 0.0751623 |
| residual_constant_train_mean | test | 5 | H2_volpct_dry | 4 | 0.748194 | 2.80348 | 2.05529 | 0.545902 | 2.7564 |
| residual_constant_train_mean | train | 5 | CO2_volpct_dry | 12 | 2.25177 | 1.56335 | -0.688418 | 1.62062 | 1.14302 |
| residual_constant_train_mean | train | 5 | CO_volpct_dry | 12 | 3.75236 | 1.53604 | -2.21632 | 3.42356 | 1.35904 |
| residual_constant_train_mean | train | 5 | H2_CO_ratio | 12 | 0.0762773 | 0.0510902 | -0.0251872 | 0.07169 | 0.0401061 |
| residual_constant_train_mean | train | 5 | H2_volpct_dry | 12 | 3.5252 | 2.74604 | -0.77916 | 3.35874 | 2.04831 |
| residual_constant_train_mean | validation | 5 | CO2_volpct_dry | 6 | 1.52011 | 0.265496 | -1.25462 | 1.50148 | 0.209299 |
| residual_constant_train_mean | validation | 5 | CO_volpct_dry | 6 | 2.26845 | 2.91487 | 0.64642 | 1.35088 | 2.71842 |
| residual_constant_train_mean | validation | 5 | H2_CO_ratio | 6 | 0.0403151 | 0.0589851 | 0.01867 | 0.0288624 | 0.0555268 |
| residual_constant_train_mean | validation | 5 | H2_volpct_dry | 6 | 1.81915 | 2.59962 | 0.780476 | 1.38522 | 2.47862 |
| residual_linear_operating | test | 3 | CO2_volpct_dry | 4 | 1.81849 | 1.63385 | -0.184642 | 1.80728 | 1.62424 |
| residual_linear_operating | test | 3 | CO_volpct_dry | 4 | 1.62178 | 2.7408 | 1.11902 | 1.57202 | 2.68534 |
| residual_linear_operating | test | 3 | H2_CO_ratio | 4 | 0.0105093 | 0.0387334 | 0.0282241 | 0.0102801 | 0.0361173 |
| residual_linear_operating | test | 3 | H2_volpct_dry | 4 | 1.02051 | 1.2138 | 0.19329 | 0.916973 | 1.02543 |
| residual_linear_operating | train | 3 | CO2_volpct_dry | 12 | 1.59476 | 0.222344 | -1.37241 | 1.40338 | 0.165672 |
| residual_linear_operating | train | 3 | CO_volpct_dry | 12 | 3.00321 | 0.242805 | -2.7604 | 2.58151 | 0.183543 |
| residual_linear_operating | train | 3 | H2_CO_ratio | 12 | 0.0872238 | 0.00284539 | -0.0843785 | 0.0814051 | 0.00215879 |
| residual_linear_operating | train | 3 | H2_volpct_dry | 12 | 4.38831 | 0.158911 | -4.2294 | 4.17063 | 0.130316 |
| residual_linear_operating | validation | 3 | CO2_volpct_dry | 6 | 0.538332 | 0.509827 | -0.0285051 | 0.507212 | 0.426329 |
| residual_linear_operating | validation | 3 | CO_volpct_dry | 6 | 2.0644 | 4.21436 | 2.14996 | 1.58907 | 4.1344 |
| residual_linear_operating | validation | 3 | H2_CO_ratio | 6 | 0.047552 | 0.102535 | 0.0549826 | 0.0271746 | 0.101198 |
| residual_linear_operating | validation | 3 | H2_volpct_dry | 6 | 2.40905 | 4.77319 | 2.36414 | 1.43839 | 4.72355 |
| residual_linear_operating | test | 5 | CO2_volpct_dry | 4 | 0.323173 | 2.203 | 1.87983 | 0.267195 | 2.19588 |
| residual_linear_operating | test | 5 | CO_volpct_dry | 4 | 1.57297 | 3.67581 | 2.10284 | 1.4708 | 3.64028 |
| residual_linear_operating | test | 5 | H2_CO_ratio | 4 | 0.022008 | 0.0506108 | 0.0286028 | 0.0185229 | 0.0487569 |
| residual_linear_operating | test | 5 | H2_volpct_dry | 4 | 0.748194 | 1.54638 | 0.798182 | 0.545902 | 1.41739 |
| residual_linear_operating | train | 5 | CO2_volpct_dry | 12 | 2.25177 | 0.21716 | -2.03461 | 1.62062 | 0.164847 |
| residual_linear_operating | train | 5 | CO_volpct_dry | 12 | 3.75236 | 0.228531 | -3.52383 | 3.42356 | 0.169948 |
| residual_linear_operating | train | 5 | H2_CO_ratio | 12 | 0.0762773 | 0.00280494 | -0.0734724 | 0.07169 | 0.00201363 |
| residual_linear_operating | train | 5 | H2_volpct_dry | 12 | 3.5252 | 0.162071 | -3.36313 | 3.35874 | 0.129476 |
| residual_linear_operating | validation | 5 | CO2_volpct_dry | 6 | 1.52011 | 0.694699 | -0.825414 | 1.50148 | 0.653422 |
| residual_linear_operating | validation | 5 | CO_volpct_dry | 6 | 2.26845 | 3.57724 | 1.30879 | 1.35088 | 3.51223 |
| residual_linear_operating | validation | 5 | H2_CO_ratio | 6 | 0.0403151 | 0.0920505 | 0.0517354 | 0.0288624 | 0.0910155 |
| residual_linear_operating | validation | 5 | H2_volpct_dry | 6 | 1.81915 | 4.37761 | 2.55846 | 1.38522 | 4.34097 |

## Boundary

Evidence boundary: direct outlet residual correction comparator only; black-box outlet patching fit on train split only; not trained physics-guided hybrid, bounded alpha_j result, learned multiplier interpretability, weak-label proof, production control, final temporal generalization, or Yulin/slurry generalization.

This comparator can quantify apparent outlet-error reduction from direct residual patching. It does not identify reaction mechanisms, train alpha_j(x_t), prove temporal generalization, or support production control.
