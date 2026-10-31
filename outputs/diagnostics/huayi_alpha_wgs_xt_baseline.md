# Huayi alpha_wgs(x_t) minimal scaffold

This diagnostic fits a tiny WGS-only bounded alpha scaffold on the frozen train split only.

- Model form: alpha_wgs_ridge_logit
- Feature set: o_c_ratio,coal_flow_kg_s,o2_flow_kg_s,carrier_gas_flow_kg_s,carrier_to_coal_ratio,pressure_pa,T_in_K
- Bounds: [0.02,2.0]
- Selection/training method: finite_candidate_train_only_ridge_logit_scaffold; one nonzero operating-feature slope or constant
- Regularization: 0.001
- Selected candidate: carrier_to_coal_ratio_neg_slope
- Selected nonzero feature: carrier_to_coal_ratio
- Train objective: 2.23998 -> 1.00025
- Validation excluded from training: True
- Test excluded from training: True

## Metrics

| split | target | n | fixed RMSE | alpha RMSE | delta fixed | global WGS RMSE | delta global | alpha median | bound hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | CO2_volpct_dry | 4 | 1.81849 | 2.2586 | 0.44011 | 2.39247 | -0.133867 | 0.26504 | 0 |
| test | CO_volpct_dry | 4 | 1.62178 | 2.63329 | 1.01151 | 2.93574 | -0.30245 | 0.26504 | 0 |
| test | H2_CO_ratio | 4 | 0.0105093 | 0.0165677 | 0.00605845 | 0.0207142 | -0.00414651 | 0.26504 | 0 |
| test | H2_volpct_dry | 4 | 1.02051 | 0.606679 | -0.413826 | 0.543219 | 0.0634602 | 0.26504 | 0 |
| train | CO2_volpct_dry | 12 | 1.59476 | 2.09474 | 0.499977 | 1.72133 | 0.373401 | 0.264714 | 0 |
| train | CO_volpct_dry | 12 | 3.00321 | 3.07783 | 0.0746252 | 2.95452 | 0.123317 | 0.264714 | 0 |
| train | H2_CO_ratio | 12 | 0.0872238 | 0.0829079 | -0.0043159 | 0.083934 | -0.00102602 | 0.264714 | 0 |
| train | H2_volpct_dry | 12 | 4.38831 | 4.17457 | -0.213742 | 4.23019 | -0.055615 | 0.264714 | 0 |
| validation | CO2_volpct_dry | 6 | 0.538332 | 0.605755 | 0.0674229 | 0.581149 | 0.0246066 | 0.275569 | 0 |
| validation | CO_volpct_dry | 6 | 2.0644 | 2.34906 | 0.284661 | 2.68439 | -0.335328 | 0.275569 | 0 |
| validation | H2_CO_ratio | 6 | 0.047552 | 0.0497481 | 0.00219612 | 0.0537128 | -0.00396464 | 0.275569 | 0 |
| validation | H2_volpct_dry | 6 | 2.40905 | 2.45553 | 0.0464827 | 2.61016 | -0.15463 | 0.275569 | 0 |

## Alpha distribution

| split | median | mean | std | status |
|---|---:|---:|---:|---|
| test | 0.26504 | 0.266399 | 0.00386304 | bound_hits=0/4 |
| train | 0.264714 | 0.314492 | 0.122155 | bound_hits=0/12 |
| validation | 0.275569 | 0.267444 | 0.01883 | bound_hits=0/6 |

## Boundary

Evidence boundary: minimal WGS-only bounded alpha_wgs(x_t) scaffold selected on train split only; not full hybrid, multi-alpha calibration, learned multiplier interpretability, weak-label proof, production control, final temporal generalization, or Yulin/slurry generalization.

This scaffold can test whether a low-capacity bounded WGS alpha path is operational. It does not establish full hybrid calibration, learned multiplier interpretability, production control, or temporal generalization.
