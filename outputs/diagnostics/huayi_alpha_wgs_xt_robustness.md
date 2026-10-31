# Huayi alpha_wgs(x_t) robustness and identifiability stress audit

This audit reuses the round44 WGS-only finite-candidate bounded-alpha scaffold.

## Regularization sweep

| regularization | selected candidate | held-out improved vs fixed | held-out improved vs global WGS | held-out mean delta vs fixed | held-out mean delta vs global |
|---:|---|---:|---:|---:|---:|
| 0 | carrier_to_coal_ratio_neg_slope | 1/8 | 6/8 | 0.180574 | -0.105793 |
| 0.0001 | carrier_to_coal_ratio_neg_slope | 1/8 | 6/8 | 0.180574 | -0.105793 |
| 0.001 | carrier_to_coal_ratio_neg_slope | 1/8 | 6/8 | 0.180574 | -0.105793 |
| 0.01 | carrier_to_coal_ratio_neg_slope | 1/8 | 6/8 | 0.180574 | -0.105793 |
| 0.1 | carrier_to_coal_ratio_neg_slope | 1/8 | 6/8 | 0.180574 | -0.105793 |

## Feature stability

| rank | candidate | feature | train objective | selection objective |
|---:|---|---|---:|---:|
| 1 | carrier_to_coal_ratio_neg_slope | carrier_to_coal_ratio | 1.00025 | 1.00074 |
| 2 | constant_alpha_0.3 | none | 1.00036 | 1.00085 |
| 3 | carrier_gas_flow_kg_s_neg_slope | carrier_gas_flow_kg_s | 1.0027 | 1.00318 |
| 4 | o_c_ratio_neg_slope | o_c_ratio | 1.00923 | 1.00972 |
| 5 | coal_flow_kg_s_pos_slope | coal_flow_kg_s | 1.0347 | 1.03518 |

Leave-one-feature-out selected candidates:
- drop_o_c_ratio: carrier_to_coal_ratio_neg_slope
- drop_coal_flow_kg_s: carrier_to_coal_ratio_neg_slope
- drop_o2_flow_kg_s: carrier_to_coal_ratio_neg_slope
- drop_carrier_gas_flow_kg_s: carrier_to_coal_ratio_neg_slope
- drop_carrier_to_coal_ratio: constant_alpha_0.3
- drop_pressure_pa: carrier_to_coal_ratio_neg_slope
- drop_T_in_K: carrier_to_coal_ratio_neg_slope

## Null control

- Null kind: deterministic_train_feature_permutation_carrier_to_coal_ratio
- Selected candidate under null: constant_alpha_0.3
- Train objective under null: 1.00036

## WGS sensitivity proxy

| target | positive | negative | zero | median fd gradient |
|---|---:|---:|---:|---:|
| CO2_volpct_dry | 5 | 17 | 0 | -1.59162 |
| CO_volpct_dry | 16 | 6 | 0 | 2.02868 |
| H2_CO_ratio | 7 | 15 | 0 | -0.0190966 |
| H2_volpct_dry | 9 | 13 | 0 | -0.316322 |

## Boundary

Evidence boundary: robustness and identifiability stress audit for minimal WGS-only alpha_wgs(x_t) scaffold; not full hybrid training, multi-alpha calibration, weak-label coal-quality modeling, learned multiplier interpretability proof, production control, or final temporal generalization evidence.

This audit is designed to stress stability and identifiability of a minimal scaffold. It does not establish learned multiplier interpretability, final temporal generalization, or production control.
