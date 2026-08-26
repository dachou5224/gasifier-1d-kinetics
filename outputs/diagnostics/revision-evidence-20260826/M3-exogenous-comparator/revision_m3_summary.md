# Revision M3 Exogenous-Only WGS Comparator

- Frozen split consumed: True
- Configuration: `reference`
- Selected train-only candidate: `coal_flow_mean_negative_slope`
- Forbidden contemporaneous outputs excluded: syngas_flow_mean, co_vol_pct, h2_vol_pct, co2_vol_pct, h2_co_ratio, pred_dry_flow_nm3_h
- Null control (cyclic shift 7 of the selected train feature): train objective 2.102621 vs control 2.102712; `control_explains=False` only by 9.1e-5. This is not a material train-fit advantage.
- Test RMSE under the selected exogenous comparator: CO 18.3688, CO2 4.8281, H2 26.3702, H2/CO 0.4110 (n=17 completed, 0 error/timeout). Versus the M2 fixed-physics test RMSE (CO 18.7319, CO2 4.5407) this is a CO decrease and a CO2 increase, the same qualitative trade-off as measurement-conditioned WGS.
- Forward-prediction claim: not supported. A single frozen split plus a near-null train control does not justify an online or pre-prediction claim.

Evidence boundary: revision evidence only; not production control, learned multi-alpha, z_d value, oxidation-channel updating, cross-gasifier generalization, or manuscript claim text
