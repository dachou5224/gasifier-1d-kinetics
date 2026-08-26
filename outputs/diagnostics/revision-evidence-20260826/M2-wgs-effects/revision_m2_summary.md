# Revision M2 WGS Effect Sizes

- Source commit: `a2955d2a2f6e850b88d8b206f465d810e1ad86ab`
- Seed: 20260826
- Bootstrap replicates: 2000
- Gate recommendation: `audit_supported_candidate`

- test `global_bounded_wgs_scalar` `CO_dry_vol_pct`: RMSE 18.4379 vs fixed 18.7319; delta_rmse=-0.294064 (-1.57%); paired block-bootstrap 95% CI [-0.306871, -0.281159].
- test `global_bounded_wgs_scalar` `CO2_dry_vol_pct`: RMSE 4.77237 vs fixed 4.54069; delta_rmse=0.231683 (+5.1%); paired block-bootstrap 95% CI [0.222816, 0.239963].
- test `low_capacity_alpha_wgs_xt` `CO_dry_vol_pct`: RMSE 18.3933 vs fixed 18.7319; delta_rmse=-0.338671 (-1.81%); paired block-bootstrap 95% CI [-0.358006, -0.319352].
- test `low_capacity_alpha_wgs_xt` `CO2_dry_vol_pct`: RMSE 4.80839 vs fixed 4.54069; delta_rmse=0.267701 (+5.9%); paired block-bootstrap 95% CI [0.253670, 0.280945].

Evidence boundary: revision evidence only; not production control, learned multi-alpha, z_d value, oxidation-channel updating, cross-gasifier generalization, or manuscript claim text
