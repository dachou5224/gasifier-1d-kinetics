# Revision M4 Local Estimability

- Scaling: target scales {'CO_volpct_dry': 10.0, 'H2_volpct_dry': 10.0, 'CO2_volpct_dry': 10.0, 'H2_CO_ratio': 0.1}; parameter scale = declared perturbation
- Rank tolerance: relative singular value > 0.001
- Configuration: `reference`
- Cosine similarity is not used as an identifiability decision.

Channel evidence:
              alpha_name                                evidence_field
               alpha_wgs                      insufficiently_supported
    alpha_char_oxidation                      insufficiently_supported
alpha_volatile_oxidation locally_estimable_under_selected_measurements
 alpha_char_gasification                      insufficiently_supported

Evidence boundary: revision evidence only; not production control, learned multi-alpha, z_d value, oxidation-channel updating, cross-gasifier generalization, or manuscript claim text
