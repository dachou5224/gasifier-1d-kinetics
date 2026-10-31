# Huayi adaptive gate figure source-data bundle

This round49 bundle extracts figure-ready source data from existing round41-round48 diagnostics only.
It does not run new solvers, train new models, search new hyperparameters, or change channel statuses.

## Generated CSV row counts

- `huayi_adaptive_gate_fig4_metric_matrix.csv`: 192 rows
- `huayi_adaptive_gate_fig4_method_ladder.csv`: 6 rows
- `huayi_adaptive_gate_channel_status.csv`: 8 rows
- `huayi_reaction_family_gate_map.csv`: 8 rows
- `huayi_alpha_similarity_for_fig5.csv`: 9 rows
- `huayi_multialpha_activation_for_fig5.csv`: 21 rows
- `huayi_gate_figure_source_manifest.csv`: 24 rows

## Fig. 4 method labels

WGS alpha_wgs(x_t), direct residual, fixed physics, global WGS, global char oxidation, global multi-alpha

## Channel statuses

audit_only, future_data_needed, locked, provisionally_open

## Not available from existing artifacts

- huayi_reaction_family_gate_map.measured_signal for alpha_char_gasification: not_available_from_existing_artifact because no clean grouped op_conds path exists.
- huayi_adaptive_gate_channel_status z_d evidence: not_available_from_existing_artifact beyond round40 protocol; no validated z_d diagnostic exists.
- production-control readiness, learned kinetic identification, and final temporal generalization: not_available_from_existing_artifact by design.

## Evidence Boundary

This is source-data extraction for a gate-diagnostic case study: source-data extraction for a gate-diagnostic case study only; not new modeling, final temporal generalization, production-control evidence, learned kinetic identification, weak-label coal-quality value, or cross-gasifier generalization.

Forbidden framing: new model performance, final predictive accuracy, production-control readiness, learned WGS kinetic identification, learned multi-alpha success, weak-label coal-quality value, final temporal generalization, or Yulin/slurry/cross-gasifier generalization.
