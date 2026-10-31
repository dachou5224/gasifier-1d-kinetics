# Bounded hybrid-correction smoke protocol

This diagnostic uses the existing WGS reaction-family multiplier as a bounded scalar correction smoke.
It intentionally avoids first-cell source-term hooks because those would require new solver/source interfaces.

- Correction: WGS_CatalyticFactor_bounded_reaction_family
- Bounds: [0.02, 2.0]
- Target: weighted_outlet_CO_H2_CO2_squared_error
- Summary rows: 22
- Trajectory rows: 110
- Batch recommendation: supplementary_feasibility_figure_candidate
- Loss-decreased cases: 22/22
- Bound hits: lower=0, upper=0

## Batch summary

| scope | phase | n_cases | n_loss_decreased | median_loss_delta | median_relative_loss_delta | lower_hits | upper_hits | recommendation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| all | all | 22 | 22 | -0.292473 | -0.617916 | 0 | 0 | supplementary_feasibility_figure_candidate |
| phase | phase1 | 8 | 8 | -0.510348 | -0.589415 | 0 | 0 | supplementary_feasibility_figure_candidate |
| phase | phase2 | 4 | 4 | -0.292473 | -0.607899 | 0 | 0 | supplementary_feasibility_figure_candidate |
| phase | phase3 | 6 | 6 | -0.157557 | -0.911876 | 0 | 0 | supplementary_feasibility_figure_candidate |
| phase | phase4 | 4 | 4 | -0.00256572 | -0.078864 | 0 | 0 | supplementary_feasibility_figure_candidate |

## Case rows

| case_id | N | initial | final | loss_initial | loss_final | status | poor_final |
|---|---:|---:|---:|---:|---:|---|---:|
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-28 | 3 | 1 | 0.213122 | 1.6805 | 0.720146 | ok_loss_decreased | 1 |
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-29 | 3 | 1 | 0.245834 | 1.06812 | 0.443693 | ok_loss_decreased | 1 |
| huayi_sy3_phase1_wgs1__shift_evening_handover_2022-09-28 | 3 | 1 | 0.26239 | 1.13182 | 0.451266 | ok_loss_decreased | 1 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29D_010 | 3 | 1 | 0.294324 | 0.765897 | 0.276606 | ok_loss_decreased | 1 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29D_011 | 3 | 1 | 0.307499 | 0.734919 | 0.263713 | ok_loss_decreased | 1 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29N_012 | 3 | 1 | 0.35405 | 0.683118 | 0.277192 | ok_loss_decreased | 1 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-29N_014 | 3 | 1 | 0.59599 | 0.306498 | 0.147732 | ok_loss_decreased | 1 |
| huayi_sy3_phase1_wgs1__plateau_2022-09-30D_017 | 3 | 1 | 0.26492 | 0.923754 | 0.392349 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase2_wgs1__shift_morning_handover_2022-10-02 | 3 | 1 | 0.382152 | 0.479835 | 0.103329 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase2_wgs1__shift_morning_handover_2022-10-04 | 3 | 1 | 0.552771 | 0.48346 | 0.275021 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase2_wgs1__plateau_2022-10-02N_010 | 3 | 1 | 0.335463 | 0.53417 | 0.0965332 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03D_012 | 3 | 1 | 0.61409 | 0.455419 | 0.303604 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-06 | 3 | 1 | 0.20789 | 0.73642 | 0.060571 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-07 | 3 | 1 | 0.592367 | 0.179392 | 0.0164508 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase3_wgs1__shift_morning_handover_2022-10-09 | 3 | 1 | 0.663076 | 0.122725 | 0.0113734 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase3_wgs1__shift_evening_handover_2022-10-07 | 3 | 1 | 0.606017 | 0.165475 | 0.0133016 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase3_wgs1__plateau_2022-10-07N_012 | 3 | 1 | 0.624967 | 0.150614 | 0.0127336 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_015 | 3 | 1 | 0.139059 | 1.96105 | 0.448491 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-10 | 3 | 1 | 0.979625 | 0.042022 | 0.0416046 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-11 | 3 | 1 | 1.19503 | 0.0581229 | 0.0212415 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase4_wgs1__plateau_2022-10-10N_006 | 3 | 1 | 0.965349 | 0.0222323 | 0.0210441 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11D_007 | 3 | 1 | 0.936616 | 0.0378122 | 0.0338689 | ok_loss_decreased | 1 |

Protocol decision: first-cell source-term correction remains design-only in this round; the implemented smoke uses an existing reaction-family parameter so public solver APIs and source-term semantics are unchanged.

Evidence boundary: bounded fixed-physics correction smoke protocol only; not a trained hybrid model, calibrated accuracy, production control recommendation, learned multiplier interpretability claim, causal proof, or Yulin/slurry generalization.
