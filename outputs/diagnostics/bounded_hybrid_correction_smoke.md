# Bounded hybrid-correction smoke protocol

This diagnostic uses the existing WGS reaction-family multiplier as a bounded scalar correction smoke.
It intentionally avoids first-cell source-term hooks because those would require new solver/source interfaces.

- Correction: WGS_CatalyticFactor_bounded_reaction_family
- Bounds: [0.02, 2.0]
- Target: weighted_outlet_CO_H2_CO2_squared_error
- Summary rows: 2
- Trajectory rows: 10

| case_id | N | initial | final | loss_initial | loss_final | status | poor_final |
|---|---:|---:|---:|---:|---:|---|---:|
| huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-28 | 3 | 1 | 0.213119 | 1.6805 | 0.720145 | ok_loss_decreased | 1 |
| huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-10 | 3 | 1 | 0.979623 | 0.0420221 | 0.041605 | ok_loss_decreased | 1 |

Protocol decision: first-cell source-term correction remains design-only in this round; the implemented smoke uses an existing reaction-family parameter so public solver APIs and source-term semantics are unchanged.

Evidence boundary: bounded fixed-physics correction smoke protocol only; not a trained hybrid model, calibrated accuracy, production control recommendation, learned multiplier interpretability claim, causal proof, or Yulin/slurry generalization.
