# Huayi expanded corrected ModelCase diagnostic audit

This artifact audits the Huayi four-phase corrected-contract preview as a paper-facing fixed-physics diagnostic dataset candidate.

- Manifest candidate segments: 59
- Manifest independence groups: 22
- Manifest READY representative ModelCases: 22
- Preview contract audit passed: True
- Ready rows: 22/22
- Phases: phase1=8, phase2=4, phase3=6, phase4=4
- N_cells baseline coverage: 3, 5
- Adapter uses primary corrected coal flow: 22/22
- Adapter avoids C-F monitoring sum: 22/22
- Steam not-applicable and zero adapter feed: 22/22
- Feature-version provenance drift rows: 0/22

## Provenance note

The four-phase previews now expose phase-specific `feature_version` values. The model adapter consumes the numeric coal-property columns directly; feature-version matching is recorded here as provenance protection.

## Phase4 caveat

Phase4 is included as expanded dry-powder diagnostic coverage. It is not sufficient by itself for generalization claims or unweighted train/validation interpretation.

## Numeric summary

- fallback_count distribution: {0: 44}
- poor_convergence_count distribution: {1: 44}
- cell0_cost range: 0.00558781 to 0.0184164
- T_out_degC range: 1134.398 to 1418.374
- delta_CO_vol_pct range: -2.426 to 6.279
- delta_H2_vol_pct range: -6.561 to 3.937
- delta_CO2_vol_pct range: -5.289 to 2.117
- delta_dry_flow_kmol_s range: -0.464882 to -0.183355
- apparent gross gas-phase C recovery range: 0.833162 to 1.045911
- C-F-misuse gross recovery range: 0.696049 to 0.801708

Evidence boundary: multi-case dry-powder fixed-physics diagnostic coverage only. This is not calibrated accuracy, carbon-closure proof, hybrid validation, or Yulin/slurry generalization.
