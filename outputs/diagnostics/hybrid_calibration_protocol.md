# Hybrid calibration protocol freeze

- Protocol version: `round40_v1`
- Claim boundary: protocol freeze only; not a trained hybrid model, calibrated accuracy, production control, learned multiplier interpretability, weak-label coal-quality proof, temporal generalization result, or Yulin/slurry generalization

## Readiness Audit

- Huayi READY ModelCases: 22/22
- Phase counts: {'phase1': 8, 'phase2': 4, 'phase3': 6, 'phase4': 4}
- Baseline rows: 44 with N_cells=[3, 5]
- Available target columns: target_CO_vol_pct, target_H2_vol_pct, target_CO2_vol_pct, target_N2_vol_pct, target_dry_flow_kmol_s, target_H2_CO_ratio
- Targets sufficient for CO/H2/CO2/H2CO: True
- Fixed-physics fallback distribution: {0: 44}
- Fixed-physics poor-convergence distribution: {1: 44}
- Cell0 cost range: [0.0055878141771886, 0.0184163829937229]
- Bounded WGS batch: 22/22 loss decreased; final theta range [0.1390586763362116, 1.1950335887455406]

### Split Blockers

- Current paper-facing set has only 22 representative ModelCases; adequate for protocol smoke and leave-future-out diagnostics, too small for a high-capacity trained hybrid.
- Representative rows span four Huayi phases but are not an exhaustive steady-window training set.
- Sparse LIMS/coal labels require timestamp alignment rules and train-only preprocessing before they can be used as z_d weak labels.
- Yulin/slurry data are out of scope for this protocol unless a separate ModelCase contract is built.

## Candidate Alpha Groups

| alpha | existing mapping | bounds | train without public API change | sensitivity | identifiability risk |
|---|---|---:|---|---|---|
| alpha_wgs | op_conds.WGS_CatalyticFactor | [0.02, 2.0] | True | CO, H2, CO2, H2/CO strongest; moderate outlet T through reaction heat. | High with O/C, steam/water availability, CO2 carrier, and coal-quality weak labels; must report alpha distribution and bound hits. |
| alpha_char_oxidation | op_conds.CharCombustionRateFactor | [0.05, 1.0] | True | outlet T, CO, CO2, dry flow; can interact strongly with O/C response. | Very high with O/C ratio and first-cell initialization; should be introduced after WGS-only baseline and monitored with cell0 cost. |
| alpha_volatile_oxidation | op_conds.CO_OxidationFactor and op_conds.H2_OxidationFactor | [0.2, 1.2] | True | H2, CO, CO2, H2/CO; possibly T through oxidation heat. | High with WGS and char oxidation; should use staged ablation and bound-hit reporting. |
| alpha_char_gasification | none as a clean grouped op_conds alpha | [0.1, 5.0] | False | CO, H2, dry syngas flow, carbon sanity; weaker direct effect on early O2 zone. | Very high with coal reactivity, particle diameter, residence time, and carbon-balance incompleteness. |

## Split Policy

- Minimum dataset: Use the larger Huayi steady-window ModelCase set once available; 22 representative rows are for smoke/protocol and leave-future-out diagnostics only.
- Chronological 22-case diagnostic split: {'train': 'phase1 + phase2 representative cases (12 rows)', 'validation': 'phase3 representative cases (6 rows)', 'test': 'phase4 representative cases (4 rows)', 'rationale': 'preserves future holdout by campaign phase/date; intentionally small and diagnostic-only.'}
- Leave-future-out: For full steady-window set, train on earliest contiguous blocks, tune on the next chronological block, and report the latest block once. No random shuffling before split.
- Leave-regime-out: Not robust with only 22 representatives. Feasible only after defining regimes by O/C/load/carrier/phase and ensuring at least one complete future-like regime is held out.

### LIMS Weak-Label Leakage Policy

- Align LIMS/coal-quality labels by timestamp using only labels available at or before the ModelCase timestamp.
- No interpolation or smoothing may use future labels relative to validation/test rows.
- Fit normalization, PCA/latent encoders, imputation medians, and lag windows on train blocks only.
- Include LIMS shuffle, delay, and dropout controls before claiming weak-label value.

## Baseline Matrix

| baseline | effort | runtime | evidence boundary |
|---|---|---|---|
| fixed_physics | existing | minutes for 22 cases at N=3/5; larger steady-window set may be hours | diagnostic baseline only, not calibrated accuracy |
| direct_outlet_residual_correction | small | low; no inner differentiable solver training | statistical residual baseline, not mechanistic alpha_j evidence |
| direct_kinetic_parameter_correction | medium | medium; repeated solver calls per candidate alpha | bounded global calibration baseline, not learned hybrid interpretability |
| bounded_alpha_xt | medium-high | high if trained through solver; start with small N_cells and cached fixed branches | hybrid readiness until validated on future holdout |
| bounded_alpha_xt_zd | high | high; do only after x_t baseline is stable | no weak-label value claim unless controls beat x_t-only baseline on future holdout |
| time_embedding_only_control | small-medium | low-medium | detects temporal memorization; not physics correction |
| lims_shuffle_delay_dropout_controls | medium | medium; repeats alpha training for controls | required before any weak-label coal-quality claim |

## Metric Schema

- Required targets: CO_volpct_dry, H2_volpct_dry, CO2_volpct_dry, H2_CO_ratio
- Optional targets: T_out_degC, dry_syngas_flow_kmol_s, N2_volpct_dry
- Per-target metrics: RMSE, MAE, bias, n
- Diagnostic fields: case_id, phase, dataset_id, split, N_cells, fallback_count, poor_convergence_count, poor_convergence_cell_index, cell0_cost, cell0_max_abs_residual, carbon_sanity_status, gross_apparent_gas_phase_C_recovery, bound_hit_count, bound_hit_fraction, alpha_min, alpha_median, alpha_max, alpha_p05, alpha_p95, status, evidence_boundary

## Runtime Risks

- Training through the implicit solver may be slow; start with N_cells=3 smoke and one alpha before multi-alpha/full N_cells.
- Persistent cell0 residual-quality flag can dominate diagnostics; report it separately and do not hide it by threshold changes.
- Alpha groups are correlated with O/C, load, carrier ratio, and coal weak labels; staged ablations are mandatory.
- Bound hits may mimic performance gains; report alpha distribution and hit frequency by split/phase.
- The 22 representative rows are too small for high-capacity learned alpha networks; use larger steady-window set before accuracy claims.
- LIMS labels are sparse and temporally delayed; future leakage is the highest risk for z_d experiments.
- JAX/IFT coverage is partial; full production training may need fallback to fixed-branch probes or careful custom VJP expansion.

## Recommended Next Model Round

- Title: Implement split manifest + fixed-physics and global bounded-alpha baseline scaffolding

Tasks:
- Create a chronological split manifest for Huayi ModelCases without random shuffling.
- Implement metric schema writer using existing fixed-physics outputs.
- Run WGS-only global bounded alpha baseline on train/validation/test split as a non-learned calibration baseline.
- Add leakage-guard tests before any alpha(x_t,z_d) experiment.

Do not start yet:
- MLP training on all candidate alphas
- weak-label coal latent claims
- production-control framing

Evidence boundary: protocol freeze only; not a trained hybrid model, calibrated accuracy, production control, learned multiplier interpretability, weak-label coal-quality proof, temporal generalization result, or Yulin/slurry generalization.
