# Revision Evidence Boundary — 2026-08-26

Dataset: `R52-C-75` (142 windows = 81 train / 44 validation / 17 test). Frozen chronological split consumed. Generation source commit: `a2955d2a2f6e850b88d8b206f465d810e1ad86ab`. Seed: `20260826` (bootstrap and null shift only; not used to select variants, features, or thresholds).

This package is revision evidence only. It is not production control, learned WGS identification, learned multi-alpha success, a z_d value, oxidation-channel updating, cross-gasifier generalization, or manuscript claim text.

## 1. Did cell-0 treatment change output metrics, selected alpha/feature or gate status?

No. Screened variants (`reference`, `refined_first_cv`, `alt_cell0_init`, `tighter_tol`) all kept `poor_flag_rate=1.0` on the nine representative windows. Outlet error was not used for selection. The selected configuration is `reference` (N=1, current minimize path). Gate status: `stable`. Train-only reselection still chose global `alpha_wgs=0.05` and `syngas_flow_mean_negative_slope`. Full-split fixed-physics: 142/142 `execution_status=completed`, 0 error/timeout; residual quality remains `poor` on every window. `numerical_quality_unresolved=True`.

## 2. What are the exact test-set CO and CO2 RMSE changes and paired intervals?

Primary block length 8 from train residual ACF (lags 1–8 all above 1/e; neighbors 7 and 9 reported). 2000 paired moving-block bootstrap replicates.

Global bounded WGS vs fixed physics, test, n=17:

- CO: RMSE 18.4379 vs 18.7319; ΔRMSE = −0.294064 (−1.57%); 95% CI [−0.306871, −0.281159]
- CO2: RMSE 4.77237 vs 4.54069; ΔRMSE = +0.231683 (+5.10%); 95% CI [0.222816, 0.239963]

Measurement-conditioned `alpha_wgs(x_t)` vs fixed physics, test:

- CO: RMSE 18.3933 vs 18.7319; ΔRMSE = −0.338671 (−1.81%); 95% CI [−0.358006, −0.319352]
- CO2: RMSE 4.80839 vs 4.54069; ΔRMSE = +0.267701 (+5.90%); 95% CI [0.253670, 0.280945]

H2 and H2/CO deltas are numerically ~0. Gate recommendation: `audit_supported_candidate` (CO improvement coexists with CO2 deterioration). Dry syngas flow remains a supplementary metadata diagnostic.

## 3. Does the exogenous-only comparator add any defensible forward-prediction evidence?

No. Allowed features were declared before fitting: `o_c_ratio`, `coal_flow_mean`, `carrier_to_coal_ratio`, `pressure_mpa`. Outlet composition, dry/wet syngas flow and `syngas_flow_mean` were excluded. Train-only selection chose `coal_flow_mean_negative_slope`. The registered cyclic-shift-7 null is essentially tied on train (objective 2.102621 vs 2.102712). A single-split CO/CO2 trade-off that is not distinguished from this null does not support a forward-prediction or online-operation claim.

## 4. Which channels are locally estimable under the declared measurement/scaling design?

Scaled sensitivity uses target scales {CO, H2, CO2 vol% dry: 10; H2/CO: 0.1} and the declared per-channel perturbation. Rank tolerance: relative singular value > 0.001. Cosine similarity is not used as an identifiability decision. No channel is labelled `identifiable=true`.

| channel | evidence_field |
|---|---|
| `alpha_wgs` | `insufficiently_supported` (mean abs scaled sensitivity ~1.8e-4) |
| `alpha_char_oxidation` | `insufficiently_supported` (local FD ~0 on the N=1 Huayi path; profile flat above ~0.1) |
| `alpha_volatile_oxidation` | `locally_estimable_under_selected_measurements` |
| `alpha_char_gasification` | `insufficiently_supported` (design-only; no clean grouped path) |

Pooled numerical rank is 2 of 3 with `condition_number=inf` because the third singular value is 0. Profiles are diagnostics, not confidence intervals.

## 5. Which N=5 derivative pairs pass or fail parity?

36 pairs = 4 phases × {WGS_CatalyticFactor, oxygen_flow, coal_mass_flow} × {CO, H2, CO2 mol/s}. Predeclared pass: rel ≤ 1e-4 or abs ≤ 1e-4; near-zero |grad|<1e-8 uses abs ≤ 1e-6. `JaxSeedCell0FromMinimize=False`.

- Pass: 35/36
- Fail: 1/36 — phase 1 `coal_mass_flow → CO_mol_s` (rel_error=1.33e-4, abs_error=6.91e-4)

All four N=5 JAX solves have `residual_max_abs` in 6.15–20.06. This is selected-path FD vs JAX/jacfwd consistency, not a full-solver differentiability proof. The failed pair remains visible and blocks a blanket trust-layer claim.

## 6. Are all registered tests clean?

Yes. Data-preparation registered paper-facing suite (2026-07-16 lock, seven files): **40 passed in 23.07 s**. No xfail, skip, or narrowed selection. The stale moisture assertion was updated to match production `config/coal_assumption_v0.yaml` (`used_in_v0: true` → `assumed_lims`). No production contract or generated audit hashes were changed.

## 7. Which manuscript claims must be weakened regardless of outcome?

Weaken or remove, even after this package:

1. Language that treats the Huayi N=1 baseline as a fully converged first-cell solve.
2. Any forward-prediction or online-operation claim that uses contemporaneous `syngas_flow_mean` or outlet composition as a feature.
3. Identifiability or observability claims based only on cosine similarity, a finite condition number, or non-collinearity.
4. Statements that bounded WGS is generally better than fixed physics; report the CO/CO2 trade-off with the intervals above.
5. Learned WGS kinetics, learned multi-alpha success, oxidation-channel updating, z_d, production closed-loop control, or cross-gasifier generalization.
6. Full-solver differentiability or IFT proof from the N=5 probe; keep “selected-path gradient consistency”, and do not hide the failed pair or the large N=5 residuals.
7. Advertising a known 39/40 data-prep test state (now 40/40 after the stale-assertion fix).
8. Local estimability of `alpha_wgs` or `alpha_char_oxidation` under the selected dry-gas measurements at N=1.

Allowed descriptive statements: residual-quality flags remain unresolved under the screened numerical variants; WGS comparators show a small test-set CO RMSE decrease and a larger CO2 RMSE increase with paired CIs; an exogenous-only comparator was run and does not support a forward-prediction claim; only volatile oxidation is locally estimable under the declared screen; 35/36 selected N=5 pairs pass the predeclared FD/JAX threshold.
