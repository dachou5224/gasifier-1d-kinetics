# Huayi hybrid diagnostic evidence bundle

This bundle collates existing round41-round47 diagnostics. It does not add new modeling.

## Included methods

- alpha_wgs_xt_scaffold
- direct_residual_constant_train_mean
- direct_residual_linear_operating
- fixed_physics
- global_char_oxidation_alpha
- global_multialpha:alpha_char_oxidation
- global_multialpha:alpha_char_oxidation+alpha_volatile_oxidation
- global_multialpha:alpha_volatile_oxidation
- global_multialpha:alpha_wgs
- global_multialpha:alpha_wgs+alpha_char_oxidation
- global_multialpha:alpha_wgs+alpha_char_oxidation+alpha_volatile_oxidation
- global_multialpha:alpha_wgs+alpha_volatile_oxidation
- global_wgs_alpha

## Held-out metric headline

| method | held-out rows | improved vs fixed | improved vs global WGS | mean delta vs fixed | mean delta vs WGS |
|---|---:|---:|---:|---:|---:|
| alpha_wgs_xt_scaffold | 8 | 1/8 | 6/8 | 0.180577 | -0.10579 |
| direct_residual_constant_train_mean | 16 | 3/16 | 2/16 | 0.740909 | 0.292106 |
| direct_residual_linear_operating | 16 | 3/16 | 3/16 | 0.849968 | 0.425691 |
| fixed_physics | 16 | 0/16 | 7/16 | 0 | -0.286367 |
| global_char_oxidation_alpha | 8 | 1/8 | 4/8 | 0.305433 | 0.0190653 |
| global_multialpha:alpha_char_oxidation | 8 | 2/8 | 3/8 | 0.716361 | 0.429994 |
| global_multialpha:alpha_char_oxidation+alpha_volatile_oxidation | 8 | 2/8 | 3/8 | 0.716369 | 0.430002 |
| global_multialpha:alpha_volatile_oxidation | 8 | 2/8 | 3/8 | 0.716368 | 0.43 |
| global_multialpha:alpha_wgs | 8 | 3/8 | 7/8 | 0.101223 | -0.185144 |
| global_multialpha:alpha_wgs+alpha_char_oxidation | 8 | 3/8 | 7/8 | 0.101223 | -0.185144 |
| global_multialpha:alpha_wgs+alpha_char_oxidation+alpha_volatile_oxidation | 8 | 3/8 | 7/8 | 0.101219 | -0.185148 |
| global_multialpha:alpha_wgs+alpha_volatile_oxidation | 8 | 3/8 | 7/8 | 0.101207 | -0.18516 |
| global_wgs_alpha | 8 | 1/8 | 0/8 | 0.286367 | 0 |

## Claim-safe labels

| result | label | safe wording |
|---|---|---|
| verification_first_split_baseline | diagnostic_only | The fixed-physics model provides a split-aware diagnostic reference with solver-quality fields. |
| wgs_alpha_xt_scaffold | supports_scaffold | A minimal bounded WGS-only scaffold can be run leakage-safely and compared against global scalars. |
| direct_residual_warning | overfit_warning | Direct residual patching is a black-box comparator that improves train rows but shows mixed held-out behavior. |
| multi_alpha_stop_signal | negative_multi_alpha | The conservative global multi-alpha diagnostic collapses to WGS-only and supports stopping before learned multi-alpha on 22 representatives. |
| unsupported_accuracy_claims | unsupported_for_accuracy | These artifacts support diagnostic scaffolding and protocol decisions, not accuracy benchmarking. |

## Recommendation

- current_path: stay WGS-only or stop before learned multi-alpha; expand data/protocol before z_d
- figure_claim_boundary: present hybrid sequence as diagnostic scaffold and stopping rule, not performance benchmark

## Boundary

Evidence boundary: hybrid diagnostic evidence bundle only; not new modeling, learned multi-alpha training, weak-label coal-quality modeling, production control, or final temporal-generalization evidence.

Forbidden framing: calibrated accuracy, learned multiplier interpretability proof, production control, final temporal generalization, weak-label coal-quality proof, or Yulin/slurry generalization.
