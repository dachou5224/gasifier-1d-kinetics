# Huayi bounded global multi-alpha diagnostic

This diagnostic selects global scalar multipliers on the frozen train split only.

- Volatile mapping: alpha_volatile_oxidation maps to both op_conds.CO_OxidationFactor and op_conds.H2_OxidationFactor
- Recommendation: stop_and_expand_data_before_learned_multi_alpha

## Subset selection

| subset | alpha_wgs | alpha_char | alpha_vol | train objective | delta | bound hits | compensation |
|---|---:|---:|---:|---:|---:|---:|---|
| alpha_wgs | 0.3 | 1 | 1 | 1.00271 | -1.17062 | 0 | single_or_mild_move+heldout_target_tradeoff |
| alpha_char_oxidation | 1 | 1 | 1 | 2.17332 | 0 | 1 | single_or_mild_move+bound_hit+heldout_target_tradeoff |
| alpha_volatile_oxidation | 1 | 1 | 1 | 2.1733 | 0 | 0 | single_or_mild_move+heldout_target_tradeoff |
| alpha_wgs+alpha_char_oxidation | 0.3 | 1 | 1 | 1.00271 | -1.1706 | 1 | single_or_mild_move+bound_hit+heldout_target_tradeoff |
| alpha_wgs+alpha_volatile_oxidation | 0.3 | 1 | 1 | 1.0027 | -1.17062 | 0 | single_or_mild_move+heldout_target_tradeoff |
| alpha_char_oxidation+alpha_volatile_oxidation | 1 | 1 | 1 | 2.17332 | 0 | 1 | single_or_mild_move+bound_hit+heldout_target_tradeoff |
| alpha_wgs+alpha_char_oxidation+alpha_volatile_oxidation | 0.3 | 1 | 1 | 1.00271 | -1.17061 | 1 | single_or_mild_move+bound_hit+heldout_target_tradeoff |

## Held-out metric headline

| subset | improved vs fixed | improved vs global WGS | mean delta vs fixed | mean delta vs WGS |
|---|---:|---:|---:|---:|
| alpha_char_oxidation | 2/8 | 3/8 | 0.716361 | 0.429994 |
| alpha_char_oxidation+alpha_volatile_oxidation | 2/8 | 3/8 | 0.716369 | 0.430002 |
| alpha_volatile_oxidation | 2/8 | 3/8 | 0.716368 | 0.43 |
| alpha_wgs | 3/8 | 7/8 | 0.101223 | -0.185144 |
| alpha_wgs+alpha_char_oxidation | 3/8 | 7/8 | 0.101223 | -0.185144 |
| alpha_wgs+alpha_char_oxidation+alpha_volatile_oxidation | 3/8 | 7/8 | 0.101219 | -0.185148 |
| alpha_wgs+alpha_volatile_oxidation | 3/8 | 7/8 | 0.101207 | -0.18516 |

## Boundary

Evidence boundary: bounded global multi-alpha scalar diagnostic selected on train split only; not learned alpha_j(x_t), weak-label coal-quality modeling, learned multiplier interpretability proof, production control, or final temporal-generalization evidence.

This is not learned alpha_j(x_t), weak-label coal-quality modeling, learned multiplier interpretability proof, production control, or final temporal-generalization evidence.
