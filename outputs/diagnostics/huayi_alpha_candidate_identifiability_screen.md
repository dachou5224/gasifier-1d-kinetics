# Huayi alpha candidate identifiability screen

This fixed-design screen perturbs feasible alpha groups one at a time under the frozen split.

## Per-alpha sensitivity headline

| alpha | mean | max | min split sign consistency | failures | recommendation |
|---|---:|---:|---:|---:|---|
| alpha_wgs | 2.30898 | 12.222 | 0.625 | 0 | include-with-caution |
| alpha_char_oxidation | 0.770942 | 2.59521 | 0.625 | 0 | include-with-caution |
| alpha_volatile_oxidation | 7.20713 | 19.2577 | 0.5 | 0 | include-with-caution |
| alpha_char_gasification | nan | nan | nan | nan | design-only |

## Cross-alpha similarity

| alpha_i | alpha_j | cosine | Pearson | flag |
|---|---|---:|---:|---|
| alpha_char_oxidation | alpha_volatile_oxidation | -0.526887 | -0.526762 | separable_or_weak |
| alpha_char_oxidation | alpha_wgs | -0.0773115 | -0.0773851 | separable_or_weak |
| alpha_volatile_oxidation | alpha_wgs | -0.253723 | -0.253723 | separable_or_weak |

## Split sign consistency

| alpha | split | target | positive | negative | median gradient |
|---|---|---|---:|---:|---:|
| alpha_char_oxidation | test | CO2_volpct_dry | 0 | 4 | -1.30206 |
| alpha_char_oxidation | test | CO_volpct_dry | 4 | 0 | 2.08135 |
| alpha_char_oxidation | test | H2_CO_ratio | 0 | 4 | -0.0250626 |
| alpha_char_oxidation | test | H2_volpct_dry | 0 | 4 | -0.725207 |
| alpha_char_oxidation | train | CO2_volpct_dry | 0 | 12 | -0.583215 |
| alpha_char_oxidation | train | CO_volpct_dry | 12 | 0 | 1.02056 |
| alpha_char_oxidation | train | H2_CO_ratio | 0 | 12 | -0.0119325 |
| alpha_char_oxidation | train | H2_volpct_dry | 1 | 11 | -0.281258 |
| alpha_char_oxidation | validation | CO2_volpct_dry | 0 | 6 | -1.69288 |
| alpha_char_oxidation | validation | CO_volpct_dry | 5 | 1 | 2.37186 |
| alpha_char_oxidation | validation | H2_CO_ratio | 2 | 4 | -0.0256797 |
| alpha_char_oxidation | validation | H2_volpct_dry | 2 | 4 | -0.600115 |
| alpha_volatile_oxidation | test | CO2_volpct_dry | 0 | 4 | -2.18999 |
| alpha_volatile_oxidation | test | CO_volpct_dry | 0 | 4 | -12.6791 |
| alpha_volatile_oxidation | test | H2_CO_ratio | 4 | 0 | 0.288685 |
| alpha_volatile_oxidation | test | H2_volpct_dry | 4 | 0 | 12.8908 |
| alpha_volatile_oxidation | train | CO2_volpct_dry | 2 | 10 | -0.886052 |
| alpha_volatile_oxidation | train | CO_volpct_dry | 0 | 12 | -11.1503 |
| alpha_volatile_oxidation | train | H2_CO_ratio | 12 | 0 | 0.226454 |
| alpha_volatile_oxidation | train | H2_volpct_dry | 12 | 0 | 10.6197 |
| alpha_volatile_oxidation | validation | CO2_volpct_dry | 1 | 5 | -1.37354 |
| alpha_volatile_oxidation | validation | CO_volpct_dry | 0 | 6 | -15.9844 |
| alpha_volatile_oxidation | validation | H2_CO_ratio | 6 | 0 | 0.352019 |
| alpha_volatile_oxidation | validation | H2_volpct_dry | 6 | 0 | 15.4404 |
| alpha_wgs | test | CO2_volpct_dry | 4 | 0 | 2.56682 |
| alpha_wgs | test | CO_volpct_dry | 0 | 4 | -6.08739 |
| alpha_wgs | test | H2_CO_ratio | 4 | 0 | 0.101747 |
| alpha_wgs | test | H2_volpct_dry | 4 | 0 | 3.64574 |
| alpha_wgs | train | CO2_volpct_dry | 1 | 11 | -2.40232 |
| alpha_wgs | train | CO_volpct_dry | 10 | 2 | 3.54636 |
| alpha_wgs | train | H2_CO_ratio | 3 | 9 | -0.0374623 |
| alpha_wgs | train | H2_volpct_dry | 4 | 8 | -1.11336 |
| alpha_wgs | validation | CO2_volpct_dry | 0 | 6 | -1.59154 |
| alpha_wgs | validation | CO_volpct_dry | 6 | 0 | 2.02878 |
| alpha_wgs | validation | H2_CO_ratio | 0 | 6 | -0.0190977 |
| alpha_wgs | validation | H2_volpct_dry | 1 | 5 | -0.316346 |

## Boundary

Evidence boundary: candidate alpha identifiability screen only; not multi-alpha training, learned multiplier interpretation proof, weak-label coal-quality modeling, production control, or final temporal-generalization evidence.

Recommendations are for the next diagnostic experiment only. They are not final manuscript claims.
