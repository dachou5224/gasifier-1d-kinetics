# Huayi fixed-physics split metrics

| model | split | N | target | n | RMSE | MAE | bias | fallback | poor |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| fixed_physics | test | 3 | CO2_volpct_dry | 4 | 1.81849 | 1.80728 | 1.80728 | 0 | 4 |
| fixed_physics | test | 3 | CO_volpct_dry | 4 | 1.62178 | 1.57202 | -1.57202 | 0 | 4 |
| fixed_physics | test | 3 | H2_CO_ratio | 4 | 0.0105093 | 0.0102801 | -0.00425097 | 0 | 4 |
| fixed_physics | test | 3 | H2_volpct_dry | 4 | 1.02051 | 0.916973 | -0.916973 | 0 | 4 |
| fixed_physics | train | 3 | CO2_volpct_dry | 12 | 1.59476 | 1.40338 | 0.417317 | 0 | 12 |
| fixed_physics | train | 3 | CO_volpct_dry | 12 | 3.00321 | 2.58151 | 2.54621 | 0 | 12 |
| fixed_physics | train | 3 | H2_CO_ratio | 12 | 0.0872238 | 0.0814051 | -0.0694766 | 0 | 12 |
| fixed_physics | train | 3 | H2_volpct_dry | 12 | 4.38831 | 4.17063 | -3.38837 | 0 | 12 |
| fixed_physics | validation | 3 | CO2_volpct_dry | 6 | 0.538332 | 0.507212 | 0.507212 | 0 | 6 |
| fixed_physics | validation | 3 | CO_volpct_dry | 6 | 2.0644 | 1.58907 | 0.307161 | 0 | 6 |
| fixed_physics | validation | 3 | H2_CO_ratio | 6 | 0.047552 | 0.0271746 | -0.023388 | 0 | 6 |
| fixed_physics | validation | 3 | H2_volpct_dry | 6 | 2.40905 | 1.43839 | -1.43839 | 0 | 6 |
| fixed_physics | test | 5 | CO2_volpct_dry | 4 | 0.323173 | 0.267195 | 0.267195 | 0 | 4 |
| fixed_physics | test | 5 | CO_volpct_dry | 4 | 1.57297 | 1.4708 | -1.4708 | 0 | 4 |
| fixed_physics | test | 5 | H2_CO_ratio | 4 | 0.022008 | 0.0185229 | 0.0185229 | 0 | 4 |
| fixed_physics | test | 5 | H2_volpct_dry | 4 | 0.748194 | 0.545902 | 0.545902 | 0 | 4 |
| fixed_physics | train | 5 | CO2_volpct_dry | 12 | 2.25177 | 1.62062 | -1.62062 | 0 | 12 |
| fixed_physics | train | 5 | CO_volpct_dry | 12 | 3.75236 | 3.42356 | 3.42356 | 0 | 12 |
| fixed_physics | train | 5 | H2_CO_ratio | 12 | 0.0762773 | 0.07169 | -0.0566395 | 0 | 12 |
| fixed_physics | train | 5 | H2_volpct_dry | 12 | 3.5252 | 3.35874 | -2.2105 | 0 | 12 |
| fixed_physics | validation | 5 | CO2_volpct_dry | 6 | 1.52011 | 1.50148 | -1.50148 | 0 | 6 |
| fixed_physics | validation | 5 | CO_volpct_dry | 6 | 2.26845 | 1.35088 | 1.22244 | 0 | 6 |
| fixed_physics | validation | 5 | H2_CO_ratio | 6 | 0.0403151 | 0.0288624 | -0.0119536 | 0 | 6 |
| fixed_physics | validation | 5 | H2_volpct_dry | 6 | 1.81915 | 1.38522 | -0.325164 | 0 | 6 |

Evidence boundary: split-aware fixed-physics diagnostic baseline only; not calibrated accuracy.
