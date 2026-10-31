# IFT vs direct-unrolled AD cost diagnostic

This is method-cost evidence for a fixed-branch JAX fixture, not calibration accuracy evidence.

- Rows: 6
- Status counts: ok=5, timeout=1
- IFT OK N_cells: 1, 2, 3, 4
- Direct-unrolled OK N_cells: 1

| method | N | status | cold_s | warm_s | grad | ref_grad | rel_diff | residual_max |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ift_custom_vjp | 1 | ok | 10.7 | 0.000276 | -0.63955 | nan | nan | 2.996e-06 |
| ift_custom_vjp | 2 | ok | 9.73 | 0.000562 | -1.84954 | nan | nan | 2.996e-06 |
| ift_custom_vjp | 3 | ok | 11.4 | 0.000577 | -1.91432 | nan | nan | 8.708e-06 |
| ift_custom_vjp | 4 | ok | 11 | 0.000543 | -2.56675 | nan | nan | 8.708e-06 |
| direct_unrolled_ad | 1 | ok | 56.2 | 0.000721 | -0.63953 | -0.63955 | 3.032e-05 | 2.996e-06 |
| direct_unrolled_ad | 2 | timeout | nan | nan | nan | nan | nan | nan |

Conclusion: in this fixture, IFT/custom VJP is the scalable path to test across larger scan lengths. Direct-unrolled AD is useful for small-N agreement checks but is bounded by compile/runtime cost.

Evidence boundary: fixed-branch HeatLossPercent -> final_T_K benchmark only. Do not interpret as full-reactor performance, training, or calibration evidence.
