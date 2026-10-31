# End-to-end gradient smoke

This is end-to-end gradient smoke evidence, not calibration accuracy evidence.

- Parameter: HeatLossPercent
- Steps: 8
- Initial loss: 4.5
- Final loss: 2.5817899
- Overall decrease: True
- Per-step decrease: True

| step | theta | loss | grad | step_size | next_loss | T_out_K | residual_max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 4.5 | -5.5486 | 0.055486 | 4.19745 | 2014.300 | 1.199e-07 |
| 1 | 2.05549 | 4.19745 | -5.35705 | 0.0535705 | 3.91542 | 2014.197 | 1.201e-07 |
| 2 | 2.10906 | 3.91542 | -5.17229 | 0.0517229 | 3.6525 | 2014.098 | 1.204e-07 |
| 3 | 2.16078 | 3.6525 | -4.99407 | 0.0499407 | 3.40739 | 2014.003 | 1.207e-07 |
| 4 | 2.21072 | 3.40739 | -4.82215 | 0.0482215 | 3.17885 | 2013.910 | 1.209e-07 |
| 5 | 2.25894 | 3.17885 | -4.6563 | 0.046563 | 2.96577 | 2013.821 | 1.212e-07 |
| 6 | 2.3055 | 2.96577 | -4.49628 | 0.0449628 | 2.76708 | 2013.735 | 1.214e-07 |
| 7 | 2.35047 | 2.76708 | -4.34189 | 0.0434189 | 2.58179 | 2013.652 | 1.217e-07 |

Evidence boundary: fixed-branch two-cell JAX/IFT smoke only. Do not interpret as hybrid calibration, full-reactor training, or industrial validation.
