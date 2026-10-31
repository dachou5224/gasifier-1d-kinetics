# P0-B 网格判断

所有已跑网格的 cell-0 都超过 1e-4。不把更细网格选成更真实的主模型。后续通道归因继续停用。

| N | split | target | RMSE | n |
|---:|---|---|---:|---:|
| 3 | test | CO_dry_vol_pct | 1.22808 | 17 |
| 3 | test | H2_dry_vol_pct | 0.993446 | 17 |
| 5 | test | CO_dry_vol_pct | 3.46371 | 17 |
| 5 | test | H2_dry_vol_pct | 2.24088 | 17 |
| 9 | test | CO_dry_vol_pct | 3.40505 | 17 |
| 9 | test | H2_dry_vol_pct | 3.10382 | 17 |

出口干基 H2 的中位数（vol%）：
- N=3: 23.0601
- N=5: 26.6494
- N=9: 27.5943

下游单元离开燃烧区的窗口数：426 / 426。
