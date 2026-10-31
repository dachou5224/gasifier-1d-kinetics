# P0-C 判断

9 个窗口都以 ftol 停在非零残差上，接受解邻域内的可行步不能再把 cost 降下 0.1%。非零残差来自限幅后的方程在这个可行域里没有根，不是求解器停早了。

出口 H2 的消耗主导项：H2_Ox_combustion_zone_instant_oxidation。

| window_id | stop | cost | relative drop | pO2 atm | H2 out mol/s | H2 driver | off cost | off H2 mol/s | off H2 vol% | identity gap |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003 | ftol | 0.00669088 | 1.01e-14 | 17.89 | 3.60018e-14 | H2_Ox_combustion_zone_instant_oxidation | 2.89001e-26 | 1.71951e-12 | 6.47124 | 1.58e-11 |
| R52-C-75__P1__huayi_sy3_phase1_wgs1__shift_morning_handover_2022-09-28__W001 | ftol | 0.0052189 | 3.32e-16 | 18.52 | 3.13766e-16 | H2_Ox_combustion_zone_instant_oxidation | 1.08213e-27 | 3.89377e-12 | 6.71174 | 1.25e-12 |
| R52-C-75__P2__huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03D_011__W001 | ftol | 0.0140297 | 1.36e-15 | 17.15 | 2.82455e-15 | H2_Ox_combustion_zone_instant_oxidation | 1.45538e-27 | 1.27898e-12 | 9.66529 | 3.86e-12 |
| R52-C-75__P2__huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03N_013__W001 | ftol | 0.011091 | 5.47e-15 | 17.53 | 1.9142e-14 | H2_Ox_combustion_zone_instant_oxidation | 5.61834e-28 | -9.09495e-12 | 8.98909 | 4.69e-13 |
| R52-C-75__P3__huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_014__W004 | ftol | 0.0147643 | 2.35e-15 | 16.9 | 1.84129e-14 | H2_Ox_combustion_zone_instant_oxidation | 4.59099e-28 | 5.82645e-12 | 10.5158 | 1.66e-12 |
| R52-C-75__P3__huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_015__W001 | ftol | 0.00881989 | 1.32e-14 | 17.95 | 1.0926e-13 | H2_Ox_combustion_zone_instant_oxidation | 8.41032e-26 | 6.4432e-11 | 9.12909 | 2.27e-11 |
| R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-10N_006__W001 | ftol | 0.0139251 | 8.72e-16 | 16.06 | 4.58123e-15 | H2_Ox_combustion_zone_instant_oxidation | 4.01793e-28 | -6.90648e-12 | 10.6384 | 1.14e-13 |
| R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11N_009__W002 | ftol | 0.0147737 | 6.69e-15 | 15.96 | 5.0485e-14 | H2_Ox_combustion_zone_instant_oxidation | 5.73228e-26 | -4.64979e-11 | 10.8667 | 2.04e-11 |
| R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__shift_evening_handover_2022-10-11__W001 | ftol | 0.0158419 | 6.57e-16 | 15.88 | 3.17038e-15 | H2_Ox_combustion_zone_instant_oxidation | 9.23246e-27 | 1.06866e-11 | 11.1524 | 9.98e-12 |

燃烧区分支是入口可用氧分压高于 0.05 atm 时，用瞬时挥发分氧化替换 CH4/CO/H2 动力学速率。
诊断重解只把这个分支关掉，平衡式、热解产率、phi、f_co2 和 H2_OxidationFactor 都没有改。
cost 平坦的判据：边界内的梯度方向和各坐标相对步都不能把 0.5||r||^2 降低 0.1% 以上。
