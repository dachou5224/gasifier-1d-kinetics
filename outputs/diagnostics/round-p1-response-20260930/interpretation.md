# P1 通道响应

36 行里 interpretable_solved 有 36 行，not_interpretable_due_to_residual 有 0 行。
selected_N 仍是 3。N=5 和 N=9 只比较 CO 差值的符号。

N=3 的差值范围：

| 通道 | CO vol% | CO2 vol% | H2 vol% | 干气 Nm3/h |
|---|---:|---:|---:|---:|
| alpha_wgs | 0.0237664 至 0.0285777 | -0.0147981 至 -0.0120864 | -0.0139411 至 -0.011685 | -20.7231 至 -15.3612 |
| alpha_char_oxidation | -0.00529934 至 0.00859414 | -0.00720694 至 0.00237993 | -0.000301536 至 0.00351443 | 0.543116 至 5.17271 |
| alpha_volatile_oxidation | -0.120641 至 0.196716 | 0.189285 至 0.335826 | -0.392739 至 -0.222136 | -719.808 至 -289.109 |

相对 N=3，同一窗口同一通道的 CO 差值：

N=5 与 N=3 同号 7/12。N=9 与 N=3 同号 5/12。
不同号的行：
- R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003 / alpha_wgs：N=3 0.0237664，N=5 0.00993988，N=9 -0.000953965
- R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003 / alpha_char_oxidation：N=3 0.00400332，N=5 -0.00631136，N=9 -0.00462381
- R52-C-75__P1__huayi_sy3_phase1_wgs1__plateau_2022-09-28D_008__W003 / alpha_volatile_oxidation：N=3 0.0450931，N=5 -0.00637348，N=9 0.0910385
- R52-C-75__P2__huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03N_013__W001 / alpha_wgs：N=3 0.0256053，N=5 0.0147336，N=9 -0.000952446
- R52-C-75__P2__huayi_sy3sy2_phase2_wgs1__plateau_2022-10-03N_013__W001 / alpha_char_oxidation：N=3 0.00859414，N=5 -0.00589632，N=9 -0.00495458
- R52-C-75__P3__huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_014__W004 / alpha_wgs：N=3 0.0285777，N=5 0.00751922，N=9 -0.00100915
- R52-C-75__P3__huayi_sy3sy2_phase3_wgs1__plateau_2022-10-09D_014__W004 / alpha_volatile_oxidation：N=3 0.0127388，N=5 -0.0714487，N=9 0.0624502
- R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11N_009__W002 / alpha_wgs：N=3 0.0258573，N=5 -0.00174615，N=9 -0.00229694
- R52-C-75__P4__huayi_sy3sy2_phase4_wgs1__plateau_2022-10-11N_009__W002 / alpha_volatile_oxidation：N=3 -0.120641，N=5 -0.125561，N=9 0.0562564
