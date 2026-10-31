# FD-vs-IFT/JAX gradient parity diagnostic

This is fixed-branch gradient-consistency evidence, not hybrid calibration accuracy evidence.

- Rows: 20
- OK rows: 20
- Blocked rows: 0
- Supported parameters: HeatLossPercent, carrier_gas_flow, coal_mass_flow, oxygen_flow
- Blocked parameters: none
- Outputs: final_CO2_mol_s, final_CO_mol_s, final_H2_mol_s, final_T_K, final_dry_flow_mol_s
- Max abs error among OK rows: 0.001881
- Max rel error among OK rows: 0.00393117

| parameter | output | fd_grad | ift_or_ad_grad | abs_error | rel_error | status |
|---|---|---:|---:|---:|---:|---|
| HeatLossPercent | final_T_K | -1.84953 | -1.84953 | 9.671e-07 | 5.229e-07 | ok |
| HeatLossPercent | final_CO_mol_s | -0.00590802 | -0.00590802 | 2.501e-09 | 4.232e-07 | ok |
| HeatLossPercent | final_H2_mol_s | 0.00618652 | 0.00618652 | 4.437e-09 | 7.172e-07 | ok |
| HeatLossPercent | final_CO2_mol_s | 0.00557348 | 0.00557348 | 3.165e-09 | 5.679e-07 | ok |
| HeatLossPercent | final_dry_flow_mol_s | 0.00537819 | 0.00537819 | 4.787e-09 | 8.901e-07 | ok |
| oxygen_flow | final_T_K | 89.7979 | 89.7979 | 3.993e-05 | 4.447e-07 | ok |
| oxygen_flow | final_CO_mol_s | 0.705704 | 0.705706 | 1.459e-06 | 2.068e-06 | ok |
| oxygen_flow | final_H2_mol_s | -0.336617 | -0.336617 | 6.212e-08 | 1.845e-07 | ok |
| oxygen_flow | final_CO2_mol_s | 0.493949 | 0.493948 | 6.979e-07 | 1.413e-06 | ok |
| oxygen_flow | final_dry_flow_mol_s | 0.878147 | 0.878147 | 6.818e-07 | 7.765e-07 | ok |
| carrier_gas_flow | final_T_K | -17.1983 | -17.1985 | 1.912e-04 | 1.112e-05 | ok |
| carrier_gas_flow | final_CO_mol_s | 0.00139336 | 0.00139884 | 5.478e-06 | 3.931e-03 | ok |
| carrier_gas_flow | final_H2_mol_s | -0.143093 | -0.143091 | 1.370e-06 | 9.573e-06 | ok |
| carrier_gas_flow | final_CO2_mol_s | 0.914093 | 0.91409 | 2.095e-06 | 2.292e-06 | ok |
| carrier_gas_flow | final_dry_flow_mol_s | 0.758729 | 0.758733 | 4.697e-06 | 6.190e-06 | ok |
| coal_mass_flow | final_T_K | -664.663 | -664.661 | 1.881e-03 | 2.830e-06 | ok |
| coal_mass_flow | final_CO_mol_s | 5.37263 | 5.37259 | 4.543e-05 | 8.456e-06 | ok |
| coal_mass_flow | final_H2_mol_s | 3.41001 | 3.41 | 1.149e-05 | 3.371e-06 | ok |
| coal_mass_flow | final_CO2_mol_s | -0.875514 | -0.875497 | 1.696e-05 | 1.938e-05 | ok |
| coal_mass_flow | final_dry_flow_mol_s | 7.86293 | 7.86289 | 3.956e-05 | 5.031e-06 | ok |

Evidence boundary: fixed-branch two-cell JAX/IFT fixture. Do not interpret as full-reactor gradient parity, training, or calibrated accuracy.
