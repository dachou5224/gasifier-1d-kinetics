# Huayi O/C response curve diagnostic

This diagnostic perturbs molar O/C by changing oxygen flow at fixed dry coal load and fixed carrier boundary.

- Cases: 22
- Phases: phase1, phase2, phase3, phase4
- Response rows: 110
- OK response rows: 110
- Figure-axis rows: 660
- O/C multipliers: 0.96, 0.98, 1.00, 1.02, 1.04
- Solver path: GasifierSystem.solve(solver_method='minimize', jacobian_mode='scipy')
- Local slopes use the centered 0.98/1.02 O/C multiplier points around baseline.
- IFT/JAX parity is not claimed for this full 22-case response grid; this is fixed-physics finite-perturbation response evidence.

| target | mean_abs_local_slope_per_relative_multiplier |
|---|---:|
| T_out_K | 802.203 |
| CO_volpct_dry | 28.3126 |
| H2_volpct_dry | 20.9913 |
| CO2_volpct_dry | 5.9457 |
| dry_syngas_flow_kmol_s | 1.05034 |
| H2_CO_ratio | 0.501326 |

Evidence boundary: fixed-physics O/C response evidence on Huayi dry-powder ModelCases; supports plant-coordinate influence tracing and figure generation only; not calibrated accuracy, causal proof, trained hybrid validation, production control recommendation, or Yulin/slurry generalization.
Engineering-coordinate boundary: engineering-coordinate fixed-physics influence tracing only; not trained hybrid, calibrated accuracy, causal proof, production online deployment, full-reactor 22-case IFT parity, independent plant manipulation of carrier gas or heat loss, or Yulin/slurry generalization.
