# Figure/table-ready diagnostics bundle

This bundle reshapes existing model diagnostics for manuscript plotting and tables.
It does not add model functionality, train a hybrid model, or claim calibration accuracy.

| artifact | rows | source | purpose |
|---|---:|---|---|
| fig2_baseline_mismatch | 10 | outputs/diagnostics/pilot_fixed_physics_diagnostic_baseline.csv | fixed-physics predicted-vs-target and delta bars |
| fig3_gradient_parity_matrix | 20 | outputs/diagnostics/gradient_parity_diagnostic.csv | FD-vs-IFT parity matrix or heatmap |
| fig3_ift_cost_scaling | 6 | outputs/diagnostics/ift_cost_diagnostic.csv | IFT vs direct/unrolled AD cost-scaling panel |
| fig3_gradient_smoke_trajectory | 8 | outputs/diagnostics/end_to_end_gradient_smoke.csv | end-to-end gradient descent smoke trajectory |
| table_sensitivity_ranking | 4 | outputs/diagnostics/fixed_physics_sensitivity_ranking.csv | fixed-physics sensitivity ranking table |
| ift_cost_protocol | 43 | outputs/diagnostics/ift_cost_protocol.md | metadata |

Can support:
- pilot fixed-physics mismatch characterization
- sensitivity-prioritized correction design
- FD-vs-IFT/JAX gradient parity evidence
- IFT/custom-VJP method-cost rationale
- end-to-end gradient smoke evidence

Cannot support:
- hybrid calibration accuracy
- trained MLP or learned multiplier performance
- industrial generalization
- final Nature-style figure aesthetics without separate plotting/QA.
