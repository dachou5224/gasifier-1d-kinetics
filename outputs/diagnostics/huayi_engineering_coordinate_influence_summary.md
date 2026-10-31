# Huayi engineering-coordinate influence tracing

This diagnostic translates raw feed/carrier/heat-loss sensitivities into plant-relevant engineering coordinates.

- Cases: 22
- Rows: 440
- OK rows: 440
- O/C ratio is treated as the primary plant-facing operating coordinate.
- Carrier gas is treated as a coal-transport-coupled boundary, not an independent gasifier control priority.
- HeatLossPercent is treated as a model-calibration coordinate, not a plant operating knob.

| rank | coordinate | class | mean_abs_norm_sens | max_abs_norm_sens | rule |
|---:|---|---|---:|---:|---|
| 1 | O_C_ratio_at_fixed_load | plant_operating | 0.65927 | 1.06415 | perturb O2 flow to change molar O/C while dry coal load and carrier boundary remain fixed |
| 2 | load_at_fixed_O_C | load | 0.233966 | 0.966244 | scale dry coal, O2, and carrier gas together to preserve baseline O/C and carrier/coal ratios |
| 3 | carrier_to_coal_ratio_boundary | transport_boundary | 0.172193 | 0.683471 | perturb carrier gas relative to fixed dry coal flow; boundary sensitivity, not independent plant priority |
| 4 | HeatLossPercent_calibration | model_calibration | 0.0645095 | 0.509433 | perturb HeatLossPercent calibration parameter; not a plant operating knob |

Interpretation boundary: coordinate ranking supports diagnostic localization and candidate online-correction axis selection. It is not a trained hybrid, calibrated accuracy result, causal proof, production online deployment, full-reactor 22-case IFT parity, independent carrier/heat-loss plant manipulation claim, or Yulin/slurry generalization.

Evidence boundary: engineering-coordinate fixed-physics influence tracing only; not trained hybrid, calibrated accuracy, causal proof, production online deployment, full-reactor 22-case IFT parity, independent plant manipulation of carrier gas or heat loss, or Yulin/slurry generalization.

## Coordinate finite-difference subtable

The accompanying sensitivity-detail CSV stores the centered finite-difference probes used to build the engineering-coordinate influence table. The coordinates are reporting axes, not a claim that all underlying raw inputs are independent plant manipulated variables.
