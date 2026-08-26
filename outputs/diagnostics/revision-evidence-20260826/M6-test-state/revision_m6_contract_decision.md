# Revision M6 Contract Decision

Decision: **update the stale assertion; keep the production contract unchanged.**

`audit_coal_assumption_readiness()` reads `config/coal_assumption_v0.yaml`. The moisture entry `proximate_analysis.moisture_ar` is marked `used_in_v0: true` because it converts weighfeeder as-received coal mass to dry feed:

`m_dry = m_ar × (1 - M_ar)`

With `used_in_v0=true` and `source.type=lims_table`, the contract returns `assumed_lims`. The previous test expected `measured_unused_in_v0`, which is the branch for `used_in_v0: false`. That expectation was stale relative to the production YAML and the generated audits already in `outputs/model_cases/*.coal_assumption_model_input_audit.yaml`.

No generated audit files were regenerated. No numerical source-data hashes changed. No ModelCase contract fields were altered.

The registered paper-facing suite now reports 40/40 passing.
