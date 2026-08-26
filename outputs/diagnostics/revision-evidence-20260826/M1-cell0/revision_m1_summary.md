# Revision M1 Cell-0 Residual-Quality Robustness

- Dataset: `R52-C-75`
- Frozen split consumed: True
- Source commit: `a2955d2a2f6e850b88d8b206f465d810e1ad86ab`
- Seed: 20260826 (not used for variant selection)
- Poor-quality threshold: cost > 0.0001
- Representative windows: 9 covering phases [1, 2, 3, 4]
- Selection rule: zero execution errors/timeouts; a variant is chosen over reference only if it lowers the poor-flag rate or reduces median cell0 cost by more than 1%; outlet error is not used
- Selected configuration: `reference`
- Gate status: `stable`
- Numerical-quality unresolved: True

No screened configuration removed or materially reduced the cell-0 quality flag. The numerical-quality concern remains unresolved and manuscript claims that treat the baseline as a fully converged first-cell solve should be reduced.

Train-only reselection under reference produced global alpha values [0.05].

Evidence boundary: revision evidence only; not production control, learned multi-alpha, z_d value, oxidation-channel updating, cross-gasifier generalization, or manuscript claim text

Forbidden claims: no production closed-loop control; no learned WGS kinetic identification; no learned multi-alpha success; no z_d value; no oxidation-channel updating; no cross-gasifier generalization.
