# Phase 8B-R replacement Job 1 execution audit

- Execution SHA: `e321223b6ee2c79407192ff4be3087fd83ec5ef8`
- Slurm job: `179988`, `COMPLETED 0:0`, elapsed `16406.27 s`.
- Backend and GPU: CuPy on NVIDIA GeForce RTX 5090 (`33,668,988,928` bytes).
- Output: `z=1.300 m`, `15,000` records, no required numeric-array NaN/Inf.
- Configuration: `full_isaacs_eq27`, Heun, Strang, and `diag_operator_energy=true`.  The sole difference from the formal ON configuration is the diagnostic observability field.

## Passed contracts

- The float32-aware coordinate audit passed: the archive end point is `0.4000000004` ULP from `1.300 m`, and the accepted-step distance is within its accumulated half-ULP bound.
- Legacy Raman absorption is disabled and `alpha_R_applied_max_z=0`.
- Raman closure passes: p99 step residual `1.990885304985568e-4 < 1e-3`; cumulative residual `2.245331188532873e-6 < 5e-3`.
- Every propagation step contains two Raman substeps and four Raman convolutions.
- All six opt-in operator-energy checkpoint histories are finite and aligned; their telescope residual is exactly `0 J`, and the final checkpoint equals `U_z` exactly.
- Near-focus total-energy closure passes: `1.2044126022869913% < 2%`.

## Failed contract and cause evidence

The final total-energy closure is `1.4176516743495496%`, above the locked `1%` maximum.  The runtime reconstruction reproduces this number exactly and the runtime energy histories are internally self-consistent.

The added checkpoints locate the missing energy in the two nominally lossless linear half steps: their combined field-energy change is `-3.094249404966831e-5 J`.  The final unaccounted field loss is `3.0784489354118705e-5 J`, agreeing to the scale expected from checkpoint/storage precision.  The two Raman substeps together lose `3.966363146901131e-5 J`, independently consistent with the passing Raman Eq. (10) closure.

This is therefore classified as `production_numerical_dissipation`, not a Raman-operator closure failure.  Independent reintegration from archived full fields remains unavailable because the production archive intentionally does not store a full `(z,t,x,y)` field history.

## Decision

`Job 1` has completed, but the Phase 8B-R total-energy contract failed.  The workflow is stopped: Job 2 was neither prepared nor submitted.  The required next action is to correct or explicitly account for linear-half-step energy loss, validate it in a short smoke test, and obtain new authorization before another full propagation.
