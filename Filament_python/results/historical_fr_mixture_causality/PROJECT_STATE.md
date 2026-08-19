# historical_fr_mixture — pre-April f_R mixture causal test

- Operational baseline: `HEAD` at preparation time (branch `codex/historical-fr-mixture`).
- Frozen physical baseline: `e11d13f103c484953c0f733aa9b410bff385b2b5`.
- Single causal variable: `raman.operator_mode = "historical_fr_mixture"` versus the production `legacy_split` baseline.
- Authorized production jobs: exactly one.
- No performance optimization, profiling smoke, grid/step change, or non-Raman ablation is in scope.
- Large `.npz` output stays on the HPC run directory; only small json/csv/md reports are committed here.
