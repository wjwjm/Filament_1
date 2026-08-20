# historical_fr_mixture — pre-April f_R mixture causal test

## Submission working-directory guard (2026-08-20)

- Failed allocation `215595` performed no propagation: Slurm opened its default
  `slurm-215595.out` in the staging repository before the batch script started,
  so the intentional clean-worktree guard exited with code 22.
- The submission entry point is now `tools/submit_historical_fr_mixture_job.sh`.
  It creates `RUN_DIR` first, then passes both `--chdir="$RUN_DIR"` and
  explicit `--output/--error` paths under `RUN_DIR`.  This is an execution-path
  correction only; it does not alter the configuration or any physical model.

- Operational baseline: `HEAD` at preparation time (branch `codex/historical-fr-mixture`).
- Frozen physical baseline: `e11d13f103c484953c0f733aa9b410bff385b2b5`.
- Single causal variable: `raman.operator_mode = "historical_fr_mixture"` versus the production `legacy_split` baseline.
- Authorized production jobs: exactly one.
- No performance optimization, profiling smoke, grid/step change, or non-Raman ablation is in scope.
- Large `.npz` output stays on the HPC run directory; only small json/csv/md reports are committed here.
