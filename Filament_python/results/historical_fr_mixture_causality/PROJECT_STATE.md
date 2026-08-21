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

## Completed 120 fs causal run (job 215812)

- Completed on 2026-08-20 with exit code `0:0` on `m4gn1401` (RTX 5090), using
  execution SHA `fbbf18972ddd5fd8db97a140cfe8aa4460490bff`; the raw 30 MB NPZ
  remains at the HPC run directory.
- Postprocess gate passed.  The configuration provenance, historical kernel
  scalars, finite Raman phase/absorption diagnostics, and 15,000 z records all
  passed their checks.
- At `rho_e=1e22 m^-3`, the mixture onset was `-13.395 cm`, versus current
  production `-16.412 cm` and PyCAP `-14.027 cm`: a `+3.017 cm` shift from
  production, ending `0.632 cm` later than PyCAP.
- The peak density/position was `2.035e22 m^-3` at `-9.855 cm`, compared with
  production `6.461e22 m^-3` at `-14.440 cm` and PyCAP `6.455e22 m^-3` at
  `-12.184 cm`.  Thus the phase-only substitution moves onset substantially
  toward PyCAP but does not reproduce the PyCAP peak-density evolution.
