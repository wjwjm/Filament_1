# CPU P1R2 J1 diagnostics

This tool runs only the approved J1 optical diagnostic window. It reuses the
P1R asset loader and provenance checks, then starts three independent child
processes (`baseline-a`, `baseline-b`, `profile`). Every child reloads the same
sealed source and PRE0 input and executes exactly eight complete scheduled
intervals. The two baseline outputs are compared by canonical hashes
immediately after `baseline-b`; a mismatch blocks the profile stage. A
baseline/profile mismatch is retained as `NUMERICAL_DIFFERENCE`.

The run retains the existing `measure_performance` and
`diag_bk_nee_profile` flags. The additional profile uses private function
namespaces and an FFT proxy; it does not replace `numpy.fft`, module-level
`xp`, or the scientific equations. It records inclusive/exclusive timings,
FFT transform and auxiliary calls, callsites, axes, input shapes, and dtypes.
The profile covers selected propagation, Raman, ionization, diagnostic, and
deposition functions. Remaining propagation time is explicitly
`UNATTRIBUTED`. Effective `linear_precision_strategy` is recorded; J1 refuses
`unitary_projection`.

The in-memory PRE adapter preserves all three float64 fields. For initial
PRE0, its actual `delta_n` canonical hash must match the sealed input hash and
its velocity fields must be exact positive zero. It does not claim disk-backed
or production lifecycle status. Final E, POST fields, diagnostics, profile,
and receipts are written to new files only.

The coordinator accepts only explicit phase J1 and the P1R one-node CPU
allocation. It uses fresh child processes, file-backed child logs, a 45-minute
window limit per stage, a 15-minute allocation reserve, an 80% RSS latch that
blocks later stages, and bounded child termination at 90% RSS. PSS is sampled
every 10 seconds and at stage end. Output status remains diagnostic-only and
never grants production qualification.

Full-run command (the parent-provided launcher supplies the approved paths
and allocation):

```powershell
& $PythonExe -s -B -m tools.cpu_p1r2.cpu_p1r2_j1 `
  --phase J1 --bench-root $BenchRoot --input-root $InputRoot `
  --vendor-dir $VendorDir --output-root $OutputRoot
```

The `--entry-smoke` mode launches independent parent/child processes on a
deterministic tiny synthetic grid and exercises the full Raman/Eq. 27,
ionization, deposition, observer, profile, output, and hash path. Its receipt is
marked `SOFTWARE_SMOKE_ONLY`; it is not evidence from the sealed optical input
or a real benchmark.

Focused tests live in `test_cpu_p1r2_j1.py` and are registered in
`tools/run_local_tests.ps1`. Run them only through the repository wrapper,
after its required backend and sanity gates.

Numerical equality excludes only the four explicit wall-clock diagnostic arrays
(`raman_operator_walltime_step_s`, `linear_walltime_step_s`,
`ionization_walltime_step_s`, `total_walltime_step_s`); these arrays remain saved,
hashed, and listed in comparisons. All other numeric diagnostic arrays are
compared exactly. The default tiny test uses the sealed rotational
`full_isaacs_eq27` mode, Strang splitting, Kerr and self-steepening; a separate
tiny test additionally covers `full_isaacs_eq27_complete`. Neither changes the
sealed execution configuration.
