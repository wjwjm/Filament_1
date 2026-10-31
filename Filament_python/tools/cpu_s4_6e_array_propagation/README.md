# Private A/P24 propagation experiment

Only the frozen Raman intensity expression is changed: complex absolute,
square, and coefficient multiplication run as three separate axis-zero
worker phases. The original coefficient AST, FFT/IFFT, IIR, Eq10 path,
reductions, precision, expression order, and output contract remain
source-bound. One factory owns one persistent pool, which is joined before
the original normal-output call and application completion anchor.

`protocol.py` freezes the 14-window R/A sequence and resources. `entry.py`
admits local W1 `[384,17,19]` engineering windows separately from Slurm A24
`[384,117,129]`. Each domain accepts only two-interval qualification or
eight-interval OFF windows. Full-size execution is disabled in this module;
the small-grid gate must be accepted by the parent before any later extension.

`prepare.py` uses `provenance_v2` on a clean committed checkout, inheriting
the existing source closure and frozen input/reference records. It does not
submit or authorize a job. `hpc.py --preflight` checks the fixed interpreter,
dependencies, committed source, and batch entry without creating run roots.
Normal execution requires the parent-controlled explicit resource
authorization, fixed allocation, source identities, and node/input gates.

Local validation uses `run_local_tests.ps1`; backend precedes sanity and
targeted tests. The profile is off for both roles, and scalar worker records
are bounded to 128 and drained after each call. Default I0 and
`production_qualified=false` are preserved.
