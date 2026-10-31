# Private Eq10/P48 propagation experiment

Only the four Eq10 pointwise operations are changed. The original L1 FFT,
global maximum, reduction, expression order, precision and output contract
remain source-bound. One factory owns one pool. The pool is joined before
the original normal-output call and application completion anchor.

`protocol.py` freezes the 14-window sequence and resources. `entry.py`
admits local W1 `[384,17,19]` engineering windows separately from Slurm W48
`[384,117,129]`. Each domain accepts only two-interval qualification or
eight-interval OFF windows. The old S4-5B entry is unchanged.

`prepare.py` uses `provenance_v2` on a clean committed checkout, inheriting
the S4-6C source closure and frozen S4-5B input/reference records. It does
not submit or authorize a job. `hpc.py --preflight` checks the real fixed
interpreter, dependency versions, committed source and batch entry without
creating run directories. Normal execution requires a separate explicit
`RESOURCE_AUTHORIZATION.json`, the fixed allocation, source identities and
node/input gates. `cpu.sbatch` contains the bounded resources; no submission
or automatic retry implementation is provided.

After staging, `HPC_EXECUTION_PLAN.json` binds `source_root`, `python`,
`system_git`, `query_git_launcher` and `execution_commit`. Use the existing
`git_readonly_compat.launcher_source` for the new checkout. Pass raw hashes
as `S46D_HPC_PLAN_SHA256` and `S46D_EXECUTION_SHA256`; source text identity
uses Git blobs and canonical LF, never Windows checkout raw hashes.

Local validation uses `run_local_tests.ps1` with the existing CPU O1 test
environment. Backend precedes sanity and targeted tests. The ordinary
targeted set uses `s46d and not real_entry_micro`. After committing and
preparing an execution report, set `S46D_REAL_ENTRY_REPORT` and select
`s46d_real_entry_micro` to run R2/B2/R8/B8 as four fresh W1 processes.

`analysis.analyze` consumes exactly 14 accepted cases in order. It reports
all four paired seconds and savings, body ordinals 2..8, application costs,
POST pacing and observed R/R drift. A valid negative result completes this
experiment. No extra samples, old performance samples or cost subtraction
are allowed. First POST explanation and final acceptance remain manual.

Raw arrays, failed windows and exact JSONL indexes remain at the execution
site. The scalar accepted-case records are sufficient for offline metric
recalculation. Default I0 and `production_qualified=false` are preserved.
