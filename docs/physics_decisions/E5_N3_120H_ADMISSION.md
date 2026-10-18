# N3/K8048/H4 final admission

The N3 entry uses one node, six GPUs, 48 CPUs, site-default memory, and a
120-hour allocation. It excludes g0603, requests no fixed node, and disables
requeue. A rejected 120-hour request stops this entry; no shorter fallback is
authorized. There is at most one actual submission after all static gates pass.

`E5AProductionSpec.final_gate_manifest_path` is an optional orchestration field.
Omitting it preserves the legacy admission payload. The new N3 launcher requires
it, binds its identity into admission, and requires `science_child_timeout_s =
432000` and `storage_lock_timeout_s = 163`. Lifecycle locking remains 10 seconds.
This is not a physical configuration field in config.py/config_schema.py.

The external manifest binds the actual implementation, N3 inputs, unchanged
journal/lifecycle performance qualification, source-derived time budget, and
encoded journal capacity. Explicit capacity objects are added to the existing
budget object table without deducting its metadata allowance. The 300/64 GiB
policies and 8 GiB safety margin remain in force.

Static admission is sealed before submission. Actual resource/GPU/environment
checks and admission persistence occur in the sole allocation before large
initialization. Failure stops without recovery or resubmission. Local recovery
fixtures do not qualify arbitrary production restarts. Physical operators,
frozen inputs and exact comparison semantics are unchanged. E5_1C and E5_2
remain OPEN until formal completion and terminal review.
