# P1R CPU-only HR-4 retest

This adapter runs the approved 768-task H workload on one 2-node/192-core Slurm allocation. Every receipt is labeled `P1R_DIAGNOSTIC_ONLY` and `NOT_PRODUCTION_QUALIFIED`. It does not implement the blocked optical branch, run a production propagation, or claim that the historical NEXT records were produced by a GPU backend.

## Frozen workload

The pre-submission `WORKLOAD_768.json` contains 16 repetitions × 48 real S5 POST screens in repetition-major order. Each task binds the POST raw-byte SHA256, canonical three-field hashes, hydro parameter hash, and its fixed 1-node and 2-node worker assignments. Parameter hashes use the repository `provenance_v2.raw_sha256` over sorted compact UTF-8 JSON; input and output files use `raw_sha256_file`; array fields use `KHz_filament.hr4e_timestep.sha256_array`.

Generate the workload offline from the accepted manifest and verified input records with `build_workload(details, records)`, serialize that returned object once, and freeze its raw SHA before staging. At execution, `--workload` is mandatory and the runner validates the exact object against the accepted manifest; it will not generate a replacement workload. The manifest itself is checked by the frozen P1 loader against the accepted raw SHA. Do not alter the manifest, records, physical parameters, or workload to make a run pass.

## Allocation and execution

`cpu_p1r_hydro.sbatch` requests partition `v6_384`, account `bscc-t`, QoS `normal`, two exclusive nodes, one 96-CPU task per node, 320 GiB per node, and 90 minutes. It checks Bash syntax and calls `audit_batch_entry.py` before the runner creates any output root. The launcher uses the fixed Python 3.12.8 interpreter, disables GPU selection, limits BLAS/thread pools to one thread, and uses only the supplied repository, vendor, input, and workload paths.

The batch arguments are:

```text
BENCH_ROOT OUTPUT_ROOT VENDOR_DIR INPUT_ROOT WORKLOAD_768.json
```

`INPUT_ROOT/candidate_input_manifest.json` is the frozen input manifest. The caller must provide an unused output-root leaf with an existing parent. No job is submitted by this implementation task.

The controller first runs a real two-rank `srun` probe. It requires separate rank receipts in a shared directory, distinct hostnames, 96 verified physical cores per rank, and the fixed interpreter. H1 runs `1N-A → 2N-A → 2N-B → 1N-B`; the 1-node stages use 96 workers × 8 tasks, and the 2-node stages use 2 × 96 workers × 4 tasks. Each worker binds to one physical core before loading the input and performs the unchanged CPU NumPy HR-4 call for 1000 steps with `require_stable=True`.

Stage service time runs from `srun` launch until the controller validates every expected task receipt, output NPZ raw SHA, and canonical field hash. Cross-stage field and observable comparison is timed separately. A canonical hash mismatch or any nonzero numeric observable delta preserves its metrics and stops remaining/optional stages with `NUMERICAL_DIFFERENCE`. Scheduler step teardown is checked by the named step; unresolved teardown prevents a later stage. Each node stops at 80% of its 320 GiB requested memory. The controller reserves 300 seconds at the end and 300 seconds for each remaining stage, and estimates a new layout from the 394-second historical baseline or measured same-layout service ×1.25 without assuming that two nodes halve runtime.

H2 is optional after all four H1 stages complete. It reuses the first 96 tasks with 24 workers, comparing NUMA-0 placement and four-NUMA-node striped placement in ABBA order. It is skipped if hardware or the remaining budget is insufficient.

## Output and interpretation

The output root is non-overwriting. It contains `environment.json`, `input_acceptance.json`, separate Slurm-step stdout/stderr files, exclusive per-rank probe receipts, per-stage/per-node task NPZs and receipts, `performance_summary.csv`, `cross_stage_comparison.jsonl`, stage summaries, and a final `status.json`. Per-task timings separate input load/hash, kernel wall/CPU, output write/hash, and service wall. Node CPU time is explicitly a sampled peak of process-tree CPU at one instant, not cumulative CPU; final allocated/consumed CPU accounting must come from Slurm `TotalCPU` and accounting evidence.

Historical NEXT comparisons are diagnostic only. The receipt states `historical_next_backend_identity=NOT_ASSERTED_BY_ACCEPTED_INPUT_MANIFEST`; no GPU-match or production qualification is inferred. CPU layout equality, complete scheduler steps, valid hashes, and passing software tests do not qualify CPU N5 science.

For the login-node software entry check, use the same fixed interpreter/vendor path and an unused output root:

```text
python cpu_p1r_hydro.py --entry-smoke --output-root <unused-output-root> --vendor-dir <vendor-dir>
```

The entry smoke uses only synthetic 6×5 fields, executes the frozen HR-4 function for 1000 steps in a child process, writes an NPZ and receipt, then rechecks raw and canonical hashes in the parent. It is marked `SOFTWARE_SMOKE_ONLY_SYNTHETIC` and is not a real-input benchmark.
