# HR-4E-5 HPC audit pack

**Snapshot date:** 2026-09-16
**Purpose:** make the code under review, the queued S5-FINAL job, historic
S3/S4R/S5 resource evidence, storage position, and software evidence
auditable before a formal HR-4E-5 campaign is authorized.

This is a read-only evidence pack, not an S5 result or a formal E5 submission
plan.  Values explicitly marked **missing** were not inferred from requested
resources, queue state, or a different job.

## 1. Current S5-FINAL: source, submission, and effective inputs

### 1.1 Code-to-job alignment

| Item | Verified value |
| --- | --- |
| GitHub review branch | `origin/codex/hr4e5s-s5-lifecycle` |
| Review base at collection | `8755a4eab0b64c3cbef4680a35529a1e4ab60a89` |
| Queued S5-FINAL execution SHA | `cd456ff8413cbc041d2d60b9b64007a1554028a1` |
| Relation | `cd456ff` is an ancestor of the pushed review tip; the new design/audit documents are not runtime inputs to job `244700`. |
| Remote worktree | `/data/run01/scvi806/user_Wangjimin/projects/Filament_1_hr4e5s_s5_final_cd456ff` |
| Remote worktree HEAD | `cd456ff8413cbc041d2d60b9b64007a1554028a1` |
| Remote tracked-tree check | clean: `git status --porcelain=v1 --untracked-files=all` returned no records. |
| Run root | `/data/run01/scvi806/user_Wangjimin/projects/hr4e5s_s5_final_cd456ff_r3` |
| Initial job | `244700`, `PENDING (Priority)` at this snapshot; controller `WAIT_FOR_DUAL_CLAIMS`, `signal_sent=false`, `recovery_job_id=null`. |

The job therefore uses a committed, clean execution worktree.  It is not using
the newly pushed documentation commit at runtime, and it has no recorded
runtime override of its tracked source at this snapshot.

### 1.2 Actual submission realization

The receipt records one initial submission:

```text
case_id                 S5_FINAL_WORKER_LOSS
case_mode               initial
job_id                  244700
resources               1 optical GPU + 2 hydro GPUs
execution_sha           cd456ff8413cbc041d2d60b9b64007a1554028a1
```

The scheduler independently reports:

```text
Command = .../Filament_python/tools/hr4e5s_s5_final.sbatch
WorkDir = .../hr4e5s_s5_final_cd456ff_r3
StdOut  = .../initial-244700.out
StdErr  = .../initial-244700.err
ReqTRES = cpu=24, mem=378000M, gres/gpu=3
TimeLimit = 12:00:00
QOS = normal; Partition = gpu
```

The committed submit entry constructs this initial allocation as:

```text
sbatch --parsable --job-name=e5s-s5-final-initial
  --chdir=<RUN_ROOT> --gres=gpu:3 --ntasks=3
  --output=<RUN_ROOT>/initial-%j.out --error=<RUN_ROOT>/initial-%j.err
  --export=ALL,EXPECTED_GIT_SHA=<SHA>,REPO_DIR=<REPO>,RUN_ROOT=<RUN_ROOT>,
           CASE_MODE=initial,INPUT_MANIFEST=<RUN_ROOT>/s5_final_input_manifest.json,
           LUT_WORKSPACE=<RUN_ROOT>/lut_workspace,
           REFERENCE_CASE_ROOT=<clean-reference-root>
  <REPO>/Filament_python/tools/hr4e5s_s5_final.sbatch
```

The monitor manifest binds `<clean-reference-root>` to
`/data/run01/scvi806/user_Wangjimin/projects/hr4e5s_s5r2_aef5a76_r1/clean`
and names `hydro_consumer_1` as the only permitted external-loss target.
The recovery job may be submitted only by that monitor after quiescence and
the frozen recovery-effects receipts; it has not been submitted.

### 1.3 Code and input hashes

The r3 provenance record uses Git-blob OID plus canonical-LF SHA-256 for
tracked text, and raw-byte SHA-256 for external inputs.

| Runtime input | Git blob OID / SHA-256 |
| --- | --- |
| `hr4e5s_streaming.py` | `d6c1341791fdd7fd131136bac7939997e24f94d8` / `69db846dcf41f4f131e5506e94bdae5a8bbfbc029fc7c6cb97238fd0fc60366c` |
| `hr4e5s_s5_final.py` | `0d6f72d3aa2ca90e72516f350000aa874a0847b9` / `bd358869f74e46037453057eaa5aa426fe457b9561b1beaf4962ce9a74ffd12d` |
| `hr4e5s_s5_final.sbatch` | `042ee2dc78877bbf5b09a4020a7876a800074850` / `72db54d2915359a842e9286192ac51055c7da56a078e1f2a79942bc45b0b16af` |
| S5-FINAL monitor | `db1ef45e921c63b34b08d91b412d96fb59df1d4f` / `9663a26f57caa2141a66a7561209abb7ff1bcb9fffa63e5d3dadae563c325bdb` |
| S5-FINAL submit entry | `18397e0c1b95e7fb57c570bbc79cd245641870e0` / `e13fcd4e527a2c29693c36c9c7aa54025a2f6f27c00ab5bb399ff694a598b653` |
| Source configuration (raw bytes) | `real_post_input_config.json`: `eaec83ad326a29af95881912db76a7e5c26943dec9e0d6876ca9c9cc2a100c22` |
| Source state file (raw bytes) | `E1B_hr3b_source.hr3b_delta_n_th.npy`: `70677c01564ad089d985214e2767a221f10c5e1ef97f42a9f367a89edf81f467` |
| Source state array | `5990da24bec80937bf3be9985777b3797be9adffedb776dd2f2e840b118d8ad9` |
| Source manifest (raw bytes) | `3e2c7557e5e8027ac1d9fb42411fe885fe3ea13bef9ee60e91538462adaf32d2` |
| Prepared input manifest | `280d8d1f47b8cf4cdca1ae51bea24706a461fe31562e9fa2d0a3d2ab6b5f5a8a` |
| LUT-workspace report | `5ef79cc1802300720f6de68dd1a1d14b1ae9fc393815216f0e19eae703ce4936` |

The effective S5 input is 48 contiguous source screens `7998..8045`, on a
`351 x 301` float64 grid at `dx=dy=10 um`.  It fixes hydro block size 8,
queue depth 16, `dt_hydro=1 us`, 1,000 hydro steps, `chi=2.17e-5 m^2/s`,
`nu=1.5e-5 m^2/s`, and `n0=1.00027`.  The source identity is one frozen
E1B source plus one identical pulse; it is a 48-screen qualification input,
not a formal E5 full-z/multi-pulse configuration.  Fault injection defaults
to `DISABLED`.

## 2. Representative completed evidence and resource statistics

### 2.1 Scheduler accounting

| Stage / job | Topology | Terminal scheduler evidence | Elapsed |
| --- | --- | --- | --- |
| S3 batch `233394` | 1 GPU / 8 CPU | `COMPLETED / 0:0` | 03:06:26 |
| S3 stream repeat 1 `234320` | 1 optical + 1 hydro / 16 CPU | `COMPLETED / 0:0` | 03:04:55 |
| S3 stream repeat 2 `234321` | 1 optical + 1 hydro / 16 CPU | `COMPLETED / 0:0` | 03:48:42 |
| S4R replay `236541` | 6 hydro / 48 CPU | `COMPLETED / 0:0` | 00:12:50 |
| S4R stream `237236` | 1 optical + 2 hydro / 24 CPU | `COMPLETED / 0:0` | 03:44:29 |
| S4R stream `237235` | 1 optical + 4 hydro / 40 CPU | `COMPLETED / 0:0` | 02:59:28 |
| S4R stream `237234` | 1 optical + 6 hydro | `CANCELLED by 1812`, before execution | 00:00:00 |
| S5 clean reference `241098` | 1 optical + 1 hydro / 16 CPU | `COMPLETED / 0:0` | 03:03:53 |
| S5 F03 recovery `244318` | 1 optical + 1 hydro / 16 CPU | `COMPLETED / 0:0` | 03:02:11 |
| S5 F04 recovery `244319` | 1 optical + 1 hydro / 16 CPU | `COMPLETED / 0:0` | 03:03:54 |
| S5 F05 recovery `244345` | 1 optical + 1 hydro / 16 CPU | `COMPLETED / 0:0` | 03:00:41 |
| Earlier S5-FINAL `244497` | 1 optical + 2 hydro / 24 CPU | `FAILED / 1:0`, launcher defect before useful work | 00:00:18 |
| Current S5-FINAL `244700` | 1 optical + 2 hydro / 24 CPU | `PENDING`; no allocation yet | 00:00:00 |

`sacct` reports each completed job's requested/allocated TRES but leaves
`MaxRSS`, `TotalCPU`, `MaxDiskRead`, and `MaxDiskWrite` empty.  Requested
memory is therefore not treated as observed peak host memory.

### 2.2 Existing application-level telemetry

| Evidence | Verified statistics | What it supports |
| --- | --- | --- |
| S3 | Each streaming repeat completed 432/432 exact comparisons; both had zero shape, dtype, hash, array, and scientific-provenance mismatches. | S3 batch/stream exact equivalence for its fixed 48-screen scope. |
| S4R 1+2 (`237236`) | Application stream 13,431.190 s; optical active 13,425.556 s; hydro span 149.566 s; hydro rate 0.320888 screens/s; median hydro block 47.465 s. | Existing timing envelope for the fixed 48-screen stream only. |
| S4R 1+4 (`237235`) | Application stream 10,720.670 s; optical active 10,714.485 s; hydro span 83.364 s; hydro rate 0.575570 screens/s; median hydro block 36.537 s. | The preferred tested topology's fixed-scope timing. |
| S4R six-hydro replay (`236541`) | 384 screens; hydro service span 483.430 s; 0.794324 screens/s; 1,152/1,152 exact comparisons; capacity ratio 1.0504. | Replay qualification, not a full streaming endurance claim. |
| S5 F03/F04/F05 | Each corresponding recovery job reached 432/432 exact PASS in the accepted R4 evidence. | Representative single-worker recovery evidence, not the pending two-hydro-worker S5-FINAL result. |

The S4R 1+6 point has no runtime statistics because it was cancelled before
execution.  `244700` has no stdout, stderr, GPU-memory, I/O, or application
telemetry yet because it has not started.  No S3/S4R/S5 evidence captured a
device-event separation inside `advance_hr4_single_screen`; reported hydro
envelopes include host/device transfer and GPU compute where applicable.

## 3. Partition, QoS, and allocation implications

Current `gpu` partition evidence:

- Five nodes, 40 GPUs and 320 CPUs in total; `DefCpuPerGPU=8`.
- `DefMemPerCPU=MaxMemPerCPU=15,750 MiB`.
- Partition `MaxTime=UNLIMITED`; its current S5-FINAL `12:00:00` limit is the
  submitted job limit, not a demonstrated partition ceiling.
- Job `244700` uses `QOS=normal`.  Its visible `normal` QoS fields did not
  expose numeric walltime/TRES/job-count limits.  The account also exposes a
  `gpugpu` QoS record with `MaxTRES=node=16`, but this is not the QoS recorded
  for `244700`.

The completed fixed-window runs show that a normal S3/S4R or S5 clean/recovery
leg fits inside a 12-hour allocation.  S5-FINAL itself is deliberately a
**two-allocation** contract: the externally killed initial job must fail fast
and a newly submitted recovery job completes the test.  It cannot be counted
as a single-allocation end-to-end workflow.  Its cross-job path remains
unqualified until `244700` runs and the controller's terminal audit passes.

No full-z/multi-pulse formal E5 duration can be inferred from these 48-screen
timings.  The pulse count, full-z screen count, retained diagnostics, and
formal topology are not yet authorized.

## 4. Storage position, quota evidence, and retention boundary

### 4.1 Observed space and existing roots

| Item | Observed value |
| --- | --- |
| Filesystem for `/data/run01/.../projects` | JuiceFS, 300 GiB total / 121 GiB used / 180 GiB available (41% used) |
| Current S5-FINAL r3 preflight root | 14 MiB |
| S5 clean-reference root | 1.2 GiB |
| S3 evidence root | 2.4 GiB |
| S5 r3 LUT preflight free-space observation | 192,939,368,448 bytes (about 179.7 GiB) |

The full-z storage requirement must be preflighted rather than extrapolated
from the 48-screen directories.  The HR-4C implementation's own storage
model is:

```text
one three-field generation = 3 * K * Ny * Nx * dtype_bytes
authoritative plus staging  = 2 * one three-field generation
```

For an illustrative, **not-yet-authorized** 15,000-interval full-z state at
the current `351 x 301` float64 geometry, that is 35.422 GiB for one
three-field generation and 70.845 GiB for the six current/next field slots,
before NPZ products, diagnostics, cached LUTs, filesystem metadata, or any
retained historical checkpoints.  A formal E5 input manifest must determine
the actual `K`, state/checkpoint-retention policy, and a distinct storage
budget before run-root creation.

### 4.2 Missing site-level information

The login host has no `quota` command, so no user/project quota was verified.
`df` is filesystem-wide free space, not a user allocation.  No site retention
policy or automatic purge rule was available from the checked repository or
the read-only scheduler/filesystem evidence.  Existing S3/S4R/S5 roots remain
preserved; this pack authorizes no cleanup.

## 5. Software environment record

The S5 batch entry fixes the interpreter to
`/data/home/scvi806/.conda/envs/Filament_python/bin/python`, activates the
`Filament_python` Conda environment, requires a CuPy backend, and sets
`UPPE_USE_GPU=1`, `PYTHONPATH=<REPO>/Filament_python`, and eight OMP,
OpenBLAS, and MKL threads per Slurm task.

Read from that fixed interpreter on 2026-09-16:

| Component | Observed value | Limit |
| --- | --- | --- |
| Python | 3.13.7 | Live interpreter read, not yet a job-prologue receipt for `244700`. |
| CuPy | 13.6.0 | Same limitation. |
| NumPy | 2.3.3 | Same limitation. |
| SciPy | 1.16.3 | Same limitation. |
| CUDA driver/runtime, GPU model, GPU memory | **missing** | `nvidia-smi` is unavailable on the login node; existing S4 submission records did not capture these fields. |
| Container image/digest | **missing** | No container reference appears in the S5 batch entry or its preflight record. |

Before a formal E5 job, capture the exact interpreter/package set, GPU model,
driver/CUDA runtime, and visible-device mapping in a run-local prologue
receipt.  That addition must be separately reviewed if it changes the frozen
runtime entry or diagnostics contract.

## 6. Evidence sources and follow-up boundary

Primary sources are the remote S5 r3 `scontrol`, `sacct`, monitor, preflight,
input-manifest, provenance, LUT, and filesystem records; local S3/S4R
closeout manifests and copied telemetry supply the completed-history table.
The governing S5 and formal-E5 boundaries remain:

- [`hr4e5s_s5_fault_contract.md`](hr4e5s_s5_fault_contract.md)
- [`formal_hr4e5_task_design_draft.md`](formal_hr4e5_task_design_draft.md)

When S5-FINAL reaches a terminal controller state, append a separate,
verified S5 terminal audit.  Do not overwrite this snapshot, resubmit a job,
or promote formal E5 from design to execution merely because this information
pack is complete.
