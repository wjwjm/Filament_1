# Isaacs complete Eq. (27) project state

## C1 closure

- C1 implementation and closure commit: `459dd108b9873b0e8b18fe83111f386993cf5b9f`
- C1 gate: **PASS**; full propagation: not run; Slurm submission: none.
- Production/default paths and Raman parameters remain unchanged.

## C2 preparation

- Status: **prepared_not_submitted**.
- Prepared candidate: `120fs_talebpour_isaacs_complete_eq27.json`; the sole config diff is `raman.operator_mode: full_isaacs_eq27 -> full_isaacs_eq27_complete`.
- Authorized work: exactly one full 120 fs GPU job, 0 scans, 0 profiling jobs. On 2026-08-22, a read-only papp_cloud/HPC audit verified the account, scheduler records, GPU partition, and fixed comparator source hashes; no remote write, upload, or Slurm submission was performed.
- Parent scientific decision: **C2 admitted with explicit interpretation limits**. The candidate tests the complete combined Eq.27 implementation. Relative to the scalar baseline, electronic Kerr moves from the central scalar phase/shock approximation into the combined electronic+rotational Strang half-stages. This is accepted as part of the complete operator form, but the final result must not claim to isolate derivative algebra from finite-`dz` stage placement or electronic-rotational Heun coupling.
- Comparator precision qualification: fallback jobs `180748/180749` used `mixed_precision`, while the locked mother/candidate configuration retains its baseline default linear precision. Together with fallback staging provenance, this prevents describing the comparison as a strict same-run A/B pair.
- Current full Eq.27 comparator job `180748` and Raman-OFF comparator job `180749` are recorded as `fallback_verified_non_strict` provenance only through the fixed raw NPZ/metadata chain: the exact NPZ, job-metadata, diagnostic-report, configuration, execution-SHA, case, and RTX5090 evidence are hash-locked in `submission_manifest.json`. `prepare_isaacs_eq27_fallback_comparator_audit.py` derives axial/extras CSVs directly from those NPZs; caller-supplied CSVs, reports, metadata, or alternate raw paths are not accepted. The comparison re-derives the CSV hashes from raw NPZ before classification. Invalid jobs `179706` and `179988` remain excluded from physical classification.
- Candidate postprocessing writes axial/extras CSVs before publishing `isaacs_complete_eq27_reaudit.json`; the reaudit records job/status/gate, operator, provenance, numerical admission, and artifact paths/SHA256. The fixed fallback chain additionally records scheduler `COMPLETED 0:0` evidence without fabricating terminal state in job metadata.
- Any generated comparison report must explicitly qualify this fallback class as non-strict; a missing/nonfinite/crossing/overlap/numerical evidence gate stops with `insufficient_evidence` rather than producing `not_supported`.
- This C2 task supersedes the old Phase8C-0 suspension for this narrowly scoped operator test; it does not validate, revive, or reinterpret the old invalid jobs.

## C2 submission binding

- Submission manifest: `submission_manifest.json`; `campaign_id: isaacs_complete_eq27_c2`; fixed remote campaign root: `/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2`.
- The manifest records `expected_git_sha: null`, `execution_lock_required: true`, the derived configuration path and SHA256, `jobs_authorized: 1`, and `jobs_submitted: 0`; its resolution is `external execution_lock generated after final source commit` so the manifest does not self-reference its eventual commit.
- `create_isaacs_complete_eq27_execution_lock.py` reads this committed manifest only from a clean worktree, computes the actual clean HEAD plus manifest/config SHA256 values, and writes `status: authorized_not_consumed` to an external lock (default: `.git/codex-locks/...`). It never submits a job.
- The submit wrapper requires and hashes that external execution lock, then validates its schema/campaign/root, non-empty HEAD SHA, clean source HEAD, fixed source/derived config paths and hashes, unique flattened operator diff, fixed jobs/resources/GPU, manifest hash, fixed PyCAP path/SHA, fixed C1 gate artifacts/ancestor, and `authorized_not_consumed` status. It exports campaign/manifest/execution-lock/global-lock provenance to the batch metadata. It first creates a controlled provisional `RUN_DIR`, then atomically creates one campaign-wide consumed lock under the fixed root before invoking `sbatch`; a different `RUN_DIR` cannot bypass the consumed lock.
- Submission uses `sbatch --hold --parsable`; the strict numeric job id is recorded in read-only `RUN_DIR/job_receipt.json` together with the reservation token and manifest/lock/config bindings, then released with `scontrol release`. The batch script requires and validates that receipt against `SLURM_JOB_ID`, submission/global reservation tokens, and all fixed bindings before the single propagation. Receipt creation or release failure preserves the campaign lock, RUN_DIR, and held job with an independent failure record; submission/global records are never edited after `sbatch`.
- Any local preparation failure before `sbatch` removes only the marker-owned empty RUN_DIR and global lock record created by this invocation. Any nonzero or empty-job-id `sbatch` result is treated as ambiguous: the global lock and RUN_DIR remain, and `sbatch_failure_record.txt` records the exit code and bindings.
- Direct GitHub clone/fetch/`ls-remote` transport failed (GnuTLS timeout); the current source input is therefore an external, machine-bound verified Git bundle staging provenance JSON with schema `khz_filament.isaacs_complete_eq27.staging_provenance.v1` and method `verified_git_bundle_after_remote_github_transport_failure`.
- The staging provenance class is `verified_bundle_non_strict`: it binds the external provenance file SHA, expected Git SHA, branch, `github_push_verified: true`, bundle path/SHA, and non-empty remote-failure logs to the clean execution HEAD. The wrapper and batch revalidate it; the held receipt and candidate raw chain bind its path/SHA/method/source class. This source does not equal verified remote GitHub push/fetch provenance.
- No physical model, propagation configuration, coordinate/onset definition, comparison threshold, frozen result, manifest, submission record, or global consumed record was changed by this provenance repair; no GitHub/HPC connection, Slurm submission, propagation, or commit was performed.

## Existing C1 audit record

- Baseline SHA: `c9d9b952c4c23d6839374bdc5de184f0cd389eb3`
- Audit current HEAD (C1 closure commit): `459dd108b9873b0e8b18fe83111f386993cf5b9f`
- Branch: `codex/isaacs-raman-reclosure`
- Git dirty: `True`
- Implementation diff hash (sha256): `c08056a474cd0fc53d37a9fdb5afd14055a8aa2dc3feb0ae1ca70a9e25f244b7`
- Changed paths:
  - `Filament_python/KHz_filament/Config_explain.md`
  - `Filament_python/KHz_filament/README.md`
  - `Filament_python/KHz_filament/config.py`
  - `Filament_python/KHz_filament/config_normalize.py`
  - `Filament_python/KHz_filament/diagnostics.py`
  - `Filament_python/KHz_filament/propagate.py`
  - `Filament_python/KHz_filament/raman.py`
  - `Filament_python/results/isaacs_complete_eq27/PROJECT_STATE.md`
  - `Filament_python/results/isaacs_complete_eq27/c1_closure_summary.json`
  - `Filament_python/results/isaacs_complete_eq27/c1_operator_report.md`
  - `Filament_python/tests/test_isaacs_complete_eq27.py`
  - `Filament_python/tools/audit_isaacs_complete_eq27.py`
- Scope: complete Eq.27 electronic+rotational operator closure only
- Raman parameter change: none
- Production/default change: none
- Full propagation: not run
- Slurm submission: none
- Overall C1 gate: **PASS**
- C2 status: parent scientific decision completed; the single authorized job remains unsubmitted until the final clean commit, execution lock, and submission gates pass
