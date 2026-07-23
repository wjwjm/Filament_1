# Phase 8C-A A3 baseline profiling smoke

Status: completed and stopped as authorized. This is the only A3 GPU allocation; it did not prepare a Test A production job or enter A4.

## Provenance and resource record

- Source and remote branch SHA: `4fdedbc4d7a7fb1970d026a56ca27ec297d5ec0c`
- Operational baseline: `f70c5f48dd11e6db2376604751c8b13afdc1cd2f`
- Frozen physics baseline: `e11d13f103c484953c0f733aa9b410bff385b2b5`
- Staging: `/data/run01/scvi806/user_Wangjimin/phase8c_a3_staging/Filament_1_4fdedbc4d7a7_attempt3_sparse`
- Slurm: Job `180573`, `COMPLETED`, exit `0:0`, elapsed `00:03:20`
- Resources: one RTX 5090 GPU, eight CPU threads, N50 site-default 126,000 MB host memory, one-hour limit.

The staging HEAD, runtime SHA self-check, clean worktree/index, Git fsck, and both requested ancestry checks passed. The existing dirty remote repository was not used or modified. The original partial staging and a failed full-checkout attempt were preserved rather than reused.

## A3 contract and numerical result

Both fresh processes used the full `512×512×384` grid, 20 accepted steps to nominal `z=0.002 m`, BK-NEE `mixed_precision`, full Isaacs Eq.27 Raman ON, legacy Raman phase/absorption OFF, electronic Kerr/self-steepening/plasma/ionization ON. The two generated configurations were identical after removal of `propagation.diag_bk_nee_profile`.

Both runs completed 20 steps with no NaN/Inf. The maximum linear half-step residual was `2.1169e-11 J`; Raman step-closure p99 was `1.5019e-4`, cumulative closure `2.8315e-5`, and legacy `alpha_R` was zero. The Eq.27 path retained two Raman operator substeps and four convolutions per z-step.

The selected linear strategy was `mixed_precision`, not R5 `unitary_projection`. The frozen Raman operator did report pre-existing `raman_energy_projection_*` diagnostics (maximum three internal iterations); this is recorded explicitly for scope review and was not introduced by an A3 configuration change.

## Runtime, memory, and profiling

| Run | Mean step | 15,000-step linear projection | Peak reserved GPU memory |
| --- | ---: | ---: | ---: |
| Profiling off, cold-start | 2.9477 s | 12.28 h | 32.638 GB / 33.669 GB (96.94%) |
| Profiling on, warm-cache | 1.7465 s | 7.28 h | 32.636 GB / 33.669 GB (96.93%) |

The profiling-on process ran second in the same allocation, after driver/JIT/cache warm-up. Its apparent negative timing overhead is therefore invalid as a comparison and is not treated as a speedup. Neither observed projection meets the prior `< 6.4 h` performance gate; peak reserved GPU memory also exceeds an 85% guardrail.

Within the profiled BK-NEE stages, forward and inverse spatial FFTs dominate: 28.02% and 26.38%, respectively; transfer-kernel preparation is 14.13%. The profiler recorded 8.777 s explicit synchronization time. Stage intervals include synchronization boundaries, so their sum and the synchronization total overlap; the published naive accounting residual is observational only.

## Evidence assets

- `a3_evidence/a3_profile_summary.json` — remote lightweight aggregate.
- `a3_evidence/a3_staging_provenance.json` — SHA/staging audit.
- `a3_evidence/a3_runtime_preflight.json` — pre-GPU runtime guard.
- `a3_evidence/a3_config_pair_manifest.json` — identical-config proof.
- `a3_evidence/profile_off/` and `a3_evidence/profile_on/` — lightweight metrics, config audits, and diagnostic reports only; no NPZ was downloaded.

Next action: stop. A4, another smoke, and all full 15,000-step Test A jobs remain unprepared and unsubmitted.
