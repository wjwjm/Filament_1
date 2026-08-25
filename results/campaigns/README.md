# Campaign evidence

`results/campaigns/<campaign_id>/` contains small, explicitly published
evidence and its provenance. It is not a mirror of HPC raw data.

- Complete local derived artifacts live in the ignored
  `.artifacts/<campaign_id>/` directory.
- HPC remains authoritative for raw NPZ/MAT/HDF5 data and full scheduler
  receipts.
- Only files selected by `publish-plan --allow <specific-pattern>` may be
  copied into `artifacts/` here.
- `publish-plan` is dry-run by default and will not overwrite a differing
  destination.
- Raw binary results, caches, credentials, and secret-like paths are rejected
  from GitHub evidence manifests.

The stable cross-end references are the campaign ID, execution Git SHA,
requested/resolved configuration SHA256 values, and artifact-manifest SHA256.
Historical `Filament_python/results/*` directories remain in place and are
mechanically listed in `legacy_registry.json` with status
`legacy_unclassified`.

Phase 8C HPC legacy evidence is represented by two compact campaign records:

- `20260723_phase8c_a3_enablement_v01`
- `20260723_phase8c_b_full_raman_v01`

Each record preserves separate `legacy_runs` and `legacy_staging` path mappings,
manifest/receipt hashes and execution/config evidence without publishing raw
NPZ, staging checkouts or scientific reclassification. Full small receipts and
manifests remain in `.artifacts/<campaign_id>/hpc_relocation/`.

Phase 8B HPC legacy evidence is represented by five compact campaign records:

- `20260721_phase8b_preflight_smoke_v01`
- `20260721_phase8b_r1_job1_audit_v01`
- `20260721_phase8b_r2_job1_attempts_v01`
- `20260722_phase8b_r4_linear_smoke_v01`
- `20260722_phase8b_r5_precision_v01`

The records preserve all eight source components, including the R1 initial/v2
sequence and the R2 original/replacement/replacement-retry sequence. Archive
acceptance records management completeness only: smoke, audit, diagnostic and
precision evidence is not promoted to a new scientific conclusion. Full small
receipts and manifests remain in `.artifacts/<campaign_id>/hpc_relocation/`.

The three pre-final historical FR roots are represented by one precursor
campaign, `20260820_historical_fr_mixture_precursor_attempts_v01`. It preserves
the empty `1200Z` root and the separate `1220Z` and `1350Z` checkouts with
their distinct Git HEAD values. This record is not merged into or interpreted
as replacing `20260820_historical_fr_mixture_final_v01`.

The five top-level domains formerly under the dirty legacy repository's
`Filament_python/outputs/` directory are represented by five compact campaign
records. Each complete domain is preserved as an `outputs` component, including
NPZ, MAT, source tar files, logs, embedded source copies and failed-preflight
evidence. Archive acceptance records management completeness only and does not
revise the scientific conclusions contained in the legacy reports.

The former dirty-repository `.codex_stage_bundles` root is represented by the
diagnostic/development campaign
`20260716_vacuum_focus_profile_scan_attempts_v01` plus a
`legacy_source_packages` component appended to
`20260715_vacuum_focus_validation_ft90_v01`. The four scan directories retain
their attempt ordering without selecting a final scientific result; the FT90
tar/zip packages are provenance only. Full small manifests and receipts remain
in the corresponding ignored `.artifacts/<campaign_id>/hpc_relocation/`
directories.

The 42 earliest registered single-pulse filament jobs are grouped mechanically
under `20260430_single_pulse_filament_early_attempts_v01`. Each job has a
separate `attempts/<job_id>/` directory containing its scheduler log and, for
26 jobs, the associated MAT result. Six completed jobs with no MAT and ten
failed or cancelled jobs remain explicit in the registry; archive acceptance
does not select a final result or establish scientific validity. The legacy
LUT diagnostics, reusable rate-table cache, two LUT build logs and three MATLAB
Live Scripts were excluded from that campaign and handled by the final
repository-freeze batch described below.

The final dirty HPC repository has now been frozen at
`legacy/repository/Filament_1_dirty_bb592ef`. Its complete `.git`, dirty
worktree, untracked files, `wang-local-wip` branch and two stashes are retained
for reference only; it is not an execution repository. Before the atomic move,
the final residuals were separated into the
`20260401_ion_lut_validation_diagnostic_v01` campaign, the shared legacy
`cache/rate_tables` snapshot and the archived MATLAB Live Script reference
set. Future HPC execution must use a clean checkout from GitHub `main` at an
explicit commit or immutable tag.
