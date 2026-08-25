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
