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
