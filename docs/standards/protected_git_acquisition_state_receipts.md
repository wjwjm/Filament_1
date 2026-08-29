# Protected Git acquisition state receipts

Protected clone/fetch operations use a small persistent receipt so a lost
caller response is not mistaken for failure.

- Schema: `filament.hpc_git_acquisition.v2`.
- Required identity: UUID `operation_id`, absolute `state_file` under
  `staging-root` and outside the final checkout target.
- State flow: `started → acquiring → checkout_verified → completed`; any
  operation failure records `failed`.
- Every update writes a temporary sibling file and atomically renames it over
  the same state path. No partial JSON is considered a receipt.
- `--inspect-state --state-file` is read-only. A missing file is
  `unknown_no_receipt`, not `failed`; do not retry clone, delete the target, or
  automatically switch to a bundle.
- Bundle fallback accepts only `<bundle>.verified`. A `.part.<operation-id>`
  file must first pass SHA256, `git bundle verify`, and expected ref/HEAD checks,
  followed by an atomic rename to `.verified` outside this helper.
- The PowerShell wrapper reports `completed` only from a valid remote final
  JSON receipt. Missing or malformed final JSON is `unknown_no_receipt`.

This mechanism does not add a service, retry loop, scheduler action, or remote
database. It does not alter production physics, configs, or frozen results.
