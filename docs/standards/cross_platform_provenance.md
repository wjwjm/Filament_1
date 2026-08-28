# Cross-platform provenance and line-ending hashes

## Authority and invariant

`Filament_python/tools/hpc_ops/provenance_v2.py` is the only authoritative
implementation of file hashing for new Filament HPC manifests. Call it as a
CLI or import its helpers; do not reproduce canonicalisation or SHA256 logic in
launchers, postprocessors, or campaign-specific Python.

Every file is classified before hashing:

| Classification | Intended files | Required binding | `hash_scope` |
| --- | --- | --- | --- |
| `tracked_text` | committed Git text | Git blob OID and canonical-LF SHA256 | `git_blob_oid+canonical_lf_sha256` |
| `external` | external files and exact-byte binary artifacts, including deliberately binary Git artifacts | raw-byte SHA256 | `raw_bytes` |

The manifest-level `hash_scope` is `classified_by_record`. LF and CRLF
checkouts of the same committed text therefore have the same canonical digest,
while raw-byte hashes remain line-ending and byte sensitive.

## Classified v2 schema

```json
{
  "schema": "filament.provenance.v2",
  "version": 2,
  "repository": "repository identity without credentials",
  "repository_path": "/checkout/path/when-created",
  "head": "full Git object ID",
  "branch": "named-branch",
  "hash_scope": "classified_by_record",
  "line_endings": {
    "tracked_create": "canonical-LF-from-Git-blob",
    "tracked_validate": "canonical-LF-worktree-match",
    "external": "raw-bytes"
  },
  "records": [
    {
      "path": "Filament_python/configs/example.json",
      "classification": "tracked_text",
      "hash_scope": "git_blob_oid+canonical_lf_sha256",
      "git_blob_oid": "0123456789abcdef0123456789abcdef01234567",
      "canonical_lf_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    {
      "path": "/absolute/path/to/external/result.npz",
      "classification": "external",
      "hash_scope": "raw_bytes",
      "raw_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    }
  ],
  "created_at_utc": "2026-08-27T00:00:00Z"
}
```

The `repository_path` records creation context but is not an equality gate,
because Windows and Linux checkout roots differ. Repository identity, branch,
HEAD, each tracked Git blob, canonical content, and each external raw digest
are gates. Strict validation requires the exact classified schema. The older
v2 `tracked_text`/`external` grouped shape remains readable only when
`--require-hash-scope` is not requested.

## Creation and validation

Create a new manifest only after the tracked inputs are committed and the
worktree is clean. Write the manifest outside the repository so it cannot make
its own creation dirty:

```text
provenance_v2.py create --repo <repo> --output <external-manifest> \
  --tracked <repo-relative-text-paths> \
  --external <absolute-binary-or-external-paths>
```

For tracked text, creation hashes the committed blob and confirms that the
canonical-LF worktree bytes match it. It does not rewrite LF, CRLF, or any
file. For external/binary records, the exact file bytes are hashed.

New HPC campaigns must validate with `--require-hash-scope` before creating a
run directory, lock, receipt, execution manifest, or scheduler job. The
campaign may copy the already validated provenance manifest into its run root;
that copied manifest is an external exact-byte artifact and is bound by raw
SHA256. A tracked planning manifest remains a tracked-text record and must not
be rebound using the Windows checkout's raw SHA256.

## HR-2E enforcement and compatibility

The new classified path is enforced by:

- `hpc_preflight.sh --provenance-manifest ...`;
- `submit_hr2e_schedule_convergence.sh --provenance-manifest ...`;
- `hr2e_schedule_convergence.sbatch` before its run-directory/cache creation;
- `hr2e_schedule_convergence.py` and
  `postprocess_hr2e_schedule_convergence.sh` for classified job metadata and
  execution manifests.

New HR-2E preparation emits `khz_filament.hr2e.stage1_preflight.v2`; new jobs
emit job metadata and execution manifest v2. The analysis path keeps explicit
read-only support for already frozen v1 HR-2E outputs. Older C2, Phase 8,
historical launchers, receipts, and locks are not migrated by this change.

## Frozen `.gitattributes` exceptions

These exceptions retain their historical checkout bytes and must not be
normalised or rewritten:

- binary: `Filament_python/results/isaacs_complete_eq27/provenance_221822/execution_lock_43ac6b4.json`;
- CRLF: `Filament_python/results/isaacs_complete_eq27/c1_closure_summary.json`;
- CRLF: `Filament_python/results/isaacs_complete_eq27/c1_operator_report.md`;
- CRLF: `Filament_python/results/isaacs_complete_eq27/submission_manifest.json`;
- CRLF: `Filament_python/results/density_translation_width/density_translation_width_20260715_002/paper_pycap_120fs.csv`;
- explicit LF: `Filament_python/results/isaacs_complete_eq27/provenance_221822/SUBMISSION_LOCK`;
- explicit LF: `Filament_python/results/isaacs_complete_eq27/provenance_221822/submission_record.txt`.

All seven files under
`Filament_python/results/isaacs_complete_eq27/provenance_221822/` remain frozen
and are never rewritten or migrated to v2.

## Project-independent Codex global rule draft

> Classify every input before hashing. For Git-tracked text, bind the committed
> Git blob OID and a canonical-LF SHA256; never use platform-specific checkout
> bytes as the cross-platform identity. For external or binary artifacts, bind
> exact raw-byte SHA256. Every new manifest must declare its hash scope. Preserve
> legacy frozen receipts and line-ending exceptions byte-for-byte; validate
> them through their existing compatibility path rather than rewriting them.
