# Campaign management tools

`manage.py` is a local, standard-library-only record-keeping tool for the
three-end Filament_1 workflow. It does not run a simulation, submit Slurm, or
connect to HPC/GitHub.

The stable identity of a campaign is its `campaign_id`, execution Git SHA,
configuration SHA256 values, and artifact-manifest SHA256 values. Campaign IDs
use the form `YYYYMMDD_<topic>_<variant>_vNN`, for example
`20260825_demo_case_v01`.

Typical local flow:

```powershell
python tools/campaign/manage.py init 20260825_demo_case_v01
python tools/campaign/manage.py publish-config 20260825_demo_case_v01 input.json --kind requested
python tools/campaign/manage.py publish-config 20260825_demo_case_v01 resolved.json --kind resolved
python tools/campaign/manage.py build-manifest 20260825_demo_case_v01
python tools/campaign/manage.py check 20260825_demo_case_v01 --level lite
python tools/campaign/manage.py publish-plan 20260825_demo_case_v01 --allow metrics/*.csv
python tools/campaign/manage.py publish-plan 20260825_demo_case_v01 --allow metrics/*.csv --apply
```

`init` creates campaign metadata, requested/resolved working directories under
`configs/experiments/<id>/`, and the ignored local artifact root
`.artifacts/<id>/`. `publish-config` reads an input without changing it, writes
the reviewed snapshot under `results/campaigns/<id>/configs/`, and refuses
credential-like keys, authenticated URLs, and absolute paths. `build-manifest` hashes regular
files in sorted relative-path order and rejects symlinks. `publish-plan` is a
dry-run by default; `--apply` copies only explicitly allowlisted files and
never overwrites a differing destination.

Validation is intentionally staged:

| level | boundary |
| --- | --- |
| `lite` | local campaign metadata and optional hash format checks |
| `submit` | staging checkout, config hashes, batch-entry receipt, HPC target path |
| `publish` | terminal scheduler evidence, successful attempt, derived manifest, safe evidence |
| `archive` | raw/derived manifests, explicit publication status, HPC path |

Receipts are cached at `.artifacts/<id>/.validation/` using the actual campaign
JSON, configuration/manifest and referenced-file hashes, live staging state,
batch-audit receipt, GitHub evidence state, validation level, and execution
SHA. Changing any of those inputs creates a new receipt. Receipts
are bookkeeping only and are ignored by Git.

`register-legacy` reads the frozen repository inventory and writes a mechanical
`results/campaigns/legacy_registry.json`. It records the existing
`Filament_python/results/` directories without moving, deleting, or
reinterpreting their contents.
