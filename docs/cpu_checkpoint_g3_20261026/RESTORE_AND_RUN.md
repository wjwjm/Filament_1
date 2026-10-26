# CPU G3 research checkpoint

State: `G3_SMALLGRID_QUALIFIED / FULLSIZE_NOT_YET_CONFIRMED / PRODUCTION_FALSE`.
Default remains I0. This branch is not merged into main and does not qualify
full-distance propagation, focus-region behavior, multiple pulses or Hydro.

The scientific subtree is `821fec5bdacd832b0347e1f2e84285cb54172e57`.
The historical CPU deployment commit is
`66801ad435d6ec1967338cee4eb82af2a152389c`; the I factory anchor is
`3d4cbb54bc216c3fcfcd85bdd9c77a250e4bb0c3`. These are distinct from this
checkpoint's actual commit. Source guards read the real checkout's Git blobs;
no historical Git directory or pretend HEAD is needed.

## Required external data

No scientific arrays, environment binaries, caches or review ZIPs are tracked.
The three shape-independent LUT artifacts are listed with exact sizes and
raw SHA256 in `EXTERNAL_DATA_MANIFEST.json` and `prepare_data.py`. Obtain them
from the existing frozen CPU input archive or its custodian, preserving the
listed relative names. There is no automatic network retrieval. Missing or
changed artifacts are rejected before a target data directory is created.

From the checkout, using an explicitly configured external Python environment:

```powershell
$env:PYTHONPATH=Join-Path (Get-Location) 'Filament_python'
& $PythonExe -s -B -m tools.cpu_s4_checkpoint_support.prepare_data `
  --source-lut-root $FrozenLutRoot --data-root $NewExternalDataRoot
if($LASTEXITCODE -ne 0){throw 'External data verification failed'}
& '.\Filament_python\tools\cpu_s4_checkpoint_support\run_portable_tests.ps1' `
  -PythonExe $PythonExe -DataRoot $NewExternalDataRoot -OutputRoot $NewTestOutput
```

`$PythonExe`, `$FrozenLutRoot`, `$NewExternalDataRoot` and `$NewTestOutput` must
be explicit absolute paths. Test outputs/caches must be outside the checkout.
Use the repository's original local-test runner, not bare pytest. The wrapper
runs backend, import and sanity before the directed tests. Local Numba pool,
FFT and Raman workers are explicitly one; this is not a local W48 claim.

The directed set retains the original 30 lifecycle/science/strict/boundary
checks plus the two final entry checks. It adds source-location/Git binding
checks and a real micro propagation followed by a writer fault and recovery.
It includes real 2/8 interval micro cases and Nt384 micro-space cases. It does
not rerun the historical full test collection or the small-grid HPC matrix.

Environment used for acceptance: Python 3.12.8, NumPy 2.2.5, SciPy 1.15.2,
Numba 0.61.2, llvmlite 0.44.0, psutil 7.0.0, threadpoolctl 3.6.0; generic CPU,
workqueue, BLAS1, complex128, fastmath=False. Scientific caches are separated
by checkout/role and remain external. Explicit `-O`/`PYTHONOPTIMIZE` refusals
are tested rather than relying on assertions for execution admission.

## Source and evidence boundaries

`CHECKPOINT_DEPENDENCIES.json` maps legacy relative source names/raw identities
to repository paths. Git tracked text additionally uses its real blob OID and
canonical-LF SHA256; current manifest receipts are generated from the committed
checkout. Raw historical receipts are retained in the protected archive, not
rewritten or published. All frozen G/E/F/D/R8 mathematics and the G3 lifecycle
callsite remain byte-identical; only location/identity bindings are adapted.

Support copies `g2_hpc_entry` are used only for `_quiet_runtime` and unchanged
normal scientific output persistence. Its historical `run()`/CLI and obsolete
task authority are not this checkpoint's entry point; `g2_hpc_common` is not
an implicit external dependency. New full-size execution requires its own
explicit confirmation plan, source commit, input and actual node gates.

The small-grid report, original full-precision timing CSVs and exact index are
small copies here. Full original comparison records/arrays stay in the frozen
R14 archive, whose size/hash is in `SMALLGRID_EVIDENCE_SUMMARY.json`.
Scientific exactness and a Git push are separate qualifications.
