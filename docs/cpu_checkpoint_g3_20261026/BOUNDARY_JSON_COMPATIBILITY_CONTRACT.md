# R1 boundary JSON compatibility contract

The R1 repair is limited to the private boundary receipt load boundary. It does not modify `g2_boundary.py`, the frozen CSV mapping contract, `g1_exact`, `numeric_gate`, or any propagation and physical model code.

The writer historically serializes a receipt with `sort_keys=True`. The reader therefore accepts any JSON member order, including the historical sorted order, but reads the seven dynamic scalar values by their fixed names:

`z`, `interval_index`, `dz_try`, `U0_baseline`, `U_now`, `adaptive_rejection_count`, `safety_mode_trigger_count`.

The loader uses an object-pairs hook that rejects duplicate JSON keys before a normal dictionary can overwrite one. It then checks the exact boundary schema, 112-field count, six array records, 99 diagnostic records, fixed post-field names, scalar key set, and scalar types. For each referenced NPY member, the pre-array gate verifies a confined relative path and reads only the NPY header for shape and dtype. No array payload is loaded until all metadata checks pass.

The six array and 99 diagnostic records retain the source mapping order and the
writer's exact flat member names (`<field_name>.npy`); path traversal, renamed
members, or a field outside the frozen CSV mapping is rejected before the
header walk.

The receipt's mapping identity is checked against the existing raw-byte CSV
identity before the compatibility result is reported; the CSV file is neither
rewritten nor re-ordered.

The native-layout and alias sections are also structural gates: native layouts must
retain the complete source layout shape/dtype/stride/flag/pointer mapping, alias
rows must have the four expected name/boolean members, and the linear-result
mapping must have its fixed container/source/layout members. Malformed metadata
therefore fails with a named gate before any header or payload read, rather than
being deferred to a later dictionary comparison.

After validation, the loader constructs an in-memory scalar view in the fixed semantic order without rounding, stringifying, defaulting, aliasing, or changing dtype semantics. The source JSON order, receipt path, raw receipt identity when supplied, canonical order, scalar types, and “values unchanged” assertion are returned as `json_compatibility` metadata. The original receipt bytes are never rewritten.

Signed zero is checked through the unchanged scientific comparator. Quoted
non-finite markers remain the existing `_safe_to_value` conversion; the focused
test verifies their source/canonical IEEE metadata and records the frozen
comparator's existing `allow_nan=False` serialization rejection rather than
turning that behavior into a new exact rule.

`inspect_boundary_metadata(..., require_members=False)` is reserved for historical receipt-only review when archived NPY files are unavailable. It reports `scientific_comparison_status=NOT_RUN_METADATA_ONLY`; this status cannot satisfy a scientific boundary or full-size qualification gate. `load_boundary_capture()` always requires all members and is the only path used by `compare_current_boundary_pair()`.

The focused regression covers sorted and insertion/reversed JSON order, duplicate-key rejection, metadata failure before `numpy.load`, and metadata-only historical receipts. It uses tiny temporary arrays solely to exercise the disk round trip; it is not a propagation or full-size qualification test.

The formal parent runner should register the three new test files through
`PYTEST_ADDOPTS` after its backend/import/sanity gates. The repository wrapper
remains the only supported local entry point; the pre-existing targeted list
is intentionally unchanged by this bounded patch. `$TestPython` below is the
operator-provided frozen test interpreter variable, not a repository path:

```powershell
$TestPython = '<frozen-filament-test-python>'
$env:NUMBA_NUM_THREADS = '1'
$env:NUMBA_CPU_NAME = 'generic'
$env:NUMBA_THREADING_LAYER = 'workqueue'
foreach ($Name in 'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS') {
    Set-Item "Env:$Name" '1'
}
$env:S43SG_REPORT_ROOT = '<new-report-directory-outside-checkout>'
$env:NUMBA_CACHE_DIR = '<new-cache-directory-outside-checkout>'
$env:CPU_S4_DATA_ROOT = '<frozen-local-input-directory>'
$env:CPU_BOUNDARY_HISTORICAL_RECEIPT = '<original-readonly-boundary-receipt>'
$env:S43G2_SOURCE_ROOT = (Get-Location).Path
$env:S43F_SOURCE_ROOT = (Get-Location).Path
$env:PYTHONPYCACHEPREFIX = '<new-pycache-directory-outside-checkout>'
foreach ($Prefix in 'S43G2','S43G1','S43F','S43E') {
    Set-Item "Env:${Prefix}_REPORT_ROOT" $env:S43SG_REPORT_ROOT
    Set-Item "Env:${Prefix}_STAGE" 'R1_LOCAL'
}
$env:S43SG_STAGE = 'R1_LOCAL'
$TestPaths = @(
    'Filament_python/tools/cpu_s4_checkpoint_support/test_boundary_json_compat.py',
    'Filament_python/tools/cpu_s4_fullsize_confirmation/test_boundary_json_real_roundtrip.py',
    'Filament_python/tools/cpu_s4_fullsize_confirmation/test_r1_controller_gates.py'
)
$PreviousPytestAddopts = $env:PYTEST_ADDOPTS
try {
    & .\Filament_python\tools\run_local_tests.ps1 -Mode backend -PythonExe $TestPython
    & .\Filament_python\tools\run_local_tests.ps1 -Mode import -PythonExe $TestPython
    & .\Filament_python\tools\run_local_tests.ps1 -Mode sanity -PythonExe $TestPython
    $env:PYTEST_ADDOPTS = $TestPaths -join ' '
    & .\Filament_python\tools\run_local_tests.ps1 -Mode targeted `
        -TargetedFilter 'boundary_json or real_collector or r1_controller' -PythonExe $TestPython
} finally {
    $env:PYTEST_ADDOPTS = $PreviousPytestAddopts
}
```

The real micro case is a local-pool-1 engineering check at `Nt=384` and
transverse shape `[17,19]`; it is not a full-size or HPC qualification gate.

The R1 controller has a separate lifecycle schema and task identity. Its node gate performs the strict reference boundary metadata/header check after source, sealed-input, and runtime gates and before creating a run directory or launching propagation. The lifecycle default remains `I0`, and review remains `MANUAL_REVIEW_PENDING`.

The controller accepts only the R1 task/schema. Its budget fields are explicit:
`prior_allocated_core_hours=14.72`, `additional_allocated_core_hours`,
`inflight_reserved_core_hours`, `next_submission_reserved_core_hours`, and
`additional_allocated_core_hours_limit=129.28`; both the additional cap and the
overall 144-core-hour cap are checked without epsilon. The reference receipt path,
raw SHA256, and byte size must be present and agree with the matching
`reference_records` entry before the node metadata gate is entered.
