param(
 [Parameter(Mandatory=$true)][string]$PythonExe,
 [Parameter(Mandatory=$true)][string]$DataRoot,
 [Parameter(Mandatory=$true)][string]$OutputRoot
)
$ErrorActionPreference='Stop'
$repoRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
if($env:SLURM_JOB_ID -or $env:PYTHONOPTIMIZE){throw 'LOCAL_UNOPTIMIZED_TEST_ENTRY_ONLY'}
if(-not [IO.Path]::IsPathFullyQualified($DataRoot) -or -not [IO.Path]::IsPathFullyQualified($OutputRoot)){throw 'EXPLICIT_ABSOLUTE_PATHS_REQUIRED'}
$out=[IO.Path]::GetFullPath($OutputRoot)
if($out.StartsWith($repoRoot+[IO.Path]::DirectorySeparatorChar)){throw 'TEST_OUTPUT_MUST_BE_OUTSIDE_CHECKOUT'}
if(Test-Path -LiteralPath $out){throw 'NEW_TEST_OUTPUT_REQUIRED'}
New-Item -ItemType Directory -Path $out,(Join-Path $out 'tmp')|Out-Null
foreach($name in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')){Set-Item -LiteralPath "Env:$name" -Value '1'}
$env:NUMBA_NUM_THREADS='1'
$env:NUMBA_CPU_NAME='generic'
$env:NUMBA_THREADING_LAYER='workqueue'
$env:UPPE_USE_GPU='0'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
$env:CPU_S4_DATA_ROOT=$DataRoot
$env:S43SG_REPORT_ROOT=$out
$env:S43SG_STAGE='checkpoint_restore'
$env:S43SG_LOCAL_ENGINEERING='1'
$env:NUMBA_CACHE_DIR=Join-Path $out 'cache/numba'
$env:PYTHONPYCACHEPREFIX=Join-Path $out 'cache/pycache'
foreach($name in @('S43G2_SOURCE_ROOT','S43F_SOURCE_ROOT','S43C_SOURCE_ROOT')){Set-Item -LiteralPath "Env:$name" -Value $repoRoot}
foreach($prefix in @('S43G2','S43G1','S43F','S43E')){Set-Item -LiteralPath "Env:${prefix}_REPORT_ROOT" -Value $out;Set-Item -LiteralPath "Env:${prefix}_STAGE" -Value 'checkpoint_restore'}
$runner=Join-Path $repoRoot 'Filament_python/tools/run_local_tests.ps1'
Push-Location -LiteralPath $repoRoot
try {
 foreach($mode in @('backend','import','sanity')){
  & $runner -Mode $mode -PythonExe $PythonExe
  if($LASTEXITCODE -ne 0){throw "CHECKPOINT_GATE_FAILED:$mode"}
 }
 $names=@('test_lifecycle.py','test_science.py','test_real_contracts.py','test_boundary.py','test_public_strict.py','test_entry.py','test_portable_binding.py','test_entry_cleanup.py')
 $tests=$names|ForEach-Object{"Filament_python/tools/cpu_s4_3g3_smallgrid/$_"}
 $nativeOut=$out.Replace('\','/')
 $env:PYTEST_ADDOPTS="--basetemp=$nativeOut/tmp/restore --junitxml=$nativeOut/RESTORE_TESTS.xml $($tests -join ' ') -x"
 & $runner -Mode targeted -PythonExe $PythonExe -TargetedFilter 'smallgrid or checkpoint'
 if($LASTEXITCODE -ne 0){throw 'CHECKPOINT_DIRECTED_RESTORE_FAILED'}
} finally {Pop-Location}
