param([ValidateSet('Windows','Linux')][string]$Platform,[string]$Revision='final')
$ErrorActionPreference='Stop'
$repo=(Get-Location).Path
$report=Join-Path $repo 'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028'
$env:S46A_REPORT_ROOT=$report
$env:CPU_S4_DATA_ROOT=Join-Path $report 'external_data/input_bundle'
$env:CPU_BOUNDARY_HISTORICAL_RECEIPT=Join-Path $report 'external_data/G2_BOUNDARY_RECEIPT.json'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:BLIS_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'
$env:NUMBA_NUM_THREADS='1'; $env:NUMBA_CPU_NAME='generic'; $env:NUMBA_THREADING_LAYER='workqueue'; $env:UPPE_USE_GPU='0'; $env:PYTHONUTF8='1'
$env:S43G2_SOURCE_ROOT=$repo; $env:S43F_SOURCE_ROOT=$repo; $env:S43C_SOURCE_ROOT=$repo
git rev-parse HEAD
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$actual='D:\Filament_1\.local_envs\cpu_o1_py312\python.exe'
$probe=$actual
if($Platform -eq 'Linux'){$probe=Join-Path $PSScriptRoot 'LinuxPython.ps1'}
$env:S46A_TEST_STAGE='backend_'+$Revision
& '.\Filament_python\tools\run_local_tests.ps1' -Mode backend -PythonExe $probe
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$env:S46A_TEST_STAGE='sanity_'+$Revision
& '.\Filament_python\tools\run_local_tests.ps1' -Mode sanity -PythonExe $probe
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
if($Platform -eq 'Windows'){
  & $actual -s -B -m compileall Filament_python/KHz_filament
  if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
  $probe=Join-Path $PSScriptRoot 'FocusedPython.ps1'
}
$env:S46A_TEST_STAGE='records_'+$Revision
& '.\Filament_python\tools\run_local_tests.ps1' -Mode targeted -PythonExe $probe
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$env:S46A_TEST_STAGE='micro_'+$Revision
& '.\Filament_python\tools\run_local_tests.ps1' -Mode targeted -PythonExe $probe
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
git rev-parse HEAD
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
