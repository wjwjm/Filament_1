[CmdletBinding()]
param(
 [string]$PythonExe='D:\Filament_1\.local_envs\cpu_o1_py312\python.exe',
 [Parameter(Mandatory=$true)][string]$EvidenceRoot
)
$ErrorActionPreference='Stop'
if(Test-Path -LiteralPath $EvidenceRoot){throw 'Use a new non-overwriting EvidenceRoot'}
New-Item -ItemType Directory -Path $EvidenceRoot | Out-Null
$envRoot=Split-Path -Parent $PythonExe
$previous=@{}
$settings=@{PATH="$envRoot;$envRoot\Library\bin;"+$env:PATH;OMP_NUM_THREADS='1';MKL_NUM_THREADS='1';OPENBLAS_NUM_THREADS='1';NUMBA_THREADING_LAYER='workqueue';NUMBA_CPU_NAME='generic';NUMBA_CACHE_DIR=(Join-Path (Split-Path -Parent $envRoot) 'cpu_o1_cache');CPU_O1_EVIDENCE_DIR=(Join-Path $EvidenceRoot 'comparisons')}
foreach($key in $settings.Keys){$previous[$key]=[Environment]::GetEnvironmentVariable($key,'Process');[Environment]::SetEnvironmentVariable($key,$settings[$key],'Process')}
$runner=Join-Path (Split-Path -Parent $PSScriptRoot) 'run_local_tests.ps1'
$filamentRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
try {
 foreach($mode in @('backend','sanity')){
  & $runner -Mode $mode -PythonExe $PythonExe *> (Join-Path $EvidenceRoot "$mode.log")
  if($LASTEXITCODE -ne 0){throw "$mode failed"}
 }
 & $PythonExe -s -B -m compileall -q (Join-Path $filamentRoot 'KHz_filament') $PSScriptRoot *> (Join-Path $EvidenceRoot 'compileall.log')
 if($LASTEXITCODE -ne 0){throw 'compileall failed'}
 & $runner -Mode targeted -TargetedFilter cpu_o1 -PythonExe $PythonExe *> (Join-Path $EvidenceRoot 'targeted.log')
 if($LASTEXITCODE -ne 0){throw 'targeted failed'}
} finally {
 foreach($key in $settings.Keys){[Environment]::SetEnvironmentVariable($key,$previous[$key],'Process')}
}
