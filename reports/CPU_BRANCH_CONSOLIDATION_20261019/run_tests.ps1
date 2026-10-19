param([Parameter(Mandatory=$true)][string]$RunTag,[Parameter(Mandatory=$true)][string]$Filter)
$ErrorActionPreference='Stop'
$reportRoot=$PSScriptRoot
$sourceRoot=Split-Path (Split-Path $reportRoot -Parent) -Parent
$py='D:/Filament_1/.local_envs/cpu_o1_py312/python.exe'
$env:PYTHONDONTWRITEBYTECODE='1';$env:NUMBA_CPU_NAME='generic';$env:NUMBA_THREADING_LAYER='workqueue'
$env:NUMBA_CACHE_DIR=(Join-Path $reportRoot 'jit_cache').Replace('\','/')
$env:PYTEST_ADDOPTS='--basetemp='+((Join-Path $reportRoot ('pytest_'+$RunTag)).Replace('\','/'))
$receipt=Join-Path $reportRoot ($RunTag+'_TEST_RECEIPT.json')
if(Test-Path -LiteralPath $receipt){throw 'Refuse overwrite test receipt'}
$checks=@();$started=[DateTime]::UtcNow.ToString('o');$failure=$null
Push-Location $sourceRoot
try{
 foreach($mode in @('backend','sanity','targeted')){
  $parameters=@{Mode=$mode;PythonExe=$py}
  if($mode -eq 'targeted'){$parameters.TargetedFilter=$Filter}
  & Filament_python/tools/run_local_tests.ps1 @parameters *> (Join-Path $reportRoot ($RunTag+'_'+$mode+'.log'))
  $code=$LASTEXITCODE;$checks+=@{mode=$mode;exit_code=$code}
  if($code -ne 0){throw "$mode failed"}
 }
 & $py -s -B -m compileall -q Filament_python/KHz_filament Filament_python/tools/cpu_o1/cpu_s4 *> (Join-Path $reportRoot ($RunTag+'_syntax.log'))
 $code=$LASTEXITCODE;$checks+=@{mode='syntax';exit_code=$code};if($code -ne 0){throw 'syntax failed'}
}catch{$failure=$_.ToString()}finally{Pop-Location}
@{started_utc=$started;ended_utc=[DateTime]::UtcNow.ToString('o');python=$py;filter=$Filter;checks=$checks;failure=$failure;status=$(if($failure){'FAIL'}else{'PASS'})}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $receipt -Encoding utf8
if($failure){throw $failure}
