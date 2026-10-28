param()
$ErrorActionPreference='Stop'
$entry=Join-Path $PSScriptRoot 'linux_dispatch.py'
$nativeEntry=$entry.Replace('\','/').Replace('C:/','/mnt/c/').Replace('D:/','/mnt/d/')
$request=Join-Path $PSScriptRoot ('linux_request_'+[DateTime]::UtcNow.Ticks+'.json')
$names=@('S46A_REPORT_ROOT','S46A_TEST_STAGE','CPU_S4_DATA_ROOT','CPU_BOUNDARY_HISTORICAL_RECEIPT','NUMBA_NUM_THREADS','NUMBA_CPU_NAME','NUMBA_THREADING_LAYER','UPPE_USE_GPU','PYTHONPATH','PYTHONNOUSERSITE')
$settings=[ordered]@{}
foreach($name in $names) {$settings[$name]=[Environment]::GetEnvironmentVariable($name)}
[ordered]@{argv=@($args|ForEach-Object{[string]$_});environment=$settings;repo=(Get-Location).Path}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $request -Encoding utf8NoBOM
$nativeRequest=$request.Replace('\','/').Replace('C:/','/mnt/c/').Replace('D:/','/mnt/d/')
& wsl.exe @('--distribution','Ubuntu','--exec','/usr/bin/python3','-s','-B',$nativeEntry,$nativeRequest)
$global:LASTEXITCODE=$LASTEXITCODE
if ($LASTEXITCODE -ne 0) {throw "LINUX_TEST_EXIT:$LASTEXITCODE"}
