param()
$ErrorActionPreference='Stop'
$incoming=@($args | ForEach-Object {[string]$_})
if ($incoming.Count -lt 4 -or $incoming[0] -ne '-s' -or $incoming[1] -ne '-B' -or $incoming[2] -ne '-m' -or $incoming[3] -ne 'pytest') {
    throw 'FINAL_REPLAY_ADAPTER_REQUIRES_REPOSITORY_TARGETED_ENTRY'
}
$actual='D:\Filament_1\.local_envs\cpu_o1_py312\python.exe'
$target=Join-Path (Get-Location).Path 'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028/CPU_S4_6A_REVIEW_20261029_002/replay_tests.py'
$receipt=Join-Path $PSScriptRoot 'WINDOWS_DELIVERY_TARGETED_A02.xml'
& $actual @('-s','-B','-m','pytest','-p','no:cacheprovider','-q','-x','--junitxml',$receipt,$target)
if ($LASTEXITCODE -ne 0) {throw "FINAL_REPLAY_TEST_EXIT:$LASTEXITCODE"}
