param()
$ErrorActionPreference='Stop'
# Invoked by the repository runner, which owns PYTHONPATH and user-site isolation.
# This fixed target adapter avoids collecting unrelated historical tests.
$actual='D:\Filament_1\.local_envs\cpu_o1_py312\python.exe'
$incoming=@($args | ForEach-Object { [string]$_ })
if ($incoming.Count -lt 4 -or $incoming[0] -ne '-s' -or $incoming[1] -ne '-B' -or $incoming[2] -ne '-m' -or $incoming[3] -ne 'pytest') {
    throw 'FOCUSED_ADAPTER_REQUIRES_REPOSITORY_TARGETED_ENTRY'
}
$root=(Get-Location).Path
$targets=@((Join-Path $root 'Filament_python/tools/cpu_s4_6a/test_records.py'),(Join-Path $root 'Filament_python/tools/cpu_s4_6a/test_gpu_manager.py'))
if ($env:S46A_TEST_STAGE.StartsWith('micro')) { $targets=@(Join-Path $root 'Filament_python/tools/cpu_s4_6a/test_micro.py') }
if ($env:S46A_TEST_STAGE.StartsWith('replay')) { $targets=@(Join-Path $root 'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028/replay_tests.py') }
$report=Join-Path $root 'reports/CPU_S4_6A_MATCHED_CPU_GPU_20261028'
New-Item -ItemType Directory -Path (Join-Path $report 'tmp') -Force | Out-Null
$junit=Join-Path $report ("WINDOWS_"+$env:S46A_TEST_STAGE.ToUpper()+'.xml')
Remove-Item Env:PYTEST_ADDOPTS -ErrorAction SilentlyContinue
& $actual @('-s','-B','-m','pytest','-p','no:cacheprovider','-q','-x','--junitxml',$junit,'--basetemp',(Join-Path $report ('tmp/windows_'+$env:S46A_TEST_STAGE))) @targets
$global:LASTEXITCODE=$LASTEXITCODE
if ($LASTEXITCODE -ne 0) { throw "FOCUSED_TEST_EXIT:$LASTEXITCODE" }
