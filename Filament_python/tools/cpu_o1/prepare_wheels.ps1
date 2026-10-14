$ErrorActionPreference='Stop'
$wheelRoot='D:\Filament_1\.local_envs\cpu_o1_artifacts\wheels'
New-Item -ItemType Directory -Path $wheelRoot -Force | Out-Null
$packages=[ordered]@{numpy='2.2.5';scipy='1.15.2';numba='0.61.2';llvmlite='0.44.0';pytest='8.3.5';pluggy='1.5.0';iniconfig='2.1.0';packaging='24.2';colorama='0.4.6';psutil='7.0.0';threadpoolctl='3.6.0'}
$records=@()
foreach($entry in $packages.GetEnumerator()){
 $metadata=Invoke-RestMethod -Uri "https://pypi.org/pypi/$($entry.Key)/$($entry.Value)/json" -TimeoutSec 45
 $matches=@($metadata.urls | Where-Object { $_.filename -match '(cp312-cp312-win_amd64|cp37-abi3-win_amd64|py3-none-any|py2.py3-none-any)\.whl$' })
 if($matches.Count -ne 1){throw "Ambiguous wheel $($entry.Key): $($matches.Count)"}
 $wheel=$matches[0]; $destination=Join-Path $wheelRoot $wheel.filename
 Invoke-WebRequest -Uri $wheel.url -OutFile $destination -TimeoutSec 90
 $actual=(Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
 if($actual -ne $wheel.digests.sha256){throw "Wheel digest mismatch $($entry.Key)"}
 $records += [pscustomobject]@{name=$entry.Key;version=$entry.Value;filename=$wheel.filename;url=$wheel.url;hash_scope='raw_bytes';sha256=$actual;bytes=(Get-Item -LiteralPath $destination).Length}
 Write-Output "VERIFIED $($entry.Key)==$($entry.Value)"
 $records | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath 'D:\Filament_1\reports\CPU_O1_LOCAL_20261014\DEPENDENCY_WHEELS.json' -Encoding utf8
}
$records | ForEach-Object { "$($_.name)==$($_.version) --hash=sha256:$($_.sha256)" } | Set-Content -LiteralPath 'D:\Filament_1\reports\CPU_O1_LOCAL_20261014\requirements-hashed.txt' -Encoding utf8
