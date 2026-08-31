[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('import', 'backend', 'targeted', 'sanity')]
    [string]$Mode,

    [string]$PythonExe = 'C:\Users\wangj\.conda\envs\filament-local-test\python.exe'
)

$ErrorActionPreference = 'Stop'

$filamentPythonRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $filamentPythonRoot

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Explicit interpreter not found: $PythonExe"
}

$sanityTest = Join-Path $filamentPythonRoot 'tests\test_sanity.py'
$targetedTests = @(
    $sanityTest,
    (Join-Path $filamentPythonRoot 'tests\test_air_dispersion.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr3c_interpulse_diffusion.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr3c_pingpong_streaming.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr3c_state_machine.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr3c_closeout_runner.py'),
    (Join-Path $filamentPythonRoot 'tests\test_longitudinal_contract.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr2c_raman_deposition.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr2d_unified_deposition_ledger.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr3a_thermalization.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr3b_post_acoustic_slow_state.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr4a_contract_scaffolding.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr4b_single_screen.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr4c_state_lifecycle.py'),
    (Join-Path $filamentPythonRoot 'tests\test_runner_multipulse_orchestration.py'),
    (Join-Path $filamentPythonRoot 'tests\test_phase8b_raman_diagnostics.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hpc_execution_guardrails.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hpc_git_source_state.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr2e_schedule_convergence.py'),
    (Join-Path $filamentPythonRoot 'tests\test_hr2e_error_localization.py')
)

$previousPythonPath = $env:PYTHONPATH
$hadPythonPath = Test-Path Env:PYTHONPATH
$previousPythonNoUserSite = $env:PYTHONNOUSERSITE
$hadPythonNoUserSite = Test-Path Env:PYTHONNOUSERSITE

# This is a self-contained, local test environment: do not inherit arbitrary
# user-level module paths or user site-packages into its dependency resolution.
$env:PYTHONPATH = $filamentPythonRoot
$env:PYTHONNOUSERSITE = '1'

Push-Location -LiteralPath $repoRoot
try {
    switch ($Mode) {
        'import' {
            $importProbe = 'import KHz_filament; import numpy; import pytest; from KHz_filament.confio import load_all; assert load_all is not None; print(1)'
            & $PythonExe -s -B -c $importProbe
        }
        'backend' {
            $backendProbe = @'
import numpy as np

x = (
    np.arange(64, dtype=np.float64)
    + 1j * np.arange(64, dtype=np.float64)[::-1]
).astype(np.complex128, copy=False)
assert x.dtype == np.complex128

y = np.multiply(x, np.complex128(1.25 - 0.5j))
y = np.add(y, np.complex128(0.25 + 0.75j))
reshaped = y.reshape(8, 8)
assert reshaped.shape == (8, 8)

spectrum = np.fft.fft(y, n=64)
roundtrip = np.fft.ifft(spectrum, n=64)
assert roundtrip.dtype == np.complex128
assert np.isfinite(roundtrip).all()

error = float(np.max(np.abs(roundtrip - y)))
assert error <= 1e-12, error
print(f"backend_probe_passed dtype={y.dtype} shape={reshaped.shape} max_error={error:.6e}")
'@
            $started = [DateTimeOffset]::Now.ToString('o')
            Write-Output "[backend] started=$started executable=$PythonExe"
            & $PythonExe -s -B -c $backendProbe
            $backendExitCode = $LASTEXITCODE
            $finished = [DateTimeOffset]::Now.ToString('o')
            Write-Output "[backend] finished=$finished exit_code=$backendExitCode"
            if ($backendExitCode -ne 0) {
                throw "Backend NumPy probe failed or crashed with exit code $backendExitCode."
            }
        }
        'sanity' {
            & $PythonExe -s -B -m pytest -p no:cacheprovider -q $sanityTest
        }
        'targeted' {
            & $PythonExe -s -B -m pytest -p no:cacheprovider -q @targetedTests
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Mode $Mode failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    if ($hadPythonPath) {
        $env:PYTHONPATH = $previousPythonPath
    } else {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    if ($hadPythonNoUserSite) {
        $env:PYTHONNOUSERSITE = $previousPythonNoUserSite
    } else {
        Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
    }
}
