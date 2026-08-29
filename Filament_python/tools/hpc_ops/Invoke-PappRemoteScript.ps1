[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('scvi806')]
    [string]$Account,

    [Parameter(Mandatory = $true)]
    [string]$RemoteRoot,

    [Parameter(Mandatory = $true)]
    [string]$LocalScript,

    [string[]]$ArgumentList = @(),

    [ValidateSet('ReadOnly', 'Write')]
    [string]$Mode = 'ReadOnly',

    [switch]$AllowRemoteWrite,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$global:LASTEXITCODE = 0

$AccountRoots = @{
    scvi806 = '/data/run01/scvi806'
}

function New-StatusJson {
    param(
        [bool]$Ok,
        [string]$State,
        [string]$Message = '',
        [hashtable]$Extra = @{}
    )
    $record = [ordered]@{
        schema = 'filament.hpc_ops.remote_exec.v1'
        ok = $Ok
        state = $State
    }
    if ($Message) { $record.message = $Message }
    foreach ($key in $Extra.Keys) { $record[$key] = $Extra[$key] }
    $record | ConvertTo-Json -Compress -Depth 8
}

function Assert-SafeRemoteRoot {
    param([string]$Value, [string]$ExpectedRoot)
    if ($Value.Contains('\')) {
        throw 'remote root must use POSIX separators'
    }
    $normalized = $Value
    if ($normalized -match '[\x00-\x1F\x7F]' -or $normalized.Contains('..')) {
        throw 'remote root contains an unsafe path component'
    }
    $shellMeta = @(';', '|', '&', '`', '$', '(', ')', '{', '}', '<', '>', '!', '"', "'", '\', '*', '?')
    foreach ($character in $shellMeta) {
        if ($normalized.Contains($character)) {
            throw 'remote root contains shell metacharacters'
        }
    }
    if ($normalized -ne $ExpectedRoot -and -not $normalized.StartsWith($ExpectedRoot + '/', [StringComparison]::Ordinal)) {
        throw 'remote root does not match account mapping'
    }
    return $normalized.TrimEnd('/')
}

function Assert-LocalScript {
    param([string]$Value)
    if ($Value -match '[\x00-\x1F\x7F]') { throw 'local script path contains a control character' }
    $item = Get-Item -LiteralPath $Value -ErrorAction Stop
    if (-not $item.PSIsContainer -and $item.Length -ge 0 -and
        -not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { return $item }
    throw 'local script must be a regular file'
}

function Convert-ToWslPath {
    param([string]$WindowsPath)
    $forwardSlash = $WindowsPath.Replace('\', '/')
    $converted = & wsl.exe -- wslpath -a -u -- $forwardSlash 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $converted) { throw 'Windows to WSL path conversion failed' }
    return ([string]$converted).Trim()
}

function Invoke-PappTransport {
    param(
        [ValidateSet('scp', 'ssh')]
        [string]$Operation,
        [string[]]$OperationArgumentList,
        [switch]$CaptureOutput
    )
    $pappBinary = $env:PAPP_CLOUD_BIN
    if (-not $pappBinary) { throw 'PAPP_CLOUD_BIN is not configured' }
    if ($pappBinary -match '[\x00-\x1F\x7F]' -or -not $pappBinary.StartsWith('/') -or $pappBinary.StartsWith('~')) {
        throw 'PAPP_CLOUD_BIN must be an absolute WSL POSIX path'
    }
    $transportArguments = @($pappBinary, $Operation) + @($OperationArgumentList)
    if ($CaptureOutput) {
        # ReadOnly preflight stdout is a schema-checked JSON contract.  Drop
        # transport stderr because SSH/client diagnostics can contain remote
        # endpoint details and must not be mixed into that contract.
        $captured = @(& wsl.exe -- @transportArguments 2>$null)
        $exitCode = $LASTEXITCODE
        return [pscustomobject]@{
            exit_code = $exitCode
            output = $captured
        }
    }
    & wsl.exe -- @transportArguments *> $null
    if ($LASTEXITCODE -ne 0) { throw "remote transport operation failed: $Operation" }
}

function Convert-RemotePreflightReport {
    param([object[]]$CapturedOutput)
    $lines = @($CapturedOutput | ForEach-Object { [string]$_ } | Where-Object { $_.Trim().Length -gt 0 })
    if ($lines.Count -ne 1) { throw 'remote preflight did not return one JSON object' }
    try {
        $report = $lines[0] | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw 'remote preflight returned invalid JSON'
    }
    if ($null -eq $report -or $report -is [System.Array] -or $report.schema -ne 'filament.hpc_preflight.v1') {
        throw 'remote preflight returned an unexpected report schema'
    }
    foreach ($property in @('ok', 'account', 'remote_root', 'source_class', 'checks', 'errors')) {
        if ($null -eq $report.PSObject.Properties[$property]) {
            throw 'remote preflight report is incomplete'
        }
    }
    return $report
}

function Convert-RemoteWriteReceipt {
    param([object[]]$CapturedOutput)
    $lines = @($CapturedOutput | ForEach-Object { [string]$_ } | Where-Object { $_.Trim().Length -gt 0 })
    if ($lines.Count -ne 1) { return $null }
    try {
        $report = $lines[0] | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        return $null
    }
    if ($null -eq $report -or $report -is [System.Array]) { return $null }
    foreach ($property in @('schema', 'ok', 'state')) {
        if ($null -eq $report.PSObject.Properties[$property]) { return $null }
    }
    return $report
}

function New-DispatcherScript {
    @'
#!/usr/bin/env bash
set -euo pipefail
manifest="$1"
script="$2"
support="$3"
expected_script_sha="$4"
expected_support_sha="$5"
expected_manifest_sha="$6"
run_dir="$(dirname -- "$manifest")"
args_file=""
cleanup() {
    status=$?
    trap - EXIT HUP INT TERM
    if [[ -n "$args_file" ]]; then rm -f -- "$args_file" || true; fi
    rm -f -- "$manifest" "$script" "$support" "$0" || true
    rmdir -- "$run_dir" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT HUP INT TERM
test -f "$manifest"
test -f "$script"
test -f "$support"
test "$(stat -c '%a' -- "$run_dir")" = 700
test "$(stat -c '%a' -- "$manifest")" = 600
test "$(stat -c '%a' -- "$script")" = 600
test "$(stat -c '%a' -- "$support")" = 600
test "$(stat -c '%a' -- "$0")" = 700
actual_manifest_sha="$(sha256sum -- "$manifest" | awk '{print $1}')"
actual_script_sha="$(sha256sum -- "$script" | awk '{print $1}')"
actual_support_sha="$(sha256sum -- "$support" | awk '{print $1}')"
test "$actual_manifest_sha" = "$expected_manifest_sha"
test "$actual_script_sha" = "$expected_script_sha"
test "$actual_support_sha" = "$expected_support_sha"
args_file="$(mktemp "$run_dir/args.XXXXXX")"
chmod 600 -- "$args_file"
if ! python3 - "$manifest" "$expected_script_sha" "$expected_support_sha" >"$args_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    document = json.load(handle)
if document.get("schema") != "filament.hpc_ops.args.v1":
    raise SystemExit("invalid argument manifest schema")
if document.get("script_sha256") != sys.argv[2]:
    raise SystemExit("script hash binding failed")
if document.get("proxy_env_sha256") != sys.argv[3]:
    raise SystemExit("proxy helper hash binding failed")
for value in document.get("arguments", []):
    if not isinstance(value, str):
        raise SystemExit("argument manifest contains a non-string")
    print(value)
PY
then
    exit 1
fi
mapfile -t args < "$args_file"
child_status=0
bash -- "$script" "${args[@]}" || child_status=$?
exit "$child_status"
'@
}

function Write-Utf8NoBomFile {
    param([string]$Path, [string]$Text)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Get-Sha256Text {
    param([string]$Text)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $bytes = $encoding.GetBytes($Text)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

$validatedRemoteRoot = $null
$scriptItem = $null
$remoteStagingCreated = $false
$target = $null
$remoteDir = $null
$failureStage = 'initialization'
try {
    $failureStage = 'validate-inputs'
    $Account = $Account.ToLowerInvariant()
    if ($Mode -eq 'Write' -and -not $AllowRemoteWrite) {
        throw 'Write mode requires -AllowRemoteWrite'
    }
    $expectedRoot = $AccountRoots[$Account]
    $validatedRemoteRoot = Assert-SafeRemoteRoot -Value $RemoteRoot -ExpectedRoot $expectedRoot
    $resolvedLocalScript = (Resolve-Path -LiteralPath $LocalScript -ErrorAction Stop).ProviderPath
    $scriptItem = Assert-LocalScript -Value $resolvedLocalScript
    $fixedReadOnlyScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'hpc_preflight.sh') -ErrorAction Stop).ProviderPath
    $fixedProxyScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'hpc_proxy_env.sh') -ErrorAction Stop).ProviderPath
    $proxyScriptItem = Assert-LocalScript -Value $fixedProxyScript
    if ($Mode -eq 'ReadOnly' -and -not $resolvedLocalScript.Equals($fixedReadOnlyScript, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'ReadOnly mode only permits the repository hpc_preflight.sh'
    }
    foreach ($argument in $ArgumentList) {
        if ($null -eq $argument -or $argument -match '[\x00-\x1F\x7F]') {
            throw 'argument list contains a null or control character'
        }
    }

    $scriptHash = (Get-FileHash -LiteralPath $scriptItem.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $proxyScriptHash = (Get-FileHash -LiteralPath $proxyScriptItem.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $runId = [Guid]::NewGuid().ToString('N')
    $remoteDir = "$validatedRemoteRoot/.codex_ops/$runId"
    $remoteScript = "$remoteDir/script.sh"
    $remoteProxyScript = "$remoteDir/hpc_proxy_env.sh"
    $remoteManifest = "$remoteDir/args.json"
    $remoteDispatcher = "$remoteDir/dispatch.sh"

    $manifest = [ordered]@{
        schema = 'filament.hpc_ops.args.v1'
        script_sha256 = $scriptHash
        proxy_env_sha256 = $proxyScriptHash
        arguments = @($ArgumentList)
    }
    $manifestText = (($manifest | ConvertTo-Json -Compress -Depth 8) + "`n")
    $manifestHash = Get-Sha256Text -Text $manifestText

    if ($DryRun) {
        $extra = @{
            account = $Account
            mode = $Mode
            dry_run = $true
            remote_root = $validatedRemoteRoot
            argument_count = @($ArgumentList).Count
            script_sha256 = $scriptHash
            proxy_env_sha256 = $proxyScriptHash
            argument_manifest_sha256 = $manifestHash
            would_upload = @('script', 'hpc_proxy_env.sh', 'args_manifest', 'dispatcher')
            would_execute = $true
        }
        New-StatusJson -Ok $true -State 'dry_run' -Extra $extra
        $global:LASTEXITCODE = 0
        return
    }

    $temporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) "filament_hpc_ops_$runId"
    New-Item -ItemType Directory -Path $temporaryDirectory -Force | Out-Null
    try {
        $manifestPath = Join-Path $temporaryDirectory 'args.json'
        $dispatcherPath = Join-Path $temporaryDirectory 'dispatch.sh'
        Write-Utf8NoBomFile -Path $manifestPath -Text $manifestText
        Write-Utf8NoBomFile -Path $dispatcherPath -Text (New-DispatcherScript)
        $manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

        $wslScript = Convert-ToWslPath -WindowsPath $scriptItem.FullName
        $wslProxyScript = Convert-ToWslPath -WindowsPath $proxyScriptItem.FullName
        $wslManifest = Convert-ToWslPath -WindowsPath $manifestPath
        $wslDispatcher = Convert-ToWslPath -WindowsPath $dispatcherPath
        $target = "$Account@nc-n50r5"

        # papp_cloud forwards this argument through a local WSL shell before
        # the remote shell sees it. Escape every dollar expansion once so
        # command substitutions and variable references are evaluated only on
        # the remote host, never against the local WSL filesystem or identity.
        $mkdirCommand = 'set -eu; resolved_root=\$(realpath -e -- ''{0}''); test "\$resolved_root" = ''{0}''; test -d "\$resolved_root"; test ! -L ''{0}''; umask 077; if test ! -e ''{0}/.codex_ops''; then mkdir -m 700 -- ''{0}/.codex_ops''; fi; test -d ''{0}/.codex_ops''; test ! -L ''{0}/.codex_ops''; test "\$(stat -c %u -- ''{0}/.codex_ops'')" = "\$(id -u)"; test "\$(stat -c %a -- ''{0}/.codex_ops'')" = 700; mkdir -m 700 -- ''{1}''' -f $validatedRemoteRoot, $remoteDir
        $failureStage = 'remote-mkdir'
        Invoke-PappTransport -Operation ssh -OperationArgumentList @($target, $mkdirCommand)
        $remoteStagingCreated = $true
        $failureStage = 'upload-script'
        Invoke-PappTransport -Operation scp -OperationArgumentList @($wslScript, "$target`:$remoteScript")
        $failureStage = 'upload-proxy-helper'
        Invoke-PappTransport -Operation scp -OperationArgumentList @($wslProxyScript, "$target`:$remoteProxyScript")
        $failureStage = 'upload-argument-manifest'
        Invoke-PappTransport -Operation scp -OperationArgumentList @($wslManifest, "$target`:$remoteManifest")
        $failureStage = 'upload-dispatcher'
        Invoke-PappTransport -Operation scp -OperationArgumentList @($wslDispatcher, "$target`:$remoteDispatcher")

        # The uploaded dispatcher performs all raw SHA checks before it runs
        # the user script. RemoteRoot and generated names contain no shell
        # metacharacters; user arguments remain in the JSON manifest.
        $remoteCommand = "umask 077; chmod 700 -- '$remoteDir'; chmod 600 -- '$remoteScript' '$remoteProxyScript' '$remoteManifest'; chmod 700 -- '$remoteDispatcher'; bash -- '$remoteDispatcher' '$remoteManifest' '$remoteScript' '$remoteProxyScript' '$scriptHash' '$proxyScriptHash' '$manifestHash'"
        $remoteReport = $null
        $remoteExitCode = 0
        if ($Mode -eq 'ReadOnly') {
            $failureStage = 'execute-readonly-dispatcher'
            $remoteResult = Invoke-PappTransport -Operation ssh -OperationArgumentList @($target, $remoteCommand) -CaptureOutput
            $remoteExitCode = $remoteResult.exit_code
            $remoteReport = Convert-RemotePreflightReport -CapturedOutput $remoteResult.output
            if ($remoteExitCode -ne 0 -and [bool]$remoteReport.ok) {
                throw 'remote preflight exit status disagreed with its report'
            }
            $statusOk = ($remoteExitCode -eq 0 -and [bool]$remoteReport.ok)
            $statusState = if ($statusOk) { 'completed' } else { 'remote_report' }
            New-StatusJson -Ok $statusOk -State $statusState -Extra @{
                account = $Account
                mode = $Mode
                dry_run = $false
                remote_root = $validatedRemoteRoot
                argument_count = @($ArgumentList).Count
                script_sha256 = $scriptHash
                proxy_env_sha256 = $proxyScriptHash
                argument_manifest_sha256 = $manifestHash
                remote_report = $remoteReport
                remote_exit_code = $remoteExitCode
            }
            if (-not $statusOk) {
                $global:LASTEXITCODE = 1
                exit 1
            }
        }
        else {
            $failureStage = 'execute-write-dispatcher'
            $remoteResult = Invoke-PappTransport -Operation ssh -OperationArgumentList @($target, $remoteCommand) -CaptureOutput
            $remoteExitCode = $remoteResult.exit_code
            $remoteReport = Convert-RemoteWriteReceipt -CapturedOutput $remoteResult.output
            if ($null -eq $remoteReport) {
                # A transport timeout or lost stdout receipt is not evidence of
                # remote failure. Preserve the distinction for state-file
                # inspection by the caller.
                New-StatusJson -Ok $false -State 'unknown_no_receipt' -Extra @{
                    account = $Account
                    mode = $Mode
                    dry_run = $false
                    remote_root = $validatedRemoteRoot
                    argument_count = @($ArgumentList).Count
                    script_sha256 = $scriptHash
                    proxy_env_sha256 = $proxyScriptHash
                    argument_manifest_sha256 = $manifestHash
                    remote_exit_code = $remoteExitCode
                    remote_receipt_lines = @($remoteResult.output).Count
                }
                $global:LASTEXITCODE = 2
                exit 2
            }
            $statusOk = ($remoteExitCode -eq 0 -and [bool]$remoteReport.ok -and [string]$remoteReport.state -eq 'completed')
            $statusState = if ($statusOk) { 'completed' } else { 'rejected_or_failed' }
            New-StatusJson -Ok $statusOk -State $statusState -Extra @{
                account = $Account
                mode = $Mode
                dry_run = $false
                remote_root = $validatedRemoteRoot
                argument_count = @($ArgumentList).Count
                script_sha256 = $scriptHash
                proxy_env_sha256 = $proxyScriptHash
                argument_manifest_sha256 = $manifestHash
                remote_report = $remoteReport
                remote_exit_code = $remoteExitCode
            }
            if (-not $statusOk) {
                $global:LASTEXITCODE = 1
                exit 1
            }
        }
        $global:LASTEXITCODE = 0
    }
    finally {
        if ($remoteStagingCreated -and $target -and $remoteDir) {
            # The dispatcher normally removes its own files.  This fixed,
            # best-effort cleanup also handles SCP or dispatcher-start failure
            # without ever deleting outside the generated mode-700 directory.
            $remoteCleanup = "rm -f -- '$remoteScript' '$remoteProxyScript' '$remoteManifest' '$remoteDispatcher'; rmdir -- '$remoteDir' 2>/dev/null || true"
            try {
                Invoke-PappTransport -Operation ssh -OperationArgumentList @($target, $remoteCleanup)
            }
            catch {
                # Preserve the primary operation result; a later inventory can
                # detect an unreachable private staging directory.
            }
        }
        if (Test-Path -LiteralPath $temporaryDirectory) {
            Remove-Item -LiteralPath $temporaryDirectory -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
catch {
    # Do not echo exception text: transport errors can contain remote URLs or
    # command fragments. The stable schema remains safe for automation.
    New-StatusJson -Ok $false -State 'rejected_or_failed' -Message 'remote operation was rejected or failed' -Extra @{
        failure_stage = $failureStage
    }
    $global:LASTEXITCODE = 1
    exit 1
}
