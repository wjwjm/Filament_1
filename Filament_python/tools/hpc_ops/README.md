# HPC execution guardrails

This directory contains low-level, non-physical execution helpers for the
Filament_1 HPC workflow. They do not submit jobs, create campaign locks, or
modify simulation results.

## Remote script wrapper

`Invoke-PappRemoteScript.ps1` accepts only an account enum, a mapped remote
root, a local script path, a string-array argument list, `ReadOnly`/`Write`
mode, `-AllowRemoteWrite`, and `-DryRun`. `ReadOnly` is deliberately limited
to the repository's exact `hpc_preflight.sh` path; it may create a private
mode-700 staging directory under `.codex_ops`, but cannot run an arbitrary
write script. Other scripts require `-Mode Write -AllowRemoteWrite`. The
wrapper rejects a symlinked remote root; an existing `.codex_ops` must already
be owned by the remote user and have mode 700, and its permissions are not
silently changed. The wrapper never accepts an arbitrary PowerShell or SSH command. Arguments are
placed in a JSON manifest and are not interpolated into the remote command
line. The PowerShell wrapper enforces mode and the ReadOnly allowlist. The
uploaded fixed dispatcher checks the script, fixed proxy helper, and argument
manifest SHA256 values before invoking the script, and removes only the files
and mode-700 run directory created for that invocation. A best-effort wrapper
cleanup covers transfer or dispatcher-start failures.

Use a dry run before any network operation:

```powershell
& .\Filament_python\tools\hpc_ops\Invoke-PappRemoteScript.ps1 `
  -Account scvi806 `
  -RemoteRoot /data/run01/scvi806/user_Wangjimin/example `
  -LocalScript .\Filament_python\tools\hpc_ops\hpc_preflight.sh `
  -ArgumentList @('--help') `
  -DryRun
```

An actual run requires `PAPP_CLOUD_BIN` to be an absolute WSL path to the
approved papp_cloud client. The wrapper uses argument arrays for `wsl.exe`,
SCP, and SSH. It emits only `filament.hpc_ops.remote_exec.v1` status JSON and
never prints proxy values, tokens, or authenticated URLs. In ReadOnly mode it
accepts exactly one schema-valid `filament.hpc_preflight.v1` object from remote
stdout and nests it as `remote_report`; transport stderr is discarded.

## Proxy loader

`hpc_proxy_env.sh` reads only `http_proxy`/`https_proxy` (including uppercase
and optional `export` syntax) from a user-owned mode-600 regular file. It does
not source or evaluate that file. Unknown keys, control characters, command
substitution, semicolons, and malformed HTTP(S) URLs are rejected. The
`hpc_proxy_git_ls_remote` requires an expected full SHA and exact ref, applies
a timeout that must be an integer from 1 through 300 seconds, and requires the
`timeout` utility. It accepts exactly one `<expected_head>\t<ref>` line from
`git ls-remote`; failures do not expose the captured error text. Proxy URLs
may contain authentication information when they are in the external mode-
600 secret file; credentials in a GitHub URL are forbidden.

The preferred secret-file location is a placeholder policy, not a committed
credential:

```text
/data/run01/scvi806/user_Wangjimin/.secrets/github_proxy.env
```

The file must contain only placeholder-shaped assignments such as
`http_proxy=https://proxy.example.invalid:8443`; never put a token in a Git
URL or in this repository.

## Read-only preflight

`hpc_preflight.sh` checks the account/root mapping, requires the repository to
resolve inside that non-symlinked remote root, verifies repository HEAD/branch
and clean state, and requires the exact configured Miniforge
`Filament_python` prefix and interpreter with NumPy/CuPy. It also checks
required Git/SHA256/Slurm tools and a proxy `git ls-remote`
probe. If the proxy path fails, an explicitly supplied Git bundle is accepted
only after its raw SHA256, `git bundle verify`, expected ref, and expected
HEAD pass. In that case the JSON reports
`verified_bundle_non_strict`. No run directory, lock, receipt, `sbatch`, or
production output is created.

The stdout contract is `filament.hpc_preflight.v1`; `--json` is a compatible
no-argument flag and does not write a second report or any arbitrary path.

## Protected Git source acquisition

`hpc_git_source.sh` exposes fixed `clone`/`fetch` arguments with explicit
account, remote root, and staging root. Targets are restricted to that staging
root. `--source-mode auto` preserves the proxy-first behavior; `proxy-only`
requires a successful exact proxy probe, while `bundle-only` requires an
explicit bundle and SHA256 and skips proxy access entirely. Bundle acquisition
still copies to a private mode-600 snapshot and checks its raw SHA/ref/HEAD.
Clone mode builds a clean expected-branch
checkout in a private temporary directory and atomically renames it to a
nonexistent target. Fetch mode requires an already clean target at the expected
branch/HEAD and only updates `FETCH_HEAD`; it never resets or merges. It cannot
push or submit a scheduler job and emits `filament.hpc_ops.git_source.v1` JSON.

## Provenance v2

`provenance_v2.py` provides `create` and `validate` subcommands. Tracked text
records the committed Git blob OID and a canonical-LF SHA256. External files
record exact raw-byte SHA256. Creation requires a clean repository, committed
tracked paths, and LF worktree bytes; validation uses canonical LF so an
otherwise identical CRLF checkout can be audited. Existing v1 execution
locks, receipts, and the `221822` result provenance are not rewritten.
