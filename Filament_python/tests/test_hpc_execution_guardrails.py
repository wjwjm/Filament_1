"""Low-cost tests for the Sol–Luna/HPC execution guardrails."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
HPC_OPS = ROOT / "Filament_python" / "tools" / "hpc_ops"
PROVENANCE_PATH = HPC_OPS / "provenance_v2.py"


def _load_provenance_module():
    spec = importlib.util.spec_from_file_location("provenance_v2_guardrail", PROVENANCE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True, capture_output=True)
    _git(repo, "config", "user.email", "guardrail@example.invalid")
    _git(repo, "config", "user.name", "Guardrail Test")
    (repo / "tracked.md").write_bytes(b"alpha\nbeta\n")
    _git(repo, "add", "--", "tracked.md")
    _git(repo, "commit", "-m", "guardrail fixture")
    return repo


def _wsl_executable() -> str | None:
    """Return a working WSL executable when native POSIX tools are absent."""

    if os.name != "nt":
        return None
    executable = shutil.which("wsl.exe")
    if not executable:
        return None
    probe = subprocess.run(
        [executable, "-e", "bash", "-c", "exit 0"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return executable if probe.returncode == 0 else None


def _wsl_path(path: Path) -> str:
    executable = shutil.which("wsl.exe")
    assert executable
    windows_path = str(path).replace("\\", "/")
    result = subprocess.run(
        [executable, "--", "wslpath", "-a", "-u", "--", windows_path],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _run_posix_stdin(script: str, *args: str | Path) -> subprocess.CompletedProcess[str]:
    """Run a fixture through native bash or WSL without inline shell quoting."""

    script = script.replace("\r\n", "\n")
    bash = shutil.which("bash") if os.name != "nt" else None
    if bash:
        command = [bash, "-s", "--"]
        command_args = [str(value) for value in args]
    else:
        wsl = _wsl_executable()
        if not wsl:
            pytest.skip("neither native bash nor a working WSL bash is available")
        command = [wsl, "-e", "bash", "-s", "--"]
        command_args = [_wsl_path(value) if isinstance(value, Path) else str(value) for value in args]
    result = subprocess.run(
        command + command_args,
        input=script.encode("utf-8"),
        capture_output=True,
    )
    return subprocess.CompletedProcess(
        result.args,
        result.returncode,
        result.stdout.decode("utf-8", errors="replace"),
        result.stderr.decode("utf-8", errors="replace"),
    )


def test_provenance_v2_canonical_and_raw_hashes(tmp_path: Path):
    module = _load_provenance_module()
    assert module.canonical_lf_sha256(b"a\r\nb\n") == module.canonical_lf_sha256(b"a\nb\n")
    assert module.raw_sha256(b"a\r\nb\n") != module.raw_sha256(b"a\nb\n")

    repo = _git_repo(tmp_path)
    external = tmp_path / "external.bin"
    external.write_bytes(b"binary\r\n")
    manifest_path = tmp_path / "manifest.json"

    manifest = module.create_manifest(repo, manifest_path, ["tracked.md"], [str(external)])
    assert manifest["schema"] == "filament.provenance.v2"
    assert manifest["tracked_text"][0]["path"] == "tracked.md"
    assert len(manifest["tracked_text"][0]["git_blob_oid"]) == 40
    module.validate_manifest(repo, manifest_path)

    # Validation is canonical-LF by design: a CRLF checkout retains the same
    # tracked blob and canonical digest only when non-strict mode is explicit.
    (repo / "tracked.md").write_bytes(b"alpha\r\nbeta\r\n")
    with pytest.raises(module.ProvenanceError):
        module.validate_manifest(repo, manifest_path)
    module.validate_manifest(repo, manifest_path, require_clean=False)
    with pytest.raises(module.ProvenanceError):
        module.create_manifest(repo, tmp_path / "reject-crlf.json", ["tracked.md"], [])

    (repo / "tracked.md").write_bytes(b"alpha\nbeta\n")
    (repo / "untracked.txt").write_text("untracked", encoding="utf-8")
    with pytest.raises(module.ProvenanceError):
        module.create_manifest(repo, tmp_path / "reject-dirty.json", ["tracked.md"], [])
    (repo / "untracked.txt").unlink()

    manifest_path_2 = tmp_path / "manifest-2.json"
    module.create_manifest(repo, manifest_path_2, ["tracked.md"], [str(external)])
    external.write_bytes(b"tampered\n")
    with pytest.raises(module.ProvenanceError):
        module.validate_manifest(repo, manifest_path_2)


def test_provenance_v2_rejects_external_symlink_on_validate(tmp_path: Path):
    module = _load_provenance_module()
    repo = _git_repo(tmp_path)
    external = tmp_path / "external.bin"
    external.write_bytes(b"locked\n")
    manifest_path = tmp_path / "manifest.json"
    module.create_manifest(repo, manifest_path, [], [str(external)])
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"locked\n")
    external.unlink()
    try:
        external.symlink_to(replacement)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(module.ProvenanceError):
        module.validate_manifest(repo, manifest_path, require_clean=False)


def test_provenance_v2_external_symlink_rejected_via_posix_fixture():
    fixture = r'''
set -euo pipefail
tool="$1"
root=$(mktemp -d)
trap 'rm -rf -- "$root"' EXIT
repo="$root/repo"
mkdir -- "$repo"
git -C "$repo" init -b main >/dev/null
git -C "$repo" config user.email guardrail@example.invalid
git -C "$repo" config user.name Guardrail
printf 'tracked\n' > "$repo/tracked.md"
git -C "$repo" add -- tracked.md
git -C "$repo" commit -m fixture >/dev/null
printf 'external\n' > "$root/external.bin"
python3 "$tool" create --repo "$repo" --output "$root/manifest.json" --external "$root/external.bin" >/dev/null
mv -- "$root/external.bin" "$root/replacement.bin"
ln -s -- "$root/replacement.bin" "$root/external.bin"
set +e
python3 "$tool" validate --repo "$repo" --manifest "$root/manifest.json" --non-strict >/dev/null 2>&1
rc=$?
set -e
test "$rc" -ne 0
printf 'external-symlink-rejected\n'
'''
    result = _run_posix_stdin(fixture, PROVENANCE_PATH)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "external-symlink-rejected"


def test_provenance_v2_rejects_untracked_and_uncommitted_paths(tmp_path: Path):
    module = _load_provenance_module()
    repo = _git_repo(tmp_path)
    missing = tmp_path / "missing.bin"
    with pytest.raises(module.ProvenanceError):
        module.create_manifest(repo, tmp_path / "missing.json", ["missing.md"], [])
    with pytest.raises(module.ProvenanceError):
        module.create_manifest(repo, tmp_path / "empty.json", [], [])
    with pytest.raises(module.ProvenanceError):
        module.create_manifest(repo, tmp_path / "missing-external.json", [], [str(missing)])


def test_toml_agent_contracts_preserve_runtime_boundaries():
    import tomllib

    expected = {
        "filament_mapper": ("gpt-5.6-luna", "max", "read-only"),
        "filament_worker": ("gpt-5.6-luna", "max", "workspace-write"),
        "filament_numerical_reviewer": ("gpt-5.6-luna", "max", "read-only"),
        "filament_tester": ("gpt-5.6-luna", "max", "workspace-write"),
    }
    fields = (
        "task_boundary",
        "evidence",
        "files_changed",
        "commands_and_exit_codes",
        "tests",
        "unverified",
        "parent_decisions",
    )
    for name, (model, effort, sandbox) in expected.items():
        path = ROOT / ".codex" / "agents" / f"{name}.toml"
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        assert (document["model"], document["model_reasoning_effort"], document["sandbox_mode"]) == (
            model,
            effort,
            sandbox,
        )
        for field in fields:
            assert field in document["developer_instructions"]


def test_shell_helpers_parse_without_execution():
    for path in (
        HPC_OPS / "hpc_proxy_env.sh",
        HPC_OPS / "hpc_preflight.sh",
        HPC_OPS / "hpc_git_source.sh",
    ):
        result = _run_posix_stdin('bash -n -- "$1"\n', path)
        assert result.returncode == 0, result.stderr


def test_proxy_loader_rejects_unknown_key_without_printing_values(tmp_path: Path):
    bash = shutil.which("bash")
    if not bash or os.name == "nt":
        pytest.skip("secure POSIX mode-600 proxy fixture requires POSIX bash")
    helper = HPC_OPS / "hpc_proxy_env.sh"
    env_file = tmp_path / "proxy.env"
    env_file.write_text(
        "export http_proxy=http://proxy.example.invalid:8080\n"
        "https_proxy=https://proxy.example.invalid:8443\n",
        encoding="utf-8",
    )
    env_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    good = subprocess.run([bash, str(helper), str(env_file)], capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["loaded"] is True
    assert "proxy.example.invalid" not in good.stdout

    bad = tmp_path / "bad.env"
    bad.write_text(
        "http_proxy=http://proxy.example.invalid:8080\n"
        "https_proxy=https://proxy.example.invalid:8443\n"
        "UNEXPECTED=value\n",
        encoding="utf-8",
    )
    bad.chmod(stat.S_IRUSR | stat.S_IWUSR)
    result = subprocess.run([bash, str(helper), str(bad)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "proxy.example.invalid" not in result.stdout + result.stderr

def test_proxy_probe_requires_exact_head_ref_and_timeout():
    if os.name == "nt" and not _wsl_executable():
        pytest.skip("proxy probe fixture requires native POSIX bash or a working WSL bash")
    fixture = r'''
set -euo pipefail
helper="$1"
root=$(mktemp -d)
trap 'rm -rf -- "$root"' EXIT
env_file="$root/proxy.env"
printf '%s\n' 'http_proxy=http://proxy.example.invalid:8080' 'https_proxy=https://proxy.example.invalid:8443' > "$env_file"
chmod 600 -- "$env_file"
. "$helper"
fakebin="$root/bin"
mkdir -- "$fakebin"
real_git=$(command -v git)
cat > "$fakebin/git" <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
    if [[ "$arg" == ls-remote ]]; then
        printf '%s\t%s\n' "$PROBE_HEAD" "$PROBE_REF"
        exit 0
    fi
done
exec "$REAL_GIT" "$@"
EOF
chmod 700 -- "$fakebin/git"
export PROBE_HEAD=0123456789abcdef0123456789abcdef01234567
export PROBE_REF=refs/heads/main
export REAL_GIT="$real_git"
PATH="$fakebin:$PATH"
hpc_proxy_git_ls_remote https://github.com/example/repo.git "$PROBE_REF" "$PROBE_HEAD" 1
if hpc_proxy_git_ls_remote https://github.com/example/repo.git "$PROBE_REF" 0000000000000000000000000000000000000000 1; then exit 20; fi
if hpc_proxy_git_ls_remote https://github.com/example/repo.git "$PROBE_REF" "$PROBE_HEAD" 0; then exit 21; fi
if hpc_proxy_git_ls_remote https://github.com/example/repo.git "$PROBE_REF" "$PROBE_HEAD" 301; then exit 22; fi
if hpc_proxy_git_ls_remote https://github.com/example/repo.git "$PROBE_REF" "$PROBE_HEAD"; then exit 23; fi
printf 'probe-fixture-ok\n'
'''
    result = _run_posix_stdin(fixture, HPC_OPS / "hpc_proxy_env.sh")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "probe-fixture-ok"


def test_preflight_uses_verified_bundle_after_proxy_failure(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("preflight integration fixture requires a POSIX shell")
    bash = shutil.which("bash")
    assert bash
    account_root = tmp_path / "account"
    remote_root = account_root / "guardrail"
    remote_root.mkdir(parents=True)
    repo = _git_repo(remote_root)
    head = _git(repo, "rev-parse", "HEAD")
    bundle = tmp_path / "repo.bundle"
    subprocess.run(["git", "-C", str(repo), "bundle", "create", str(bundle), "main"], check=True)
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("sbatch", "sacct", "scontrol"):
        executable = fake_bin / command
        executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    real_git = shutil.which("git")
    assert real_git
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do [[ \"$arg\" == ls-remote ]] && exit 1; done\n"
        "exec \"$REAL_GIT\" \"$@\"\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    miniforge = tmp_path / "miniforge"
    conda_hook = miniforge / "etc" / "profile.d"
    conda_hook.mkdir(parents=True)
    environment_bin = miniforge / "envs" / "Filament_python" / "bin"
    environment_bin.mkdir(parents=True)
    environment_python = environment_bin / "python"
    environment_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    environment_python.chmod(0o700)
    (conda_hook / "conda.sh").write_text(
        "conda() {\n"
        "  [[ \"$1\" == activate && \"$2\" == Filament_python ]] || return 1\n"
        f"  export CONDA_PREFIX='{miniforge}/envs/Filament_python'\n"
        "  export PATH=\"$CONDA_PREFIX/bin:$PATH\"\n"
        "}\n",
        encoding="utf-8",
    )
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    preflight = tool_dir / "hpc_preflight.sh"
    preflight.write_text(
        (HPC_OPS / "hpc_preflight.sh")
        .read_text(encoding="utf-8")
        .replace("/data/run01/scvi806", str(account_root))
        .replace("/data/apps/miniforge/25.3.0-3", str(miniforge)),
        encoding="utf-8",
    )
    shutil.copy2(HPC_OPS / "hpc_proxy_env.sh", tool_dir / "hpc_proxy_env.sh")
    proxy_env = tmp_path / "proxy.env"
    proxy_env.write_text(
        "http_proxy=http://proxy.example.invalid:8080\n"
        "https_proxy=https://proxy.example.invalid:8443\n",
        encoding="utf-8",
    )
    proxy_env.chmod(0o600)
    environment = os.environ.copy()
    environment["PATH"] = str(fake_bin) + os.pathsep + environment.get("PATH", "")
    environment["REAL_GIT"] = real_git
    environment["HPC_PROXY_GIT_TIMEOUT_SECONDS"] = "1"
    command = [
        bash,
        str(preflight),
        "--account",
        "scvi806",
        "--remote-root",
        str(remote_root),
        "--repo",
        str(repo),
        "--expected-head",
        head,
        "--expected-branch",
        "main",
        "--proxy-env",
        str(proxy_env),
        "--github-url",
        "https://github.com/example/repo.git",
        "--github-ref",
        "refs/heads/main",
        "--bundle",
        str(bundle),
        "--bundle-sha",
        bundle_sha,
        "--json",
    ]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "filament.hpc_preflight.v1"
    assert report["ok"] is True
    assert report["source_class"] == "verified_bundle_non_strict"
    assert "proxy.example.invalid" not in result.stdout + result.stderr

def test_preflight_bundle_fallback_via_wsl_fixture():
    if os.name != "nt":
        pytest.skip("WSL-specific fixture is only needed on Windows")
    if not _wsl_executable():
        pytest.skip("WSL-specific fixture requires a working WSL bash")
    fixture = r'''
set -euo pipefail
preflight="$1"
root=$(mktemp -d)
trap 'rm -rf -- "$root"' EXIT
    account_root="$root/account"
    remote_root="$account_root/guardrail"
    mkdir -p -- "$remote_root"
    tool_dir="$root/tool"
    mkdir -- "$tool_dir"
    sed -e "s#/data/run01/scvi806#$account_root#g" -e "s#/data/apps/miniforge/25.3.0-3#$root/miniforge#g" -e "s#/data/home/scvi806/.conda/envs/Filament_python#$root/filament-env#g" "$preflight" > "$tool_dir/hpc_preflight.sh"
    cp -- "$2" "$tool_dir/hpc_proxy_env.sh"
    chmod 700 -- "$tool_dir/hpc_preflight.sh"
    preflight="$tool_dir/hpc_preflight.sh"
    repo="$remote_root/repo"
mkdir -- "$repo"
git -C "$repo" init -b main >/dev/null
git -C "$repo" config user.email guardrail@example.invalid
git -C "$repo" config user.name Guardrail
printf 'fixture\n' > "$repo/tracked.txt"
git -C "$repo" add -- tracked.txt
git -C "$repo" commit -m fixture >/dev/null
head=$(git -C "$repo" rev-parse HEAD)
bundle="$root/repo.bundle"
git -C "$repo" bundle create "$bundle" main >/dev/null
bundle_sha=$(sha256sum "$bundle" | awk '{print $1}')
fakebin="$root/bin"
mkdir -- "$fakebin"
real_git=$(command -v git)
cat > "$fakebin/git" <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
    if [[ "$arg" == ls-remote ]]; then exit 1; fi
done
exec "$REAL_GIT" "$@"
EOF
chmod 700 -- "$fakebin/git"
for command in sbatch sacct scontrol; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "$fakebin/$command"
    chmod 700 -- "$fakebin/$command"
done
miniforge="$root/miniforge"
mkdir -p "$miniforge/etc/profile.d"
mkdir -p "$root/filament-env/bin"
cat > "$miniforge/etc/profile.d/conda.sh" <<'EOF'
conda() {
    [[ "$1" == activate && "$2" == Filament_python ]] || return 1
    export CONDA_PREFIX="$FILAMENT_ENV_PREFIX_FIXED"
    export PATH="$CONDA_PREFIX/bin:$PATH"
}
EOF
cat > "$root/filament-env/bin/python" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod 700 -- "$root/filament-env/bin/python"
proxy="$root/proxy.env"
printf '%s\n' 'http_proxy=http://proxy.example.invalid:8080' 'https_proxy=https://proxy.example.invalid:8443' > "$proxy"
chmod 600 -- "$proxy"
export REAL_GIT="$real_git" FILAMENT_ENV_PREFIX_FIXED="$root/filament-env"
    PATH="$fakebin:$PATH"
    "$preflight" --account scvi806 --remote-root "$remote_root" --repo "$repo" --expected-head "$head" --expected-branch main --proxy-env "$proxy" --github-url https://github.com/example/repo.git --github-ref refs/heads/main --bundle "$bundle" --bundle-sha "$bundle_sha" --json
'''
    result = _run_posix_stdin(
        fixture,
        HPC_OPS / "hpc_preflight.sh",
        HPC_OPS / "hpc_proxy_env.sh",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "filament.hpc_preflight.v1"
    assert report["ok"] is True
    assert report["source_class"] == "verified_bundle_non_strict"
    assert report["checks"]["proxy_or_bundle"] is True
    assert "proxy.example.invalid" not in result.stdout + result.stderr


def test_git_source_bundle_clone_fetch_and_wrong_head_cleanup():
    fixture = r'''
set -euo pipefail
source_helper="$1"
proxy_helper="$2"
root=$(mktemp -d)
trap 'rm -rf -- "$root"' EXIT
account_root="$root/account"
remote_root="$account_root/user_Wangjimin"
staging="$remote_root/staging"
mkdir -p -- "$staging"

# Keep the production account mapping intact in the repository file while
# making this isolated fixture runnable without privileged /data access.
mkdir -- "$root/tool"
sed "s#/data/run01/scvi806#$account_root#g" "$source_helper" > "$root/tool/hpc_git_source.sh"
cp -- "$proxy_helper" "$root/tool/hpc_proxy_env.sh"
chmod 700 -- "$root/tool/hpc_git_source.sh"

repo="$remote_root/source"
mkdir -- "$repo"
git -C "$repo" init -b main >/dev/null
git -C "$repo" config user.email guardrail@example.invalid
git -C "$repo" config user.name Guardrail
printf 'fixture\n' > "$repo/tracked.txt"
git -C "$repo" add -- tracked.txt
git -C "$repo" commit -m fixture >/dev/null
head=$(git -C "$repo" rev-parse HEAD)
bundle="$staging/source.bundle"
git -C "$repo" bundle create "$bundle" main >/dev/null
bundle_sha=$(sha256sum -- "$bundle" | awk '{print $1}')
proxy="$remote_root/proxy.env"
printf '%s\n' 'http_proxy=http://proxy.example.invalid:8080' 'https_proxy=https://proxy.example.invalid:8443' > "$proxy"
chmod 600 -- "$proxy"

fakebin="$root/bin"
mkdir -- "$fakebin"
real_git=$(command -v git)
cat > "$fakebin/git" <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
    if [[ "$arg" == ls-remote ]]; then : > "$PROBE_MARKER"; exit 1; fi
    if [[ "${FAIL_GIT_STATUS:-0}" == 1 && "$arg" == status ]]; then exit 97; fi
done
exec "$REAL_GIT" "$@"
EOF
chmod 700 -- "$fakebin/git"
export REAL_GIT="$real_git"
export PROBE_MARKER="$root/probe-called"
PATH="$fakebin:$PATH"

target="$staging/checkout"
clone_json=$("$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode clone --url https://github.com/example/repo.git --ref refs/heads/main --expected-head "$head" --expected-branch main --proxy-env "$proxy" --target "$target" --bundle "$bundle" --bundle-sha "$bundle_sha" --timeout-seconds 1)
python3 - "$clone_json" "$head" <<'PY'
import json, sys
report = json.loads(sys.argv[1])
assert report["ok"] is True
assert report["source_class"] == "verified_bundle_non_strict"
assert report["target_head"] == sys.argv[2]
assert report["target_branch"] == "main"
assert report["fetch_head"] == sys.argv[2]
PY
test -z "$(git -C "$target" status --porcelain=v1 --untracked-files=all)"

bundle_only_target="$staging/checkout_bundle_only"
rm -f -- "$PROBE_MARKER"
bundle_only_json=$("$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode clone --source-mode bundle-only --url https://github.com/example/repo.git --ref refs/heads/main --expected-head "$head" --expected-branch main --target "$bundle_only_target" --bundle "$bundle" --bundle-sha "$bundle_sha" --timeout-seconds 1)
python3 - "$bundle_only_json" "$head" <<'PY'
import json, sys
report = json.loads(sys.argv[1])
assert report["ok"] is True
assert report["source_class"] == "verified_bundle_non_strict"
assert report["source_mode"] == "bundle-only"
assert report["target_head"] == sys.argv[2]
assert report["target_branch"] == "main"
assert report["fetch_head"] == sys.argv[2]
PY
test ! -e "$PROBE_MARKER"
test -z "$(git -C "$bundle_only_target" status --porcelain=v1 --untracked-files=all)"

fetch_json=$("$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode fetch --url https://github.com/example/repo.git --ref refs/heads/main --expected-head "$head" --expected-branch main --proxy-env "$proxy" --target "$target" --bundle "$bundle" --bundle-sha "$bundle_sha" --timeout-seconds 1)
python3 - "$fetch_json" "$head" <<'PY'
import json, sys
report = json.loads(sys.argv[1])
assert report["ok"] is True
assert report["fetch_head"] == sys.argv[2]
PY

export FAIL_GIT_STATUS=1
set +e
"$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode fetch --url https://github.com/example/repo.git --ref refs/heads/main --expected-head "$head" --expected-branch main --proxy-env "$proxy" --target "$target" --bundle "$bundle" --bundle-sha "$bundle_sha" --timeout-seconds 1 >/dev/null
status_failure_rc=$?
set -e
unset FAIL_GIT_STATUS
test "$status_failure_rc" -ne 0

wrong_target="$staging/wrong"
set +e
"$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode clone --url https://github.com/example/repo.git --ref refs/heads/main --expected-head 0000000000000000000000000000000000000000 --expected-branch main --proxy-env "$proxy" --target "$wrong_target" --bundle "$bundle" --bundle-sha "$bundle_sha" --timeout-seconds 1 >/dev/null
wrong_rc=$?
set -e
test "$wrong_rc" -ne 0
test ! -e "$wrong_target"
if compgen -G "$staging/.git-source.*" >/dev/null; then exit 31; fi
if compgen -G "$staging/.verified-bundle.*" >/dev/null; then exit 32; fi
if compgen -G "$staging/.bundle-verify.*" >/dev/null; then exit 33; fi
printf 'git-source-fixture-ok\n'
'''
    result = _run_posix_stdin(
        fixture,
        HPC_OPS / "hpc_git_source.sh",
        HPC_OPS / "hpc_proxy_env.sh",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "git-source-fixture-ok"


def test_powershell_wrapper_dry_run_is_argument_array_only(tmp_path: Path):
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("PowerShell is not available on this host")
    wrapper = HPC_OPS / "Invoke-PappRemoteScript.ps1"
    wrapper_ps = str(wrapper).replace("'", "''")
    script_ps = str(HPC_OPS / "hpc_preflight.sh").replace("'", "''")
    harness = tmp_path / "invoke-wrapper.ps1"
    harness.write_text(
        f"& '{wrapper_ps}' -Account scvi806 "
        f"-RemoteRoot '/data/run01/scvi806/user_Wangjimin/guardrail' "
        f"-LocalScript '{script_ps}' "
        "-ArgumentList @('space arg','中文','$HOME','double\"quote',\"single'quote\") -DryRun\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(harness)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["schema"] == "filament.hpc_ops.remote_exec.v1"
    assert report["dry_run"] is True
    assert report["argument_count"] == 5
    assert len(report["argument_manifest_sha256"]) == 64
    assert len(report["proxy_env_sha256"]) == 64
    assert "hpc_proxy_env.sh" in report["would_upload"]


def test_powershell_wrapper_rejects_mismatched_account_root(tmp_path: Path):
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("PowerShell is not available on this host")
    wrapper = HPC_OPS / "Invoke-PappRemoteScript.ps1"
    script = tmp_path / "script.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(wrapper),
            "-Account",
            "scvi806",
            "-RemoteRoot",
            "/publicfs01/fs1-t/home/t0s000727/not-allowed",
            "-LocalScript",
            str(script),
            "-DryRun",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.returncode != 0
    assert report["schema"] == "filament.hpc_ops.remote_exec.v1"
    assert report["ok"] is False


def test_powershell_readonly_rejects_arbitrary_script(tmp_path: Path):
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("PowerShell is not available on this host")
    wrapper = HPC_OPS / "Invoke-PappRemoteScript.ps1"
    script = tmp_path / "arbitrary.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(wrapper),
            "-Account",
            "scvi806",
            "-RemoteRoot",
            "/data/run01/scvi806/user_Wangjimin/guardrail",
            "-LocalScript",
            str(script),
            "-Mode",
            "ReadOnly",
            "-DryRun",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.returncode != 0
    assert report["ok"] is False


def test_job_221822_v1_provenance_remains_tracked_and_unchanged():
    frozen = ROOT / "Filament_python" / "results" / "isaacs_complete_eq27" / "provenance_221822"
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(frozen.relative_to(ROOT))],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    assert len(tracked) == 7
    assert all((ROOT / path).is_file() for path in tracked)
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(frozen.relative_to(ROOT))],
        check=False,
    )
    assert unchanged.returncode == 0


def test_secret_scan_of_new_guardrails():
    paths = [
        HPC_OPS / "Invoke-PappRemoteScript.ps1",
        HPC_OPS / "hpc_proxy_env.sh",
        HPC_OPS / "hpc_preflight.sh",
        HPC_OPS / "hpc_git_source.sh",
        HPC_OPS / "provenance_v2.py",
        HPC_OPS / "README.md",
        ROOT / "docs" / "experience" / "sol_luna_hpc_execution_playbook.md",
        ROOT / "docs" / "experience" / "2026-08-22_isaacs_eq27_c2_postmortem.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "gh" + "p_" not in text
    assert "wjm" + "@2023" not in text
    assert "https://user:password" + "@" not in text
    assert "Invoke-Expression" not in (HPC_OPS / "Invoke-PappRemoteScript.ps1").read_text(encoding="utf-8")
    assert "cmd /c" not in (HPC_OPS / "Invoke-PappRemoteScript.ps1").read_text(encoding="utf-8")


def test_powershell_wrapper_defers_bootstrap_substitutions_to_remote_shell():
    wrapper = (HPC_OPS / "Invoke-PappRemoteScript.ps1").read_text(encoding="utf-8")
    mkdir_line = next(line for line in wrapper.splitlines() if "$mkdirCommand =" in line)
    assert "resolved_root=\\$(realpath" in mkdir_line
    assert '"\\$resolved_root"' in mkdir_line
    assert '"\\$(stat -c %u' in mkdir_line
    assert '"\\$(id -u)' in mkdir_line
    assert '"\\$(stat -c %a' in mkdir_line
    assert 'test -d --' not in mkdir_line
    assert 'test ! -e --' not in mkdir_line
