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
BATCH_ENTRY_AUDIT = HPC_OPS / "audit_batch_entry.py"


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


def _run_batch_entry_audit(batch: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BATCH_ENTRY_AUDIT),
            "--batch",
            str(batch),
            "--fixed-python",
            "/opt/filament/bin/python",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _require_batch_audit_shell() -> None:
    if os.name == "nt":
        native_bash = shutil.which("bash")
        system_bash = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "bash.exe"
        has_native = bool(native_bash and Path(native_bash).resolve() != system_bash.resolve())
        if not has_native and not _wsl_executable():
            pytest.skip("batch-entry audit requires native POSIX bash or WSL")
    elif not shutil.which("bash"):
        pytest.skip("batch-entry audit requires bash")


def test_batch_entry_audit_rejects_bare_python_before_conda(tmp_path: Path):
    _require_batch_audit_shell()
    batch = tmp_path / "bare_python.sbatch"
    batch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "python -c 'print(1)'\n"
        "source /opt/conda.sh\n"
        "conda activate Filament_python\n",
        encoding="utf-8",
    )
    result = _run_batch_entry_audit(batch)
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["schema"] == "filament.hpc_batch_entry_audit.v1"
    assert report["status"] == "bare_python_before_activation"
    assert report["bare_python_commands"] == [{"command": "python", "line": 3}]


def test_batch_entry_audit_accepts_fixed_python_and_post_activation_python(tmp_path: Path):
    _require_batch_audit_shell()
    batch = tmp_path / "fixed_python.sbatch"
    batch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "readonly FIXED_PYTHON=/opt/filament/bin/python\n"
        '"${FIXED_PYTHON}" -c \'print(1)\'\n'
        "source /opt/conda.sh\n"
        "conda activate Filament_python\n"
        "python -c 'print(2)'\n",
        encoding="utf-8",
    )
    result = _run_batch_entry_audit(batch)
    report = json.loads(result.stdout)
    assert result.returncode == 0
    assert report["status"] == "passed"
    assert report["activation_line"] == 6
    assert not report["bare_python_commands"]
    assert report["qualified_python_commands"] == [
        {"command": "${FIXED_PYTHON}", "line": 4}
    ]


def test_batch_entry_audit_rejects_bash_syntax_error(tmp_path: Path):
    _require_batch_audit_shell()
    batch = tmp_path / "syntax_error.sbatch"
    batch.write_text("#!/usr/bin/env bash\nif true; then\n", encoding="utf-8")
    result = _run_batch_entry_audit(batch)
    report = json.loads(result.stdout)
    assert result.returncode == 2
    assert report["status"] == "bash_syntax_failed"


@pytest.mark.parametrize(
    "command",
    [
        "if python -c 'print(1)'; then true; fi",
        "env python -c 'print(1)'",
        "command python -c 'print(1)'",
        "sudo python -c 'print(1)'",
    ],
)
def test_batch_entry_audit_rejects_wrapped_or_conditional_python(tmp_path: Path, command: str):
    _require_batch_audit_shell()
    batch = tmp_path / "wrapped_python.sbatch"
    batch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{command}\n"
        "conda activate Filament_python\n",
        encoding="utf-8",
    )
    result = _run_batch_entry_audit(batch)
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["status"] == "bare_python_before_activation"
    assert report["bare_python_commands"]


def test_batch_entry_audit_does_not_treat_function_definition_as_activation(tmp_path: Path):
    _require_batch_audit_shell()
    batch = tmp_path / "function_activation.sbatch"
    batch.write_text(
        "#!/usr/bin/env bash\n"
        "activate_env() {\n"
        "  conda activate Filament_python\n"
        "}\n"
        "python -c 'print(1)'\n",
        encoding="utf-8",
    )
    result = _run_batch_entry_audit(batch)
    report = json.loads(result.stdout)
    assert result.returncode == 1
    assert report["activation_line"] is None
    assert report["bare_python_commands"] == [{"command": "python", "line": 5}]


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

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True, capture_output=True)
    _git(repo, "config", "user.email", "guardrail@example.invalid")
    _git(repo, "config", "user.name", "Guardrail Test")
    (repo / ".gitattributes").write_text("tracked.md text eol=crlf\n", encoding="utf-8")
    (repo / "tracked.md").write_bytes(b"alpha\r\nbeta\r\n")
    _git(repo, "add", "--", ".gitattributes", "tracked.md")
    _git(repo, "commit", "-m", "CRLF fixture")
    assert _git(repo, "status", "--porcelain=v1") == ""
    external = tmp_path / "external.bin"
    external.write_bytes(b"binary\r\n")
    manifest_path = tmp_path / "manifest.json"

    manifest = module.create_manifest(repo, manifest_path, ["tracked.md"], [str(external)])
    assert manifest["schema"] == "filament.provenance.v2"
    assert manifest["hash_scope"] == "classified_by_record"
    tracked = module.lookup_record(
        manifest, "tracked.md", classification="tracked_text", require_hash_scope=True
    )
    assert tracked["hash_scope"] == "git_blob_oid+canonical_lf_sha256"
    assert len(tracked["git_blob_oid"]) == 40
    assert tracked["canonical_lf_sha256"] == module.canonical_lf_sha256(b"alpha\nbeta\n")
    external_record = module.lookup_record(
        manifest, str(external.resolve()), classification="external", require_hash_scope=True
    )
    assert external_record["hash_scope"] == "raw_bytes"
    module.validate_manifest(repo, manifest_path, require_hash_scope=True)

    # LF and CRLF bytes are one tracked-text identity. This manual LF rewrite is
    # dirty under eol=crlf on Windows, so the clean-worktree gate still rejects
    # it while explicit non-strict validation confirms the canonical binding.
    (repo / "tracked.md").write_bytes(b"alpha\nbeta\n")
    assert _git(repo, "status", "--porcelain=v1") == "M tracked.md"
    with pytest.raises(module.ProvenanceError, match="worktree must be clean"):
        module.validate_manifest(repo, manifest_path, require_hash_scope=True)
    module.validate_manifest(
        repo, manifest_path, require_clean=False, require_hash_scope=True
    )

    (repo / "tracked.md").write_bytes(b"alpha\r\nbeta\r\n")
    assert _git(repo, "status", "--porcelain=v1") == ""
    (repo / "untracked.txt").write_text("untracked", encoding="utf-8")
    with pytest.raises(module.ProvenanceError):
        module.create_manifest(repo, tmp_path / "reject-dirty.json", ["tracked.md"], [])
    (repo / "untracked.txt").unlink()

    manifest_path_2 = tmp_path / "manifest-2.json"
    module.create_manifest(repo, manifest_path_2, ["tracked.md"], [str(external)])
    external.write_bytes(b"tampered\n")
    with pytest.raises(module.ProvenanceError):
        module.validate_manifest(repo, manifest_path_2)


def test_provenance_v2_strict_schema_rejects_wrong_scope_blob_and_hash(tmp_path: Path):
    module = _load_provenance_module()
    repo = _git_repo(tmp_path)
    source = tmp_path / "source.json"
    module.create_manifest(repo, source, ["tracked.md"], [])
    original = json.loads(source.read_text(encoding="utf-8"))

    mutations = []
    top_scope = json.loads(json.dumps(original))
    top_scope["hash_scope"] = "raw_bytes"
    mutations.append(top_scope)
    record_scope = json.loads(json.dumps(original))
    record_scope["records"][0]["hash_scope"] = "raw_bytes"
    mutations.append(record_scope)
    wrong_blob = json.loads(json.dumps(original))
    wrong_blob["records"][0]["git_blob_oid"] = "0" * 40
    mutations.append(wrong_blob)
    wrong_hash = json.loads(json.dumps(original))
    wrong_hash["records"][0]["canonical_lf_sha256"] = "0" * 64
    mutations.append(wrong_hash)

    for index, payload in enumerate(mutations):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with pytest.raises(module.ProvenanceError):
            module.validate_manifest(repo, path, require_hash_scope=True)


def test_provenance_v2_legacy_grouped_shape_is_compatibility_only(tmp_path: Path):
    module = _load_provenance_module()
    repo = _git_repo(tmp_path)
    classified_path = tmp_path / "classified.json"
    classified = module.create_manifest(repo, classified_path, ["tracked.md"], [])
    record = classified["records"][0]
    legacy = {
        key: value
        for key, value in classified.items()
        if key not in {"hash_scope", "records"}
    }
    legacy["tracked_text"] = [{
        "path": record["path"],
        "git_blob_oid": record["git_blob_oid"],
        "canonical_lf_sha256": record["canonical_lf_sha256"],
    }]
    legacy["external"] = []
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    module.validate_manifest(repo, legacy_path)
    with pytest.raises(module.ProvenanceError):
        module.validate_manifest(repo, legacy_path, require_hash_scope=True)


def test_external_binary_raw_hash_remains_byte_sensitive(tmp_path: Path):
    module = _load_provenance_module()
    repo = _git_repo(tmp_path)
    external = tmp_path / "artifact.bin"
    external.write_bytes(b"\x00\r\n\xff")
    manifest_path = tmp_path / "manifest.json"
    module.create_manifest(repo, manifest_path, [], [str(external)])
    external.write_bytes(b"\x00\n\xff")
    with pytest.raises(module.ProvenanceError, match="raw artifact hash mismatch"):
        module.validate_manifest(repo, manifest_path, require_hash_scope=True)


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


def test_preflight_rejects_non_scvi_account_without_side_effects():
    fixture = r'''
set -euo pipefail
preflight="$1"
root=$(mktemp -d)
trap 'rm -rf -- "$root"' EXIT
mkdir -p -- "$root/repo"
set +e
output=$(bash "$preflight" \
  --account t0s000727 \
  --remote-root /publicfs01/fs1-t/home/t0s000727 \
  --repo "$root/repo" \
  --expected-head 0123456789abcdef0123456789abcdef01234567 \
  --expected-branch main \
  --proxy-env "$root/proxy.env" \
  --github-url https://github.com/example/repo.git \
  --github-ref refs/heads/main \
  --json)
rc=$?
set -e
test "$rc" -eq 65
printf '%s' "$output" | grep -F '"account_root":false' >/dev/null
printf '%s' "$output" | grep -F 'unsupported account' >/dev/null
test ! -e "$root/.codex_ops"
'''
    result = _run_posix_stdin(fixture, HPC_OPS / "hpc_preflight.sh")
    assert result.returncode == 0, result.stderr


def test_preflight_rejects_invalid_classified_manifest_without_side_effects():
    fixture = r'''
set -euo pipefail
preflight_source="$1"
provenance_source="$2"
proxy_source="$3"
root=$(mktemp -d)
trap 'rm -rf -- "$root"' EXIT
account_root="$root/account"
remote_root="$account_root/guardrail"
repo="$remote_root/repo"
mkdir -p -- "$repo/Filament_python/tools/hpc_ops"
git -C "$repo" init -b main >/dev/null
git -C "$repo" config user.email guardrail@example.invalid
git -C "$repo" config user.name Guardrail
cp -- "$provenance_source" "$repo/Filament_python/tools/hpc_ops/provenance_v2.py"
printf 'tracked\n' > "$repo/tracked.txt"
git -C "$repo" add -- tracked.txt Filament_python/tools/hpc_ops/provenance_v2.py
git -C "$repo" commit -m fixture >/dev/null
head=$(git -C "$repo" rev-parse HEAD)
manifest="$remote_root/invalid-provenance.json"
python3 "$provenance_source" create --repo "$repo" --output "$manifest" --tracked tracked.txt >/dev/null
python3 - "$manifest" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["records"][0]["hash_scope"] = "raw_bytes"
path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
PY

fakebin="$root/bin"
mkdir -- "$fakebin"
real_git=$(command -v git)
cat > "$fakebin/git" <<'EOF'
#!/usr/bin/env bash
for arg in "$@"; do
    if [[ "$arg" == ls-remote ]]; then
        printf '%s\t%s\n' "$PREFLIGHT_HEAD" refs/heads/main
        exit 0
    fi
done
exec "$REAL_GIT" "$@"
EOF
chmod 700 -- "$fakebin/git"
for command in sbatch sacct scontrol; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "$fakebin/$command"
    chmod 700 -- "$fakebin/$command"
done

miniforge="$root/miniforge"
env_prefix="$root/filament-env"
mkdir -p -- "$miniforge/etc/profile.d" "$env_prefix/bin"
cat > "$miniforge/etc/profile.d/conda.sh" <<'EOF'
conda() {
    [[ "$1" == activate && "$2" == Filament_python ]] || return 1
    export CONDA_PREFIX="$PREFLIGHT_ENV_PREFIX"
    export PATH="$CONDA_PREFIX/bin:$PATH"
}
EOF
cat > "$env_prefix/bin/python" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == -c ]]; then exit 0; fi
exec python3 "$@"
EOF
chmod 700 -- "$env_prefix/bin/python"

tool_dir="$root/tool"
mkdir -- "$tool_dir"
sed -e "s#/data/run01/scvi806#$account_root#g" \
    -e "s#/data/apps/miniforge/25.3.0-3#$miniforge#g" \
    -e "s#/data/home/scvi806/.conda/envs/Filament_python#$env_prefix#g" \
    "$preflight_source" > "$tool_dir/hpc_preflight.sh"
cp -- "$proxy_source" "$tool_dir/hpc_proxy_env.sh"
chmod 700 -- "$tool_dir/hpc_preflight.sh"
proxy="$root/proxy.env"
printf '%s\n' 'http_proxy=http://proxy.example.invalid:8080' 'https_proxy=https://proxy.example.invalid:8443' > "$proxy"
chmod 600 -- "$proxy"
export REAL_GIT="$real_git" PREFLIGHT_HEAD="$head" PREFLIGHT_ENV_PREFIX="$env_prefix"
PATH="$fakebin:$PATH"
set +e
output=$("$tool_dir/hpc_preflight.sh" \
    --account scvi806 --remote-root "$remote_root" --repo "$repo" \
    --expected-head "$head" --expected-branch main \
    --proxy-env "$proxy" --github-url https://github.com/example/repo.git \
    --github-ref refs/heads/main --provenance-manifest "$manifest" --json)
rc=$?
set -e
test "$rc" -eq 75
printf '%s' "$output" | grep -F '"status":"failed"' >/dev/null
printf '%s' "$output" | grep -F 'provenance manifest strict validation failed' >/dev/null
test ! -e "$remote_root/run-created"
test ! -e "$remote_root/.codex_ops"
'''
    result = _run_posix_stdin(
        fixture,
        HPC_OPS / "hpc_preflight.sh",
        PROVENANCE_PATH,
        HPC_OPS / "hpc_proxy_env.sh",
    )
    assert result.returncode == 0, result.stderr


def test_hr2e_submission_provenance_gate_precedes_every_side_effect():
    launcher = (ROOT / "Filament_python" / "tools" / "submit_hr2e_schedule_convergence.sh").read_text(
        encoding="utf-8"
    )
    gate = launcher.index('provenance_v2.py" validate')
    run_directory = launcher.index('mkdir -p "$RUN_ROOT"')
    receipt = launcher.index('receipt="$RUN_ROOT/submission_receipt.tsv"')
    scheduler = launcher.index("sbatch --parsable")
    assert gate < run_directory < receipt < scheduler


def test_new_hr2e_entrypoints_do_not_bind_tracked_config_raw_bytes():
    tool_root = ROOT / "Filament_python" / "tools"
    prepare = (tool_root / "prepare_hr2e_schedule_convergence.py").read_text(encoding="utf-8")
    submit = (tool_root / "submit_hr2e_schedule_convergence.sh").read_text(encoding="utf-8")
    batch = (tool_root / "hr2e_schedule_convergence.sbatch").read_text(encoding="utf-8")
    assert "config_sha256" not in prepare
    assert "full_raman_source_sha256" not in prepare
    assert "hashlib.sha256" not in submit
    assert "EXPECTED_CONFIG_SHA256" not in submit
    assert "EXPECTED_CONFIG_SHA256" not in batch
    assert 'sha256sum "${CONFIG_PATH}"' not in batch
    assert '"hash_scope": "classified_by_record"' in prepare
    assert '"tracked_paths": tracked_paths' in prepare
    for source in (submit, batch):
        assert "canonical_lf_sha256" in source


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
        .replace("/data/apps/miniforge/25.3.0-3", str(miniforge))
        .replace(
            "/data/home/scvi806/.conda/envs/Filament_python",
            str(miniforge / "envs" / "Filament_python"),
        ),
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
operation_id=33333333-3333-3333-3333-333333333333
bundle="$staging/source.bundle.verified"
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
    if [[ "$arg" == ls-remote ]]; then exit 1; fi
    if [[ "${FAIL_GIT_STATUS:-0}" == 1 && "$arg" == status ]]; then exit 97; fi
done
exec "$REAL_GIT" "$@"
EOF
chmod 700 -- "$fakebin/git"
export REAL_GIT="$real_git"
PATH="$fakebin:$PATH"

target="$staging/checkout"
state="$staging/acquisition.json"
clone_json=$("$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode clone --url https://github.com/example/repo.git --ref refs/heads/main --expected-head "$head" --expected-branch main --proxy-env "$proxy" --target "$target" --bundle "$bundle" --bundle-sha "$bundle_sha" --operation-id "$operation_id" --state-file "$state" --timeout-seconds 1)
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

fetch_operation_id=44444444-4444-4444-4444-444444444444
fetch_state="$staging/fetch.json"
fetch_json=$("$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode fetch --url https://github.com/example/repo.git --ref refs/heads/main --expected-head "$head" --expected-branch main --proxy-env "$proxy" --target "$target" --bundle "$bundle" --bundle-sha "$bundle_sha" --operation-id "$fetch_operation_id" --state-file "$fetch_state" --timeout-seconds 1)
python3 - "$fetch_json" "$head" <<'PY'
import json, sys
report = json.loads(sys.argv[1])
assert report["ok"] is True
assert report["fetch_head"] == sys.argv[2]
PY

export FAIL_GIT_STATUS=1
set +e
"$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode fetch --url https://github.com/example/repo.git --ref refs/heads/main --expected-head "$head" --expected-branch main --proxy-env "$proxy" --target "$target" --bundle "$bundle" --bundle-sha "$bundle_sha" --operation-id "$fetch_operation_id" --state-file "$fetch_state" --timeout-seconds 1 >/dev/null
status_failure_rc=$?
set -e
unset FAIL_GIT_STATUS
test "$status_failure_rc" -ne 0

wrong_target="$staging/wrong"
wrong_state="$staging/wrong.json"
set +e
"$root/tool/hpc_git_source.sh" --account scvi806 --remote-root "$remote_root" --staging-root "$staging" --mode clone --url https://github.com/example/repo.git --ref refs/heads/main --expected-head 0000000000000000000000000000000000000000 --expected-branch main --proxy-env "$proxy" --target "$wrong_target" --bundle "$bundle" --bundle-sha "$bundle_sha" --operation-id "$operation_id" --state-file "$wrong_state" --timeout-seconds 1 >/dev/null
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


def test_powershell_wrapper_write_receipt_states_distinguish_unknown_from_failure():
    wrapper = (HPC_OPS / "Invoke-PappRemoteScript.ps1").read_text(encoding="utf-8")
    assert "function Convert-RemoteWriteReceipt" in wrapper
    assert "-CaptureOutput" in wrapper
    assert "unknown_no_receipt" in wrapper
    assert "rejected_or_failed" in wrapper
    assert "[string]$remoteReport.state -eq 'completed'" in wrapper
    assert "remote_receipt_lines" in wrapper
    # The Write path must not unconditionally report success after transport
    # returns; completion is gated by a parsed remote final receipt.
    write_branch = wrapper[wrapper.index("$failureStage = 'execute-write-dispatcher'"):]
    assert "New-StatusJson -Ok $true -State 'completed'" not in write_branch


def test_powershell_wrapper_rejects_non_scvi_account_at_parameter_binding():
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("PowerShell is not available on this host")
    wrapper = HPC_OPS / "Invoke-PappRemoteScript.ps1"
    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert "[ValidateSet('scvi806')]" in wrapper_text
    assert "t0s000727" not in wrapper_text
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(wrapper),
            "-Account",
            "t0s000727",
            "-RemoteRoot",
            "/publicfs01/fs1-t/home/t0s000727",
            "-LocalScript",
            str(HPC_OPS / "hpc_preflight.sh"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode != 0
    assert not result.stdout.strip()


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


def test_ssh_wrapper_dry_run_uses_native_transport_and_no_papp(tmp_path: Path):
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("PowerShell is not available on this host")
    wrapper = HPC_OPS / "Invoke-SshRemoteScript.ps1"
    wrapper_ps = str(wrapper).replace("'", "''")
    script_ps = str(HPC_OPS / "hpc_preflight.sh").replace("'", "''")
    harness = tmp_path / "invoke-ssh-wrapper.ps1"
    harness.write_text(
        f"& '{wrapper_ps}' -Target scvi-hpc "
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
    assert report["target"] == "scvi-hpc"
    assert report["account"] == "scvi806"
    assert report["dry_run"] is True
    source = wrapper.read_text(encoding="utf-8")
    assert "function Invoke-SshTransport" in source
    assert "BatchMode=yes" in source
    assert "PAPP_CLOUD_BIN" not in source
    assert "wsl.exe" not in source


def test_ssh_wrapper_rejects_non_scvi_target_at_parameter_binding():
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("PowerShell is not available on this host")
    wrapper = HPC_OPS / "Invoke-SshRemoteScript.ps1"
    result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(wrapper),
            "-Target",
            "t0-hpc",
            "-RemoteRoot",
            "/publicfs01/fs1-t/home/t0s000727",
            "-LocalScript",
            str(HPC_OPS / "hpc_preflight.sh"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode != 0
    assert not result.stdout.strip()


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


def test_historical_gitattributes_hash_exceptions_remain_explicit_and_unchanged():
    binary = "Filament_python/results/isaacs_complete_eq27/provenance_221822/execution_lock_43ac6b4.json"
    crlf = [
        "Filament_python/results/isaacs_complete_eq27/c1_closure_summary.json",
        "Filament_python/results/isaacs_complete_eq27/c1_operator_report.md",
        "Filament_python/results/isaacs_complete_eq27/submission_manifest.json",
        "Filament_python/results/density_translation_width/density_translation_width_20260715_002/paper_pycap_120fs.csv",
    ]
    explicit_lf = [
        "Filament_python/results/isaacs_complete_eq27/provenance_221822/SUBMISSION_LOCK",
        "Filament_python/results/isaacs_complete_eq27/provenance_221822/submission_record.txt",
    ]

    binary_attrs = _git(ROOT, "check-attr", "text", "eol", "--", binary)
    assert f"{binary}: text: unset" in binary_attrs
    for path in crlf:
        attrs = _git(ROOT, "check-attr", "text", "eol", "--", path)
        assert f"{path}: text: set" in attrs
        assert f"{path}: eol: crlf" in attrs
    for path in explicit_lf:
        attrs = _git(ROOT, "check-attr", "text", "eol", "--", path)
        assert f"{path}: text: set" in attrs
        assert f"{path}: eol: lf" in attrs

    protected = [binary, *crlf, *explicit_lf]
    unchanged = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", *protected],
        check=False,
    )
    assert unchanged.returncode == 0


def test_secret_scan_of_new_guardrails():
    paths = [
        HPC_OPS / "Invoke-SshRemoteScript.ps1",
        HPC_OPS / "Invoke-PappRemoteScript.ps1",
        HPC_OPS / "hpc_proxy_env.sh",
        HPC_OPS / "hpc_preflight.sh",
        HPC_OPS / "hpc_git_source.sh",
        HPC_OPS / "provenance_v2.py",
        HPC_OPS / "audit_batch_entry.py",
        HPC_OPS / "README.md",
        ROOT / "docs" / "experience" / "sol_luna_hpc_execution_playbook.md",
        ROOT / "docs" / "experience" / "2026-08-22_isaacs_eq27_c2_postmortem.md",
        ROOT / "docs" / "experience" / "2026-08-24_hybrid_execution_postmortem.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "gh" + "p_" not in text
    assert "wjm" + "@2023" not in text
    assert "https://user:password" + "@" not in text
    assert "PAPP_CLOUD_BIN" not in (HPC_OPS / "Invoke-SshRemoteScript.ps1").read_text(encoding="utf-8")
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
