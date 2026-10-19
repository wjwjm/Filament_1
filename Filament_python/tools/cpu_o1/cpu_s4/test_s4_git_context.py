"""Adapter Git-context gates; all Git repositories are disposable test clones."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
Q3_HEAD = "66801ad435d6ec1967338cee4eb82af2a152389c"
SCIENCE_TREE = "821fec5bdacd832b0347e1f2e84285cb54172e57"
ADMISSION_RELATIVE = Path("Filament_python/tools/cpu_o1/cpu_s4/admission.py")


def _git_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.upper().startswith("GIT_"):
            env.pop(name, None)
    return env


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], cwd=repo, env=_git_environment(),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr}")
    return result


def _copy_admission(adapter: Path) -> None:
    target = adapter / ADMISSION_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "admission.py", target)


@pytest.fixture(scope="module")
def git_adapter_base():
    # Keep clones under the short worktree root rather than a potentially long
    # user temp path; core.longpaths is needed for this repository on Windows.
    base = Path(tempfile.mkdtemp(prefix="s4g_", dir=REPO_ROOT))
    adapter = base / "adapter"
    git_dir = base / "adapter_git"
    clone = subprocess.run(
        ["git", "-c", "core.longpaths=true", "clone", "--no-hardlinks",
         f"--separate-git-dir={git_dir}", str(REPO_ROOT), str(adapter)],
        cwd=base, env=_git_environment(), capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if clone.returncode != 0:
        shutil.rmtree(base, ignore_errors=True)
        pytest.fail(f"Git integration fixture clone failed: {clone.stderr}")
    _git(adapter, "checkout", "--detach", Q3_HEAD)
    _git(adapter, "update-ref", "refs/heads/s4-test-q3", Q3_HEAD)
    _copy_admission(adapter)
    pointer = adapter / ".git"
    if pointer.is_symlink() or not pointer.is_file():
        shutil.rmtree(base, ignore_errors=True)
        pytest.fail("Git integration fixture did not create a regular .git pointer")
    pointer_bytes = pointer.read_bytes()
    yield {"base": base, "adapter": adapter, "git_dir": git_dir,
           "pointer_bytes": pointer_bytes}
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def clean_git_adapter(git_adapter_base):
    adapter = git_adapter_base["adapter"]
    pointer = adapter / ".git"
    pointer.unlink()
    pointer.write_bytes(git_adapter_base["pointer_bytes"])
    _git(adapter, "reset", "--hard", Q3_HEAD)
    yield git_adapter_base
    pointer.unlink()
    pointer.write_bytes(git_adapter_base["pointer_bytes"])
    _git(adapter, "reset", "--hard", Q3_HEAD)


def _invoke(adapter: Path, git_dir: Path, *, mode: str = "public",
            wrong_import_root: Path | None = None,
            env: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess, dict]:
    script = r'''
import importlib.util, json, pathlib, sys
adapter = pathlib.Path(sys.argv[1])
git_dir = pathlib.Path(sys.argv[2])
mode = sys.argv[3]
wrong_root = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else None
if wrong_root is not None:
    sys.path.insert(0, str(wrong_root / "Filament_python"))
    import tools.cpu_o1.cpu_o1
spec = importlib.util.spec_from_file_location(
    "s4_git_context_admission", adapter / "Filament_python/tools/cpu_o1/cpu_s4/admission.py")
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)
try:
    if mode == "wrong_tree_expectation":
        value = admission._validate_adapter_git_repository_context(
            adapter, git_dir, expected_head=admission.Q3_HEAD,
            expected_science_tree="0" * 40)
    else:
        value = admission.validate_adapter_git_context(adapter, git_dir)
    print(json.dumps({"receipt": value}, sort_keys=True))
except Exception as exc:
    print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
    raise SystemExit(3)
'''
    args = [sys.executable, "-s", "-B", "-c", script,
            str(adapter), str(git_dir), mode]
    if wrong_import_root is not None:
        args.append(str(wrong_import_root))
    result = subprocess.run(
        args, cwd=adapter, env=env or _git_environment(), capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {"error": result.stderr or result.stdout}
    return result, payload


def test_s4_git_context_separate_gitdir_and_real_source_binding(clean_git_adapter):
    adapter = clean_git_adapter["adapter"]
    git_dir = clean_git_adapter["git_dir"]
    env = _git_environment()
    env.update({
        "GIT_DIR": str(clean_git_adapter["base"] / "wrong_git"),
        "GIT_WORK_TREE": str(clean_git_adapter["base"] / "wrong_tree"),
        "GIT_OBJECT_DIRECTORY": str(clean_git_adapter["base"] / "wrong_objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(clean_git_adapter["base"] / "wrong_alternate"),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.worktree",
        "GIT_CONFIG_VALUE_0": str(clean_git_adapter["base"] / "wrong_tree"),
    })
    result, payload = _invoke(adapter, git_dir, env=env)
    assert result.returncode == 0, payload
    receipt = payload["receipt"]
    assert receipt["status"] == "PASS"
    assert receipt["head"] == Q3_HEAD
    assert receipt["science_tree"] == SCIENCE_TREE
    assert receipt["actual_git_top_level"] == str(adapter.resolve())
    assert receipt["actual_git_dir"] == str(git_dir.resolve())
    assert receipt["imported_cpu_o1"]["path"] == str(
        (adapter / "Filament_python/tools/cpu_o1/cpu_o1.py").resolve())
    assert receipt["p1r2_source_binding"]["working_tree_matches_anchor"] is True
    assert set(receipt["git_redirect_environment_cleared"]) >= {
        "GIT_DIR", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
    }


def test_s4_git_context_rejects_parent_repo_discovery(clean_git_adapter):
    adapter = clean_git_adapter["adapter"]
    nested = adapter / "no_dot_git"
    nested.mkdir()
    try:
        # The gate imports its own deployed helper before it checks the target
        # repository root, so provide the same helper in this negative fixture.
        _copy_admission(nested)
        result, payload = _invoke(nested, clean_git_adapter["git_dir"])
    finally:
        shutil.rmtree(nested)
    assert result.returncode == 3
    assert "regular .git pointer" in payload["error"]


def test_s4_git_context_rejects_pointer_to_wrong_gitdir(clean_git_adapter):
    adapter = clean_git_adapter["adapter"]
    wrong_git = clean_git_adapter["base"] / "wrong_git"
    wrong_git.mkdir()
    pointer = adapter / ".git"
    pointer.write_text(
        "gitdir: " + str(wrong_git) + "\n",
        encoding="ascii")
    result, payload = _invoke(adapter, clean_git_adapter["git_dir"])
    assert result.returncode == 3
    assert "differs from configured adapter_git_dir" in payload["error"]


def test_s4_git_context_rejects_wrong_head(clean_git_adapter):
    adapter = clean_git_adapter["adapter"]
    _git(adapter, "-c", "user.name=S4 test", "-c",
         "user.email=s4@example.invalid", "commit", "--allow-empty",
         "-m", "test wrong adapter head")
    result, payload = _invoke(adapter, clean_git_adapter["git_dir"])
    assert result.returncode == 3
    assert "HEAD differs from frozen Q3 identity" in payload["error"]


def test_s4_git_context_rejects_science_tree_identity_mismatch(clean_git_adapter):
    result, payload = _invoke(
        clean_git_adapter["adapter"], clean_git_adapter["git_dir"],
        mode="wrong_tree_expectation")
    assert result.returncode == 3
    assert "science tree differs from frozen Q3 identity" in payload["error"]


def test_s4_git_context_rejects_tracked_source_drift(clean_git_adapter):
    adapter = clean_git_adapter["adapter"]
    tracked = adapter / "Filament_python/tools/cpu_o1/cpu_o1.py"
    tracked.write_bytes(tracked.read_bytes() + b"\n# unapproved drift\n")
    result, payload = _invoke(adapter, clean_git_adapter["git_dir"])
    assert result.returncode == 3
    assert "unstaged changes" in payload["error"]


def test_s4_git_context_rejects_cpu_o1_import_outside_adapter(clean_git_adapter):
    result, payload = _invoke(
        clean_git_adapter["adapter"], clean_git_adapter["git_dir"],
        wrong_import_root=REPO_ROOT)
    assert result.returncode == 3
    assert "imported tools.cpu_o1.cpu_o1 is outside adapter root" in payload["error"]


def test_s4_git_context_rejects_missing_p1r2_anchor_object(git_adapter_base):
    base = Path(tempfile.mkdtemp(prefix="s4m_", dir=REPO_ROOT))
    adapter = base / "adapter"
    git_dir = base / "adapter_git"
    try:
        clone = subprocess.run(
            ["git", "-c", "core.longpaths=true", "clone", "--no-local",
             "--depth=1", "--no-checkout",
             "--single-branch", "--branch", "s4-test-q3",
             f"--separate-git-dir={git_dir}",
             str(git_adapter_base["adapter"]), str(adapter)],
            cwd=base, env=_git_environment(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        assert clone.returncode == 0, clone.stderr
        _git(adapter, "checkout", "--detach", Q3_HEAD)
        _copy_admission(adapter)
        shallow = git_dir / "shallow"
        assert shallow.is_file()
        # Hide the shallow marker so the frozen checker must detect the missing
        # historical anchor object itself rather than merely rejecting depth=1.
        shallow.unlink()
        result, payload = _invoke(adapter, git_dir)
        assert result.returncode == 3
        assert "frozen P1-R2 source binding failed" in payload["error"]
    finally:
        shutil.rmtree(base, ignore_errors=True)
