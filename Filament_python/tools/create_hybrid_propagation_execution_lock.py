#!/usr/bin/env python3
"""Create the external, single-use execution lock for the 0.60 m campaign.

The lock binds a clean committed Git HEAD to the prepared manifest and both
case configuration hashes.  It is intentionally written outside the tracked
tree (by default below ``.git/codex-locks``), and this module never submits a
job or consumes the lock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
FILAMENT_ROOT = REPO / "Filament_python"
DEFAULT_MANIFEST = FILAMENT_ROOT / "results" / "hybrid_propagation_validation" / "submission_manifest.json"
DEFAULT_OUTPUT = REPO / ".git" / "codex-locks" / "hybrid_propagation_validation_0p60.execution_lock.json"
CAMPAIGN_ID = "hybrid_propagation_validation_0p60"
REMOTE_CAMPAIGN_ROOT = "/data/run01/scvi806/user_Wangjimin/hybrid_propagation_validation_0p60"
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 5090"
LOCK_SCHEMA = "khz_filament.hybrid_propagation_validation.execution_lock.v1"
EXPECTED_RESOLUTION = "external execution_lock generated after final source commit"


class ExecutionLockError(RuntimeError):
    """Raised when a campaign cannot be bound to an execution lock."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionLockError(f"git {' '.join(args)} failed: {exc}") from exc
    return result.stdout.strip()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionLockError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExecutionLockError(f"{label} must be a JSON object: {path}")
    return payload


def _prepare_module():
    path = Path(__file__).with_name("prepare_hybrid_propagation_validation.py")
    spec = importlib.util.spec_from_file_location("hybrid_prepare_for_lock", path)
    if spec is None or spec.loader is None:
        raise ExecutionLockError(f"cannot load preparation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    # Preparation stores paths relative to Filament_python.  Accept the
    # absolute paths used by isolated tests as well, but never search outside
    # the declared manifest/config provenance.
    return (FILAMENT_ROOT / candidate).resolve()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError as exc:
        raise ExecutionLockError(f"path is outside repository: {path}") from exc


def _record_path(path: Path) -> str:
    """Use repo-relative provenance in production, absolute paths in fixtures."""
    try:
        return _repo_relative(path)
    except ExecutionLockError:
        return str(path.resolve())


def _assert_committed(path: Path) -> None:
    relative = _repo_relative(path)
    tracked = _git("ls-files", "--error-unmatch", "--", relative)
    if tracked != relative:
        raise ExecutionLockError(f"required file is not committed: {relative}")


def _check_hash(path: Path, expected: Any, label: str) -> str:
    if not path.is_file():
        raise ExecutionLockError(f"{label} does not exist: {path}")
    actual = sha256(path)
    if not isinstance(expected, str) or actual != expected:
        raise ExecutionLockError(f"{label} SHA256 mismatch: expected={expected!r} actual={actual}")
    return actual


def validate_manifest_lock(
    manifest_path: Path,
    *,
    expected_git_sha: str | None = None,
    require_clean: bool = True,
    require_committed: bool = True,
) -> dict[str, Any]:
    """Validate manifest, mother, pair configs and exact hashes.

    This function is side-effect free and is reused by the batch contract.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ExecutionLockError(f"manifest does not exist: {manifest_path}")
    manifest = _load_object(manifest_path, "campaign manifest")
    if manifest.get("schema") != "khz_filament.hybrid_propagation_validation.submission_manifest.v1":
        raise ExecutionLockError("manifest schema is invalid")
    if manifest.get("campaign_id") != CAMPAIGN_ID:
        raise ExecutionLockError("manifest campaign_id is not the fixed 0.60 m campaign")
    if manifest.get("remote_campaign_root") != REMOTE_CAMPAIGN_ROOT:
        raise ExecutionLockError("manifest remote_campaign_root is not fixed")
    if manifest.get("status") != "prepared_not_submitted":
        raise ExecutionLockError("manifest status must be prepared_not_submitted before submission")
    if manifest.get("expected_git_sha") is not None:
        raise ExecutionLockError("manifest expected_git_sha must be null before lock creation")
    if manifest.get("execution_lock_required") is not True:
        raise ExecutionLockError("manifest execution_lock_required must be true")
    if manifest.get("expected_git_sha_resolution") != EXPECTED_RESOLUTION:
        raise ExecutionLockError("manifest expected_git_sha_resolution is invalid")
    head = _git("rev-parse", "HEAD")
    if expected_git_sha and head != expected_git_sha:
        raise ExecutionLockError(f"HEAD mismatch: expected={expected_git_sha} actual={head}")
    if require_clean and _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ExecutionLockError("source worktree is not clean")

    prepare = _prepare_module()
    mother_path = FILAMENT_ROOT / prepare.SOURCE_CONFIG_REL
    mother_hash = _check_hash(mother_path, manifest.get("mother_config_sha256"), "mother config")
    if manifest.get("mother_config") != prepare.SOURCE_CONFIG_REL:
        raise ExecutionLockError("manifest mother_config is not the fixed mother")

    reference_path = _resolve_manifest_path(manifest_path, str(manifest.get("reference_config", "")))
    hybrid_path = _resolve_manifest_path(manifest_path, str(manifest.get("hybrid_config", "")))
    reference_hash = _check_hash(reference_path, manifest.get("reference_config_sha256"), "reference config")
    hybrid_hash = _check_hash(hybrid_path, manifest.get("hybrid_config_sha256"), "hybrid config")
    reference = _load_object(reference_path, "reference config")
    hybrid = _load_object(hybrid_path, "hybrid config")
    try:
        _, _, expected_diff = prepare.build_pair(_load_object(mother_path, "mother config"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ExecutionLockError(f"fixed mother pair cannot be rebuilt: {exc}") from exc
    differences = prepare.config_diff(reference, hybrid)
    if differences != expected_diff or manifest.get("strict_config_diff") != expected_diff:
        raise ExecutionLockError(f"A/B configuration diff is not the unique hybrid delta: {differences}")
    if reference.get("propagation", {}).get("propagation_mode") != "full_nonlinear_from_z0":
        raise ExecutionLockError("reference propagation mode is invalid")
    if reference.get("propagation", {}).get("z_nl_start") != 0.0:
        raise ExecutionLockError("reference z_nl_start must be zero")
    if hybrid.get("propagation", {}).get("propagation_mode") != "hybrid":
        raise ExecutionLockError("hybrid propagation mode is invalid")
    if hybrid.get("propagation", {}).get("z_nl_start") != 0.6:
        raise ExecutionLockError("hybrid z_nl_start must be exactly 0.60 m")
    for label, config in (("reference", reference), ("hybrid", hybrid)):
        if config.get("propagation", {}).get("measure_performance") is not True:
            raise ExecutionLockError(f"{label} measure_performance must be true")
        if config.get("propagation", {}).get("diag_operator_energy") is not True:
            raise ExecutionLockError(f"{label} diag_operator_energy must be true")
        if config.get("propagation", {}).get("limit_focus_window") is not False:
            raise ExecutionLockError(f"{label} limit_focus_window must be false")

    cases = manifest.get("cases")
    if not isinstance(cases, dict) or list(cases) != ["reference", "hybrid"]:
        raise ExecutionLockError("manifest case order must be reference then hybrid")
    if manifest.get("execution", {}).get("allocation_count") != 1:
        raise ExecutionLockError("manifest must authorize one allocation")
    if manifest.get("execution", {}).get("sequential") is not True:
        raise ExecutionLockError("manifest must require sequential cases")
    if manifest.get("execution", {}).get("retry_policy") != "no_retry":
        raise ExecutionLockError("manifest retry policy must be no_retry")
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise ExecutionLockError("manifest resources are missing")
    expected_resources = {
        "partition": "gpu",
        "gpu_count": 1,
        "cpu_threads": 8,
        "requested_time": "15:00:00",
        "expected_gpu_model": EXPECTED_GPU_MODEL,
    }
    for key, value in expected_resources.items():
        if resources.get(key) != value:
            raise ExecutionLockError(f"manifest resources.{key} must equal {value!r}")
    if manifest.get("additional_start_planes_authorized") != []:
        raise ExecutionLockError("additional start planes are not authorized")
    if manifest.get("pulse_train_authorized") is not False or manifest.get("round_2_authorized") is not False:
        raise ExecutionLockError("pulse-train/Round 2 work is not authorized")

    if require_committed:
        for path in (manifest_path, mother_path, reference_path, hybrid_path):
            _assert_committed(path)

    return {
        "head": head,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "mother_path": mother_path,
        "mother_config_sha256": mother_hash,
        "reference_path": reference_path,
        "reference_config_sha256": reference_hash,
        "hybrid_path": hybrid_path,
        "hybrid_config_sha256": hybrid_hash,
    }


def _atomic_create_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ExecutionLockError(f"execution lock target already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ExecutionLockError(f"execution lock target appeared during create: {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def create_lock(
    manifest_path: Path = DEFAULT_MANIFEST,
    output: Path = DEFAULT_OUTPUT,
    *,
    expected_git_sha: str | None = None,
    require_clean: bool = True,
    require_committed: bool = True,
) -> dict[str, Any]:
    """Validate and create one external lock; never submit or consume it."""
    checked = validate_manifest_lock(
        manifest_path,
        expected_git_sha=expected_git_sha,
        require_clean=require_clean,
        require_committed=require_committed,
    )
    output = Path(output).expanduser().absolute()
    git_dir = Path(_git("rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = REPO / git_dir
    git_dir = git_dir.resolve()
    try:
        output.resolve(strict=False).relative_to(REPO.resolve())
    except ValueError:
        pass
    else:
        try:
            output.resolve(strict=False).relative_to(git_dir)
        except ValueError as exc:
            raise ExecutionLockError(
                "execution lock inside the checkout is allowed only below the Git directory"
            ) from exc
    manifest = checked["manifest"]
    payload = {
        "schema": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "remote_campaign_root": REMOTE_CAMPAIGN_ROOT,
        "status": "authorized_not_consumed",
        "expected_git_sha": checked["head"],
        "manifest_path": _record_path(checked["manifest_path"]),
        "manifest_sha256": checked["manifest_sha256"],
        "mother_config": _record_path(checked["mother_path"]),
        "mother_config_sha256": checked["mother_config_sha256"],
        "reference_config": _record_path(checked["reference_path"]),
        "reference_config_sha256": checked["reference_config_sha256"],
        "hybrid_config": _record_path(checked["hybrid_path"]),
        "hybrid_config_sha256": checked["hybrid_config_sha256"],
        "case_order": ["reference", "hybrid"],
        "allocation_count": 1,
        "retry_policy": "no_retry",
        "expected_gpu_model": EXPECTED_GPU_MODEL,
        "gpu_count": 1,
        "cpu_threads": 8,
        "requested_time": "15:00:00",
        "strict_config_diff": manifest["strict_config_diff"],
        "raw_npz_policy": "retain_on_HPC_RUN_DIR_not_repository",
        "created_by": "create_hybrid_propagation_execution_lock.py",
    }
    _atomic_create_json(output, payload)
    return {**payload, "lock_path": str(output), "lock_sha256": sha256(output)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-git-sha")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(create_lock(args.manifest, args.output, expected_git_sha=args.expected_git_sha), ensure_ascii=False, indent=2))
    except (OSError, ExecutionLockError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
