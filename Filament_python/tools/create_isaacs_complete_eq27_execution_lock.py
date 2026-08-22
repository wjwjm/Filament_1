#!/usr/bin/env python3
"""Bind the committed C2 manifest to the clean source tree used for execution.

This tool only creates an external authorization record.  It never invokes
``sbatch`` or starts a propagation.  The default lock path is under the Git
metadata directory so creating the lock does not modify tracked results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
FILAMENT_ROOT = REPO / "Filament_python"
DEFAULT_MANIFEST = FILAMENT_ROOT / "results" / "isaacs_complete_eq27" / "submission_manifest.json"
DEFAULT_OUTPUT = REPO / ".git" / "codex-locks" / "isaacs_complete_eq27_c2.execution_lock.json"
FIXED_CAMPAIGN_ID = "isaacs_complete_eq27_c2"
FIXED_REMOTE_CAMPAIGN_ROOT = "/data/run01/scvi806/user_Wangjimin/isaacs_complete_eq27_c2"
LOCK_SCHEMA = "khz_filament.isaacs_complete_eq27.c2_execution_lock.v1"
EXPECTED_RESOLUTION = "external execution_lock generated after final source commit"
EXPECTED_GPU_MODEL = "NVIDIA GeForce RTX 5090"
EXPECTED_I_CAP = 1.0e19
SOURCE_CONFIG_REL = "configs/isaacs_raman_closure/120fs_talebpour_isaacs_full_operator_on.json"
DERIVED_CONFIG_REL = "results/isaacs_complete_eq27/120fs_talebpour_isaacs_complete_eq27.json"
LOCKED_BASE_CONFIG_SHA256 = "942adca964f50b689fa5985c9af46f294da7948646b246c39ca0d50238a1b02a"
PYCAP_REL = "results/density_translation_width/density_translation_width_20260715_002/paper_pycap_120fs.csv"
PYCAP_SHA256 = "9b43e75ebc08ccb0a7796829e45c6727b42ab12cd661b9a3d8d235ef89d31461"
C1_COMMIT = "459dd108b9873b0e8b18fe83111f386993cf5b9f"
C1_SUMMARY_REL = "results/isaacs_complete_eq27/c1_closure_summary.json"
C1_SUMMARY_SHA256 = "ccf6f865042651894e747f1272c5371cad8bc4bb7fd6abd11b61684a795ebcdc"
C1_REPORT_REL = "results/isaacs_complete_eq27/c1_operator_report.md"
C1_REPORT_SHA256 = "fe8b7fe99a88dde5d4c987d88d1a87dd5208461bb70ff25af6e365ef4ac7b21d"


def _prepare_module():
    path = Path(__file__).with_name("prepare_isaacs_complete_eq27_job.py")
    spec = importlib.util.spec_from_file_location("_c2_prepare_for_execution_lock", path)
    if spec is None or spec.loader is None:
        raise ExecutionLockError(f"cannot load fixed C2 config assertions: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExecutionLockError(RuntimeError):
    """Raised when the manifest or source tree is not lockable."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExecutionLockError(f"git {' '.join(args)} failed: {exc}") from exc
    return completed.stdout.strip()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError as exc:
        raise ExecutionLockError(f"path must be inside repository: {path}") from exc


def _manifest_config(manifest: dict[str, Any], manifest_path: Path) -> Path:
    value = manifest.get("derived_config")
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ExecutionLockError("manifest derived_config must be a non-empty relative path")
    config = (FILAMENT_ROOT / Path(value)).resolve()
    try:
        config.relative_to(FILAMENT_ROOT.resolve())
    except ValueError as exc:
        raise ExecutionLockError("manifest derived_config escapes Filament_python") from exc
    if not config.is_file():
        raise ExecutionLockError(f"manifest derived config not found: {config}")
    if not manifest_path.is_file():
        raise ExecutionLockError(f"manifest not found: {manifest_path}")
    return config


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionLockError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExecutionLockError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_c1_gate(manifest: dict[str, Any], head: str) -> dict[str, Any]:
    """Validate immutable C1 evidence without trusting its dirty-worktree fields."""
    if manifest.get("parent_c1_commit") != C1_COMMIT:
        raise ExecutionLockError("manifest parent_c1_commit is not the fixed C1 commit")
    expected = {
        "commit": C1_COMMIT,
        "summary_path": C1_SUMMARY_REL,
        "summary_sha256": C1_SUMMARY_SHA256,
        "report_path": C1_REPORT_REL,
        "report_sha256": C1_REPORT_SHA256,
        "overall": "PASS",
    }
    if manifest.get("c1_gate") != expected:
        raise ExecutionLockError("manifest c1_gate binding is not the fixed C1 evidence")
    try:
        _git("merge-base", "--is-ancestor", C1_COMMIT, head)
    except ExecutionLockError as exc:
        raise ExecutionLockError(f"fixed C1 commit is not an ancestor of HEAD: {head}") from exc
    summary_path = FILAMENT_ROOT / C1_SUMMARY_REL
    report_path = FILAMENT_ROOT / C1_REPORT_REL
    if not summary_path.is_file() or sha256(summary_path) != C1_SUMMARY_SHA256:
        raise ExecutionLockError("fixed C1 machine-readable summary is missing or has the wrong SHA256")
    if not report_path.is_file() or sha256(report_path) != C1_REPORT_SHA256:
        raise ExecutionLockError("fixed C1 operator report is missing or has the wrong SHA256")
    summary = _load_object(summary_path, "fixed C1 closure summary")
    if summary.get("schema") != "khz_filament.isaacs_complete_eq27.c1.v1":
        raise ExecutionLockError("fixed C1 closure summary schema is invalid")
    # Do not inspect summary.current_sha or summary.git_dirty: those fields
    # describe the audit's generation worktree, not the locked source commit.
    if summary.get("overall") != "PASS":
        raise ExecutionLockError("fixed C1 machine-readable overall gate is not PASS")
    return expected


def _validate_fixed_config_contract(manifest: dict[str, Any], derived_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Recheck the locked source and the one permitted C2 configuration delta."""
    source_rel = manifest.get("source_config")
    if source_rel != SOURCE_CONFIG_REL:
        raise ExecutionLockError(f"manifest source_config is not the locked path: {source_rel!r}")
    if manifest.get("derived_config") != DERIVED_CONFIG_REL:
        raise ExecutionLockError(f"manifest derived_config is not the locked path: {manifest.get('derived_config')!r}")
    source_path = (FILAMENT_ROOT / source_rel).resolve()
    if not source_path.is_file():
        raise ExecutionLockError(f"locked source config not found: {source_path}")
    if derived_path.resolve() != (FILAMENT_ROOT / DERIVED_CONFIG_REL).resolve():
        raise ExecutionLockError(f"derived config path is not the locked path: {derived_path}")
    source_sha = sha256(source_path)
    if source_sha != LOCKED_BASE_CONFIG_SHA256:
        raise ExecutionLockError(
            f"locked source config SHA mismatch: expected={LOCKED_BASE_CONFIG_SHA256} actual={source_sha}"
        )
    if manifest.get("locked_base_config_sha256") != LOCKED_BASE_CONFIG_SHA256:
        raise ExecutionLockError("manifest locked_base_config_sha256 is incorrect")
    if manifest.get("source_config_sha256") != source_sha:
        raise ExecutionLockError("manifest source_config_sha256 does not match actual source config")

    prepare = _prepare_module()
    source = _load_object(source_path, "locked source config")
    derived = _load_object(derived_path, "derived config")
    for label, payload in (("source", source), ("derived", derived)):
        if payload.get("ionization", {}).get("I_cap") != EXPECTED_I_CAP:
            raise ExecutionLockError(f"{label} config ionization.I_cap is not the fixed {EXPECTED_I_CAP:g}")
    try:
        prepare._assert_fixed(source)
        prepare._assert_fixed(derived)
    except (AssertionError, KeyError, TypeError) as exc:
        raise ExecutionLockError(f"fixed C2 configuration assertion failed: {exc}") from exc
    differences = prepare.config_diff(source, derived)
    expected_diff = [{
        "path": "raman.operator_mode",
        "full_isaacs_eq27": prepare.SOURCE_MODE,
        "full_isaacs_eq27_complete": prepare.COMPLETE_MODE,
    }]
    if differences != expected_diff:
        raise ExecutionLockError(f"C2 flattened config diff is not the unique operator delta: {differences}")
    if manifest.get("strict_config_diff") != expected_diff:
        raise ExecutionLockError("manifest strict_config_diff does not match the fixed C2 delta")
    if manifest.get("operator_modes") != {"source": prepare.SOURCE_MODE, "candidate": prepare.COMPLETE_MODE}:
        raise ExecutionLockError("manifest operator_modes do not match the fixed C2 delta")
    if manifest.get("single_causal_variable") != "electronic Eq.27 operator form":
        raise ExecutionLockError("manifest single_causal_variable is incorrect")
    return source_path, differences


def _validate_fixed_run_contract(manifest: dict[str, Any]) -> None:
    expected_counts = {
        "jobs_authorized": 1,
        "jobs_submitted": 0,
        "full_jobs_authorized": 1,
        "full_propagation_jobs_authorized": 1,
        "full_production_jobs_submitted": 0,
        "scan_jobs_authorized": 0,
        "profiling_jobs_authorized": 0,
        "optimization_jobs_authorized": 0,
    }
    for key, expected in expected_counts.items():
        if manifest.get(key) != expected:
            raise ExecutionLockError(f"manifest {key} must equal {expected!r}")
    for key in ("parameter_scan_authorized", "profiling_authorized"):
        if manifest.get(key) is not False:
            raise ExecutionLockError(f"manifest {key} must be false for the fixed C2 job")
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise ExecutionLockError("manifest resources must be an object")
    for key, expected in {
        "partition": "gpu",
        "gpu_count": 1,
        "cpu_threads": 8,
        "requested_time": "15:00:00",
        "expected_gpu_model": EXPECTED_GPU_MODEL,
    }.items():
        if resources.get(key) != expected:
            raise ExecutionLockError(f"manifest resources.{key} must equal {expected!r}")
    shared = manifest.get("shared_resources")
    if not isinstance(shared, dict):
        raise ExecutionLockError("manifest shared_resources must be an object")
    for key, expected in {
        "gpu_count": 1,
        "cpu_threads": 8,
        "expected_gpu_model": EXPECTED_GPU_MODEL,
    }.items():
        if shared.get(key) != expected:
            raise ExecutionLockError(f"manifest shared_resources.{key} must equal {expected!r}")
    walltime = manifest.get("walltime_policy")
    if not isinstance(walltime, dict) or walltime.get("partition") != "gpu" or walltime.get("requested_time") != "15:00:00":
        raise ExecutionLockError("manifest walltime_policy does not match the fixed C2 job")


def _validate_fixed_pycap(manifest: dict[str, Any]) -> Path:
    comparison = manifest.get("comparison_inputs")
    if not isinstance(comparison, dict) or comparison.get("pycap_120fs") != PYCAP_REL:
        raise ExecutionLockError("manifest PyCAP path is not the fixed C2 input")
    if comparison.get("pycap_120fs_sha256") != PYCAP_SHA256:
        raise ExecutionLockError("manifest PyCAP SHA256 is not the fixed C2 input")
    path = (FILAMENT_ROOT / PYCAP_REL).resolve()
    if not path.is_file():
        raise ExecutionLockError(f"fixed PyCAP input does not exist: {path}")
    actual = sha256(path)
    if actual != PYCAP_SHA256:
        raise ExecutionLockError(f"fixed PyCAP SHA256 mismatch: expected={PYCAP_SHA256} actual={actual}")
    return path


def validate_manifest_lock(
    manifest_path: Path,
    lock_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_lock_sha256: str | None = None,
    expected_git_sha: str | None = None,
    require_clean: bool = True,
    require_committed_manifest: bool = False,
) -> dict[str, Any]:
    """Re-open and validate the complete manifest/lock binding.

    This is intentionally side-effect free and is used by the batch script as
    a second authorization boundary, so direct ``sbatch`` cannot replace the
    submit wrapper's caller-supplied environment with a different config or
    lock.  It does not create or consume a lock.
    """
    manifest_path = manifest_path.expanduser().resolve()
    lock_path = lock_path.expanduser().resolve()
    expected_manifest_path = (FILAMENT_ROOT / "results" / "isaacs_complete_eq27" / "submission_manifest.json").resolve()
    if manifest_path != expected_manifest_path:
        raise ExecutionLockError(f"manifest path is not fixed: {manifest_path}")
    if not manifest_path.is_file():
        raise ExecutionLockError(f"manifest not found: {manifest_path}")
    if require_committed_manifest:
        manifest_rel = _repo_relative(manifest_path)
        _git("ls-files", "--error-unmatch", "--", manifest_rel)
    actual_manifest_sha256 = sha256(manifest_path)
    if expected_manifest_sha256 and actual_manifest_sha256 != expected_manifest_sha256:
        raise ExecutionLockError(
            f"manifest SHA mismatch: expected={expected_manifest_sha256} actual={actual_manifest_sha256}"
        )
    manifest = _load_object(manifest_path, "C2 manifest")
    if manifest.get("schema") != "khz_filament.isaacs_complete_eq27.c2_submission_manifest.v1":
        raise ExecutionLockError("manifest schema is invalid")
    if manifest.get("campaign_id") != FIXED_CAMPAIGN_ID:
        raise ExecutionLockError("manifest campaign_id is not the fixed C2 campaign")
    if manifest.get("remote_campaign_root") != FIXED_REMOTE_CAMPAIGN_ROOT:
        raise ExecutionLockError("manifest remote_campaign_root is not the fixed C2 root")
    if manifest.get("status") != "prepared_not_submitted":
        raise ExecutionLockError("manifest status is not prepared_not_submitted")
    if manifest.get("expected_git_sha") is not None:
        raise ExecutionLockError("manifest expected_git_sha must be null")
    if manifest.get("execution_lock_required") is not True:
        raise ExecutionLockError("manifest execution_lock_required must be true")
    if manifest.get("expected_git_sha_resolution") != EXPECTED_RESOLUTION:
        raise ExecutionLockError("manifest expected_git_sha_resolution is invalid")
    head = _git("rev-parse", "HEAD")
    if expected_git_sha and head != expected_git_sha:
        raise ExecutionLockError(f"actual HEAD does not match expected Git SHA: {head}")
    if require_clean and _git("status", "--porcelain=v1"):
        raise ExecutionLockError("source worktree is not clean")
    c1_gate = _validate_c1_gate(manifest, head)
    config = _manifest_config(manifest, manifest_path)
    source, differences = _validate_fixed_config_contract(manifest, config)
    _validate_fixed_run_contract(manifest)
    pycap = _validate_fixed_pycap(manifest)
    declared_config_sha = manifest.get("derived_config_sha256")
    actual_config_sha = sha256(config)
    if declared_config_sha != actual_config_sha:
        raise ExecutionLockError("manifest derived_config_sha256 does not match actual config")

    if not lock_path.is_file():
        raise ExecutionLockError(f"execution lock not found: {lock_path}")
    actual_lock_sha256 = sha256(lock_path)
    if expected_lock_sha256 and actual_lock_sha256 != expected_lock_sha256:
        raise ExecutionLockError(
            f"execution lock SHA mismatch: expected={expected_lock_sha256} actual={actual_lock_sha256}"
        )
    lock = _load_object(lock_path, "execution lock")
    if lock.get("schema") != LOCK_SCHEMA:
        raise ExecutionLockError("execution lock schema is invalid")
    for key, expected in {
        "campaign_id": FIXED_CAMPAIGN_ID,
        "remote_campaign_root": FIXED_REMOTE_CAMPAIGN_ROOT,
        "status": "authorized_not_consumed",
        "expected_gpu_model": EXPECTED_GPU_MODEL,
        "manifest_path": _repo_relative(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "source_config_path": SOURCE_CONFIG_REL,
        "source_config_sha256": sha256(source),
        "config_path": DERIVED_CONFIG_REL,
        "config_sha256": actual_config_sha,
        "derived_config_path": DERIVED_CONFIG_REL,
        "derived_config_sha256": actual_config_sha,
        "strict_config_diff": differences,
        "pycap_path": PYCAP_REL,
        "pycap_sha256": PYCAP_SHA256,
        "operator_mode": "full_isaacs_eq27_complete",
        "use_raman_full_operator": True,
    }.items():
        if lock.get(key) != expected:
            raise ExecutionLockError(f"execution lock {key} does not match the fixed binding")
    lock_head = lock.get("expected_git_sha")
    if not isinstance(lock_head, str) or not lock_head.strip() or lock_head != head:
        raise ExecutionLockError("execution lock expected_git_sha does not match actual clean HEAD")
    if lock.get("c1_gate") != c1_gate:
        raise ExecutionLockError("execution lock c1_gate does not match fixed C1 evidence")
    resources = manifest.get("resources")
    if lock.get("resources") != resources:
        raise ExecutionLockError("execution lock resources do not match manifest resources")
    return {
        "manifest": manifest,
        "lock": lock,
        "manifest_path": manifest_path,
        "lock_path": lock_path,
        "manifest_sha256": actual_manifest_sha256,
        "lock_sha256": actual_lock_sha256,
        "head": head,
        "source_path": source,
        "config_path": config,
        "config_sha256": actual_config_sha,
        "pycap_path": pycap,
        "c1_gate": c1_gate,
    }


def create_lock(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Validate the committed manifest and write one external execution lock."""
    manifest_path = manifest_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    manifest_rel = _repo_relative(manifest_path)
    if not manifest_path.is_file():
        raise ExecutionLockError(f"manifest not found: {manifest_path}")
    if _git("status", "--porcelain=v1"):
        raise ExecutionLockError("worktree must be clean before creating execution lock")
    try:
        _git("ls-files", "--error-unmatch", "--", manifest_rel)
    except ExecutionLockError as exc:
        raise ExecutionLockError(f"manifest is not committed: {manifest_rel}") from exc

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionLockError(f"cannot parse manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ExecutionLockError("manifest top-level value is not an object")
    if manifest.get("schema") != "khz_filament.isaacs_complete_eq27.c2_submission_manifest.v1":
        raise ExecutionLockError("manifest schema is not the C2 submission manifest")
    if manifest.get("campaign_id") != FIXED_CAMPAIGN_ID:
        raise ExecutionLockError("manifest campaign_id is not the fixed C2 campaign")
    if manifest.get("remote_campaign_root") != FIXED_REMOTE_CAMPAIGN_ROOT:
        raise ExecutionLockError("manifest remote_campaign_root is not the fixed C2 root")
    if manifest.get("status") != "prepared_not_submitted":
        raise ExecutionLockError("manifest status is not prepared_not_submitted")
    if manifest.get("expected_git_sha") is not None:
        raise ExecutionLockError("manifest expected_git_sha must be null before lock creation")
    if manifest.get("execution_lock_required") is not True:
        raise ExecutionLockError("manifest execution_lock_required must be true")
    if manifest.get("expected_git_sha_resolution") != EXPECTED_RESOLUTION:
        raise ExecutionLockError("manifest expected_git_sha_resolution is incorrect")

    config = _manifest_config(manifest, manifest_path)
    source_config, differences = _validate_fixed_config_contract(manifest, config)
    _validate_fixed_run_contract(manifest)
    pycap_path = _validate_fixed_pycap(manifest)
    head = _git("rev-parse", "HEAD")
    c1_gate = _validate_c1_gate(manifest, head)
    manifest_hash = sha256(manifest_path)
    source_config_hash = sha256(source_config)
    config_hash = sha256(config)
    declared_config_hash = manifest.get("derived_config_sha256")
    if declared_config_hash != config_hash:
        raise ExecutionLockError(
            "manifest derived_config_sha256 does not match actual config: "
            f"declared={declared_config_hash} actual={config_hash}"
        )

    payload = {
        "schema": LOCK_SCHEMA,
        "campaign_id": FIXED_CAMPAIGN_ID,
        "remote_campaign_root": FIXED_REMOTE_CAMPAIGN_ROOT,
        "status": "authorized_not_consumed",
        "expected_git_sha": head,
        "parent_c1_commit": C1_COMMIT,
        "c1_gate": c1_gate,
        "manifest_path": manifest_rel,
        "manifest_sha256": manifest_hash,
        "source_config_path": SOURCE_CONFIG_REL,
        "source_config_sha256": source_config_hash,
        "config_path": manifest["derived_config"],
        "config_sha256": config_hash,
        "derived_config_path": manifest["derived_config"],
        "derived_config_sha256": config_hash,
        "strict_config_diff": differences,
        "expected_gpu_model": EXPECTED_GPU_MODEL,
        "pycap_path": PYCAP_REL,
        "pycap_sha256": sha256(pycap_path),
        "jobs_authorized": manifest["jobs_authorized"],
        "full_jobs_authorized": manifest["full_jobs_authorized"],
        "resources": manifest["resources"],
        "operator_mode": "full_isaacs_eq27_complete",
        "use_raman_full_operator": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        "--lock-path",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="external JSON lock path (default: under .git/codex-locks)",
    )
    args = parser.parse_args(argv)
    try:
        payload = create_lock(args.manifest, args.output)
    except ExecutionLockError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps({"status": payload["status"], "lock_path": str(args.output.resolve()), "expected_git_sha": payload["expected_git_sha"]}, indent=2))


if __name__ == "__main__":
    main()
