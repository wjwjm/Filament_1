"""Fail-closed metadata admission for the CPU-S4 preparation tools.

The default preflight validates source/manifests and input metadata without
reading the fourteen optical assets.  Full raw-byte verification is an
explicit allocation-stage operation.  Neither outcome unlocks real execution.
"""
from __future__ import annotations

import importlib
import sys
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Callable


SCHEMA = "filament.cpu_s4.admission.v1"
EXPECTED_INPUT_COUNT = 14
EXPECTED_INPUTS_RAW_SHA256 = "219e6450438ced67be772efb554f3d124f55cb5c382958894519d19feaf1587a"
Q3_HEAD = "66801ad435d6ec1967338cee4eb82af2a152389c"
SCIENCE_TREE = "821fec5bdacd832b0347e1f2e84285cb54172e57"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
S4_TOOL_PREFIX = "Filament_python/tools/cpu_o1/cpu_s4/"
# Historical contract retained for provenance. Current source identity is bound
# by the complete raw-byte adapter manifest below, not by these old profile bytes.
LEGACY_PROTECTED_OVERLAYS = {
    "__init__.py": "92025a6fdd71dabe0363bf25b3b284891477b0124a2a40982980285133e520d8",
    "profile.py": "5d9aff8dc0b4d7dabdc9b8e3b2062f29d2425b37e7cb4cf3cc18c6950a4d81d6",
    "test_cpu_s4_profile.py": "c8ce2f243849311e957adec19280ed02a877ebe87be2be7e5b3bf3dab38e6a53",
}
REQUIRED_S4_OVERLAY_FILES = frozenset({
    "__init__.py", "admission.py", "profile.py", "supervisor.py",
    "output_inventory.py", "memory_limits.py", "timing_instrumentation.py",
    "launcher.py", "runtime_gate.py", "storage_budget.py",
})


class AdmissionBlocked(RuntimeError):
    """A requested preflight could not establish every required identity."""


_GIT_CONTEXT_REDIRECT_ENV = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY", "GIT_OBJECT_DIRECTORY_RELATIVE",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM", "GIT_NAMESPACE",
    "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM",
})


def _clear_inherited_git_context_environment() -> list[str]:
    """Remove inherited Git settings that can redirect lookup or inject config.

    The launcher and its children share this environment.  Only variable names
    are reported; any values may contain local paths or credentials.
    """
    removed = []
    for name in tuple(os.environ):
        upper = name.upper()
        if (upper in _GIT_CONTEXT_REDIRECT_ENV
                or upper.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))):
            os.environ.pop(name, None)
            removed.append(name)
    return sorted(removed)


def _run_adapter_git(adapter_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(adapter_root), *args], cwd=adapter_root,
        env=os.environ.copy(), capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )


def _adapter_git_output(adapter_root: Path, *args: str) -> str:
    result = _run_adapter_git(adapter_root, *args)
    if result.returncode != 0:
        raise AdmissionBlocked(
            "adapter Git query failed: " + " ".join(args)
            + f" (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def _resolve_git_path(adapter_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = adapter_root / path
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise AdmissionBlocked(f"adapter Git path is unavailable: {value}") from exc


def _marker_exists(path: Path) -> bool:
    return (path.exists() or path.is_symlink()
            or getattr(path, "is_junction", lambda: False)())


def _validate_adapter_git_repository_context(
        adapter_root: str | os.PathLike[str],
        adapter_git_dir: str | os.PathLike[str], *,
        expected_head: str, expected_science_tree: str) -> dict[str, Any]:
    """Verify the .git pointer and frozen checkout identity before imports.

    Expected identities are parameters only on this private helper so tests can
    exercise mismatch branches against a real repository.  The public runtime
    gate always supplies the frozen Q3 constants.
    """
    removed_environment = _clear_inherited_git_context_environment()
    try:
        root = Path(adapter_root).resolve(strict=True)
        configured_raw = Path(adapter_git_dir)
    except (OSError, TypeError, ValueError) as exc:
        raise AdmissionBlocked("adapter Git context paths are invalid") from exc
    if not root.is_dir():
        raise AdmissionBlocked("adapter root is not a directory")
    if not configured_raw.is_absolute():
        raise AdmissionBlocked("adapter_git_dir must be an absolute configured path")
    if configured_raw.is_symlink() or getattr(configured_raw, "is_junction", lambda: False)():
        raise AdmissionBlocked("adapter_git_dir must not be a link")
    try:
        configured = configured_raw.resolve(strict=True)
    except OSError as exc:
        raise AdmissionBlocked("configured adapter Git database is missing") from exc
    if not configured.is_dir():
        raise AdmissionBlocked("configured adapter Git database is not a directory")
    try:
        configured.relative_to(root)
    except ValueError:
        pass
    else:
        raise AdmissionBlocked("adapter Git database must be outside adapter root")

    git_pointer = root / ".git"
    if (git_pointer.is_symlink()
            or getattr(git_pointer, "is_junction", lambda: False)()
            or not git_pointer.is_file()):
        raise AdmissionBlocked("adapter requires a regular .git pointer file")
    try:
        pointer_bytes = git_pointer.read_bytes()
        pointer_lines = pointer_bytes.decode("ascii", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AdmissionBlocked("adapter .git pointer is unreadable or malformed") from exc
    if len(pointer_lines) != 1 or not pointer_lines[0].startswith("gitdir: "):
        raise AdmissionBlocked("adapter .git pointer must contain one gitdir line")
    pointer_value = pointer_lines[0][len("gitdir: "):]
    if not pointer_value:
        raise AdmissionBlocked("adapter .git pointer has an empty gitdir")
    pointer_target = _resolve_git_path(root, pointer_value)
    if pointer_target != configured:
        raise AdmissionBlocked("adapter .git pointer differs from configured adapter_git_dir")

    actual_root = _resolve_git_path(
        root, _adapter_git_output(root, "rev-parse", "--show-toplevel"))
    actual_git_dir = _resolve_git_path(
        root, _adapter_git_output(root, "rev-parse", "--git-dir"))
    if actual_root != root:
        raise AdmissionBlocked("adapter Git top-level differs from configured adapter root")
    if actual_git_dir != configured:
        raise AdmissionBlocked("adapter Git database differs from configured adapter_git_dir")

    if _marker_exists(configured / "commondir"):
        raise AdmissionBlocked("adapter Git database must not share a common directory")
    if _marker_exists(configured / "shallow"):
        raise AdmissionBlocked("adapter Git database must contain complete history")
    if _marker_exists(configured / "objects" / "info" / "alternates"):
        raise AdmissionBlocked("adapter Git database must not use alternates")
    if any((configured / "objects" / "pack").glob("*.promisor")):
        raise AdmissionBlocked("adapter Git database must not use promisor objects")
    if (configured / "info" / "grafts").exists():
        raise AdmissionBlocked("adapter Git database must not use grafted history")
    if any((configured / "refs" / "replace").rglob("*")):
        raise AdmissionBlocked("adapter Git database must not replace frozen objects")
    try:
        config_bytes = (configured / "config").read_bytes()
    except OSError as exc:
        raise AdmissionBlocked("adapter Git database config is unreadable") from exc
    config_lower = config_bytes.lower()
    if (re.search(rb"(?im)^\s*partialclone\s*=", config_lower)
            or re.search(rb"(?im)^\s*promisor\s*=\s*true\s*(?:[#;].*)?$",
                         config_lower)):
        raise AdmissionBlocked("adapter Git database must not use partial/promisor objects")

    head = _adapter_git_output(root, "rev-parse", "--verify", "HEAD^{commit}")
    if head != expected_head:
        raise AdmissionBlocked("adapter Git HEAD differs from frozen Q3 identity")
    science_tree = _adapter_git_output(
        root, "rev-parse", "--verify",
        f"{expected_head}:Filament_python/KHz_filament")
    if science_tree != expected_science_tree:
        raise AdmissionBlocked("adapter Git science tree differs from frozen Q3 identity")

    unstaged = _run_adapter_git(root, "diff", "--no-ext-diff", "--quiet", "--")
    staged = _run_adapter_git(
        root, "diff", "--cached", "--no-ext-diff", "--quiet", "HEAD", "--")
    if unstaged.returncode not in (0, 1) or staged.returncode not in (0, 1):
        raise AdmissionBlocked("adapter tracked-source drift could not be checked")
    if unstaged.returncode == 1:
        raise AdmissionBlocked("adapter tracked files have unstaged changes")
    if staged.returncode == 1:
        raise AdmissionBlocked("adapter tracked files have staged changes")

    if str(root / "Filament_python") not in sys.path:
        sys.path.insert(0, str(root / "Filament_python"))
    try:
        from tools.hpc_ops import provenance_v2 as pv
        pointer_sha256 = pv.raw_sha256(pointer_bytes)
    except Exception as exc:
        raise AdmissionBlocked(
            f"authoritative raw-byte hash implementation is unavailable: {type(exc).__name__}: {exc}") from exc

    return {
        "schema": "filament.cpu_s4.adapter_git_context.v1",
        "status": "PASS",
        "adapter_root": str(root),
        "adapter_git_dir": str(configured),
        "git_pointer": {
            "path": ".git", "kind": "regular_file",
            "hash_scope": "raw_bytes",
            "raw_sha256": pointer_sha256,
        },
        "actual_git_top_level": str(actual_root),
        "actual_git_dir": str(actual_git_dir),
        "git_common_dir": str(configured),
        "head": head,
        "science_tree": science_tree,
        "tracked_unstaged_changes": 0,
        "tracked_staged_changes": 0,
        "history_mode": "complete_independent_no_alternates",
        "git_redirect_environment_cleared": removed_environment,
        "checks": {
            "regular_git_pointer_matches_config": "PASS",
            "adapter_is_git_top_level": "PASS",
            "external_independent_git_database": "PASS",
            "frozen_q3_head_and_science_tree": "PASS",
            "complete_history_without_shared_objects": "PASS",
            "tracked_worktree_and_index_clean": "PASS",
        },
    }


def validate_adapter_git_context(
        adapter_root: str | os.PathLike[str],
        adapter_git_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate the external Git database and the actual adapter source import.

    This is a runtime admission gate, not a source-copy fallback.  The S4
    overlay's ordinary .git pointer must already be covered by the raw manifest.
    """
    receipt = _validate_adapter_git_repository_context(
        adapter_root, adapter_git_dir, expected_head=Q3_HEAD,
        expected_science_tree=SCIENCE_TREE)
    root = Path(receipt["adapter_root"])
    adapter_python = root / "Filament_python"
    if not adapter_python.is_dir():
        raise AdmissionBlocked("adapter Filament_python import root is missing")
    adapter_python_text = str(adapter_python)
    sys.path[:] = [entry for entry in sys.path if entry != adapter_python_text]
    sys.path.insert(0, adapter_python_text)
    try:
        cpu_o1 = importlib.import_module("tools.cpu_o1.cpu_o1")
    except Exception as exc:
        raise AdmissionBlocked(
            f"adapter CPU-O1 module could not be imported: {type(exc).__name__}: {exc}") from exc
    module_path = Path(getattr(cpu_o1, "__file__", "")).resolve(strict=True)
    expected_module = (adapter_python / "tools/cpu_o1/cpu_o1.py").resolve(strict=True)
    if module_path != expected_module or not module_path.is_relative_to(root):
        raise AdmissionBlocked("imported tools.cpu_o1.cpu_o1 is outside adapter root")
    binding_function = getattr(cpu_o1, "_validate_p1r2_source_binding", None)
    if not callable(binding_function):
        raise AdmissionBlocked("adapter CPU-O1 source binding validator is unavailable")
    try:
        source_binding = binding_function()
    except Exception as exc:
        raise AdmissionBlocked(
            "adapter frozen P1-R2 source binding failed: "
            f"{type(exc).__name__}: {exc}") from exc
    anchor_commit = getattr(cpu_o1, "P1R2_SOURCE_ANCHOR_COMMIT", None)
    anchor_tree = getattr(cpu_o1, "P1R2_SOURCE_ANCHOR_TREE", None)
    if (not isinstance(source_binding, dict)
            or source_binding.get("anchor_commit") != anchor_commit
            or source_binding.get("anchor_tree") != anchor_tree
            or source_binding.get("working_tree_matches_anchor") is not True):
        raise AdmissionBlocked("adapter frozen P1-R2 source binding receipt is invalid")
    receipt["imported_cpu_o1"] = {
        "module": "tools.cpu_o1.cpu_o1",
        "path": str(module_path),
        "status": "PASS",
    }
    receipt["p1r2_source_binding"] = source_binding
    receipt["checks"]["adapter_module_import_and_frozen_binding"] = "PASS"
    return receipt


def _strict_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False, indent=2) + "\n").encode("utf-8")


def write_json_new(path: str | os.PathLike[str], value: dict[str, Any]) -> None:
    """Write a strict JSON receipt once; existing evidence is never replaced."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(_strict_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _input_relative_path(raw_path: str) -> PurePosixPath:
    """Return a canonical path below the frozen ``input_optical`` directory."""
    if not isinstance(raw_path, str) or "\\" in raw_path:
        raise AdmissionBlocked("expected input path must be a POSIX path")
    parts = PurePosixPath(raw_path).parts
    try:
        index = parts.index("input_optical")
    except ValueError as exc:
        raise AdmissionBlocked(
            f"expected input path is outside input_optical: {raw_path}") from exc
    tail = parts[index + 1:]
    if not tail or any(part in {"", ".", ".."} for part in tail):
        raise AdmissionBlocked(f"unsafe expected input path: {raw_path}")
    return PurePosixPath(*tail)


def _parse_expected_inputs(payload: bytes, *, expected_sha256: str) -> dict[str, Any]:
    """Parse the sealed Q1 artifact; the digest argument is private for fixtures."""
    if str(Path(__file__).resolve().parents[3]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.hpc_ops import provenance_v2 as pv
    actual_sha256 = pv.raw_sha256(payload)
    if actual_sha256 != expected_sha256:
        raise AdmissionBlocked("EXPECTED_INPUTS artifact SHA256 is not the reviewed identity")
    try:
        expected = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionBlocked("EXPECTED_INPUTS is not valid UTF-8 JSON") from exc
    if (not isinstance(expected, dict)
            or set(expected) != {"hash_scope", "source_provenance_sha256",
                                 "source_head", "records"}
            or expected.get("hash_scope") != "raw_bytes"):
        raise AdmissionBlocked("EXPECTED_INPUTS must use the frozen raw_bytes schema")
    source_sha = expected.get("source_provenance_sha256")
    source_head = expected.get("source_head")
    records = expected.get("records")
    if not isinstance(source_sha, str) or not _SHA256.fullmatch(source_sha):
        raise AdmissionBlocked("EXPECTED_INPUTS historical source hash is invalid")
    if not isinstance(source_head, str) or not source_head:
        raise AdmissionBlocked("EXPECTED_INPUTS historical source head is invalid")
    if not isinstance(records, list) or len(records) != EXPECTED_INPUT_COUNT:
        raise AdmissionBlocked(
            f"EXPECTED_INPUTS must contain exactly {EXPECTED_INPUT_COUNT} objects")

    by_path: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if (not isinstance(record, dict)
                or set(record) != {"path", "classification", "hash_scope", "raw_sha256"}
                or record.get("classification") != "external"
                or record.get("hash_scope") != "raw_bytes"):
            raise AdmissionBlocked(f"EXPECTED_INPUTS record {index} has an invalid shape")
        digest = record.get("raw_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise AdmissionBlocked(f"EXPECTED_INPUTS record {index} has an invalid SHA256")
        relative = _input_relative_path(record.get("path", "")).as_posix()
        if relative in by_path:
            raise AdmissionBlocked(f"duplicate EXPECTED_INPUTS path: {relative}")
        by_path[relative] = {**record, "relative_path": relative}
    if len(by_path) != EXPECTED_INPUT_COUNT:
        raise AdmissionBlocked("EXPECTED_INPUTS path count is not fourteen")
    return {
        "artifact_sha256": actual_sha256,
        "origin": {
            "source_provenance_sha256": source_sha,
            "source_head": source_head,
            "scope": "HISTORICAL_Q1_ORIGIN_ONLY",
        },
        "records": by_path,
    }


def _manifest_input_records(manifest: dict[str, Any], input_root: Path, pv) -> dict[str, dict]:
    """Index strict provenance-v2 external records by frozen relative path."""
    root = input_root.resolve(strict=True)
    found: dict[str, dict] = {}
    for record in pv.iter_records(manifest, require_hash_scope=True):
        if not isinstance(record, dict) or record.get("classification") != "external":
            continue
        if record.get("hash_scope") != "raw_bytes":
            raise AdmissionBlocked("external provenance record is not raw_bytes")
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            raise AdmissionBlocked("external provenance path is invalid")
        candidate = Path(raw_path)
        try:
            relative = candidate.resolve(strict=False).relative_to(root)
            key = PurePosixPath(*relative.parts).as_posix()
        except (OSError, ValueError):
            key = _input_relative_path(raw_path).as_posix()
        if key in found:
            raise AdmissionBlocked(f"duplicate external provenance path: {key}")
        found[key] = record
    return found


def _verify_input_records(expected_by_path: dict[str, dict],
                          manifest_by_path: dict[str, dict], input_root: Path, *,
                          verify_raw_hashes: bool,
                          raw_sha256_file: Callable[[Path], str]) -> list[dict[str, Any]]:
    """Match the exact fourteen-record set and optionally read each file's bytes."""
    if set(expected_by_path) != set(manifest_by_path):
        missing = sorted(set(expected_by_path) - set(manifest_by_path))
        extra = sorted(set(manifest_by_path) - set(expected_by_path))
        raise AdmissionBlocked(f"fourteen-input path set differs; missing={missing}; extra={extra}")
    root = input_root.resolve(strict=True)
    verified: list[dict[str, Any]] = []
    for relative, expected in sorted(expected_by_path.items()):
        manifest_record = manifest_by_path[relative]
        expected_hash = expected["raw_sha256"]
        if (manifest_record.get("hash_scope") != "raw_bytes"
                or manifest_record.get("raw_sha256") != expected_hash):
            raise AdmissionBlocked(f"Q3 manifest identity differs for input: {relative}")
        candidate = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise AdmissionBlocked(f"resolved input escapes input-root: {relative}") from exc
        if candidate.is_symlink() or not candidate.is_file():
            raise AdmissionBlocked(f"input is not a regular non-symlink file: {relative}")
        actual_hash = raw_sha256_file(candidate) if verify_raw_hashes else None
        passed = actual_hash == expected_hash if verify_raw_hashes else None
        verified.append({
            "path": str(candidate),
            "relative_path": relative,
            "size_bytes": candidate.stat().st_size,
            "hash_scope": "raw_bytes",
            "expected_raw_sha256": expected_hash,
            "actual_raw_sha256": actual_hash,
            "metadata_status": "PASS",
            "identity_status": ("PASS" if passed else "FAIL_HASH_MISMATCH")
            if verify_raw_hashes else "DEFERRED_TO_EXPLICIT_ALLOCATION_VALIDATION",
            "shape_dtype": "DEFERRED_TO_FROZEN_SINGLE_MODE_CHILD",
        })
        if verify_raw_hashes and not passed:
            raise AdmissionBlocked(f"raw SHA256 mismatch for input: {relative}")
    return verified


def _validate_provenance_metadata(root: Path, manifest_path: Path, pv, *,
                                  require_clean: bool) -> tuple[dict, list[dict]]:
    """Validate provenance-v2 metadata while optionally deferring raw externals."""
    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissionBlocked(f"cannot read provenance manifest: {manifest_path}") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema") != pv.SCHEMA
            or manifest.get("version") != pv.VERSION
            or set(manifest) != pv.STRICT_TOP_LEVEL_KEYS):
        raise AdmissionBlocked("provenance-v2 manifest metadata is invalid")
    if manifest.get("repository_path") != str(root):
        raise AdmissionBlocked("provenance manifest is bound to a different root")
    try:
        repository = pv._safe_repository_identity(
            pv._git(root, "remote", "get-url", "origin", check=False) or root.name)
        branch = pv._git(root, "symbolic-ref", "--short", "-q", "HEAD", check=False)
        head = pv._git(root, "rev-parse", "HEAD")
        if manifest.get("repository") != repository:
            raise AdmissionBlocked("provenance repository identity differs from root")
        if manifest.get("branch") != branch:
            raise AdmissionBlocked("provenance branch differs from root")
        if manifest.get("head") != head:
            raise AdmissionBlocked("provenance HEAD differs from root")
        if require_clean:
            pv._ensure_clean(root)
        records = pv.iter_records(manifest, require_hash_scope=True)
        external = []
        for record in records:
            if not isinstance(record, dict):
                raise AdmissionBlocked("provenance record must be an object")
            classification = record.get("classification")
            if classification == "tracked_text":
                pv.validate_record(root, record, require_hash_scope=True)
            elif classification == "external":
                if (set(record) != {"path", "classification", "hash_scope", "raw_sha256"}
                        or record.get("hash_scope") != "raw_bytes"
                        or not isinstance(record.get("raw_sha256"), str)
                        or not _SHA256.fullmatch(record["raw_sha256"])):
                    raise AdmissionBlocked("external provenance metadata is invalid")
                path = Path(record["path"])
                if not path.is_absolute() or path.is_symlink() or not path.is_file():
                    raise AdmissionBlocked(f"external input metadata is unavailable: {path}")
                external.append(record)
            else:
                raise AdmissionBlocked("provenance record classification is invalid")
    except AdmissionBlocked:
        raise
    except Exception as exc:
        raise AdmissionBlocked(f"provenance-v2 validation failed: {type(exc).__name__}: {exc}") from exc
    return manifest, external


def _validate_adapter_raw_manifest(adapter_root: Path, manifest_path: Path, pv) -> list[dict]:
    """Validate a copied adapter snapshot from raw-byte records without Git claims."""
    root = adapter_root.resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.resolve(strict=True).read_text(encoding="utf-8"))
        records = pv.iter_records(manifest, require_hash_scope=True)
    except Exception as exc:
        raise AdmissionBlocked(
            f"cannot read provenance-v2 adapter raw manifest: {type(exc).__name__}: {exc}") from exc
    if not records:
        raise AdmissionBlocked("adapter raw manifest contains no records")
    seen: set[str] = set()
    validated = []
    for index, record in enumerate(records):
        if (not isinstance(record, dict)
                or record.get("classification") != "external"
                or record.get("hash_scope") != "raw_bytes"):
            raise AdmissionBlocked(f"adapter record {index} must be external raw_bytes")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise AdmissionBlocked(f"adapter record {index} path must be absolute")
        source = Path(raw_path)
        if source.is_symlink():
            raise AdmissionBlocked(f"adapter file must not be a symlink: {source}")
        path = source.resolve(strict=True)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise AdmissionBlocked(f"adapter raw record escapes adapter-root: {source}") from exc
        if relative in seen:
            raise AdmissionBlocked(f"duplicate adapter raw record: {relative}")
        seen.add(relative)
        try:
            pv.validate_record(root, record, require_hash_scope=True)
        except Exception as exc:
            raise AdmissionBlocked(
                f"adapter raw-byte identity failed for {relative}: {type(exc).__name__}: {exc}") from exc
        validated.append({"path": relative, "raw_sha256": record["raw_sha256"],
                          "hash_scope": "raw_bytes", "status": "PASS"})
    actual_paths = set()
    for candidate in root.rglob('*'):
        if candidate.is_symlink() or candidate.is_junction():
            raise AdmissionBlocked('snapshot contains a link: '+str(candidate))
        if candidate.is_file() and candidate.resolve()!=manifest_path.resolve():
            actual_paths.add(candidate.relative_to(root).as_posix())
    if actual_paths != seen:
        raise AdmissionBlocked('snapshot file inventory differs from adapter manifest')
    return validated


def _bind_s4_overlay_inventory(adapter_root: Path, adapter_records: list[dict], pv) -> dict[str, Any]:
    """Bind every current S4 overlay file through the reviewed raw-byte manifest.

    This replaces the stale three-file profile allowlist while preserving that
    allowlist as explicit historical metadata. The adapter manifest itself is
    separately SHA-bound by the run authorization.
    """
    root = adapter_root.resolve(strict=True)
    by_path = {row["path"]: row for row in adapter_records}
    overlay_records = sorted(
        (row for row in adapter_records if row["path"].startswith(S4_TOOL_PREFIX)),
        key=lambda row: row["path"])
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / S4_TOOL_PREFIX).rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    listed = {row["path"] for row in overlay_records}
    if actual != listed:
        raise AdmissionBlocked(
            "complete S4 overlay inventory differs from raw-byte manifest")
    required = {S4_TOOL_PREFIX + name for name in REQUIRED_S4_OVERLAY_FILES}
    if not required.issubset(listed):
        raise AdmissionBlocked(
            "adapter snapshot omits required S4 runtime files: "
            + repr(sorted(required - listed)))
    normalized = [{"path": row["path"], "hash_scope": "raw_bytes",
                   "raw_sha256": row["raw_sha256"]}
                  for row in overlay_records]
    fingerprint = pv.raw_sha256(json.dumps(normalized, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    legacy = {}
    for name, old_digest in LEGACY_PROTECTED_OVERLAYS.items():
        current = by_path.get(S4_TOOL_PREFIX + name)
        legacy[name] = {
            "historical_raw_sha256": old_digest,
            "current_raw_sha256": current["raw_sha256"] if current else None,
            "status": ("HISTORICAL_IDENTITY_MATCH" if current
                       and current["raw_sha256"] == old_digest
                       else "SUPERSEDED_BY_COMPLETE_OVERLAY_BINDING"),
        }
    return {"schema": "filament.cpu_s4.overlay_inventory.v2",
        "hash_scope": "raw_bytes", "status": "PASS",
        "record_count": len(normalized), "records": normalized,
        "inventory_sha256": fingerprint,
        "legacy_overlay_contract": legacy}


def validate_preflight(*, bench_root: str | os.PathLike[str],
                       provenance_path: str | os.PathLike[str],
                       adapter_root: str | os.PathLike[str],
                       adapter_provenance_path: str | os.PathLike[str],
                       input_root: str | os.PathLike[str],
                       expected_inputs_path: str | os.PathLike[str],
                       verify_input_hashes: bool = False) -> dict[str, Any]:
    """Check Q3/source/adapter metadata and the exact 14-input set.

    ``verify_input_hashes`` is an explicit allocation-stage bulk-read option.
    The normal path reads no optical asset contents and never unlocks execution.
    """
    if str(Path(__file__).resolve().parents[3]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.hpc_ops import provenance_v2 as pv

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "identity_scope": "METADATA_ONLY",
        "full_input_identity_verified": False,
        "real_execution": "DISABLED_PENDING_SEPARATE_UNLOCK_REVIEW",
        "bench_root": str(Path(bench_root).resolve(strict=False)),
        "adapter_root": str(Path(adapter_root).resolve(strict=False)),
        "input_root": str(Path(input_root).resolve(strict=False)),
        "checks": [],
        "inputs": [],
        "input_count_expected": EXPECTED_INPUT_COUNT,
        "input_count_verified": 0,
        "hpc_submission": "NOT_SUBMITTED",
    }
    try:
        bench = Path(bench_root).resolve(strict=True)
        adapter = Path(adapter_root).resolve(strict=True)
        inputs = Path(input_root).resolve(strict=True)
        if bench == adapter:
            raise AdmissionBlocked("benchmark source and adapter snapshot must be separate roots")
        if not bench.is_dir() or not adapter.is_dir() or not inputs.is_dir():
            raise AdmissionBlocked("benchmark, adapter, and input roots must be directories")

        expected_path = Path(expected_inputs_path).resolve(strict=True)
        if expected_path.stat().st_size > 256 * 1024:
            raise AdmissionBlocked("EXPECTED_INPUTS artifact exceeds the sealed metadata size limit")
        expected_payload = expected_path.read_bytes()
        expected = _parse_expected_inputs(
            expected_payload, expected_sha256=EXPECTED_INPUTS_RAW_SHA256)

        source_manifest_path = Path(provenance_path)
        source_manifest, external = _validate_provenance_metadata(
            bench, source_manifest_path, pv, require_clean=True)
        current_head = pv._git(bench, "rev-parse", "HEAD")
        science_tree = pv._git(bench, "rev-parse", f"{Q3_HEAD}:Filament_python/KHz_filament")
        if current_head != Q3_HEAD or source_manifest.get("head") != Q3_HEAD:
            raise AdmissionBlocked("benchmark source HEAD differs from the frozen Q3 identity")
        if science_tree != SCIENCE_TREE:
            raise AdmissionBlocked("benchmark science tree differs from the frozen Q3 identity")

        adapter_records = _validate_adapter_raw_manifest(
            adapter, Path(adapter_provenance_path), pv)
        snapshot_paths = {r['path'] for r in adapter_records}
        tool_prefix = S4_TOOL_PREFIX
        overlay_binding = _bind_s4_overlay_inventory(adapter, adapter_records, pv)
        required = {tool_prefix+n for n in REQUIRED_S4_OVERLAY_FILES}
        tracked_records = [r for r in pv.iter_records(source_manifest, require_hash_scope=True)
                           if r.get('classification') == 'tracked_text']
        required.update(r['path'] for r in tracked_records)
        if not required.issubset(snapshot_paths):
            raise AdmissionBlocked('adapter snapshot omits frozen source or S4 entry records')
        for record in tracked_records:
            if pv.canonical_lf_sha256((adapter/record['path']).read_bytes()) != record['canonical_lf_sha256']:
                raise AdmissionBlocked('import snapshot differs from clean benchmark: '+record['path'])
        manifest_inputs = _manifest_input_records(source_manifest, inputs, pv)
        if len(manifest_inputs) != EXPECTED_INPUT_COUNT:
            raise AdmissionBlocked(
                f"Q3 provenance has {len(manifest_inputs)} external inputs; expected 14")
        verified = _verify_input_records(
            expected["records"], manifest_inputs, inputs,
            verify_raw_hashes=verify_input_hashes, raw_sha256_file=pv.raw_sha256_file)
        receipt.update({
            "status": "FULL_INPUT_IDENTITY_PASS" if verify_input_hashes
            else "METADATA_READY_FULL_HASH_DEFERRED",
            "identity_scope": "FULL_RAW_BYTES" if verify_input_hashes else "METADATA_ONLY",
            "full_input_identity_verified": bool(verify_input_hashes),
            "source_provenance_sha256": pv.raw_sha256_file(Path(provenance_path)),
            "source_head": current_head,
            "science_tree": science_tree,
            "adapter_provenance_sha256": pv.raw_sha256_file(Path(adapter_provenance_path)),
            "s4_overlay_inventory": overlay_binding,
            "expected_inputs_artifact_sha256": expected["artifact_sha256"],
            "expected_inputs_origin": expected["origin"],
            "inputs": verified,
            "input_count_verified": len(verified) if verify_input_hashes else 0,
            "checks": [
                {"name": "expected_inputs_artifact", "status": "PASS",
                 "sha256": expected["artifact_sha256"], "historical_origin_preserved": True},
                {"name": "q3_benchmark_identity", "status": "PASS",
                 "head": current_head, "science_tree": science_tree, "clean": True},
                {"name": "adapter_raw_byte_manifest", "status": "PASS",
                 "record_count": len(adapter_records), "clean_required": False},
                {"name": "fourteen_input_metadata", "status": "PASS",
                 "count": len(verified), "raw_hashes_verified": bool(verify_input_hashes)},
                {"name": "real_execution_unlock", "status": "BLOCKED",
                 "reason": "separate manual unlock review required"},
            ],
        })
    except Exception as exc:
        receipt["status"] = "BLOCKED_INPUT_SOURCE_PREFLIGHT"
        receipt["blocker"] = {"type": type(exc).__name__, "message": str(exc)}
        if "verified" in locals():
            receipt["inputs"] = verified
            receipt["input_count_verified"] = sum(
                row.get("identity_status") == "PASS"
                for row in verified)
    return receipt


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--adapter-provenance", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-inputs", type=Path, required=True)
    parser.add_argument("--verify-all-input-hashes", action="store_true",
                        help="explicit allocation-stage pass that reads all 14 raw assets")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    receipt = validate_preflight(
        bench_root=args.bench_root,
        provenance_path=args.provenance,
        adapter_root=args.adapter_root,
        adapter_provenance_path=args.adapter_provenance,
        input_root=args.input_root,
        expected_inputs_path=args.expected_inputs,
        verify_input_hashes=args.verify_all_input_hashes,
    )
    if args.receipt:
        write_json_new(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if receipt["status"] in {
        "METADATA_READY_FULL_HASH_DEFERRED", "FULL_INPUT_IDENTITY_PASS"} else 2


if __name__ == "__main__":  # pragma: no cover - exercised through supervisor CLI
    raise SystemExit(main())

