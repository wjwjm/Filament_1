"""Portable source/identity bridge for the frozen G2 helper.

The copied G2 adapter retains its AST, code-object and method-binding checks.
Only report-local locator functions are bridged here so a clean checkout does
not depend on a historical reports tree.  Tracked files are resolved from the
current repository and its actual ``HEAD:path`` blob; uncommitted support
files use the exact raw-byte records in ``CHECKPOINT_DEPENDENCIES.json`` until
the parent commit adds their Git identities.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import CodeType, FunctionType, ModuleType
from typing import Any, Mapping

from tools.hpc_ops import provenance_v2 as pv


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def support_root() -> Path:
    return Path(__file__).resolve().parent


def g2_adapter_path() -> Path:
    return support_root() / "g2_adapter.py"


def ensure_support_imports() -> Path:
    root = support_root()
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root


def support_module(name: str):
    """Bind each imported support file to its actual committed source path."""
    ensure_support_imports()
    expected = (support_root() / (name + ".py")).resolve(strict=True)
    if expected.parent != support_root():
        raise RuntimeError("CHECKPOINT_SUPPORT_MODULE_NAME")
    tracked = _frozen_tracked_record_impl(_relative(expected))
    if tracked is None:
        raise RuntimeError("CHECKPOINT_SUPPORT_REQUIRES_COMMITTED_CODE:" + name)
    module = importlib.import_module(name)
    if Path(str(getattr(module, "__file__", ""))).resolve() != expected:
        raise RuntimeError("CHECKPOINT_SUPPORT_MODULE_MISMATCH:" + name)
    return module


def _dependency_manifest() -> dict[str, Any]:
    path = support_root() / "CHECKPOINT_DEPENDENCIES.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("CHECKPOINT_DEPENDENCIES_UNAVAILABLE") from exc
    if not isinstance(value, dict) or value.get("hash_scope") != "raw_bytes":
        raise RuntimeError("CHECKPOINT_DEPENDENCIES_SCOPE")
    return value


def _records() -> list[dict[str, Any]]:
    records = _dependency_manifest().get("records")
    if not isinstance(records, list):
        raise RuntimeError("CHECKPOINT_DEPENDENCIES_RECORDS")
    return [row for row in records if isinstance(row, dict)]


def _relative(path: Path) -> str:
    return path.resolve(strict=True).relative_to(repository_root()).as_posix()


def _record_for_relative(relative: str) -> dict[str, Any] | None:
    for row in _records():
        if str(row.get("new_relative", "")).replace("\\", "/") == relative:
            return row
    return None


def _git_output(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", "-C", str(repository_root()), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.returncode, result.stdout.strip()


def _frozen_tracked_record_impl(relative: str) -> dict[str, Any] | None:
    """Return a real current-HEAD binding, or ``None`` for untracked files."""
    relative = str(relative).replace("\\", "/")
    listed, listed_value = _git_output("ls-files", "--error-unmatch", "--", relative)
    if listed != 0 or listed_value != relative:
        return None
    oid_status, oid = _git_output("rev-parse", "--verify", f"HEAD:{relative}")
    if oid_status != 0 or len(oid) != 40 or any(c not in "0123456789abcdef" for c in oid.lower()):
        raise RuntimeError("G2_FORMAL_SOURCE_GIT_BINDING_FAILED")
    type_status, object_type = _git_output("cat-file", "-t", oid)
    if type_status != 0 or object_type != "blob":
        raise RuntimeError("G2_FORMAL_SOURCE_GIT_BINDING_FAILED")
    path = repository_root() / Path(*relative.split("/"))
    raw = path.read_bytes()
    canonical = pv.canonical_lf_sha256(raw)
    blob = pv._read_git_blob(repository_root(), oid)
    if pv.canonical_lf_bytes(blob) != pv.canonical_lf_bytes(raw):
        raise RuntimeError("G2_FORMAL_SOURCE_CANONICAL_BLOB_MISMATCH")
    return {
        "path": relative,
        "classification": "tracked_text",
        "hash_scope": pv.TRACKED_HASH_SCOPE,
        "git_blob_oid": oid,
        "canonical_lf_sha256": canonical,
    }


def _external_frozen_source_binding_impl(path: Path, raw: bytes) -> dict[str, Any]:
    actual = Path(path).resolve(strict=True)
    relative = _relative(actual)
    record = _record_for_relative(relative)
    if record is None:
        raise RuntimeError(f"G2_EXTERNAL_FROZEN_SOURCE_PATH_MISMATCH:{actual.name}")
    expected = str(record.get("legacy_raw_sha256", record.get("raw_sha256", ""))).lower()
    digest = pv.raw_sha256(raw)
    if digest != expected or len(raw) != int(record.get("size_bytes", -1)):
        raise RuntimeError(f"G2_EXTERNAL_FROZEN_SOURCE_HASH_MISMATCH:{actual.name}")
    canonical = pv.canonical_lf_sha256(raw)
    expected_canonical = str(record.get("canonical_lf_sha256", "")).lower()
    if expected_canonical and canonical != expected_canonical:
        raise RuntimeError(f"G2_EXTERNAL_FROZEN_SOURCE_CANONICAL_MISMATCH:{actual.name}")
    return {
        "relative_path": relative,
        "filename": str(actual),
        "classification": "external",
        "hash_scope": "raw_bytes",
        "git_blob_oid": None,
        "external_frozen_source": True,
        "external_binding_report": str(support_root() / "CHECKPOINT_DEPENDENCIES.json"),
        "expected_raw_sha256": expected,
        "source_raw_sha256": digest,
        "source_canonical_lf_sha256": canonical,
        "source_size_bytes": len(raw),
        "external_source_path_match": "exact_repo_relative",
        "legacy_relative_path": str(record.get("original_relative", "")),
    }


def _role_audit_source_contract_impl() -> dict[str, Any]:
    """Verify RoleAudit path, raw bytes, AST and nested code identity."""
    ensure_support_imports()
    record = _record_for_relative(
        "Filament_python/tools/cpu_s4_checkpoint_support/g1_segment.py"
    )
    if record is None:
        raise RuntimeError("G2_TIMING_HELPER_BINDING")
    expected_path = repository_root() / Path(
        *str(record["new_relative"]).split("/")
    )
    expected_path = expected_path.resolve(strict=True)
    tracked_binding = _frozen_tracked_record_impl(str(record["new_relative"]))
    expected_sha = str(record["legacy_raw_sha256"]).lower()
    expected_size = int(record["size_bytes"])
    try:
        module = importlib.import_module("g1_segment")
        role_audit = getattr(module, "RoleAudit", None)
        init = getattr(role_audit, "__dict__", {}).get("__init__")
        module_path = Path(str(getattr(module, "__file__"))).resolve(strict=True)
        init_code = getattr(init, "__code__", None)
        source_path = Path(str(getattr(init_code, "co_filename"))).resolve(strict=True)
        module_globals = getattr(module, "__dict__", None)
        init_globals = getattr(init, "__globals__", None)
        if (
            not isinstance(role_audit, type)
            or not callable(init)
            or not isinstance(init_code, CodeType)
            or not isinstance(module_globals, dict)
            or init_globals is not module_globals
            or module_path != expected_path
            or source_path != expected_path
        ):
            raise RuntimeError("G2_TIMING_HELPER_BINDING")
        raw = expected_path.read_bytes()
        if len(raw) != expected_size or pv.raw_sha256(raw).lower() != expected_sha:
            raise RuntimeError("G2_TIMING_HELPER_BINDING")
        canonical = pv.canonical_lf_bytes(raw)
        adapter = sys.modules.get("g2_adapter")
        find_frozen_code = getattr(adapter, "_find_frozen_code", None)
        code_structure = getattr(adapter, "_code_structure_sha256", None)
        if not callable(find_frozen_code) or not callable(code_structure):
            raise RuntimeError("G2_TIMING_HELPER_BINDING")
        frozen_init = find_frozen_code(source_path, canonical, init_code)
        if code_structure(init_code) != code_structure(frozen_init):
            raise RuntimeError("G2_TIMING_HELPER_BINDING")
        target_qualname = f"{role_audit.__qualname__}.__init__.<locals>.closed"
        nested: list[CodeType] = []

        def visit(code: CodeType) -> None:
            for value in code.co_consts:
                if isinstance(value, CodeType):
                    if str(getattr(value, "co_qualname", "")) == target_qualname:
                        nested.append(value)
                    visit(value)

        visit(init_code)
        if len(nested) != 1:
            raise RuntimeError("G2_TIMING_HELPER_BINDING")
        closed_code = nested[0]
        frozen_closed = find_frozen_code(source_path, canonical, closed_code)
        if code_structure(closed_code) != code_structure(frozen_closed):
            raise RuntimeError("G2_TIMING_HELPER_BINDING")
        return {
            "type": role_audit,
            "module": module,
            "globals": module_globals,
            "init_code": init_code,
            "closed_code": frozen_closed,
            "source_path": source_path,
            "source_raw_sha256": expected_sha,
            "source_size_bytes": expected_size,
            "receipt": {
                "module": str(getattr(module, "__name__", "")),
                "class": f"{role_audit.__module__}.{role_audit.__qualname__}",
                "source_path": str(source_path),
                "source_raw_sha256": expected_sha,
                "source_canonical_lf_sha256": pv.canonical_lf_sha256(raw),
                "hash_scope": (
                    pv.TRACKED_HASH_SCOPE
                    if tracked_binding is not None
                    else pv.EXTERNAL_HASH_SCOPE
                ),
                "git_blob_oid": (
                    None
                    if tracked_binding is None
                    else str(tracked_binding["git_blob_oid"])
                ),
                "source_size_bytes": expected_size,
                "source_record_path": str(record.get("original_relative", "")),
                "path_translation": {
                    "record_raw": str(record.get("original_relative", "")),
                    "record_portable": str(record.get("original_relative", "")),
                    "expected_runtime_path": str(expected_path),
                    "matched_complete_suffix": "Filament_python/tools/cpu_s4_checkpoint_support/g1_segment.py",
                },
                "init_qualname": str(getattr(init_code, "co_qualname", "")),
                "init_code_structure_sha256": code_structure(init_code),
                "closed_qualname": str(getattr(closed_code, "co_qualname", "")),
                "closed_code_structure_sha256": code_structure(closed_code),
                "input_binding": str(support_root() / "CHECKPOINT_DEPENDENCIES.json"),
            },
        }
    except RuntimeError:
        raise
    except BaseException as exc:
        raise RuntimeError("G2_TIMING_HELPER_BINDING") from exc


def _bridged_frozen_tracked_record(relative: str):
    return _portable_binding._frozen_tracked_record_impl(relative)


def _bridged_external_frozen_source_binding(path: Path, raw: bytes):
    return _portable_binding._external_frozen_source_binding_impl(path, raw)


def _bridged_role_audit_source_contract():
    return _portable_binding._role_audit_source_contract_impl()


def _clone_function(function: FunctionType, globals_dict: dict[str, Any]) -> FunctionType:
    clone = FunctionType(
        function.__code__, globals_dict, function.__name__, function.__defaults__, function.__closure__
    )
    clone.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
    clone.__annotations__ = getattr(function, "__annotations__", {})
    clone.__module__ = getattr(function, "__module__", None)
    clone.__qualname__ = getattr(function, "__qualname__", function.__name__)
    clone.__doc__ = getattr(function, "__doc__", None)
    return clone


def load_g2_adapter() -> ModuleType:
    """Load the copied adapter with only its report locators replaced."""
    ensure_support_imports()
    expected = g2_adapter_path().resolve(strict=True)
    if _frozen_tracked_record_impl(_relative(expected)) is None:
        raise RuntimeError("CHECKPOINT_ADAPTER_REQUIRES_COMMITTED_CODE")
    previous = sys.modules.get("g2_adapter")
    if previous is not None:
        previous_path = getattr(previous, "__file__", None)
        if previous_path is None or Path(str(previous_path)).resolve() != expected:
            raise RuntimeError("SMALLGRID_FROZEN_ADAPTER_ACTUAL_MODULE_MISMATCH")
        if getattr(previous, "_portable_checkpoint_binding", False):
            return previous
    spec = importlib.util.spec_from_file_location("_cpu_s4_checkpoint_g2_raw", expected)
    if spec is None or spec.loader is None:
        raise RuntimeError("G2_ADAPTER_SPEC_UNAVAILABLE")
    raw_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw_module)
    for name in ("g1_common", "g2_boundary", "timing_adapter"):
        support_module(name)
    module = ModuleType("g2_adapter")
    module.__file__ = str(expected)
    module.__package__ = ""
    module.__dict__.update(vars(raw_module))
    module.__dict__["__name__"] = "g2_adapter"
    module.__dict__["__file__"] = str(expected)
    module.__dict__["_REPORT_REPO_ROOT"] = repository_root()
    module.__dict__["_portable_binding"] = sys.modules[__name__]
    for name, value in tuple(module.__dict__.items()):
        if isinstance(value, FunctionType) and value.__globals__ is vars(raw_module):
            module.__dict__[name] = _clone_function(value, module.__dict__)
    module.__dict__["_frozen_tracked_record"] = _clone_function(
        _bridged_frozen_tracked_record, module.__dict__
    )
    module.__dict__["_external_frozen_source_binding"] = _clone_function(
        _bridged_external_frozen_source_binding, module.__dict__
    )
    module.__dict__["_role_audit_source_contract"] = _clone_function(
        _bridged_role_audit_source_contract, module.__dict__
    )
    module.__dict__["_portable_checkpoint_binding"] = True
    sys.modules["g2_adapter"] = module
    return module


__all__ = [
    "ensure_support_imports",
    "g2_adapter_path",
    "load_g2_adapter",
    "repository_root",
    "support_root",
]
