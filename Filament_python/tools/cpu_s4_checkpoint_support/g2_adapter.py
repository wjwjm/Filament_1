"""Private G2 role builder for the frozen CPU propagation paths.

Only this report-local module creates the G2 clone.  It first obtains the
real I0 or G3 controlled propagator from the frozen factory, then runs the
same G3 AST builder with an independent ``compile`` interceptor.  The
interceptor adds the verified timing nodes and, only for
``QUALIFY_BOUNDARY``, one explicit first-boundary capture call.  It never
changes a production module or the G3 call-site guards.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import CodeType, FunctionType
from typing import Any, Callable, Mapping

_REPORT_REPO_ROOT = Path(__file__).resolve().parents[3]

try:  # The parent runner places this tool directory directly on sys.path.
    import g2_boundary as _boundary
    import g1_common as _common
    import timing_adapter as _timing
except ImportError:  # pragma: no cover - package-style local loading
    from . import g1_common as _common
    from . import g2_boundary as _boundary
    from . import timing_adapter as _timing


SCHEMA = "filament.cpu_s4_3g2.adapter.v1"
QUALIFY_BOUNDARY = "QUALIFY_BOUNDARY"
PURE_OFF_REAL8 = "PURE_OFF_REAL8"
ROLES = ("I0", "G_CONTROLLED_INPUT")
MODES = (QUALIFY_BOUNDARY, PURE_OFF_REAL8)
_HOOK_NAME = "_g2_boundary_capture"
_LEASE_NAME = "_g2_boundary_lease"
_G_HELPER_NAMES = (
    "_g_step_linear_bk_nee",
    "_g_apply_nonlinear",
    "_g_controlled_raman_substep",
    "_g_propagation_enter",
    "_g_propagation_finalize",
    "_g_publish_result",
    "_g_reject_complete_mode",
)
_G_HELPER_CLOSURE_ROLES = {
    "_g_step_linear_bk_nee": {
        "state": "dispatch.cpu_s4_3g_state",
        "original_linear": "state._g_linear_function",
    },
    "_g_apply_nonlinear": {
        "state": "dispatch.cpu_s4_3g_state",
        "original_apply": "state._g_nonlinear_function",
    },
    "_g_controlled_raman_substep": {"state": "dispatch.cpu_s4_3g_state"},
    "_g_propagation_enter": {"state": "dispatch.cpu_s4_3g_state"},
    "_g_propagation_finalize": {"state": "dispatch.cpu_s4_3g_state"},
    "_g_publish_result": {"state": "dispatch.cpu_s4_3g_state"},
    "_g_reject_complete_mode": {"state": "dispatch.cpu_s4_3g_state"},
}
_TIMING_HELPER_NAMES = (
    "_timing_call_begin",
    "_timing_segment_begin",
    "_timing_segment_end",
    "_timing_heun_begin",
    "_timing_heun_end",
    "_timing_call_finalize",
)
_TIMING_END_NAME = "_timing_heun_end"
_ROLE_AUDIT_SOURCE_NAME = "g1_segment.py"
_FROZEN_CODE_CACHE: dict[tuple[str, str], dict[str, tuple[CodeType, ...]]] = {}
_SOURCE_BINDING_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


class G2AdapterError(RuntimeError):
    """A source, AST, binding, or lifecycle contract failed closed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ast_dump(module: ast.AST) -> str:
    return ast.dump(module, annotate_fields=True, include_attributes=False)


def _copy_function(function: Callable[..., Any], globals_dict: dict[str, Any]) -> Callable[..., Any]:
    clone = FunctionType(
        function.__code__,
        globals_dict,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
    clone.__annotations__ = getattr(function, "__annotations__", {})
    clone.__module__ = getattr(function, "__module__", None)
    clone.__qualname__ = getattr(function, "__qualname__", function.__name__)
    clone.__doc__ = getattr(function, "__doc__", None)
    return clone


def _is_boundary_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.AugAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "z"
        and isinstance(statement.op, ast.Add)
        and isinstance(statement.value, ast.Name)
        and statement.value.id == "dz_try"
    )


_EXPECTED_PROGRESS_ASSIGNMENT = ast.parse(
    _boundary.BOUNDARY_ASSIGNMENT, mode="exec"
).body[0]


def _is_progress_assignment(statement: ast.stmt) -> bool:
    """Match the frozen progress assignment, including its full expression."""
    return isinstance(statement, ast.Assign) and _ast_dump(statement) == _ast_dump(
        _EXPECTED_PROGRESS_ASSIGNMENT
    )


def _statement_matches(module: ast.AST) -> list[tuple[list[Any], int, ast.stmt]]:
    matches: list[tuple[list[Any], int, ast.stmt]] = []

    def visit(node: ast.AST, in_propagator: bool = False, in_loop: bool = False) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            in_propagator = node.name == "propagate_one_pulse"
            in_loop = False
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            in_loop = True
        for _field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if (
                        in_propagator
                        and in_loop
                        and isinstance(item, ast.stmt)
                        and _is_boundary_statement(item)
                        and index + 1 < len(value)
                        and isinstance(value[index + 1], ast.stmt)
                        and _is_progress_assignment(value[index + 1])
                    ):
                        matches.append((value, index, item))
                    if isinstance(item, ast.AST):
                        visit(item, in_propagator, in_loop)
            elif isinstance(value, ast.AST):
                visit(value, in_propagator, in_loop)

    visit(module)
    return matches


def _boundary_call(bindings: tuple[tuple[str, str], ...]) -> ast.Expr:
    keywords = [
        ast.keyword(arg="interval_index", value=ast.Name(id="interval_index", ctx=ast.Load())),
        ast.keyword(arg="E", value=ast.Name(id="E", ctx=ast.Load())),
        ast.keyword(arg="Qacc", value=ast.Name(id="Qacc", ctx=ast.Load())),
        ast.keyword(arg="Qacc_raman", value=ast.Name(id="Qacc_raman", ctx=ast.Load())),
        ast.keyword(arg="linear_result", value=ast.Name(id="linear_result", ctx=ast.Load())),
        ast.keyword(
            arg="need_energy",
            value=ast.Name(id="diag_linear_halfstep_energy", ctx=ast.Load()),
        ),
        ast.keyword(
            arg="need_profile",
            value=ast.Name(id="diag_bk_nee_profile", ctx=ast.Load()),
        ),
        ast.keyword(arg="thermal_slow_state", value=ast.Name(id="thermal_slow_state", ctx=ast.Load())),
        ast.keyword(arg="state_after", value=ast.Name(id="state_after", ctx=ast.Load())),
        ast.keyword(arg="z", value=ast.Name(id="z", ctx=ast.Load())),
        ast.keyword(arg="dz_try", value=ast.Name(id="dz_try", ctx=ast.Load())),
        ast.keyword(arg="U0_baseline", value=ast.Name(id="U0_baseline", ctx=ast.Load())),
        ast.keyword(arg="U_now", value=ast.Name(id="U_now", ctx=ast.Load())),
        ast.keyword(
            arg="adaptive_rejection_count",
            value=ast.Name(id="adaptive_rejection_count", ctx=ast.Load()),
        ),
        ast.keyword(
            arg="safety_mode_trigger_count",
            value=ast.Name(id="safety_mode_trigger_count", ctx=ast.Load()),
        ),
        ast.keyword(
            arg="diagnostic_values",
            value=ast.Tuple(
                elts=[ast.Name(id=source, ctx=ast.Load()) for _, source in bindings],
                ctx=ast.Load(),
            ),
        ),
    ]
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id=_HOOK_NAME, ctx=ast.Load()),
            args=[ast.Name(id=_LEASE_NAME, ctx=ast.Load())],
            keywords=keywords,
        )
    )


def _insert_boundary(module: ast.Module, bindings: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    matches = _statement_matches(module)
    if len(matches) != 1:
        raise G2AdapterError(f"BOUNDARY_ASSIGNMENT_ANCHOR_COUNT:{len(matches)}")
    statements, index, anchor = matches[0]
    hook = _boundary_call(bindings)
    ast.copy_location(hook, anchor)
    statements.insert(index + 1, hook)
    ast.fix_missing_locations(module)
    return {
        "assignment": _boundary.BOUNDARY_ASSIGNMENT,
        "anchor_lineno": int(getattr(anchor, "lineno", -1)),
        "anchor_col_offset": int(getattr(anchor, "col_offset", -1)),
        "inserted_statement": _HOOK_NAME,
        "explicit_array_names": list(_boundary.SCIENCE_ARRAY_FIELDS),
        "explicit_post_names": list(_boundary.POST_FIELDS),
        "explicit_science_alias_fields": list(_boundary.SCIENCE_ARRAY_FIELDS)
        + [f"POST.{name}" for name in _boundary.POST_FIELDS],
        "explicit_linear_result_name": "linear_result",
        "explicit_linear_energy_flag_name": "diag_linear_halfstep_energy",
        "explicit_linear_profile_flag_name": "diag_bk_nee_profile",
        "explicit_dynamic_scalar_names": list(_boundary.DYNAMIC_SCALARS),
        "explicit_diagnostic_count": len(bindings),
        "anchor_count": 1,
    }


def _is_boundary_call(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == _HOOK_NAME
    )


def _strip_boundary(module: ast.Module) -> tuple[ast.Module, int]:
    restored = copy.deepcopy(module)
    count = 0

    def visit(node: ast.AST) -> None:
        nonlocal count
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                kept = []
                for item in value:
                    if isinstance(item, ast.stmt) and _is_boundary_call(item):
                        count += 1
                        continue
                    kept.append(item)
                    if isinstance(item, ast.AST):
                        visit(item)
                setattr(node, field_name, kept)
            elif isinstance(value, ast.AST):
                visit(value)

    visit(restored)
    return restored, count


def _structure_without_timing_ast(module: ast.Module) -> ast.Module:
    """Mirror the frozen timing stripper while retaining an AST object."""
    restored = copy.deepcopy(module)
    function_node = _timing._function(restored, "propagate_one_pulse")
    body = list(function_node.body)
    for index in range(len(body) - 1):
        if not _timing._is_timer_call(body[index], "_timing_call_begin"):
            continue
        wrapper = body[index + 1]
        if not isinstance(wrapper, ast.Try) or len(wrapper.finalbody) != 1:
            continue
        if not _timing._is_timer_call(wrapper.finalbody[0], "_timing_call_finalize"):
            continue
        function_node.body = body[:index] + list(wrapper.body) + body[index + 2 :]
        break

    def strip(node: ast.AST) -> None:
        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                if value and all(isinstance(item, ast.stmt) for item in value):
                    kept = [item for item in value if not _timing._is_timer_call(item)]
                    setattr(node, field_name, kept)
                    for item in kept:
                        strip(item)
                else:
                    for item in value:
                        if isinstance(item, ast.AST):
                            strip(item)
            elif isinstance(value, ast.AST):
                strip(value)

    strip(restored)
    return restored


def _make_compile_interceptor(
    *,
    role: str,
    mode: str,
    bindings: tuple[tuple[str, str], ...],
) -> Callable[..., Any]:
    injected = False

    def interceptor(source: Any, filename: Any, compile_mode: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal injected
        if not injected and isinstance(source, ast.Module):
            baseline = copy.deepcopy(source)
            timing_receipt = _timing._inject_timing(source, role=role)
            boundary_receipt = None
            if mode == QUALIFY_BOUNDARY:
                boundary_receipt = _insert_boundary(source, bindings)
                stripped, count = _strip_boundary(_structure_without_timing_ast(source))
                if count != 1:
                    raise G2AdapterError(f"BOUNDARY_STRIP_COUNT:{count}")
                if _ast_dump(stripped) != _ast_dump(baseline):
                    raise G2AdapterError("G2_AST_CHANGED_BEYOND_TIMING_AND_BOUNDARY")
            else:
                if _ast_dump(_structure_without_timing_ast(source)) != _ast_dump(baseline):
                    raise G2AdapterError("G2_PURE_OFF_AST_CHANGED_AFTER_TIMING_STRIP")
            interceptor.static_receipt = {
                "schema": SCHEMA + ".static",
                "timing": timing_receipt,
                "boundary": boundary_receipt,
                "baseline_ast_sha256": _sha256_bytes(_ast_dump(baseline).encode("utf-8")),
                "post_strip_ast_sha256": _sha256_bytes(
                    _ast_dump(_strip_boundary(_structure_without_timing_ast(source))[0]).encode("utf-8")
                )
                if mode == QUALIFY_BOUNDARY
                else _sha256_bytes(
                    _ast_dump(_structure_without_timing_ast(source)).encode("utf-8")
                ),
            }
            injected = True
        return builtins.compile(source, filename, compile_mode, *args, **kwargs)

    interceptor.static_receipt = None
    return interceptor


def _actual_code_record(function: Callable[..., Any]) -> dict[str, Any]:
    code = getattr(function, "__code__", None)
    if code is None:
        raise G2AdapterError("G2_CLONE_CODE_MISSING")
    return {
        "filename": str(Path(str(code.co_filename)).resolve()),
        "name": str(code.co_name),
        "firstlineno": int(code.co_firstlineno),
        "code_object_id": int(id(code)),
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
    }


def _source_path(function: Callable[..., Any]) -> Path:
    code = getattr(function, "__code__", None)
    if code is None:
        raise G2AdapterError("G2_SOURCE_CODE_MISSING")
    path = Path(str(code.co_filename)).resolve(strict=True)
    return path


def _code_constant_payload(value: Any) -> Any:
    """Serialize Python code constants without using process-local reprs."""
    if isinstance(value, CodeType):
        return {"kind": "code", "value": _code_structure_payload(value)}
    if value is None:
        return {"kind": "none"}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if type(value) is bool:
        return {"kind": "bool", "value": bool(value)}
    if type(value) is int:
        return {"kind": "int", "value": str(value)}
    if type(value) is float:
        return {"kind": "float", "value": value.hex()}
    if type(value) is complex:
        return {
            "kind": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bytes:
        return {"kind": "bytes", "value": value.hex()}
    if type(value) is tuple:
        return {"kind": "tuple", "items": [_code_constant_payload(item) for item in value]}
    if type(value) is frozenset:
        items = [_code_constant_payload(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"kind": "frozenset", "items": items}
    raise G2AdapterError(f"G2_CODE_CONSTANT_UNSUPPORTED:{type(value).__name__}")


def _code_structure_payload(code: CodeType) -> dict[str, Any]:
    """Return a complete, deterministic structural representation of code."""
    return {
        "co_argcount": int(code.co_argcount),
        "co_posonlyargcount": int(getattr(code, "co_posonlyargcount", 0)),
        "co_kwonlyargcount": int(code.co_kwonlyargcount),
        "co_nlocals": int(code.co_nlocals),
        "co_stacksize": int(code.co_stacksize),
        "co_flags": int(code.co_flags),
        "co_code": bytes(code.co_code).hex(),
        "co_consts": [_code_constant_payload(item) for item in code.co_consts],
        "co_names": list(code.co_names),
        "co_varnames": list(code.co_varnames),
        "co_freevars": list(code.co_freevars),
        "co_cellvars": list(code.co_cellvars),
        "co_filename": str(code.co_filename),
        "co_name": str(code.co_name),
        "co_qualname": str(getattr(code, "co_qualname", code.co_name)),
        "co_firstlineno": int(code.co_firstlineno),
        "co_linetable": bytes(getattr(code, "co_linetable", b"")).hex(),
        "co_exceptiontable": bytes(getattr(code, "co_exceptiontable", b"")).hex(),
    }


def _code_structure_sha256(code: CodeType) -> str:
    payload = _code_structure_payload(code)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_frozen_code(source_path: Path, canonical: bytes, actual: CodeType) -> CodeType:
    """Compile frozen source without executing it and locate ``co_qualname``."""
    target = str(getattr(actual, "co_qualname", ""))
    if not target:
        raise G2AdapterError("G2_FROZEN_CODE_QUALNAME_MISSING")
    actual_filename = str(getattr(actual, "co_filename", ""))
    try:
        source_resolved = Path(source_path).resolve(strict=True)
        actual_resolved = Path(actual_filename).resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise G2AdapterError("G2_FROZEN_CODE_PATH_MISMATCH") from exc
    if actual_resolved != source_resolved:
        raise G2AdapterError("G2_FROZEN_CODE_PATH_MISMATCH")
    cache_key = (actual_filename, hashlib.sha256(canonical).hexdigest())
    cached = _FROZEN_CODE_CACHE.get(cache_key)
    if cached is None:
        try:
            module_code = compile(
                canonical.decode("utf-8"),
                actual_filename,
                "exec",
                dont_inherit=True,
                optimize=0,
            )
        except BaseException as exc:
            raise G2AdapterError("G2_FROZEN_SOURCE_COMPILE_FAILED") from exc
        found: dict[str, list[CodeType]] = {}

        def visit(code: CodeType) -> None:
            qualname = str(getattr(code, "co_qualname", ""))
            found.setdefault(qualname, []).append(code)
            for value in code.co_consts:
                if isinstance(value, CodeType):
                    visit(value)

        visit(module_code)
        cached = {name: tuple(values) for name, values in found.items()}
        _FROZEN_CODE_CACHE[cache_key] = cached
    matches = list(cached.get(target, ()))
    if len(matches) != 1:
        raise G2AdapterError(f"G2_FROZEN_CODE_QUALNAME_COUNT:{target}:{len(matches)}")
    return matches[0]


def _external_frozen_source_binding(path: Path, raw: bytes) -> dict[str, Any]:
    """Resolve an explicitly archived raw-byte identity for an external source.

    The G3 builder lives in an intentionally external, report-bound adapter
    namespace rather than in the repository's tracked production tree.  A
    failed Git-blob lookup therefore cannot be treated as a source match.  It
    is admitted only when the current path and raw bytes both match the
    already-frozen G1 implementation binding; the report itself is never
    used as a substitute for the source bytes.
    """
    report_path = (
        _REPORT_REPO_ROOT
        / "reports"
        / "CPU_S4_3G2_LOCAL_20261023_160301"
        / "G1_FINAL_IMPLEMENTATION_BINDING.json"
    )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except BaseException as exc:
        raise G2AdapterError("G2_EXTERNAL_FROZEN_SOURCE_REPORT_UNAVAILABLE") from exc
    if not isinstance(report, dict) or report.get("hash_scope") != "raw_bytes":
        raise G2AdapterError("G2_EXTERNAL_FROZEN_SOURCE_REPORT_SCOPE")
    digest = hashlib.sha256(raw).hexdigest()
    expected_by_name = report.get("owned_file_sha256")
    if not isinstance(expected_by_name, dict):
        raise G2AdapterError("G2_EXTERNAL_FROZEN_SOURCE_EXPECTED_MAP")
    name = path.name
    expected = expected_by_name.get(name)
    if not isinstance(expected, str) or expected.lower() != digest:
        raise G2AdapterError(f"G2_EXTERNAL_FROZEN_SOURCE_HASH_MISMATCH:{name}")
    normalized_path = str(path.resolve()).casefold()
    def portable_relative(value: str) -> str | None:
        normalized = str(value).replace("\\", "/").casefold()
        marker = "/filament_python/"
        position = normalized.find(marker)
        if position < 0:
            return None
        return normalized[position + 1 :]

    actual_portable_relative = portable_relative(normalized_path)
    records = report.get("records")
    matched_record: dict[str, Any] | None = None
    matched_path_mode: str | None = None
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            source = record.get("source")
            if not isinstance(source, dict):
                continue
            source_path = source.get("path")
            source_hash = source.get("raw_sha256")
            if not isinstance(source_path, str) or not isinstance(source_hash, str):
                continue
            try:
                source_resolved = str(Path(source_path).resolve()).casefold()
            except BaseException:
                continue
            source_portable_relative = portable_relative(source_path)
            exact_match = source_resolved == normalized_path
            portable_match = (
                actual_portable_relative is not None
                and source_portable_relative is not None
                and actual_portable_relative == source_portable_relative
            )
            if (exact_match or portable_match) and source_hash.lower() == digest:
                matched_record = source
                matched_path_mode = "exact" if exact_match else "portable_relative"
                break
    if matched_record is None:
        raise G2AdapterError(f"G2_EXTERNAL_FROZEN_SOURCE_PATH_MISMATCH:{name}")
    expected_size = matched_record.get("size_bytes")
    if expected_size is not None and int(expected_size) != len(raw):
        raise G2AdapterError(f"G2_EXTERNAL_FROZEN_SOURCE_SIZE_MISMATCH:{name}")
    try:
        relative = path.relative_to(_REPORT_REPO_ROOT).as_posix()
    except ValueError:
        relative = str(path)
    return {
        "relative_path": relative,
        "filename": str(path),
        "hash_scope": "raw_bytes",
        "git_blob_oid": None,
        "external_frozen_source": True,
        "external_binding_report": str(report_path),
        "expected_raw_sha256": str(expected).lower(),
        "source_raw_sha256": digest,
        "source_size_bytes": len(raw),
        "external_source_path_match": matched_path_mode,
    }


def _source_relative_candidates(path: Path) -> list[tuple[Path, str]]:
    """Return runner-root/source-relative pairs in authoritative order."""
    roots: list[Path] = []
    configured_names = ("S43G2_SOURCE_ROOT", "S43F_SOURCE_ROOT")
    for name in configured_names:
        configured = os.environ.get(name)
        if not configured:
            continue
        root = Path(configured).resolve()
        if root not in roots:
            roots.append(root)
    report_root = _REPORT_REPO_ROOT.resolve()
    if report_root not in roots:
        roots.append(report_root)
    result: list[tuple[Path, str]] = []
    for root in roots:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        pair = (root, relative)
        if pair not in result:
            result.append(pair)
    return result


def _frozen_tracked_record(relative: str) -> dict[str, Any] | None:
    """Return the frozen tracked record for a source-relative path, if any."""
    manifest_path = (
        _REPORT_REPO_ROOT
        / "reports"
        / "CPU_S4_3G2_LOCAL_20261023_160301"
        / "inputs"
        / "FROZEN_SOURCE_PROVENANCE_V2.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except BaseException as exc:
        raise G2AdapterError("G2_FORMAL_SOURCE_GIT_BINDING_FAILED") from exc
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise G2AdapterError("G2_FORMAL_SOURCE_GIT_BINDING_FAILED")
    for record in records:
        if (
            isinstance(record, dict)
            and record.get("classification") == "tracked_text"
            and record.get("path") == relative
        ):
            return copy.deepcopy(record)
    return None


def _method_binding(function: Callable[..., Any]) -> dict[str, Any]:
    """Bind one method to frozen source and a complete actual code structure."""
    code = getattr(function, "__code__", None)
    if code is None:
        raise G2AdapterError("G2_FORMAL_METHOD_CODE_MISSING")
    if not isinstance(code, CodeType):
        raise G2AdapterError("G2_FORMAL_METHOD_CODE_TYPE")
    path = Path(str(code.co_filename)).resolve(strict=True)
    raw = path.read_bytes()
    canonical = _timing._provenance_v2.canonical_lf_bytes(raw)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    source_cache_key = (str(path), raw_sha256)
    cached_source = _SOURCE_BINDING_CACHE.get(source_cache_key)
    if cached_source is None:
        external_binding: dict[str, Any] | None = None
        relative: str | None = None
        blob_oid: str | None = None
        git_root: str | None = None
        tracked_candidate_seen = False
        for candidate_root, candidate_relative in _source_relative_candidates(path):
            frozen_record = _frozen_tracked_record(candidate_relative)
            if frozen_record is None:
                continue
            tracked_candidate_seen = True
            try:
                _timing._provenance_v2.validate_record(
                    candidate_root,
                    frozen_record,
                    require_hash_scope=True,
                )
                candidate_oid = str(frozen_record["git_blob_oid"])
                blob_data = _timing._provenance_v2._read_git_blob(candidate_root, candidate_oid)
                if _timing._provenance_v2.canonical_lf_bytes(blob_data) != canonical:
                    raise G2AdapterError("G2_FORMAL_SOURCE_CANONICAL_BLOB_MISMATCH")
                relative = candidate_relative
                blob_oid = candidate_oid
                git_root = str(candidate_root)
                break
            except G2AdapterError:
                raise
            except BaseException:
                raise G2AdapterError("G2_FORMAL_SOURCE_GIT_BINDING_FAILED")
        if relative is None:
            if tracked_candidate_seen:
                raise G2AdapterError("G2_FORMAL_SOURCE_GIT_BINDING_FAILED")
            try:
                external_binding = _external_frozen_source_binding(path, raw)
            except G2AdapterError as external_exc:
                raise G2AdapterError("G2_FORMAL_SOURCE_GIT_BINDING_FAILED") from external_exc
            relative = str(external_binding["relative_path"])
            blob_oid = None
        cached_source = {
            "relative": str(relative),
            "blob_oid": None if blob_oid is None else str(blob_oid),
            "git_root": git_root,
            "external_binding": None if external_binding is None else copy.deepcopy(external_binding),
        }
        _SOURCE_BINDING_CACHE[source_cache_key] = cached_source
    relative = str(cached_source["relative"])
    blob_oid = cached_source.get("blob_oid")
    external_binding = cached_source.get("external_binding")
    frozen_code = _find_frozen_code(path, canonical, code)
    actual_structure_sha256 = _code_structure_sha256(code)
    frozen_structure_sha256 = _code_structure_sha256(frozen_code)
    if actual_structure_sha256 != frozen_structure_sha256:
        raise G2AdapterError(
            f"G2_FROZEN_FUNCTION_CODE:{getattr(code, 'co_qualname', '')}"
        )
    return {
        "relative_path": relative,
        "filename": str(path),
        "name": str(code.co_name),
        "qualname": str(getattr(code, "co_qualname", code.co_name)),
        "firstlineno": int(code.co_firstlineno),
        "hash_scope": (
            str(external_binding["hash_scope"])
            if external_binding is not None
            else _timing._provenance_v2.TRACKED_HASH_SCOPE
        ),
        "git_blob_oid": None if blob_oid is None else str(blob_oid),
        "git_root": cached_source.get("git_root"),
        "git_relative_path": (
            str(relative) if blob_oid is not None else None
        ),
        "external_frozen_source": external_binding is not None,
        "external_binding_report": (
            None
            if external_binding is None
            else str(external_binding["external_binding_report"])
        ),
        "external_source_path_match": (
            None
            if external_binding is None
            else str(external_binding["external_source_path_match"])
        ),
        "expected_raw_sha256": (
            None
            if external_binding is None
            else str(external_binding["expected_raw_sha256"])
        ),
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "frozen_bytecode_sha256": hashlib.sha256(frozen_code.co_code).hexdigest(),
        "actual_code_structure_sha256": actual_structure_sha256,
        "frozen_code_structure_sha256": frozen_structure_sha256,
        "code_structure_match": True,
        "frozen_compile": {"dont_inherit": True, "optimize": 0, "target_qualname": str(getattr(code, "co_qualname", code.co_name))},
        "source_raw_sha256": raw_sha256,
        "source_canonical_lf_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _function_identity_record(
    function: Callable[..., Any],
    *,
    label: str,
    owner_module: Any | None,
    closure_roles: Mapping[str, str],
    expected_code: CodeType | None = None,
    expected_globals: Mapping[str, Any] | None = None,
    globals_owner_kind: str = "MODULE_DICT",
    backend_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe identity record for one bound helper/function.

    The live function, globals dictionary, and closure cells remain private to
    the runtime guard.  This record contains only source/code and scalar
    ownership metadata, so it can safely be copied into a case receipt.
    """
    code = getattr(function, "__code__", None)
    globals_dict = getattr(function, "__globals__", None)
    if not callable(function) or not isinstance(code, CodeType):
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    if not isinstance(globals_dict, dict):
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    if expected_globals is None:
        if owner_module is None or globals_dict is not getattr(owner_module, "__dict__", None):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        globals_owner_is_module_dict = True
    else:
        if not isinstance(expected_globals, dict) or globals_dict is not expected_globals:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        globals_owner_is_module_dict = (
            owner_module is not None
            and globals_dict is getattr(owner_module, "__dict__", None)
        )
    if expected_code is not None:
        if _code_structure_sha256(code) != _code_structure_sha256(expected_code):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    try:
        binding = _method_binding(function)
    except BaseException as exc:
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING") from exc
    freevars = tuple(str(name) for name in code.co_freevars)
    closure = getattr(function, "__closure__", None)
    if closure is None:
        if closure_roles:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    elif len(closure) != len(freevars):
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    if tuple(sorted(closure_roles)) != tuple(sorted(freevars)):
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    return {
        "label": str(label),
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(code, "co_qualname", code.co_name)),
        "filename": str(code.co_filename),
        "firstlineno": int(code.co_firstlineno),
        "bytecode_sha256": hashlib.sha256(bytes(code.co_code)).hexdigest(),
        "code_structure_sha256": _code_structure_sha256(code),
        "method_binding": binding,
        "globals_owner_module": str(globals_dict.get("__name__", "")),
        "globals_owner_kind": str(globals_owner_kind),
        "globals_owner_is_module_dict": bool(globals_owner_is_module_dict),
        "backend_binding": (
            None
            if backend_binding is None
            else copy.deepcopy(dict(backend_binding))
        ),
        "closure_freevars": list(freevars),
        "closure_roles": {str(key): str(value) for key, value in closure_roles.items()},
    }


def _closure_cells(function: Callable[..., Any]) -> dict[str, Any]:
    code = getattr(function, "__code__", None)
    closure = getattr(function, "__closure__", None)
    if not isinstance(code, CodeType) or closure is None or len(closure) != len(code.co_freevars):
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    result: dict[str, Any] = {}
    for name, cell in zip(code.co_freevars, closure):
        try:
            result[str(name)] = cell
            cell.cell_contents
        except BaseException as exc:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING") from exc
    return result


def _role_audit_source_contract() -> dict[str, Any]:
    """Bind the only externally supplied timing audit to its frozen source."""
    report_root = (
        _REPORT_REPO_ROOT
        / "reports"
        / "CPU_S4_3G2_LOCAL_20261023_160301"
    )
    binding_path = report_root / "INPUT_AND_VERSION_BINDING.json"
    try:
        document = json.loads(binding_path.read_text(encoding="utf-8"))
    except BaseException as exc:
        raise G2AdapterError("G2_TIMING_HELPER_BINDING") from exc
    records = document.get("inherited_helpers") if isinstance(document, dict) else None
    if not isinstance(records, list):
        raise G2AdapterError("G2_TIMING_HELPER_BINDING")
    expected_suffix = (
        "reports/CPU_S4_3G2_LOCAL_20261023_160301/"
        "tool/g1_segment.py"
    ).casefold()

    def portable_record_path(record: Mapping[str, Any]) -> str:
        return str(record.get("path", "")).replace("\\", "/")

    source_record = next(
        (
            record
            for record in records
            if isinstance(record, dict)
            and portable_record_path(record).casefold().endswith(expected_suffix)
        ),
        None,
    )
    if not isinstance(source_record, dict):
        raise G2AdapterError("G2_TIMING_HELPER_BINDING")
    try:
        source_record_path = str(source_record["path"])
        source_record_portable = portable_record_path(source_record)
        expected_path = (
            report_root / "tool" / _ROLE_AUDIT_SOURCE_NAME
        ).resolve(strict=True)
        expected_sha256 = str(source_record["raw_sha256"]).lower()
        expected_size = int(source_record["size_bytes"])
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise G2AdapterError("G2_TIMING_HELPER_BINDING") from exc
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
            raise G2AdapterError("G2_TIMING_HELPER_BINDING")
        raw = expected_path.read_bytes()
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest().lower() != expected_sha256:
            raise G2AdapterError("G2_TIMING_HELPER_BINDING")
        canonical = _timing._provenance_v2.canonical_lf_bytes(raw)
        frozen_init = _find_frozen_code(source_path, canonical, init_code)
        if _code_structure_sha256(init_code) != _code_structure_sha256(frozen_init):
            raise G2AdapterError("G2_TIMING_HELPER_BINDING")
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
            raise G2AdapterError("G2_TIMING_HELPER_BINDING")
        closed_code = nested[0]
        frozen_closed = _find_frozen_code(source_path, canonical, closed_code)
        if _code_structure_sha256(closed_code) != _code_structure_sha256(frozen_closed):
            raise G2AdapterError("G2_TIMING_HELPER_BINDING")
        return {
            "type": role_audit,
            "module": module,
            "globals": module_globals,
            "init_code": init_code,
            "closed_code": frozen_closed,
            "source_path": source_path,
            "source_raw_sha256": expected_sha256,
            "source_size_bytes": expected_size,
            "receipt": {
                "module": str(getattr(module, "__name__", "")),
                "class": f"{role_audit.__module__}.{role_audit.__qualname__}",
                "source_path": str(source_path),
                "source_raw_sha256": expected_sha256,
                "source_size_bytes": expected_size,
                "source_record_path": source_record_path,
                "path_translation": {
                    "record_raw": source_record_path,
                    "record_portable": source_record_portable,
                    "expected_runtime_path": str(expected_path),
                    "matched_complete_suffix": expected_suffix,
                },
                "init_qualname": str(getattr(init_code, "co_qualname", "")),
                "init_code_structure_sha256": _code_structure_sha256(init_code),
                "closed_qualname": str(getattr(closed_code, "co_qualname", "")),
                "closed_code_structure_sha256": _code_structure_sha256(closed_code),
                "input_binding": str(binding_path),
            },
        }
    except G2AdapterError:
        raise
    except BaseException as exc:
        raise G2AdapterError("G2_TIMING_HELPER_BINDING") from exc


def _bind_timing_helpers(
    clone: Callable[..., Any],
    helpers: Mapping[str, Callable[..., Any]],
    *,
    role: str,
    dispatch: Any,
    state: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind clock helpers and the sole permitted external timing audit hook."""
    try:
        if tuple(sorted(helpers)) != tuple(sorted(_TIMING_HELPER_NAMES)):
            raise G2AdapterError("G2_TIMING_HELPER_BINDING")
        clone_globals = getattr(clone, "__globals__", None)
        timing_globals = getattr(_timing, "__dict__", None)
        if not isinstance(clone_globals, dict) or not isinstance(timing_globals, dict):
            raise G2AdapterError("G2_TIMING_HELPER_BINDING")
        helper_guards: dict[str, Any] = {}
        helper_receipts: dict[str, Any] = {}
        for name in _TIMING_HELPER_NAMES:
            function = helpers.get(name)
            code = getattr(function, "__code__", None)
            if (
                not callable(function)
                or not isinstance(code, CodeType)
                or clone_globals.get(name) is not function
                or getattr(function, "__globals__", None) is not timing_globals
            ):
                raise G2AdapterError("G2_TIMING_HELPER_BINDING")
            cells = _closure_cells(function)
            if tuple(sorted(cells)) != ("clock",):
                raise G2AdapterError("G2_TIMING_HELPER_BINDING")
            clock = cells["clock"].cell_contents
            if not isinstance(clock, _timing.WindowClock):
                raise G2AdapterError("G2_TIMING_HELPER_BINDING")
            helper_guards[name] = {
                "function": function,
                "code": code,
                "globals": timing_globals,
                "closure": getattr(function, "__closure__", None),
                "cells": cells,
                "clock": clock,
            }
            helper_receipts[name] = {
                "qualname": str(getattr(code, "co_qualname", code.co_name)),
                "filename": str(code.co_filename),
                "code_structure_sha256": _code_structure_sha256(code),
                "closure_freevars": list(code.co_freevars),
                "closure_roles": {"clock": "WindowClock"},
            }
        role_audit = _role_audit_source_contract()
        role_audit_state = None
        if role != "I0":
            missing = object()
            role_audit_state = getattr(dispatch, "cpu_s4_3e_state", missing)
            if role_audit_state is missing or role_audit_state is not state:
                raise G2AdapterError("G2_TIMING_HELPER_BINDING")
        guard = {
            "clone": clone,
            "role": str(role),
            "dispatch": dispatch,
            "state": state,
            "role_audit_state": role_audit_state,
            "helpers": helper_guards,
            "role_audit": role_audit,
        }
        receipt = {
            "schema": SCHEMA + ".timing_helper_binding",
            "helper_count": len(_TIMING_HELPER_NAMES),
            "helpers": helper_receipts,
            "role_audit": copy.deepcopy(role_audit["receipt"]),
            "allowed_heun_end_closure_roles": {
                "original": "this_build_clock._timing_heun_end",
                "self": "frozen g1_segment.RoleAudit",
            },
        }
        return receipt, guard
    except G2AdapterError:
        raise
    except BaseException as exc:
        raise G2AdapterError("G2_TIMING_HELPER_BINDING") from exc


def _validate_timing_helper_binding(guard: Mapping[str, Any]) -> None:
    """Validate the six timing globals, allowing only the frozen RoleAudit hook."""
    try:
        clone = guard["clone"]
        clone_globals = getattr(clone, "__globals__", None)
        if not isinstance(clone_globals, dict):
            raise G2AdapterError("G2_TIMING_HELPER_BINDING")
        helper_guards = guard["helpers"]
        for name in _TIMING_HELPER_NAMES:
            expected = helper_guards[name]
            actual = clone_globals.get(name)
            expected_cells = _closure_cells(expected["function"])
            if expected_cells["clock"].cell_contents is not expected["clock"]:
                raise G2AdapterError("G2_TIMING_HELPER_BINDING")
            if name == _TIMING_END_NAME and actual is not expected["function"]:
                code = getattr(actual, "__code__", None)
                role_audit = guard["role_audit"]
                if (
                    not callable(actual)
                    or not isinstance(code, CodeType)
                    or getattr(actual, "__globals__", None) is not role_audit["globals"]
                    or _code_structure_sha256(code)
                    != _code_structure_sha256(role_audit["closed_code"])
                ):
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                cells = _closure_cells(actual)
                if tuple(sorted(cells)) != ("original", "self"):
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                if cells["original"].cell_contents is not expected["function"]:
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                audit = cells["self"].cell_contents
                if type(audit) is not role_audit["type"]:
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                if getattr(audit, "dispatch", None) is not guard["dispatch"]:
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                if str(getattr(audit, "role", "")) != str(guard["role"]):
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                if getattr(audit, "state", object()) is not guard["role_audit_state"]:
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                role_audit_init = role_audit["type"].__dict__.get("__init__")
                if getattr(role_audit_init, "__code__", None) is not role_audit["init_code"]:
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                continue
            if (
                actual is not expected["function"]
                or getattr(actual, "__code__", None) is not expected["code"]
                or getattr(actual, "__globals__", None) is not expected["globals"]
                or getattr(actual, "__closure__", None) is not expected["closure"]
            ):
                raise G2AdapterError("G2_TIMING_HELPER_BINDING")
            cells = _closure_cells(actual)
            expected_cells = expected["cells"]
            if tuple(sorted(cells)) != tuple(sorted(expected_cells)):
                raise G2AdapterError("G2_TIMING_HELPER_BINDING")
            for key, cell in expected_cells.items():
                if cells[key] is not cell:
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
                if key == "clock" and cell.cell_contents is not expected["clock"]:
                    raise G2AdapterError("G2_TIMING_HELPER_BINDING")
    except G2AdapterError:
        raise
    except BaseException as exc:
        raise G2AdapterError("G2_TIMING_HELPER_BINDING") from exc


def _bind_g_runtime_helpers(
    clone: Callable[..., Any],
    *,
    module: Any,
    builder: Callable[..., Any],
    builder_binding: Mapping[str, Any],
    builder_globals: Mapping[str, Any],
    dispatch: Any,
    state: Any,
    lease: _boundary.BoundaryLease | None,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the seven G3 closures and return receipt plus private guard.

    The guard is intentionally narrow.  It retains only the generated helper
    functions, their closure cells, the two state-bound original functions,
    and (for qualification) the private boundary hook/lease.  It never
    snapshots or scans the clone's complete globals dictionary.
    """
    try:
        module_dict = getattr(module, "__dict__", None)
        builder_module_dict = getattr(builder, "__globals__", None)
        if not isinstance(module_dict, dict) or builder_module_dict is not module_dict:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if getattr(dispatch, "cpu_s4_3g_state", None) is not state:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        original_linear = getattr(state, "_g_linear_function", None)
        original_apply = getattr(state, "_g_nonlinear_function", None)
        if not callable(original_linear) or not callable(original_apply):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        backend_validator = getattr(module, "_backend_binding", None)
        if not callable(backend_validator):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        try:
            backend_binding = backend_validator(original_linear, original_apply)
        except BaseException as exc:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING") from exc
        if not isinstance(backend_binding, Mapping):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        original_linear_globals = getattr(original_linear, "__globals__", None)
        if not isinstance(original_linear_globals, dict):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        linear_xp = original_linear_globals.get("xp")
        linear_fft = getattr(linear_xp, "fft", None)
        original_fft_dispatch = getattr(linear_fft, "dispatch", None)
        try:
            backend_dispatch_id = int(backend_binding["dispatch_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING") from exc
        if original_fft_dispatch is None or id(original_fft_dispatch) != backend_dispatch_id:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        for backend_owner in (
            getattr(state, "_g_function_binding", None),
            getattr(dispatch, "cpu_s4_3g_backend_binding", None),
        ):
            if not isinstance(backend_owner, Mapping):
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            try:
                if int(backend_owner["dispatch_id"]) != backend_dispatch_id:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING") from exc
        original_apply_globals = getattr(original_apply, "__globals__", None)
        if not isinstance(original_apply_globals, dict):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        source_globals = getattr(clone, "__globals__", None)
        if not isinstance(source_globals, dict):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if source_globals.get("step_linear_bk_nee_factorized") is not original_linear:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if source_globals.get("apply_nonlinear") is not original_apply:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if getattr(builder, "__closure__", None) is not None:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if not isinstance(builder_globals, Mapping):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        changed_globals: list[str] = []
        all_global_names = set(module_dict) | set(builder_globals)
        for name in sorted(all_global_names):
            if name == "compile":
                continue
            if name not in module_dict or name not in builder_globals:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if builder_globals[name] is not module_dict[name]:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if "compile" not in builder_globals:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        changed_globals.append("compile")

        maker = getattr(module, "_make_state_helpers", None)
        if not callable(maker):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        maker_binding = _method_binding(maker)
        expected_helpers = maker(state, original_linear, original_apply)
        if not isinstance(expected_helpers, Mapping):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if tuple(sorted(expected_helpers)) != tuple(sorted(_G_HELPER_NAMES)):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")

        helper_receipts: dict[str, Any] = {}
        helper_guards: dict[str, Any] = {}
        for name in _G_HELPER_NAMES:
            actual = source_globals.get(name)
            expected = expected_helpers.get(name)
            if not callable(actual) or not callable(expected):
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if getattr(actual, "__globals__", None) is not module_dict:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            expected_code = getattr(expected, "__code__", None)
            if not isinstance(expected_code, CodeType):
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if getattr(actual, "__code__", None) is not expected_code:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            closure_cells = _closure_cells(actual)
            expected_cells = _closure_cells(expected)
            expected_roles = _G_HELPER_CLOSURE_ROLES[name]
            if tuple(sorted(closure_cells)) != tuple(sorted(expected_roles)):
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if tuple(sorted(expected_cells)) != tuple(sorted(expected_roles)):
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if closure_cells["state"].cell_contents is not state:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if expected_cells["state"].cell_contents is not state:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if "original_linear" in expected_roles:
                if closure_cells["original_linear"].cell_contents is not original_linear:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
                if expected_cells["original_linear"].cell_contents is not original_linear:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if "original_apply" in expected_roles:
                if closure_cells["original_apply"].cell_contents is not original_apply:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
                if expected_cells["original_apply"].cell_contents is not original_apply:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            helper_receipts[name] = _function_identity_record(
                actual,
                label=name,
                owner_module=module,
                closure_roles=expected_roles,
                expected_code=expected_code,
            )
            helper_guards[name] = {
                "function": actual,
                "code": actual.__code__,
                "closure": actual.__closure__,
                "cells": closure_cells,
            }

        original_receipts = {
            "original_linear": _function_identity_record(
                original_linear,
                label="state._g_linear_function",
                owner_module=None,
                closure_roles={},
                expected_globals=original_linear_globals,
                globals_owner_kind="FROZEN_FACTORY_PRIVATE_NAMESPACE",
                backend_binding=backend_binding,
            ),
            "original_apply": _function_identity_record(
                original_apply,
                label="state._g_nonlinear_function",
                owner_module=None,
                closure_roles={},
                expected_globals=original_apply_globals,
                globals_owner_kind="FROZEN_FACTORY_PRIVATE_NAMESPACE",
                backend_binding=backend_binding,
            ),
        }
        if mode == QUALIFY_BOUNDARY:
            if lease is None:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if source_globals.get(_HOOK_NAME) is not _boundary.explicit_boundary_capture:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if source_globals.get(_LEASE_NAME) is not lease:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        receipt = {
            "schema": SCHEMA + ".runtime_helper_binding",
            "state_is_dispatch_cpu_s4_3g_state": True,
            "dispatch_state_relation": "dispatch.cpu_s4_3g_state",
            "helper_count": len(_G_HELPER_NAMES),
            "helpers": helper_receipts,
            "bound_originals": original_receipts,
            "builder": {
                "binding": copy.deepcopy(dict(builder_binding)),
                "globals_owner_module": str(module.__name__),
                "globals_owner_is_module_dict": True,
                "copy_globals_changed_names": changed_globals,
                "copy_globals_only_compile_change": changed_globals == ["compile"],
                "closure_freevars": [],
            },
            "helper_factory_binding": maker_binding,
            "qualify_boundary_hook_bound": mode == QUALIFY_BOUNDARY,
            "qualify_boundary_lease_bound": mode == QUALIFY_BOUNDARY,
        }
        guard = {
            "clone": clone,
            "module": module,
            "dispatch": dispatch,
            "state": state,
            "original_linear": original_linear,
            "original_apply": original_apply,
            "original_linear_globals": getattr(original_linear, "__globals__", None),
            "original_apply_globals": getattr(original_apply, "__globals__", None),
            "original_fft_dispatch": original_fft_dispatch,
            "original_linear_code": original_linear.__code__,
            "original_apply_code": original_apply.__code__,
            "helpers": helper_guards,
            "mode": str(mode),
            "boundary_hook": _boundary.explicit_boundary_capture if mode == QUALIFY_BOUNDARY else None,
            "lease": lease if mode == QUALIFY_BOUNDARY else None,
        }
        return receipt, guard
    except G2AdapterError:
        raise
    except BaseException as exc:
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING") from exc


def _validate_g_runtime_binding(guard: Mapping[str, Any]) -> None:
    """O(1) per-call check of the explicitly bound G helper objects only."""
    try:
        clone = guard["clone"]
        module = guard["module"]
        clone_globals = getattr(clone, "__globals__", None)
        module_globals = getattr(module, "__dict__", None)
        dispatch = guard["dispatch"]
        state = guard["state"]
        if not isinstance(clone_globals, dict) or module_globals is not getattr(module, "__dict__", None):
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if getattr(dispatch, "cpu_s4_3g_state", None) is not state:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        original_linear = guard["original_linear"]
        original_apply = guard["original_apply"]
        if getattr(state, "_g_linear_function", None) is not original_linear:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if getattr(state, "_g_nonlinear_function", None) is not original_apply:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if getattr(original_linear, "__code__", None) is not guard["original_linear_code"]:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if getattr(original_apply, "__code__", None) is not guard["original_apply_code"]:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if getattr(original_linear, "__globals__", None) is not guard["original_linear_globals"]:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if getattr(original_apply, "__globals__", None) is not guard["original_apply_globals"]:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        original_linear_globals = guard["original_linear_globals"]
        linear_xp = original_linear_globals.get("xp") if isinstance(original_linear_globals, dict) else None
        linear_fft = getattr(linear_xp, "fft", None)
        if getattr(linear_fft, "dispatch", None) is not guard["original_fft_dispatch"]:
            raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        for name, expected in guard["helpers"].items():
            actual = clone_globals.get(name)
            if actual is not expected["function"]:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if getattr(actual, "__code__", None) is not expected["code"]:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if getattr(actual, "__globals__", None) is not module_globals:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if getattr(actual, "__closure__", None) is not expected["closure"]:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            cells = expected["cells"]
            current_cells = _closure_cells(actual)
            if tuple(sorted(current_cells)) != tuple(sorted(cells)):
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            for role, cell in cells.items():
                if current_cells[role] is not cell:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
                if role == "state" and cell.cell_contents is not state:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
                if role == "original_linear" and cell.cell_contents is not original_linear:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
                if role == "original_apply" and cell.cell_contents is not original_apply:
                    raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
        if str(guard.get("mode")) == QUALIFY_BOUNDARY:
            if clone_globals.get(_HOOK_NAME) is not guard["boundary_hook"]:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
            if clone_globals.get(_LEASE_NAME) is not guard["lease"]:
                raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    except G2AdapterError:
        raise
    except BaseException as exc:
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING") from exc


def _formal_contract() -> dict[str, Any]:
    """Load the only PRE/POST classes admitted by the G2 wrapper.

    The class objects and method records are retained only in this private
    runtime contract.  The JSON-facing receipt uses ``receipt`` below, which
    contains no live objects or callbacks.
    """
    module = importlib.import_module("KHz_filament.hr4e5_formal_entry")
    view_type = getattr(module, "StreamingPulseReadView", None)
    hook_type = getattr(module, "StreamingPulseHook", None)
    if not isinstance(view_type, type) or not isinstance(hook_type, type):
        raise G2AdapterError("G2_FORMAL_TYPES_UNAVAILABLE")
    view_post = view_type.__dict__.get("post_fields")
    hook_call = hook_type.__dict__.get("__call__")
    if not callable(view_post) or not callable(hook_call):
        raise G2AdapterError("G2_FORMAL_METHODS_UNAVAILABLE")
    view_binding = _method_binding(view_post)
    hook_binding = _method_binding(hook_call)
    receipt = {
        "module": str(module.__name__),
        "view_class": f"{view_type.__module__}.{view_type.__qualname__}",
        "hook_class": f"{hook_type.__module__}.{hook_type.__qualname__}",
        "view_post_fields": view_binding,
        "hook_call": hook_binding,
        "required_hook_final": True,
        "required_hook_resume": False,
        "requires_same_view_instance": True,
        "requires_lifecycle_current_identity_match": True,
    }
    return {
        "view_type": view_type,
        "hook_type": hook_type,
        "view_post": view_post,
        "hook_call": hook_call,
        "view_binding": view_binding,
        "hook_binding": hook_binding,
        "receipt": receipt,
    }


def _same_method_binding(function: Any, expected: Mapping[str, Any]) -> bool:
    try:
        actual = _method_binding(function)
    except BaseException:
        return False
    return all(actual.get(key) == expected.get(key) for key in expected)


def _validate_formal_inputs(kwargs: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Reject arbitrary callback/state objects before entering propagation."""
    if not isinstance(kwargs, Mapping):
        raise G2AdapterError("G2_PROPAGATION_KWARGS_NOT_MAPPING")
    if "thermal_slow_state" not in kwargs or "post_commit_hook" not in kwargs:
        raise G2AdapterError("G2_FORMAL_PREPOST_ARGUMENTS_REQUIRED")
    view = kwargs.get("thermal_slow_state")
    hook = kwargs.get("post_commit_hook")
    view_type = contract["view_type"]
    hook_type = contract["hook_type"]
    if type(view) is not view_type:
        raise G2AdapterError("G2_THERMAL_STATE_NOT_ORIGINAL_STREAMING_VIEW")
    if type(hook) is not hook_type:
        raise G2AdapterError("G2_POST_HOOK_NOT_ORIGINAL_STREAMING_HOOK")
    if bool(getattr(hook, "final", False)) is not True:
        raise G2AdapterError("G2_POST_HOOK_FINAL_REQUIRED")
    if bool(getattr(hook, "resume", False)) is True:
        raise G2AdapterError("G2_POST_HOOK_RESUME_FORBIDDEN")
    if getattr(hook, "view", None) is not view:
        raise G2AdapterError("G2_POST_HOOK_VIEW_INSTANCE_MISMATCH")
    lifecycle = getattr(hook, "lifecycle", None)
    view_root = getattr(view, "root", None)
    lifecycle_root = getattr(lifecycle, "root", None)
    if view_root is None or lifecycle_root is None:
        raise G2AdapterError("G2_FORMAL_LIFECYCLE_ROOT_MISSING")
    if Path(view_root).resolve() != Path(lifecycle_root).resolve():
        raise G2AdapterError("G2_FORMAL_LIFECYCLE_ROOT_MISMATCH")
    lifecycle_manifest = getattr(lifecycle, "manifest", None)
    if not isinstance(lifecycle_manifest, Mapping):
        raise G2AdapterError("G2_FORMAL_LIFECYCLE_MANIFEST_MISSING")
    view_generation = getattr(view, "current_generation", None)
    view_content = getattr(view, "current_content_sha256", None)
    lifecycle_generation = lifecycle_manifest.get("current_generation")
    lifecycle_content = lifecycle_manifest.get("current_content_sha256")
    if not view_generation or not view_content or not lifecycle_generation or not lifecycle_content:
        raise G2AdapterError("G2_FORMAL_LIFECYCLE_IDENTITY_MISSING")
    if str(lifecycle_generation) != str(view_generation):
        raise G2AdapterError("G2_FORMAL_LIFECYCLE_GENERATION_MISMATCH")
    if str(lifecycle_content) != str(view_content):
        raise G2AdapterError("G2_FORMAL_LIFECYCLE_CONTENT_MISMATCH")
    bound_post = getattr(view, "post_fields", None)
    if not callable(bound_post) or getattr(bound_post, "__self__", None) is not view:
        raise G2AdapterError("G2_FORMAL_POST_METHOD_BINDING_INVALID")
    if not _same_method_binding(getattr(bound_post, "__func__", bound_post), contract["view_binding"]):
        raise G2AdapterError("G2_FORMAL_POST_METHOD_SOURCE_MISMATCH")
    bound_call = getattr(hook, "__call__", None)
    if not callable(bound_call) or getattr(bound_call, "__self__", None) is not hook:
        raise G2AdapterError("G2_FORMAL_HOOK_METHOD_BINDING_INVALID")
    if not _same_method_binding(getattr(bound_call, "__func__", bound_call), contract["hook_binding"]):
        raise G2AdapterError("G2_FORMAL_HOOK_METHOD_SOURCE_MISMATCH")
    return {
        "view_class": contract["receipt"]["view_class"],
        "hook_class": contract["receipt"]["hook_class"],
        "hook_final": True,
        "hook_resume": False,
        "same_view_instance": True,
        "view_root": str(Path(view_root).resolve()),
        "view_current_generation": str(view_generation),
        "view_current_content_sha256": str(view_content),
        "lifecycle_current_generation": str(lifecycle_generation),
        "lifecycle_current_content_sha256": str(lifecycle_content),
        "formal_post_fields_code_bound": True,
        "formal_hook_call_code_bound": True,
    }


def _clone_i0(
    source_path: Path,
    source: Callable[..., Any],
    lease: _boundary.BoundaryLease | None,
    *,
    mode: str,
    bindings: tuple[tuple[str, str], ...],
    helpers: dict[str, Callable[..., Any]],
) -> tuple[Callable[..., Any], dict[str, Any], dict[str, Any]]:
    source_text = _timing._provenance_v2.canonical_lf_bytes(source_path.read_bytes()).decode("utf-8")
    tree = ast.parse(source_text, filename=str(source_path))
    module = ast.Module(body=[copy.deepcopy(_timing._function(tree, "propagate_one_pulse"))], type_ignores=[])
    ast.fix_missing_locations(module)
    private = dict(getattr(source, "__globals__", {}))
    private.update(helpers)
    interceptor = _make_compile_interceptor(role="I0", mode=mode, bindings=bindings)
    code = interceptor(module, str(source_path), "exec")
    local_namespace: dict[str, Any] = {}
    exec(code, private, local_namespace)
    clone = local_namespace.get("propagate_one_pulse")
    if not callable(clone):
        raise G2AdapterError("I0_G2_CLONE_MISSING")
    receipt = getattr(interceptor, "static_receipt", None)
    if not isinstance(receipt, dict):
        raise G2AdapterError("I0_G2_STATIC_RECEIPT_MISSING")
    return clone, receipt, {"kind": "I0_DIRECT_SOURCE_CLONE"}


def _clone_g(
    source_path: Path,
    source: Callable[..., Any],
    lease: _boundary.BoundaryLease | None,
    *,
    mode: str,
    bindings: tuple[tuple[str, str], ...],
    helpers: dict[str, Callable[..., Any]],
    dispatch: Any,
    state: Any,
) -> tuple[Callable[..., Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    module = importlib.import_module("tools.cpu_s4_3g.controlled_callsite")
    builder = getattr(module, "_clone_function", None)
    if not callable(builder):
        raise G2AdapterError("G3_CLONE_BUILDER_UNAVAILABLE")
    if getattr(builder, "__globals__", None) is not getattr(module, "__dict__", None):
        raise G2AdapterError("G2_RUNTIME_HELPER_BINDING")
    builder_binding = _method_binding(builder)
    builder_globals = dict(getattr(builder, "__globals__", {}))
    interceptor = _make_compile_interceptor(role="G", mode=mode, bindings=bindings)
    builder_globals["compile"] = interceptor
    builder_copy = _copy_function(builder, builder_globals)
    private = dict(getattr(source, "__globals__", {}))
    private.update(helpers)
    clone, identity = builder_copy(source_path, private)
    if not callable(clone):
        raise G2AdapterError("G2_G_CLONE_MISSING")
    receipt = getattr(interceptor, "static_receipt", None)
    if not isinstance(receipt, dict):
        raise G2AdapterError("G2_G_STATIC_RECEIPT_MISSING")
    helper_binding, runtime_guard = _bind_g_runtime_helpers(
        clone,
        module=module,
        builder=builder,
        builder_binding=builder_binding,
        builder_globals=builder_globals,
        dispatch=dispatch,
        state=state,
        lease=lease,
        mode=mode,
    )
    return clone, receipt, {
        "kind": "G3_CONTROLLED_BUILDER",
        "identity": identity,
        "builder_binding": builder_binding,
        "helper_binding": helper_binding,
    }, runtime_guard


class G2Audit:
    """Small read-only object exposing clone and capture provenance."""

    def __init__(self, binding: dict[str, Any], lease: _boundary.BoundaryLease | None):
        self.binding = copy.deepcopy(binding)
        self.lease = lease

    @property
    def receipt(self) -> dict[str, Any] | None:
        return None if self.lease is None else self.lease.result()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA + ".audit",
            "binding": copy.deepcopy(self.binding),
            "capture": None if self.lease is None else self.lease.transparency(),
            "receipt": self.receipt,
        }


def _wrap_propagator(
    clone: Callable[..., Any],
    lease: _boundary.BoundaryLease | None,
    *,
    mode: str,
    formal_contract: Mapping[str, Any],
    timing_guard: Mapping[str, Any],
    runtime_guard: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    invoked = False
    expected_code = getattr(clone, "__code__", None)
    if expected_code is None:
        raise G2AdapterError("G2_ACTUAL_CODE_MISSING_AT_BUILD")

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal invoked
        if getattr(clone, "__code__", None) is not expected_code:
            raise G2AdapterError("G2_ACTUAL_CODE")
        _validate_timing_helper_binding(timing_guard)
        if runtime_guard is not None:
            _validate_g_runtime_binding(runtime_guard)
        if lease is None:
            _validate_formal_inputs(kwargs, formal_contract)
            return clone(*args, **kwargs)
        if invoked:
            raise _boundary.BoundaryLeaseError("G2_QUALIFY_PROPAGATOR_ALREADY_CALLED")
        invoked = True
        try:
            _validate_formal_inputs(kwargs, formal_contract)
            lease.activate()
            result = clone(*args, **kwargs)
            receipt = lease.result()
            field_count_ok = False
            try:
                if isinstance(receipt, dict):
                    field_count_ok = int(receipt.get("field_count", -1)) == int(
                        _boundary.EXPECTED_TOTAL_FIELD_COUNT
                    )
            except (TypeError, ValueError, OverflowError):
                field_count_ok = False
            if (
                not lease.collector.captured
                or not isinstance(receipt, dict)
                or not field_count_ok
                or receipt.get("status") != "PASS_BOUNDARY_CAPTURE_WITH_ISOLATED_SNAPSHOTS"
            ):
                raise G2AdapterError("G2_BOUNDARY_CAPTURE_MISSING")
            return result
        finally:
            lease.close()

    wrapped.__name__ = getattr(clone, "__name__", "propagate_one_pulse")
    wrapped.__qualname__ = getattr(clone, "__qualname__", wrapped.__name__)
    wrapped.__module__ = getattr(clone, "__module__", None)
    wrapped.__doc__ = getattr(clone, "__doc__", None)
    wrapped.__g2_inner__ = clone
    wrapped.__g2_mode__ = str(mode)
    wrapped.__g2_lease__ = lease
    wrapped.__g2_actual_code__ = clone.__code__
    wrapped.__g2_expected_code__ = expected_code
    wrapped.__g2_formal_contract__ = copy.deepcopy(formal_contract["receipt"])
    wrapped.__g2_source_identity__ = copy.deepcopy(
        getattr(clone, "__g2_source_identity__", {})
    )
    wrapped.__j1_inner_code__ = clone.__code__
    return wrapped


def _normalize_role(role: str) -> str:
    value = str(role).upper()
    if value == "G":
        value = "G_CONTROLLED_INPUT"
    if value not in ROLES:
        raise G2AdapterError(f"G2_ROLE_UNSUPPORTED:{role}")
    return value


def _normalize_mode(mode: str) -> str:
    value = str(mode).upper()
    if value not in MODES:
        raise G2AdapterError(f"G2_MODE_UNSUPPORTED:{mode}")
    return value


def _runtime_source_root() -> tuple[Path, str]:
    """Resolve the runner-owned source root without changing import order."""
    if os.name == "nt":
        env_name = "S43F_SOURCE_ROOT"
        configured = os.environ.get(env_name)
        root = Path(configured).resolve() if configured else _REPORT_REPO_ROOT.resolve()
    else:
        env_name = "S43G2_SOURCE_ROOT"
        configured = os.environ.get(env_name)
        if not configured:
            raise G2AdapterError("G2_RUNTIME_SOURCE_ROOT_REQUIRED:S43G2_SOURCE_ROOT")
        root = Path(configured).resolve()
    if not root.is_dir() or not (root / "Filament_python").is_dir():
        raise G2AdapterError(f"G2_RUNTIME_SOURCE_ROOT_INVALID:{env_name}")
    return root, env_name


def _runtime_module_paths(source_root: Path) -> dict[str, str]:
    """Bind the modules used by the role factory to the runner source root."""
    module_names = (
        "KHz_filament.propagate",
        "KHz_filament.linear",
        "KHz_filament.raman",
        "tools.cpu_s4_3g.controlled_callsite",
    )
    paths: dict[str, str] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if not module_file:
                raise G2AdapterError(f"G2_RUNTIME_MODULE_FILE_MISSING:{module_name}")
            path = Path(str(module_file)).resolve(strict=True)
            path.relative_to(source_root)
        except G2AdapterError:
            raise
        except BaseException as exc:
            raise G2AdapterError(f"G2_RUNTIME_MODULE_BINDING_FAILED:{module_name}") from exc
        paths[module_name] = str(path)
    return paths


def build_role(
    role: str,
    mode: str,
    *,
    capture_root: str | Path,
) -> tuple[Callable[..., Any], Any, Any, dict[str, Any], G2Audit, _boundary.BoundaryLease | None]:
    """Build one G2 role without changing production modules.

    ``QUALIFY_BOUNDARY`` returns a one-call propagator and a live lease.  The
    wrapper activates and retires the lease around that call.  The pure mode
    omits the boundary hook and returns a reusable timed clone with ``lease``
    set to ``None``.
    """
    role_name = _normalize_role(role)
    mode_name = _normalize_mode(mode)
    if capture_root is None:
        raise G2AdapterError("G2_CAPTURE_ROOT_REQUIRED")
    mapping = _boundary.load_boundary_mapping()
    bindings = tuple(mapping["diagnostic_bindings"])
    capture_root = Path(capture_root).resolve()
    role_root = capture_root / ("I0" if role_name == "I0" else "G_CONTROLLED_INPUT")
    runtime_source_root, runtime_source_root_env = _runtime_source_root()
    runtime_module_paths = _runtime_module_paths(runtime_source_root)

    # The frozen factory is the only route to the real source/state/dispatch.
    propagate_module = importlib.import_module("KHz_filament.propagate")
    propagate_one_pulse = getattr(propagate_module, "propagate_one_pulse", None)
    if not callable(propagate_one_pulse):
        raise G2AdapterError("G2_RUNTIME_PROPAGATOR_MISSING")

    factory = _common.factory(role_name)
    base_prop, dispatch = factory(propagate_one_pulse)
    source_path = _source_path(base_prop)
    _timing._validate_original_source(source_path)
    source_identity = copy.deepcopy(
        getattr(base_prop, "__cpu_s4_3g_source_binding__", None)
        or _timing._propagate_identity(source_path)
    )

    state = getattr(dispatch, "cpu_s4_3g_state", None) if role_name == "G_CONTROLLED_INPUT" else dispatch
    if state is None:
        raise G2AdapterError("G2_ROLE_STATE_MISSING")
    generation_getter = (
        (lambda: int(getattr(state, "_generation")))
        if role_name == "G_CONTROLLED_INPUT"
        else (lambda: 0)
    )
    initial_generation = int(generation_getter())

    lease = None
    if mode_name == QUALIFY_BOUNDARY:
        expected_capture_generation = (
            initial_generation + 2
            if role_name == "G_CONTROLLED_INPUT"
            else initial_generation
        )
        lease = _boundary.make_boundary_lease(
            role_name,
            diagnostic_bindings=bindings,
            source_identity=source_identity,
            mapping_identity=copy.deepcopy(mapping["identity"]),
            snapshot_root=role_root,
            state=state,
            generation=initial_generation,
            generation_getter=generation_getter,
            expected_capture_generation=expected_capture_generation,
        )

    timed_role = "G" if role_name == "G_CONTROLLED_INPUT" else "I0"
    clock = _timing.WindowClock(timed_role)
    timing_helpers: dict[str, Callable[..., Any]] = dict(_timing._clock_helpers(clock))
    helpers: dict[str, Callable[..., Any]] = dict(timing_helpers)
    if lease is not None:
        helpers[_HOOK_NAME] = _boundary.explicit_boundary_capture
        # The lease is a private scalar capability; no caller can supply it
        # through the public propagation kwargs.
        helpers[_LEASE_NAME] = lease
    runtime_guard = None
    if role_name == "G_CONTROLLED_INPUT":
        clone, static_receipt, clone_receipt, runtime_guard = _clone_g(
            source_path,
            base_prop,
            lease,
            mode=mode_name,
            bindings=bindings,
            helpers=helpers,
            dispatch=dispatch,
            state=state,
        )
    else:
        clone, static_receipt, clone_receipt = _clone_i0(
            source_path,
            base_prop,
            lease,
            mode=mode_name,
            bindings=bindings,
            helpers=helpers,
        )

    clone.__dict__.update(dict(getattr(base_prop, "__dict__", {})))
    clone.__j1_inner_code__ = clone.__code__
    clone.__timing_original_code__ = getattr(base_prop, "__code__", None)
    clone.__timing_original_j1_inner_code__ = getattr(
        base_prop, "__j1_inner_code__", getattr(base_prop, "__code__", None)
    )
    clone.__timing_role__ = timed_role
    clone.__timing_actual_code__ = clone.__code__
    clone.__g2_actual_code__ = clone.__code__
    clone.__g2_source_identity__ = copy.deepcopy(source_identity)
    clone.__g2_mode__ = mode_name
    clone.__g2_role__ = role_name
    if role_name == "G_CONTROLLED_INPUT":
        clone.__cpu_s4_3g_state__ = state
    if lease is not None:
        lease.bind_code(clone.__code__)
    timing_receipt, timing_guard = _bind_timing_helpers(
        clone,
        timing_helpers,
        role=role_name,
        dispatch=dispatch,
        state=state,
    )
    static = copy.deepcopy(static_receipt)
    boundary_static = static.get("boundary")
    binding = {
        "schema": SCHEMA,
        "role": role_name,
        "mode": mode_name,
        "source_identity": source_identity,
        "runtime_source_root": str(runtime_source_root),
        "runtime_source_root_env": runtime_source_root_env,
        "runtime_module_paths": runtime_module_paths,
        "clone_builder": clone_receipt,
        "runtime_helper_binding": (
            copy.deepcopy(clone_receipt.get("helper_binding"))
            if role_name == "G_CONTROLLED_INPUT"
            else None
        ),
        "timing_helper_binding": timing_receipt,
        "actual_code": _actual_code_record(clone),
        "base_code": _actual_code_record(base_prop),
        "static_receipt": static,
        "boundary": boundary_static,
        "mapping": {
            **copy.deepcopy(mapping["identity"]),
            "field_count": int(mapping["field_count"]),
            "diagnostic_count": len(bindings),
        },
        "lease_scope": {
            "thread_bound": True,
            "state_bound": True,
            "generation_at_build": initial_generation,
            "expected_capture_generation": (
                None
                if lease is None
                else lease.expected_capture_generation
            ),
            "capture_once_interval_zero": mode_name == QUALIFY_BOUNDARY,
            "pure_off_capture_nodes": mode_name == PURE_OFF_REAL8,
        },
    }
    formal_contract = _formal_contract()
    wrapped = _wrap_propagator(
        clone,
        lease,
        mode=mode_name,
        formal_contract=formal_contract,
        timing_guard=timing_guard,
        runtime_guard=runtime_guard,
    )
    binding["formal_contract"] = copy.deepcopy(formal_contract["receipt"])
    wrapped.__g2_binding__ = copy.deepcopy(binding)
    audit = G2Audit(binding, lease)
    return wrapped, dispatch, clock, binding, audit, lease


__all__ = [
    "SCHEMA",
    "QUALIFY_BOUNDARY",
    "PURE_OFF_REAL8",
    "G2AdapterError",
    "G2Audit",
    "build_role",
]
