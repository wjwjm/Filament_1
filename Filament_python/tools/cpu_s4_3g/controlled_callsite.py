"""Private call-site and ownership bridge for the CPU-S4-3G candidate.

This module deliberately contains no Raman or propagation expressions.  It
only binds the frozen propagation source to a private state object and keeps
the lifetime contract for a field returned by the original BK-NEE producer.
The capability objects never cross the public ``full_heun``/stage/apply API.
"""

from __future__ import annotations

import ast
import copy
import importlib
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np

from tools.hpc_ops import provenance_v2 as _provenance_v2


PROPAGATE_RELATIVE_PATH = "Filament_python/KHz_filament/propagate.py"
PROPAGATE_SOURCE_BLOB = "f156111398415c2b92e4609ccc1da075fcf5d1e3"
PROPAGATE_CANONICAL_LF_SHA256 = (
    "909fbcf5ef05d2c50b8ed27506b450d40127b3973709d9b4b6a961799dc8d1f6"
)

_BK_LINES = (649, 1248)
_APPLY_LINE = 963
_RAMAN_STAGE_LINE = 617
_RAMAN_APPLY_LINE = 618

_LINEAR_RELATIVE_PATH = "Filament_python/KHz_filament/linear.py"
_LINEAR_SOURCE_BLOB = "72878dd20daae656a5f9c8c7e39fa955ae8b30c5"
_LINEAR_CANONICAL_LF_SHA256 = (
    "93999555bf605c0be0d1867413a3d685ce8a8bfb781534401f96d5f9a388ce30"
)
_LINEAR_FUNCTION_NAME = "step_linear_bk_nee_factorized"
_LINEAR_FUNCTION_FIRST_LINE = 76
_NONLINEAR_RELATIVE_PATH = "Filament_python/KHz_filament/nonlinear.py"
_NONLINEAR_SOURCE_BLOB = "7674c50ef2d7fe90c805d068e4f4ba535e215dfd"
_NONLINEAR_CANONICAL_LF_SHA256 = (
    "cb55ec00bbdf7196c926c3563785efe65bd83e26cc767d67f42c6d4db5d57185"
)
_NONLINEAR_FUNCTION_NAME = "apply_nonlinear"
_NONLINEAR_FUNCTION_FIRST_LINE = 34
_MONITORING_MAX_TOOL_ID = 5
_BK_SCALAR_NAMES = (
    "k0",
    "omega0",
    "beta2",
    "denom_floor",
    "precision_strategy",
    "return_energy_diagnostics",
    "energy_scale",
    "return_profile_diagnostics",
)
_OWNERSHIP_METADATA_LIMIT_BYTES = 16 * 1024


class CallsiteBindingError(RuntimeError):
    """The frozen propagation call-site or backend binding is unavailable."""


class _ControlledCapability:
    """Opaque single-consumer capability issued by the bound BK producer."""

    __slots__ = (
        "_owner",
        "_field",
        "_scope",
        "_generation",
        "_thread_id",
        "_shape",
        "_dtype",
        "_strides",
        "_pointer",
        "_consuming",
        "_consumed",
        "_retired",
        "_config_token",
        "_site",
    )

    def __init__(
        self,
        owner: Any,
        field: np.ndarray,
        *,
        scope: str,
        generation: int,
        thread_id: int,
        config_token: Any,
        site: int,
    ) -> None:
        self._owner = owner
        self._field = field
        self._scope = str(scope)
        self._generation = int(generation)
        self._thread_id = int(thread_id)
        self._shape = tuple(int(s) for s in field.shape)
        self._dtype = field.dtype.str
        self._strides = tuple(int(s) for s in field.strides)
        self._pointer = int(field.__array_interface__["data"][0])
        self._consuming = False
        self._consumed = False
        self._retired = False
        self._config_token = config_token
        self._site = int(site)

    def __repr__(self) -> str:  # pragma: no cover - defensive opaque surface
        return "<controlled-capability opaque>"


def _field_layout(field: Any) -> tuple[tuple[int, ...], str, tuple[int, ...], int]:
    if type(field) is not np.ndarray:
        raise CallsiteBindingError("CONTROLLED_FIELD_TYPE_OR_SUBCLASS")
    if field.ndim != 3 or field.dtype != np.dtype("complex128"):
        raise CallsiteBindingError("CONTROLLED_FIELD_DTYPE_OR_NDIM")
    if not field.flags.c_contiguous or any(int(s) <= 0 for s in field.strides):
        raise CallsiteBindingError("CONTROLLED_FIELD_LAYOUT")
    return (
        tuple(int(s) for s in field.shape),
        field.dtype.str,
        tuple(int(s) for s in field.strides),
        int(field.__array_interface__["data"][0]),
    )


def _array_byte_interval(value: np.ndarray) -> tuple[int, int]:
    """Compute a strided NumPy view's byte bounds from O(1) metadata."""
    pointer = int(value.__array_interface__["data"][0])
    if int(value.size) == 0:
        return pointer, pointer
    low = pointer
    high = pointer
    for extent, stride in zip(value.shape, value.strides):
        delta = (int(extent) - 1) * int(stride)
        if delta < 0:
            low += delta
        else:
            high += delta
    return low, high + int(value.dtype.itemsize)


def _array_in_value(value: Any, field: np.ndarray) -> bool:
    """Check only object/alias metadata in a small diagnostic value."""
    if isinstance(value, np.ndarray):
        if value is field:
            return True
        # Both endpoints are native NumPy buffers.  Compare their byte
        # intervals directly so ownership validation remains O(1) and never
        # asks NumPy to scan arbitrary strided views.
        try:
            field_start, field_stop = _array_byte_interval(field)
            value_start, value_stop = _array_byte_interval(value)
        except Exception as exc:  # pragma: no cover - NumPy backend dependent
            raise CallsiteBindingError("DIAGNOSTIC_ALIAS_METADATA_UNAVAILABLE") from exc
        return field_start < value_stop and value_start < field_stop
    if isinstance(value, Mapping):
        return any(_array_in_value(item, field) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_array_in_value(item, field) for item in value)
    return False


def _extract_linear_field(result: Any, *, energy: bool, profile: bool) -> np.ndarray:
    """Mirror the four frozen BK-NEE return forms without copying."""
    if energy and profile:
        if not isinstance(result, tuple) or len(result) != 3:
            raise CallsiteBindingError("BK_RETURN_FORM_ENERGY_AND_PROFILE")
        field = result[0]
        diagnostics = result[1:]
    elif energy:
        if not isinstance(result, tuple) or len(result) != 2:
            raise CallsiteBindingError("BK_RETURN_FORM_ENERGY")
        field = result[0]
        diagnostics = (result[1],)
    elif profile:
        if not isinstance(result, tuple) or len(result) != 2:
            raise CallsiteBindingError("BK_RETURN_FORM_PROFILE")
        field = result[0]
        diagnostics = (result[1],)
    else:
        field = result
        diagnostics = ()
    if type(field) is not np.ndarray:
        raise CallsiteBindingError("BK_OUTPUT_NOT_NATIVE_NDARRAY")
    _field_layout(field)
    if any(_array_in_value(item, field) for item in diagnostics):
        raise CallsiteBindingError("BK_DIAGNOSTIC_RETAINS_RAW_FIELD")
    return field


def _storage_overlaps(first: np.ndarray, second: np.ndarray) -> bool:
    """Return a conservative O(1) byte-interval overlap result."""
    first_start, first_stop = _array_byte_interval(first)
    second_start, second_stop = _array_byte_interval(second)
    return first_start < second_stop and second_start < first_stop


def _function_source_binding(
    function: Callable[..., Any],
    *,
    relative_path: str,
    source_blob: str,
    canonical_lf_sha256: str,
    function_name: str,
    function_first_line: int,
    module_name: str,
) -> dict[str, Any]:
    """Bind a producer/writer to its frozen source and exact code object.

    File identity is checked once while the factory is built.  Propagation
    entry checks only the returned code object identity, so no source bytes are
    rehashed in the scientific interval.
    """
    code = getattr(function, "__code__", None)
    if code is None:
        raise CallsiteBindingError(f"{function_name.upper()}_CODE_MISSING")
    root = Path(__file__).resolve().parents[3]
    source_path = (root / relative_path).resolve(strict=True)
    if Path(str(code.co_filename)).resolve() != source_path:
        raise CallsiteBindingError(f"{function_name.upper()}_CODE_FILENAME_DRIFT")
    if str(code.co_name) != function_name or int(code.co_firstlineno) != int(function_first_line):
        raise CallsiteBindingError(f"{function_name.upper()}_CODE_ANCHOR_DRIFT")
    raw = source_path.read_bytes()
    canonical = _provenance_v2.canonical_lf_sha256(raw)
    if canonical != canonical_lf_sha256:
        raise CallsiteBindingError(f"{function_name.upper()}_SOURCE_CANONICAL_DRIFT")
    record = {
        "path": relative_path,
        "classification": "tracked_text",
        "hash_scope": _provenance_v2.TRACKED_HASH_SCOPE,
        "git_blob_oid": source_blob,
        "canonical_lf_sha256": canonical,
    }
    try:
        _provenance_v2.validate_record(root, record, require_hash_scope=True)
    except Exception as exc:
        raise CallsiteBindingError(f"{function_name.upper()}_SOURCE_BLOB_DRIFT:{exc}") from exc
    source_module = importlib.import_module(module_name)
    source_function = getattr(source_module, function_name, None)
    source_code = getattr(source_function, "__code__", None)
    if source_code is None or code is not source_code:
        raise CallsiteBindingError(f"{function_name.upper()}_CODE_OBJECT_IDENTITY_DRIFT")
    return {
        "path": relative_path,
        "git_blob_oid": source_blob,
        "canonical_lf_sha256": canonical,
        "function_name": function_name,
        "function_first_line": int(function_first_line),
        "code_filename": str(Path(str(code.co_filename)).resolve()),
        "code_object_id": int(id(code)),
    }


def _array_object_metadata(value: Any, name: str) -> tuple[Any, ...]:
    """Capture O(1) native-array identity/layout metadata for a BK parameter."""
    if type(value) is not np.ndarray:
        raise CallsiteBindingError(f"BK_{name.upper()}_NOT_NATIVE_NDARRAY")
    expected_ndim = 1 if name == "Omega" else 2
    if (
        value.ndim != expected_ndim
        or value.dtype != np.dtype("float64")
        or any(int(s) <= 0 for s in value.strides)
        or not value.flags.c_contiguous
    ):
        raise CallsiteBindingError(f"BK_{name.upper()}_DTYPE_OR_STRIDES")
    return (
        int(id(value)),
        value.dtype.str,
        tuple(int(s) for s in value.shape),
        tuple(int(s) for s in value.strides),
        int(value.__array_interface__["data"][0]),
        bool(value.flags.c_contiguous),
        bool(value.flags.writeable),
    )


def _scalar_token(value: Any, name: str) -> tuple[str, str]:
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple, dict, set)):
        raise CallsiteBindingError(f"BK_{name.upper()}_MUST_BE_SCALAR")
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise CallsiteBindingError(f"BK_{name.upper()}_NONFINITE")
    return type(value).__name__, repr(value)


def _bk_scalar_config(kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
    defaults = {
        "precision_strategy": "baseline_complex64",
        "return_energy_diagnostics": False,
        "energy_scale": None,
        "return_profile_diagnostics": False,
    }
    return tuple(
        (name, _scalar_token(kwargs.get(name, defaults.get(name)), name))
        for name in _BK_SCALAR_NAMES
    )


def _bounded_metadata_size(value: Any, *, _depth: int = 0) -> int:
    """Measure scalar/container ownership metadata while excluding arrays."""
    if isinstance(value, np.ndarray):
        return 0
    size = int(sys.getsizeof(value))
    if _depth >= 3:
        return size
    if isinstance(value, Mapping):
        return size + sum(
            _bounded_metadata_size(key, _depth=_depth + 1)
            + _bounded_metadata_size(item, _depth=_depth + 1)
            for key, item in list(value.items())[:32]
        )
    if isinstance(value, (tuple, list)):
        return size + sum(
            _bounded_metadata_size(item, _depth=_depth + 1)
            for item in list(value)[:32]
        )
    return size


def _backend_binding(
    linear_function: Callable[..., Any],
    nonlinear_function: Callable[..., Any],
) -> dict[str, Any]:
    """Require the ordinary source-bound J2 SciPy FFT dispatch.

    The class-name/module checks are intentional.  A capture dispatch, a
    profiler/trace object, or an unknown proxy does not establish the G
    lifetime contract and must stop before scientific work.
    """
    globals_dict = getattr(linear_function, "__globals__", None)
    if not isinstance(globals_dict, dict):
        raise CallsiteBindingError("LINEAR_GLOBAL_NAMESPACE_UNKNOWN")
    xp = globals_dict.get("xp")
    if type(xp).__name__ != "_XPNamespace":
        raise CallsiteBindingError("LINEAR_XP_NAMESPACE_UNKNOWN")
    xp_module = type(xp).__module__.replace("\\", ".")
    if not xp_module.endswith("cpu_p1r2_j2"):
        raise CallsiteBindingError("LINEAR_XP_NAMESPACE_NOT_J2")
    fft = getattr(xp, "fft", None)
    dispatch = getattr(fft, "dispatch", None)
    if type(dispatch).__name__ == "_CaptureDispatch":
        raise CallsiteBindingError("CAPTURE_DISPATCH_FORBIDDEN")
    if type(dispatch).__name__ != "_ScipyDispatch":
        raise CallsiteBindingError("LINEAR_FFT_DISPATCH_NOT_SCIPY")
    dispatch_module = type(dispatch).__module__.replace("\\", ".")
    if not dispatch_module.endswith("cpu_p1r2_j2"):
        raise CallsiteBindingError("LINEAR_FFT_DISPATCH_NOT_J2")
    for name in (
        "observe",
        "observer",
        "trace",
        "trace_callback",
        "callback",
        "internal_callback",
        "_internal_callback",
        "observer_callback",
        "_observer_callback",
    ):
        if globals_dict.get(name) is not None:
            raise CallsiteBindingError(f"UNKNOWN_OBSERVER_OR_TRACE:{name}")
    linear_source = _function_source_binding(
        linear_function,
        relative_path=_LINEAR_RELATIVE_PATH,
        source_blob=_LINEAR_SOURCE_BLOB,
        canonical_lf_sha256=_LINEAR_CANONICAL_LF_SHA256,
        function_name=_LINEAR_FUNCTION_NAME,
        function_first_line=_LINEAR_FUNCTION_FIRST_LINE,
        module_name="KHz_filament.linear",
    )
    nonlinear_source = _function_source_binding(
        nonlinear_function,
        relative_path=_NONLINEAR_RELATIVE_PATH,
        source_blob=_NONLINEAR_SOURCE_BLOB,
        canonical_lf_sha256=_NONLINEAR_CANONICAL_LF_SHA256,
        function_name=_NONLINEAR_FUNCTION_NAME,
        function_first_line=_NONLINEAR_FUNCTION_FIRST_LINE,
        module_name="KHz_filament.nonlinear",
    )
    return {
        "xp_namespace_type": type(xp).__name__,
        "xp_namespace_module": xp_module,
        "dispatch_type": type(dispatch).__name__,
        "dispatch_module": dispatch_module,
        "dispatch_id": int(id(dispatch)),
        "capture_dispatch_rejected": True,
        "linear_source_binding": linear_source,
        "nonlinear_source_binding": nonlinear_source,
    }


def _propagate_identity(path: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    canonical = _provenance_v2.canonical_lf_sha256(raw)
    if canonical != PROPAGATE_CANONICAL_LF_SHA256:
        raise CallsiteBindingError(
            f"propagate canonical source drift: {canonical} != {PROPAGATE_CANONICAL_LF_SHA256}"
        )
    root_result = __import__("subprocess").run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if root_result.returncode != 0:
        raise CallsiteBindingError("PROPAGATE_GIT_ROOT_UNAVAILABLE")
    root = Path(root_result.stdout.strip()).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise CallsiteBindingError("PROPAGATE_SOURCE_OUTSIDE_GIT_ROOT") from exc
    if relative != PROPAGATE_RELATIVE_PATH:
        raise CallsiteBindingError(f"PROPAGATE_RELATIVE_PATH_DRIFT:{relative}")
    validation_record = {
        "path": relative,
        "classification": "tracked_text",
        "hash_scope": _provenance_v2.TRACKED_HASH_SCOPE,
        "git_blob_oid": PROPAGATE_SOURCE_BLOB,
        "canonical_lf_sha256": canonical,
    }
    try:
        _provenance_v2.validate_record(root, validation_record, require_hash_scope=True)
    except Exception as exc:
        raise CallsiteBindingError(f"PROPAGATE_GIT_BLOB_DRIFT:{exc}") from exc
    return {
        **validation_record,
        "raw_sha256": _provenance_v2.raw_sha256(raw),
        "size_bytes": int(len(raw)),
    }


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise CallsiteBindingError(f"PROPAGATE_FUNCTION_ANCHOR:{name}:{len(matches)}")
    return matches[0]


def _clone_function(source_path: Path, namespace: dict[str, Any]) -> tuple[Callable[..., Any], dict[str, Any]]:
    """AST-clone only the frozen propagate function and four call-site bridges."""
    identity = _propagate_identity(source_path)
    source = _provenance_v2.canonical_lf_bytes(source_path.read_bytes()).decode("utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function_node = copy.deepcopy(_find_function(tree, "propagate_one_pulse"))
    nested = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.FunctionDef) and node.name == "_full_raman_substep"
    ]
    if len(nested) != 1:
        raise CallsiteBindingError("RAMAN_SUBSTEP_ANCHOR_NOT_UNIQUE")
    nested_fn = nested[0]
    branch = [
        node
        for node in nested_fn.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "full_isaacs_complete_mode"
    ]
    if len(branch) != 1 or not branch[0].orelse:
        raise CallsiteBindingError("RAMAN_ROTATIONAL_BRANCH_ANCHOR_DRIFT")
    raman_if = branch[0]
    bk_calls = sorted(
        (
            node
            for node in ast.walk(function_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "step_linear_bk_nee_factorized"
        ),
        key=lambda node: int(node.lineno),
    )
    if tuple(int(node.lineno) for node in bk_calls) != _BK_LINES:
        raise CallsiteBindingError(
            f"BK_CALLSITE_ANCHOR_DRIFT:{tuple(int(node.lineno) for node in bk_calls)}"
        )
    apply_calls = [
        node
        for node in ast.walk(function_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_nonlinear"
    ]
    if len(apply_calls) != 1 or int(apply_calls[0].lineno) != _APPLY_LINE:
        raise CallsiteBindingError("NONLINEAR_CALLSITE_ANCHOR_DRIFT")
    stage_calls = [
        node
        for node in ast.walk(nested_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isaacs_raman_stage"
    ]
    apply_raman_calls = [
        node
        for node in ast.walk(nested_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_isaacs_raman_operator_step"
    ]
    if [int(node.lineno) for node in stage_calls] != [_RAMAN_STAGE_LINE]:
        raise CallsiteBindingError("RAMAN_STAGE_ANCHOR_DRIFT")
    if [int(node.lineno) for node in apply_raman_calls] != [_RAMAN_APPLY_LINE]:
        raise CallsiteBindingError("RAMAN_APPLY_ANCHOR_DRIFT")

    for site, node in enumerate(bk_calls):
        node.func = ast.Name(id="_g_step_linear_bk_nee", ctx=ast.Load())
        node.args.insert(0, ast.Constant(value=int(site)))
    apply_calls[0].func = ast.Name(id="_g_apply_nonlinear", ctx=ast.Load())

    complete_guard = ast.parse("_g_reject_complete_mode()", mode="eval").body
    raman_if.body.insert(0, ast.Expr(value=ast.copy_location(complete_guard, raman_if.test)))
    replacement = ast.parse(
        """
first_stage, updated, diagnostics = _g_controlled_raman_substep(
    field,
    step_length,
    stage_kwargs,
    integrator=getattr(raman_conf, 'operator_integrator', 'heun'),
    transverse_cell_area=axes.dx * axes.dy,
)
""",
        mode="exec",
    ).body
    raman_if.orelse = [ast.copy_location(node, raman_if.orelse[0]) for node in replacement]

    # The entry check executes before any source-side field/FFT work.  It
    # validates only scalar/configuration metadata and never scans ``E``.
    entry_init = ast.parse("_g_entry_token = None", mode="exec").body[0]
    entry_call = ast.parse(
        "_g_entry_token = _g_propagation_enter(E, prop_conf, raman_conf)",
        mode="exec",
    ).body[0]
    insert_at = 0
    if function_node.body and isinstance(function_node.body[0], ast.Expr) and isinstance(
        function_node.body[0].value, ast.Constant
    ) and isinstance(function_node.body[0].value.value, str):
        insert_at = 1
    direct_returns = [node for node in function_node.body if isinstance(node, ast.Return)]
    if len(direct_returns) != 1 or int(direct_returns[0].lineno) != 2133:
        raise CallsiteBindingError("PROPAGATE_FINAL_RETURN_ANCHOR_DRIFT")
    direct_returns[0].value = ast.Call(
        func=ast.Name(id="_g_publish_result", ctx=ast.Load()),
        args=[direct_returns[0].value],
        keywords=[],
    )
    # Install the lexical finally only after entry succeeds.  A rejected
    # nested/cross-thread entry therefore has no cleanup frame that could
    # touch an outer active propagation; later cleanup remains token-gated.
    original_body = list(function_node.body[insert_at:])
    cleanup_call = ast.parse(
        "_g_propagation_finalize(_g_entry_token)", mode="exec"
    ).body[0]
    guarded_body = ast.Try(
        body=original_body,
        handlers=[],
        orelse=[],
        finalbody=[cleanup_call],
    )
    function_node.body = list(function_node.body[:insert_at]) + [entry_init, entry_call, guarded_body]
    ast.fix_missing_locations(function_node)
    module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(module)
    local_namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), namespace, local_namespace)
    clone = local_namespace.get("propagate_one_pulse")
    if not callable(clone):
        raise CallsiteBindingError("PROPAGATE_AST_CLONE_DID_NOT_PRODUCE_FUNCTION")
    return clone, identity


def _make_state_helpers(state: Any, original_linear: Callable[..., Any], original_apply: Callable[..., Any]):
    """Create per-state closures installed only in the cloned propagate globals."""

    def step_linear(site: int, *args: Any, **kwargs: Any) -> Any:
        return state._g_linear_call(int(site), original_linear, args, kwargs)

    def apply_nonlinear(*args: Any, **kwargs: Any) -> Any:
        return state._g_nonlinear_call(original_apply, args, kwargs)

    def controlled_substep(
        field: Any,
        step_length: Any,
        stage_kwargs: Mapping[str, Any],
        *,
        integrator: str,
        transverse_cell_area: float,
    ) -> tuple[Any, Any, Any]:
        return state._g_controlled_substep(
            field,
            step_length,
            stage_kwargs,
            integrator=integrator,
            transverse_cell_area=transverse_cell_area,
        )

    def propagation_enter(E: Any, prop_conf: Any, raman_conf: Any) -> Any:
        return state._g_propagation_enter(E, prop_conf, raman_conf)

    def publish_result(value: Any) -> Any:
        return state._g_publish_result(value)

    def propagation_finalize(entry_token: Any) -> None:
        state._g_propagation_finalize(entry_token)

    def reject_complete() -> None:
        state._reject("complete_operator_not_supported")

    return {
        "_g_step_linear_bk_nee": step_linear,
        "_g_apply_nonlinear": apply_nonlinear,
        "_g_controlled_raman_substep": controlled_substep,
        "_g_propagation_enter": propagation_enter,
        "_g_propagation_finalize": propagation_finalize,
        "_g_publish_result": publish_result,
        "_g_reject_complete_mode": reject_complete,
    }


def bind_propagator(
    base: Callable[..., Any],
    state: Any,
    *,
    source_path: Path,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Bind one G state to a source-verified transformed propagator."""
    if not callable(base):
        raise CallsiteBindingError("BASE_PROPAGATOR_NOT_CALLABLE")
    base_globals = getattr(base, "__globals__", None)
    if not isinstance(base_globals, dict):
        raise CallsiteBindingError("BASE_PROPAGATOR_GLOBAL_NAMESPACE_UNKNOWN")
    original_linear = base_globals.get("step_linear_bk_nee_factorized")
    original_apply = base_globals.get("apply_nonlinear")
    if not callable(original_linear) or not callable(original_apply):
        raise CallsiteBindingError("BOUND_PROPAGATOR_LACKS_LINEAR_OR_NONLINEAR")
    backend = _backend_binding(original_linear, original_apply)
    bind_function = getattr(state, "_g_bind_function_identity", None)
    if not callable(bind_function):
        raise CallsiteBindingError("G_FUNCTION_BINDING_HOOK_MISSING")
    bind_function(original_linear, original_apply, backend)
    namespace = dict(base_globals)
    namespace.update(_make_state_helpers(state, original_linear, original_apply))
    clone, identity = _clone_function(source_path, namespace)
    clone.__kwdefaults__ = getattr(base, "__kwdefaults__", None)
    clone.__annotations__ = getattr(base, "__annotations__", {})
    clone.__module__ = getattr(base, "__module__", "KHz_filament.propagate")
    clone.__qualname__ = "propagate_one_pulse"
    # J1 must observe the actual transformed call body.  Retain both frozen
    # source code objects separately for provenance; never advertise the old
    # base code as the callable's active observer code.
    clone.__j1_inner_code__ = clone.__code__
    clone.__cpu_s4_3g_original_code__ = getattr(base, "__code__", None)
    clone.__cpu_s4_3g_original_j1_inner_code__ = getattr(
        base, "__j1_inner_code__", getattr(base, "__code__", None)
    )
    clone.__cpu_s4_3g_backend_binding__ = copy.deepcopy(backend)
    clone.__cpu_s4_3g_source_binding__ = copy.deepcopy(identity)
    clone.__cpu_s4_3g_state__ = state
    for name in (
        "_cpu_o1_execution",
        "_cpu_o1_candidate_events",
        "_cpu_o1_rate_workspace",
        "_cpu_s4_3c_state",
        "_cpu_s4_3c_candidate",
        "_cpu_s4_3c_execution",
        "_cpu_s4_3c_source_identity",
    ):
        if hasattr(base, name):
            setattr(clone, name, getattr(base, name))
    return clone, {"propagate": identity, "backend": backend}


def source_binding(path: Path) -> dict[str, Any]:
    """Publicly useful source record without importing the scientific path."""
    return _propagate_identity(path)


def capability_type() -> type:
    """Return the private capability type for bounded identity checks."""
    return _ControlledCapability


def make_controlled_state_class(
    e_module: ModuleType,
    *,
    candidate_schema: str,
    candidate_unsupported: type[BaseException],
    handoff_rejected: type[BaseException],
) -> type:
    """Build the G-local E handoff subclass inside the private E namespace.

    The class is produced after loading E under a G-derived module name.  No
    class, cache, or method in the frozen E/F modules is modified.
    """
    base = getattr(e_module, "_HandoffState", None)
    call_context = getattr(e_module, "_CallContext", None)
    if not isinstance(base, type) or not isinstance(call_context, type):
        raise CallsiteBindingError("E_HANDOFF_STATE_OR_CONTEXT_MISSING")

    class _ControlledHandoffState(base):
        """G-private owner with a producer-signed pre/post capability."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._g_schema = str(candidate_schema)
            self._g_pending_cap: _ControlledCapability | None = None
            self._g_last_field: np.ndarray | None = None
            # Raman Heun returns a fresh updated field.  Keep the original
            # producer result and current E explicit in the private owner so
            # the nonlinear write gate validates the right object while the
            # old linear_result remains reachable through this interval.
            self._g_retained_linear_result: np.ndarray | None = None
            self._g_e_ref: np.ndarray | None = None
            self._g_owner_ref = self
            self._g_last_scope = "NONE"
            self._g_owner_generation = 0
            self._g_config_token: tuple[Any, ...] | None = None
            self._g_interval_bk_token: tuple[Any, ...] | None = None
            self._g_active_entry_token: object | None = None
            self._g_linear_function: Callable[..., Any] | None = None
            self._g_nonlinear_function: Callable[..., Any] | None = None
            self._g_linear_code: Any = None
            self._g_nonlinear_code: Any = None
            self._g_function_binding: dict[str, Any] = {}
            self._g_context_omega_ref: np.ndarray | None = None
            self._g_context_omega_meta: tuple[Any, ...] | None = None
            self._g_active_cap_scope: str | None = None
            self._g_propagation_active = False
            self._g_segment_started = False
            self._g_segment_start_wall = 0.0
            self._g_segment_start_cpu = 0.0
            self._g_segment_index = 0
            self._g_next_linear_site = 0
            # Qualification capture is an explicit private test hook.  It is
            # disabled for every normal factory/propagation call and never
            # uses an observer or trace callback.
            self._g_qualification_capture_enabled = False
            self._g_qualification_capture_records: list[dict[str, Any]] = []
            self._g_qualification_retained_array_refs = 0
            self._g_qualification_retained_array_bytes = 0
            self._g_metrics = {
                "G_ACTUAL": 0,
                "F_STRICT_FALLBACK": 0,
                "REJECTED": 0,
            }
            self._stats.update(
                {
                    "schema": str(candidate_schema),
                    "candidate_name": "G_CONTROLLED_INPUT",
                    "candidate_family": "RAMAN_CONTROLLED_CALLSITE_BORROW_V1",
                    "controlled_hash_role_order": [
                        "E_stage1",
                        "Omega_stage1",
                        "Omega_apply",
                        "Omega_stage2",
                    ],
                    "controlled_full_calls": 0,
                    "controlled_segment_calls": 0,
                    "controlled_segment_wall_s_total": 0.0,
                    "controlled_segment_process_cpu_s_total": 0.0,
                    "controlled_segment_failed_calls": 0,
                    "controlled_segment_failed_wall_s_total": 0.0,
                    "controlled_segment_failed_process_cpu_s_total": 0.0,
                    "last_controlled_segment_wall_s": 0.0,
                    "last_controlled_segment_process_cpu_s": 0.0,
                    "last_controlled_segment_failed_wall_s": 0.0,
                    "last_controlled_segment_failed_process_cpu_s": 0.0,
                    "controlled_half_calls": 0,
                    "controlled_half_pre_calls": 0,
                    "controlled_half_post_calls": 0,
                    "controlled_half_wall_s_total": 0.0,
                    "controlled_half_process_cpu_s_total": 0.0,
                    "last_controlled_half_scope": "NONE",
                    "last_controlled_half_wall_s": 0.0,
                    "last_controlled_half_process_cpu_s": 0.0,
                    "heun_core_calls": 0,
                    "heun_core_wall_s_total": 0.0,
                    "heun_core_process_cpu_s_total": 0.0,
                    "last_heun_core_wall_s": 0.0,
                    "last_heun_core_process_cpu_s": 0.0,
                    "route_counts": dict(self._g_metrics),
                    "rejected_reasons": {},
                    "controlled_hash_role_counts": {},
                    "last_controlled_hash_roles": [],
                    "last_controlled_hash_events": [],
                    "linear_producer_calls": 0,
                    "linear_producer_sites": {},
                    "post_capability_issues": 0,
                    "ownership_alias_roles": ["linear_result", "E", "owner"],
                }
            )

        def _candidate_unsupported(self, reason: str) -> BaseException:
            return candidate_unsupported(str(reason))

        def _reject(self, reason: str) -> None:
            reason = str(reason)
            self._g_metrics["REJECTED"] += 1
            self._stats["route_counts"] = dict(self._g_metrics)
            reasons = self._stats["rejected_reasons"]
            reasons[reason] = int(reasons.get(reason, 0)) + 1
            raise handoff_rejected(reason)

        def _reject_optimized(self) -> None:
            if int(getattr(sys.flags, "optimize", 0)) != 0 or os.environ.get("PYTHONOPTIMIZE"):
                self._reject("PYTHONOPTIMIZE_OR_O_FLAG_FORBIDDEN")

        def _g_bind_function_identity(
            self,
            linear_function: Callable[..., Any],
            nonlinear_function: Callable[..., Any],
            binding: Mapping[str, Any],
        ) -> None:
            if not callable(linear_function) or not callable(nonlinear_function):
                raise CallsiteBindingError("G_FUNCTION_BINDING_NOT_CALLABLE")
            linear_code = getattr(linear_function, "__code__", None)
            nonlinear_code = getattr(nonlinear_function, "__code__", None)
            if linear_code is None or nonlinear_code is None:
                raise CallsiteBindingError("G_FUNCTION_BINDING_CODE_MISSING")
            self._g_linear_function = linear_function
            self._g_nonlinear_function = nonlinear_function
            self._g_linear_code = linear_code
            self._g_nonlinear_code = nonlinear_code
            self._g_function_binding = copy.deepcopy(dict(binding))

        def _g_assert_function_identity(self) -> None:
            if (
                self._g_linear_function is None
                or self._g_nonlinear_function is None
                or self._g_linear_code is None
                or self._g_nonlinear_code is None
            ):
                raise CallsiteBindingError("G_FUNCTION_BINDING_MISSING")
            if getattr(self._g_linear_function, "__code__", None) is not self._g_linear_code:
                raise CallsiteBindingError("G_LINEAR_CODE_OBJECT_CHANGED")
            if getattr(self._g_nonlinear_function, "__code__", None) is not self._g_nonlinear_code:
                raise CallsiteBindingError("G_NONLINEAR_CODE_OBJECT_CHANGED")

        def _g_check_monitoring_state(self) -> None:
            """Reject every Python 3.12 monitoring tool in the OFF-only route.

            ``sys.gettrace``/``getprofile`` do not observe PEP 669 local
            callbacks.  A registered tool is therefore unsupported even when
            its global event mask is zero; checking both registration and the
            bounded global mask keeps the route fail-closed in O(1) time.
            """
            monitoring = getattr(sys, "monitoring", None)
            if monitoring is None:
                return
            get_tool = getattr(monitoring, "get_tool", None)
            get_events = getattr(monitoring, "get_events", None)
            if not callable(get_tool) or not callable(get_events):
                self._reject("UNKNOWN_SYS_MONITORING_API")
            try:
                max_tool_id = int(getattr(monitoring, "MAX_TOOL_ID", _MONITORING_MAX_TOOL_ID))
            except (TypeError, ValueError, OverflowError):
                self._reject("UNKNOWN_SYS_MONITORING_TOOL_RANGE")
            if max_tool_id < 0 or max_tool_id > _MONITORING_MAX_TOOL_ID:
                self._reject("UNKNOWN_SYS_MONITORING_TOOL_RANGE")
            for tool_id in range(max_tool_id + 1):
                try:
                    registered_tool = get_tool(tool_id)
                    event_mask = get_events(tool_id)
                except BaseException as exc:
                    self._reject(f"UNKNOWN_SYS_MONITORING_API:{type(exc).__name__}")
                if registered_tool is not None:
                    # Do not whitelist by the tool's reported name.  Local
                    # callbacks can be active while gettrace/getprofile are
                    # both None, so any registration is unsupported here.
                    self._reject("UNKNOWN_SYS_MONITORING_TOOL")
                try:
                    event_mask_value = int(event_mask)
                except (TypeError, ValueError, OverflowError):
                    self._reject("UNKNOWN_SYS_MONITORING_EVENTS")
                if event_mask_value != 0:
                    self._reject("UNKNOWN_SYS_MONITORING_EVENTS")

        def _g_check_observer_state(self) -> None:
            self._g_check_monitoring_state()
            if sys.gettrace() is not None:
                self._reject("UNKNOWN_SYS_TRACE")
            if sys.getprofile() is not None:
                self._reject("UNKNOWN_SYS_PROFILE")
            if self.observe_callback is not None or getattr(self.r8_state, "observe_callback", None) is not None:
                self._reject("UNKNOWN_OBSERVER_OR_TRACE")
            for namespace in (self._raman_namespace, self._candidate_namespace):
                for name in (
                    "observe",
                    "observer",
                    "trace",
                    "trace_callback",
                    "callback",
                    "internal_callback",
                    "_internal_callback",
                    "observer_callback",
                    "_observer_callback",
                ):
                    if isinstance(namespace, Mapping) and namespace.get(name) is not None:
                        self._reject(f"UNKNOWN_OBSERVER_OR_TRACE:{name}")
                if isinstance(namespace, Mapping):
                    for name, value in namespace.items():
                        lowered = str(name).lower()
                        if value is not None and any(
                            marker in lowered for marker in ("callback", "observer", "trace")
                        ):
                            self._reject(f"UNKNOWN_OBSERVER_OR_TRACE:{name}")

        def _g_check_live_control(self, *, require_active: bool = True) -> None:
            self._reject_optimized()
            if int(threading.get_ident()) != int(self._control_thread_id):
                self._reject("CONTROL_THREAD_MISMATCH")
            if require_active and not self._g_propagation_active:
                self._reject("PROPAGATION_SCOPE_NOT_ACTIVE")
            if bool(getattr(self._local, "g_controlled", False)):
                cap = self._g_pending_cap
                if cap is None or self._g_active_cap_scope is None:
                    self._reject("CONTROLLED_CAPABILITY_SCOPE_MISSING")
                if str(cap._scope) != str(self._g_active_cap_scope):
                    self._reject("CONTROLLED_CAPABILITY_SCOPE_CHANGED")
            try:
                self._g_assert_function_identity()
                self._assert_runtime_binding()
            except BaseException as exc:
                self._reject(f"RUNTIME_OR_FUNCTION_BINDING_REJECTED:{exc}")
            self._g_check_observer_state()

        def _g_ownership_metadata_bytes(self) -> int:
            cap = self._g_pending_cap
            cap_metadata = None
            if cap is not None:
                # Deliberately omit cap._field: array storage is reported only
                # by the qualification counters below, never as ownership
                # metadata.
                cap_metadata = (
                    cap._scope,
                    cap._generation,
                    cap._thread_id,
                    cap._shape,
                    cap._dtype,
                    cap._strides,
                    cap._pointer,
                    cap._consuming,
                    cap._consumed,
                    cap._retired,
                    cap._config_token,
                    cap._site,
                )
            return int(
                _bounded_metadata_size(
                    (
                        cap_metadata,
                        self._g_last_scope,
                        self._g_owner_generation,
                        self._g_config_token,
                        self._g_interval_bk_token,
                        self._g_context_omega_meta,
                    )
                )
            )

        def _g_clear_qualification_capture(self) -> None:
            self._g_qualification_capture_records.clear()
            self._g_qualification_retained_array_refs = 0
            self._g_qualification_retained_array_bytes = 0

        def _config_token(self, kwargs: Mapping[str, Any]) -> tuple[Any, ...]:
            return (
                str(kwargs.get("method", "iir")).lower(),
                str(kwargs.get("iir_sampling", "exact_piecewise_linear")).lower(),
                str(kwargs.get("integrator", "heun")).lower(),
                int(self.block_size),
                int(self.raman_workers),
            )

        def _g_set_qualification_capture(self, enabled: bool) -> None:
            """Private task-only capture switch; never part of public kwargs."""
            if int(threading.get_ident()) != int(self._control_thread_id):
                self._reject("CONTROL_THREAD_MISMATCH")
            if self._g_propagation_active:
                self._reject("CAPTURE_SWITCH_DURING_PROPAGATION")
            self._g_qualification_capture_enabled = bool(enabled)
            if not self._g_qualification_capture_enabled:
                self._g_clear_qualification_capture()

        def _g_take_qualification_capture(self) -> list[dict[str, Any]]:
            """Return and retire refs captured by already-closed scopes."""
            if int(threading.get_ident()) != int(self._control_thread_id):
                self._reject("CONTROL_THREAD_MISMATCH")
            if self._g_propagation_active or self._g_pending_cap is not None:
                self._reject("CAPTURE_READ_DURING_ACTIVE_SCOPE")
            if not self._g_qualification_capture_enabled:
                self._reject("QUALIFICATION_CAPTURE_DISABLED")
            records = list(self._g_qualification_capture_records)
            self._g_clear_qualification_capture()
            return records

        def _g_propagation_enter(self, field: Any, prop_conf: Any, raman_conf: Any) -> object:
            self._g_check_live_control(require_active=False)
            if self._g_propagation_active:
                self._reject("REENTRANT_OR_CONCURRENT_PROPAGATION")
            if prop_conf is None or raman_conf is None:
                self._reject("PROPAGATION_CONFIGURATION_MISSING")
            if type(field) is not np.ndarray:
                self._reject("PROPAGATION_FIELD_NOT_NATIVE_NDARRAY")
            _field_layout(field)
            if str(getattr(prop_conf, "linear_model", "uppe")).lower() != "bk_nee":
                self._reject("linear_model_not_bk_nee")
            if not bool(getattr(prop_conf, "use_raman_full_operator", False)):
                self._reject("full_raman_operator_disabled")
            operator_mode = str(getattr(raman_conf, "operator_mode", "legacy_split")).lower()
            if operator_mode != "full_isaacs_eq27":
                self._reject("operator_mode_not_rotational_full_isaacs_eq27")
            if str(getattr(raman_conf, "method", "iir")).lower() != "iir":
                self._reject("method_not_iir")
            if str(getattr(raman_conf, "iir_sampling", "exact_piecewise_linear")).lower() != "exact_piecewise_linear":
                self._reject("iir_sampling_not_exact_piecewise_linear")
            if str(getattr(raman_conf, "operator_integrator", "heun")).lower() != "heun":
                self._reject("integrator_not_heun")
            if str(getattr(raman_conf, "nonlinear_split_order", "after_other")).lower() != "strang":
                self._reject("split_order_not_strang")
            self._g_propagation_active = True
            self._g_pending_cap = None
            self._g_last_field = None
            self._g_retained_linear_result = None
            self._g_e_ref = field
            self._g_config_token = None
            self._g_context_omega_ref = None
            self._g_context_omega_meta = None
            self._g_active_cap_scope = None
            self._g_last_scope = "NONE"
            self._g_segment_started = False
            self._g_next_linear_site = 0
            self._g_interval_bk_token = None
            token = object()
            self._g_active_entry_token = token
            return token

        def _g_propagation_finalize(self, entry_token: Any) -> None:
            """Retire a failed/early source call without masking its error."""
            # A rejected nested or cross-thread entry receives ``None`` in its
            # own lexical frame, so it cannot finalize/abort another active
            # propagation owned by this state.
            if entry_token is self._g_active_entry_token and self._g_propagation_active:
                self._g_abort_propagation("propagation_exception_or_early_exit")

        def _g_prepare_linear_call(
            self,
            site: int,
            input_field: Any,
            kwargs: Mapping[str, Any],
        ) -> tuple[Any, ...]:
            """Validate all O(1) BK ownership/configuration metadata first."""
            self._g_check_live_control()
            if type(input_field) is not np.ndarray:
                self._reject("BK_INPUT_NOT_NATIVE_NDARRAY")
            _field_layout(input_field)
            if site == 0:
                if input_field is not self._g_e_ref:
                    self._reject("BK_PRE_INPUT_NOT_CURRENT_E")
                if self._g_last_scope not in ("NONE", "interval_closed"):
                    self._reject("BK_PRE_SCOPE_INVALID")
            else:
                if self._g_last_scope != "post_completed" or input_field is not self._g_e_ref:
                    self._reject("BK_POST_INPUT_NOT_CURRENT_E")
                if self._g_segment_started:
                    self._reject("SECOND_BK_BEFORE_POST_OR_SEGMENT_CLOSE")
            strategy = str(kwargs.get("precision_strategy", "baseline_complex64")).lower()
            if strategy != "baseline_complex64":
                self._reject("precision_strategy_not_baseline_complex64")
            try:
                dz = float(kwargs.get("dz"))
            except (TypeError, ValueError) as exc:
                self._reject(f"BK_DZ_NOT_SCALAR:{exc}")
            if not math.isfinite(dz) or dz <= 0.0:
                self._reject("BK_DZ_NONPOSITIVE_OR_NONFINITE")
            omega_meta = _array_object_metadata(kwargs.get("Omega"), "Omega")
            kperp_meta = _array_object_metadata(kwargs.get("kperp2"), "kperp2")
            scalar_config = _bk_scalar_config(kwargs)
            interval_token = (scalar_config, omega_meta, kperp_meta, dz)
            if site == 0:
                self._g_interval_bk_token = interval_token
            elif self._g_interval_bk_token != interval_token:
                self._reject("BK_INTERVAL_CONFIGURATION_OR_PARAMETER_CHANGED")
            return interval_token

        def _g_linear_call(
            self,
            site: int,
            original: Callable[..., Any],
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            self._reject_optimized()
            if site not in (0, 1):
                self._reject("UNKNOWN_BK_CALLSITE")
            if site != int(self._g_next_linear_site):
                self._reject("BK_CALLSITE_ORDER_OR_REPLAY")
            if site == 1 and (
                self._g_last_scope != "post_completed" or self._g_segment_started
            ):
                self._reject("SECOND_BK_BEFORE_POST_OR_SEGMENT_CLOSE")
            if not args:
                self._reject("BK_INPUT_MISSING")
            input_field = args[0]
            self._g_prepare_linear_call(site, input_field, kwargs)
            if site == 0 and not self._g_segment_started:
                self._g_segment_started = True
                self._g_segment_start_wall = time.perf_counter()
                self._g_segment_start_cpu = time.process_time()
            try:
                result = original(*args, **dict(kwargs))
                energy = bool(kwargs.get("return_energy_diagnostics", False))
                profile = bool(kwargs.get("return_profile_diagnostics", False))
                field = _extract_linear_field(result, energy=energy, profile=profile)
                if type(input_field) is not np.ndarray:
                    self._reject("BK_INPUT_NOT_NATIVE_NDARRAY")
                if _storage_overlaps(field, input_field):
                    self._reject("BK_OUTPUT_ALIASES_LINEAR_INPUT")
                if site == 0:
                    self._g_issue_capability(field, scope="pre", site=site)
                    self._g_retained_linear_result = field
                    self._g_e_ref = field
                else:
                    # The second BK half step is outside the Raman lease.  It
                    # closes this interval and leaves its fresh output as the
                    # next interval's ordinary input reference.
                    self._g_pending_cap = None
                    self._g_retained_linear_result = field
                    self._g_e_ref = field
                self._stats["linear_producer_calls"] += 1
                sites = self._stats["linear_producer_sites"]
                sites[str(site)] = int(sites.get(str(site), 0)) + 1
                if site == 1:
                    self._g_next_linear_site = 0
                    self._g_last_scope = "interval_closed"
                else:
                    self._g_next_linear_site = 1
                return result
            except BaseException:
                self._g_abort_propagation("linear_producer_exception")
                raise

        def _g_issue_capability(self, field: np.ndarray, *, scope: str, site: int) -> None:
            _field_layout(field)
            old = self._g_pending_cap
            if old is not None:
                old._retired = True
            self._g_owner_generation += 1
            cap = _ControlledCapability(
                self,
                field,
                scope=str(scope),
                generation=self._g_owner_generation,
                thread_id=int(threading.get_ident()),
                config_token=self._g_config_token,
                site=int(site),
            )
            self._g_pending_cap = cap
            self._g_last_field = field
            if self._g_ownership_metadata_bytes() > _OWNERSHIP_METADATA_LIMIT_BYTES:
                self._reject("OWNERSHIP_METADATA_BOUND_EXCEEDED")

        def _g_validate_capability(self, cap: Any, field: Any) -> None:
            if not isinstance(cap, _ControlledCapability):
                self._reject("CAPABILITY_TYPE_INVALID")
            if cap._scope not in ("pre", "post"):
                self._reject("CAPABILITY_SCOPE_INVALID")
            if cap._owner is not self or cap._retired or cap._consumed or cap._consuming:
                self._reject("CAPABILITY_REPLAY_OR_RETIRED")
            if cap._thread_id != int(threading.get_ident()):
                self._reject("CAPABILITY_THREAD_MISMATCH")
            if cap._generation != self._g_owner_generation:
                self._reject("CAPABILITY_GENERATION_MISMATCH")
            if cap is not self._g_pending_cap:
                self._reject("CAPABILITY_NOT_CURRENT")
            if field is not cap._field:
                self._reject("CAPABILITY_FIELD_OBJECT_MISMATCH")
            shape, dtype, strides, pointer = _field_layout(field)
            if (shape, dtype, strides, pointer) != (
                cap._shape,
                cap._dtype,
                cap._strides,
                cap._pointer,
            ):
                self._reject("CAPABILITY_FIELD_METADATA_MISMATCH")

        def _g_before_nonlinear(self, field: Any) -> None:
            self._g_check_live_control()
            if field is not self._g_e_ref or self._g_last_scope != "pre_completed":
                self._reject("NONLINEAR_INPUT_NOT_COMPLETED_PRE")
            if self._g_pending_cap is not None:
                self._reject("PRE_CAPABILITY_STILL_OPEN")
            self._g_owner_generation += 1
            self._g_last_scope = "pre_invalidated"

        def _g_nonlinear_call(
            self,
            original: Callable[..., Any],
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            if not args:
                self._reject("NONLINEAR_INPUT_MISSING")
            field = args[0]
            self._g_before_nonlinear(field)
            try:
                result = original(*args, **dict(kwargs))
                if result is not field:
                    self._reject("NONLINEAR_RESULT_NOT_IN_PLACE")
                _field_layout(result)
                self._g_e_ref = result
                self._g_issue_capability(result, scope="post", site=-1)
                self._stats["post_capability_issues"] += 1
                return result
            except BaseException:
                self._g_abort_propagation("nonlinear_exception")
                raise

        def _g_controlled_substep(
            self,
            field: Any,
            step_length: Any,
            stage_kwargs: Mapping[str, Any],
            *,
            integrator: str,
            transverse_cell_area: float,
        ) -> tuple[Any, Any, Any]:
            self._g_check_live_control()
            cap = self._g_pending_cap
            if cap is None:
                self._reject("NO_PRODUCER_CAPABILITY")
            if cap._scope not in ("pre", "post"):
                self._reject("CAPABILITY_SCOPE_INVALID")
            if cap._scope == "pre":
                if self._g_next_linear_site != 1 or self._g_segment_started is not True:
                    self._reject("PRE_SCOPE_OR_SITE_INVALID")
            elif (
                self._g_next_linear_site != 1
                or self._g_last_scope != "pre_invalidated"
                or self._g_segment_started is not True
            ):
                self._reject("POST_SCOPE_OR_SITE_INVALID")
            kwargs = dict(stage_kwargs)
            kwargs.update(
                {
                    "integrator": str(integrator),
                    "return_diagnostics": True,
                    "transverse_cell_area": float(transverse_cell_area),
                }
            )
            expected_config = self._config_token(kwargs)
            if cap._config_token is not None and cap._config_token != expected_config:
                self._reject("CAPABILITY_CONFIGURATION_MISMATCH")
            if self._g_config_token is None:
                self._g_config_token = expected_config
                cap._config_token = expected_config
            elif self._g_config_token != expected_config:
                self._reject("PROPAGATION_CONFIGURATION_CHANGED")
            half_started_wall = time.perf_counter()
            half_started_cpu = time.process_time()
            heun_started_wall = time.perf_counter()
            heun_started_cpu = time.process_time()
            finish_segment = False
            try:
                bundle = self._g_controlled_full_heun(field, step_length, kwargs, cap)
                result = bundle["result"]
                updated, diagnostics = result
                self._g_e_ref = updated
                if cap._scope == "pre":
                    # The frozen source keeps the old linear_result local
                    # until its second BK assignment.  Preserve that exact
                    # reference while rebinding the nonlinear gate to E.
                    self._g_retained_linear_result = cap._field
                    self._g_last_field = updated
                self._g_last_scope = "pre_completed" if cap._scope == "pre" else "post_completed"
                self._g_pending_cap = None
                if cap._scope == "post":
                    finish_segment = True
                    self._g_last_field = None
                return bundle["stage1"], updated, diagnostics
            except BaseException:
                self._g_abort_propagation("controlled_substep_exception")
                raise
            finally:
                half_wall = float(time.perf_counter() - half_started_wall)
                half_cpu = float(time.process_time() - half_started_cpu)
                heun_wall = float(time.perf_counter() - heun_started_wall)
                heun_cpu = float(time.process_time() - heun_started_cpu)
                self._stats["controlled_half_calls"] += 1
                self._stats["controlled_half_wall_s_total"] += half_wall
                self._stats["controlled_half_process_cpu_s_total"] += half_cpu
                if cap._scope == "pre":
                    self._stats["controlled_half_pre_calls"] += 1
                else:
                    self._stats["controlled_half_post_calls"] += 1
                self._stats["last_controlled_half_scope"] = str(cap._scope)
                self._stats["last_controlled_half_wall_s"] = half_wall
                self._stats["last_controlled_half_process_cpu_s"] = half_cpu
                self._stats["heun_core_calls"] += 1
                self._stats["heun_core_wall_s_total"] += heun_wall
                self._stats["heun_core_process_cpu_s_total"] += heun_cpu
                self._stats["last_heun_core_wall_s"] = heun_wall
                self._stats["last_heun_core_process_cpu_s"] = heun_cpu
                if finish_segment:
                    self._g_finish_segment()

        def _g_controlled_full_heun(
            self,
            field: Any,
            dz: Any,
            kwargs: Mapping[str, Any],
            cap: _ControlledCapability,
        ) -> dict[str, Any]:
            self._g_check_live_control()
            self._g_validate_capability(cap, field)
            current_thread = int(threading.get_ident())
            if current_thread != int(self._control_thread_id):
                self._reject("CONTROL_THREAD_MISMATCH")
            if not self._lock.acquire(blocking=False):
                self._reject("REENTRANT_OR_CONCURRENT_ENTRY")
            cap._consuming = True
            self._g_active_cap_scope = str(cap._scope)
            self._generation += 1
            self._hash_records = []
            context = call_context(self, self._generation, field, dict(kwargs))
            self._g_context_omega_ref = None
            self._g_context_omega_meta = None
            self._local.context = context
            self._local.g_controlled = True
            self._local.g_phase = "stage1"
            self._local.g_field = field
            self._local.g_field_token = None
            self._local.g_stage2_result = None
            context.route = "G_ACTUAL"
            context.status = "READY"
            try:
                self._g_check_live_control()
                self._internal_stage(context, field, kwargs, stage2=False)
                self._local.g_field_token = context.field_token
                self._g_check_live_control()
                result = self._internal_apply(context, field, dz, kwargs)
                if context.status != "CONSUMING" or context.predictor is None or not context.stage2_consumed:
                    self._reject("CONTROLLED_STAGE2_NOT_CONSUMED")
                self._set_controlled_hash_stats()
                if self._g_qualification_capture_enabled:
                    capture_values = (
                        field,
                        context.stage1,
                        context.predictor,
                        getattr(self._local, "g_stage2_result", None),
                    )
                    capture_refs = [
                        value
                        for value in capture_values
                        if isinstance(value, np.ndarray)
                    ]
                    self._g_qualification_retained_array_refs += len(capture_refs)
                    self._g_qualification_retained_array_bytes += int(
                        sum(int(value.nbytes) for value in capture_refs)
                    )
                    self._g_qualification_capture_records.append(
                        {
                            "generation": int(context.generation),
                            "scope": str(cap._scope),
                            "field": field,
                            "stage1": context.stage1,
                            "predictor": context.predictor,
                            "stage2": getattr(self._local, "g_stage2_result", None),
                        }
                    )
                context.status = "CONSUMED"
                cap._consuming = False
                cap._consumed = True
                cap._retired = True
                self._stats["controlled_full_calls"] += 1
                self._g_metrics["G_ACTUAL"] += 1
                self._stats["route_counts"] = dict(self._g_metrics)
                return {"stage1": context.stage1, "result": result}
            except BaseException as exc:
                cap._consuming = False
                cap._retired = True
                context.status = "FAILED"
                context.failure = f"{type(exc).__name__}:{exc}"
                self._stats["failed_full_calls"] += 1
                self._stats["route_counts"] = dict(self._g_metrics)
                raise
            finally:
                self._restore_control_thread_mask()
                self._local.context = None
                self._local.g_controlled = False
                self._local.g_phase = None
                self._local.g_field = None
                self._local.g_field_token = None
                self._local.g_stage2_result = None
                self._g_context_omega_ref = None
                self._g_context_omega_meta = None
                self._g_active_cap_scope = None
                self._lock.release()

        def _internal_apply(
            self,
            context: Any,
            field: Any,
            dz: Any,
            kwargs: Mapping[str, Any],
        ) -> Any:
            """Run the frozen apply leaf with an explicit controlled phase."""
            previous = getattr(self._local, "g_phase", None)
            if bool(getattr(self._local, "g_controlled", False)):
                self._local.g_phase = "apply"
            try:
                return super()._internal_apply(context, field, dz, kwargs)
            finally:
                if bool(getattr(self._local, "g_controlled", False)):
                    self._local.g_phase = previous

        def _record_hash(self, value: np.ndarray, role: str) -> str:
            controlled = bool(getattr(self._local, "g_controlled", False))
            if not controlled:
                return super()._record_hash(value, role)
            phase = str(getattr(self._local, "g_phase", ""))
            source_role = str(role)
            context = getattr(self._local, "context", None)
            cap = self._g_pending_cap
            if phase not in ("stage1", "apply", "stage2"):
                self._reject("CONTROLLED_HASH_PHASE_INVALID")
            if context is None or cap is None:
                self._reject("CONTROLLED_HASH_CONTEXT_OR_SCOPE_MISSING")
            if self._g_active_cap_scope is None or str(cap._scope) != str(self._g_active_cap_scope):
                self._reject("CONTROLLED_HASH_CAPABILITY_SCOPE_CHANGED")
            if source_role == "E":
                if phase != "stage1":
                    self._reject("CONTROLLED_HASH_E_SOURCE_ROLE_PHASE")
                actual_role = "E_stage1"
            elif source_role == "Omega_Omega":
                if not isinstance(getattr(context, "kwargs", None), Mapping) or context.kwargs.get("Omega") is not value:
                    self._reject("CONTROLLED_HASH_OMEGA_CONTEXT_OBJECT_MISMATCH")
                try:
                    omega_meta = _array_object_metadata(value, "Omega")
                except BaseException as exc:
                    self._reject(f"CONTROLLED_HASH_OMEGA_METADATA_INVALID:{exc}")
                if phase == "stage1":
                    if self._g_context_omega_ref is not None:
                        self._reject("CONTROLLED_HASH_OMEGA_STAGE1_REPLAY")
                    self._g_context_omega_ref = value
                    self._g_context_omega_meta = omega_meta
                else:
                    if value is not self._g_context_omega_ref:
                        self._reject("CONTROLLED_HASH_OMEGA_OBJECT_CHANGED")
                    if omega_meta != self._g_context_omega_meta:
                        self._reject("CONTROLLED_HASH_OMEGA_METADATA_CHANGED")
                actual_role = {
                    "stage1": "Omega_stage1",
                    "apply": "Omega_apply",
                    "stage2": "Omega_stage2",
                }[phase]
            else:
                self._reject(f"CONTROLLED_HASH_SOURCE_ROLE_INVALID:{source_role}")
            hasher = self._candidate_namespace.get("_array_raw_sha256")
            if not callable(hasher):
                raise CallsiteBindingError("R8_ARRAY_HASH_HELPER_MISSING")
            digest = str(hasher(value))
            self._hash_records.append(
                {
                    "role": actual_role,
                    "source_role": source_role,
                    "generation": int(context.generation),
                    "scope": str(cap._scope),
                    "logical_bytes": int(value.nbytes),
                    "shape": [int(s) for s in value.shape],
                    "dtype": value.dtype.str,
                }
            )
            return digest

        def _field_token(self, field: np.ndarray, kwargs: Mapping[str, Any]) -> dict[str, Any]:
            if bool(getattr(self._local, "g_controlled", False)) and str(
                getattr(self._local, "g_phase", "")
            ) == "apply":
                reference = getattr(self._local, "g_field", None)
                context = getattr(self._local, "context", None)
                stored = getattr(context, "field_token", None)
                if stored is None:
                    stored = getattr(self._local, "g_field_token", None)
                if field is not reference or not isinstance(stored, Mapping):
                    self._reject("CONTROLLED_APPLY_FIELD_OBJECT_MISMATCH")
                shape, dtype, strides, pointer = _field_layout(field)
                if (list(shape), dtype, list(strides), pointer) != (
                    stored.get("shape"),
                    stored.get("dtype"),
                    stored.get("strides_bytes"),
                    stored.get("data_pointer"),
                ):
                    self._reject("CONTROLLED_APPLY_FIELD_METADATA_MISMATCH")
                return {
                    "shape": list(shape),
                    "dtype": dtype,
                    "strides_bytes": list(strides),
                    "content_raw_sha256": stored["content_raw_sha256"],
                    "data_pointer": pointer,
                    "object_id": int(id(field)),
                    "parameter_token": self._parameter_token(kwargs),
                }
            return super()._field_token(field, kwargs)

        def _stage2_dispatch(self, field: Any, **kwargs: Any) -> Any:
            previous = getattr(self._local, "g_phase", None)
            if bool(getattr(self._local, "g_controlled", False)):
                self._local.g_phase = "stage2"
            try:
                result = super()._stage2_dispatch(field, **kwargs)
                if bool(getattr(self._local, "g_controlled", False)) and self._g_qualification_capture_enabled:
                    self._local.g_stage2_result = result
                return result
            finally:
                if bool(getattr(self._local, "g_controlled", False)):
                    self._local.g_phase = previous

        def _set_controlled_hash_stats(self) -> None:
            expected = ["E_stage1", "Omega_stage1", "Omega_apply", "Omega_stage2"]
            actual = [str(item.get("role")) for item in self._hash_records]
            if actual != expected:
                self._reject(f"CONTROLLED_HASH_ROLE_SEQUENCE:{actual}")
            context = getattr(self._local, "context", None)
            cap = self._g_pending_cap
            if context is None or cap is None:
                self._reject("CONTROLLED_HASH_FINAL_CONTEXT_OR_SCOPE_MISSING")
            expected_sources = ["E", "Omega_Omega", "Omega_Omega", "Omega_Omega"]
            for item, expected_role, expected_source in zip(
                self._hash_records, expected, expected_sources
            ):
                if str(item.get("source_role")) != expected_source:
                    self._reject("CONTROLLED_HASH_SOURCE_ROLE_SEQUENCE")
                if int(item.get("generation", -1)) != int(context.generation):
                    self._reject("CONTROLLED_HASH_GENERATION_MISMATCH")
                if str(item.get("scope")) != str(cap._scope):
                    self._reject("CONTROLLED_HASH_SCOPE_MISMATCH")
            logical_bytes = int(sum(int(item["logical_bytes"]) for item in self._hash_records))
            self._stats["hash_leaf_calls"] += len(actual)
            self._stats["hash_leaf_logical_bytes"] += logical_bytes
            self._stats["last_hash_leaf_calls"] = len(actual)
            self._stats["last_hash_leaf_logical_bytes"] = logical_bytes
            self._stats["last_controlled_hash_roles"] = list(actual)
            self._stats["last_controlled_hash_events"] = copy.deepcopy(self._hash_records)
            counts: dict[str, int] = {}
            for role in actual:
                counts[role] = int(counts.get(role, 0)) + 1
                self._stats["hash_leaf_role_counts"][role] = int(
                    self._stats["hash_leaf_role_counts"].get(role, 0)
                ) + 1
                self._stats["controlled_hash_role_counts"][role] = int(
                    self._stats["controlled_hash_role_counts"].get(role, 0)
                ) + 1
            self._stats["last_hash_leaf_role_counts"] = counts

        def _g_finish_segment(self) -> None:
            if not self._g_segment_started:
                self._reject("CONTROLLED_SEGMENT_NOT_STARTED")
            wall = float(time.perf_counter() - self._g_segment_start_wall)
            cpu = float(time.process_time() - self._g_segment_start_cpu)
            self._g_segment_index += 1
            self._stats["controlled_segment_calls"] += 1
            self._stats["controlled_segment_wall_s_total"] += wall
            self._stats["controlled_segment_process_cpu_s_total"] += cpu
            self._stats["last_controlled_segment_wall_s"] = wall
            self._stats["last_controlled_segment_process_cpu_s"] = cpu
            self._g_segment_started = False

        def _g_publish_result(self, value: Any) -> Any:
            self._g_check_live_control()
            if (
                self._g_pending_cap is not None
                or self._g_segment_started
                or self._g_last_scope != "interval_closed"
            ):
                self._reject("PREMATURE_PROPAGATION_PUBLISH")
            pending = self._g_pending_cap
            if pending is not None:
                pending._retired = True
            self._g_pending_cap = None
            self._g_propagation_active = False
            self._g_active_entry_token = None
            self._g_next_linear_site = 0
            self._g_interval_bk_token = None
            self._g_context_omega_ref = None
            self._g_context_omega_meta = None
            self._g_active_cap_scope = None
            self._g_last_field = None
            self._g_retained_linear_result = None
            self._g_e_ref = None
            self._g_last_scope = "RETIRED"
            return value

        def _g_abort_propagation(self, reason: str) -> None:
            if self._g_segment_started:
                failed_wall = float(time.perf_counter() - self._g_segment_start_wall)
                failed_cpu = float(time.process_time() - self._g_segment_start_cpu)
                self._stats["controlled_segment_failed_calls"] += 1
                self._stats["controlled_segment_failed_wall_s_total"] += failed_wall
                self._stats["controlled_segment_failed_process_cpu_s_total"] += failed_cpu
                self._stats["last_controlled_segment_failed_wall_s"] = failed_wall
                self._stats["last_controlled_segment_failed_process_cpu_s"] = failed_cpu
            pending = self._g_pending_cap
            if pending is not None:
                pending._retired = True
            self._g_pending_cap = None
            self._g_propagation_active = False
            self._g_active_entry_token = None
            self._g_segment_started = False
            self._g_next_linear_site = 0
            self._g_interval_bk_token = None
            self._g_context_omega_ref = None
            self._g_context_omega_meta = None
            self._g_active_cap_scope = None
            self._g_last_field = None
            self._g_retained_linear_result = None
            self._g_e_ref = None
            self._g_last_scope = "FAILED"
            self._g_clear_qualification_capture()
            self._stats["last_abort_reason"] = str(reason)

        def full_heun(self, field: Any, dz: Any, **kwargs: Any) -> dict[str, Any]:
            """Public/raw entry: always execute the original strict E path."""
            self._reject_optimized()
            full_kwargs = self._validate_full_kwargs(kwargs)
            if int(threading.get_ident()) != int(self._control_thread_id):
                self._reject("CONTROL_THREAD_MISMATCH")
            if not self._lock.acquire(blocking=False):
                self._reject("REENTRANT_OR_CONCURRENT_ENTRY")
            self._generation += 1
            self._hash_records = []
            context = call_context(self, self._generation, field, full_kwargs)
            self._local.context = context
            context.route = "F_STRICT_FALLBACK"
            context.status = "READY"
            try:
                bundle = self._strict_full_heun(field, dz, full_kwargs)
                context.status = "CONSUMED"
                self._stats["strict_full_calls"] += 1
                self._g_metrics["F_STRICT_FALLBACK"] += 1
                self._stats["route_counts"] = dict(self._g_metrics)
                return bundle
            except BaseException as exc:
                context.status = "FAILED"
                context.failure = f"{type(exc).__name__}:{exc}"
                self._stats["failed_full_calls"] += 1
                raise
            finally:
                self._restore_control_thread_mask()
                self._local.context = None
                self._lock.release()

        def metrics(self) -> dict[str, Any]:
            result = super().metrics()
            result["schema"] = str(candidate_schema)
            result["candidate_name"] = "G_CONTROLLED_INPUT"
            result["route_counts"] = dict(self._g_metrics)
            result["active_capability"] = bool(self._g_pending_cap is not None)
            ownership_bytes = self._g_ownership_metadata_bytes()
            result["active_metadata_bytes_bound"] = int(ownership_bytes)
            result["active_ownership_metadata_bytes"] = int(ownership_bytes)
            result["ownership_metadata_limit_bytes"] = int(_OWNERSHIP_METADATA_LIMIT_BYTES)
            result["ownership_metadata_within_bound"] = bool(
                ownership_bytes <= _OWNERSHIP_METADATA_LIMIT_BYTES
            )
            result["ownership_alias_roles"] = ["linear_result", "E", "owner"]
            result["controlled_segment_index"] = int(self._g_segment_index)
            result["raw_field_arrays_exposed"] = False
            result["public_full_heun_route"] = "F_STRICT_FALLBACK"
            result["qualification_capture_enabled"] = bool(self._g_qualification_capture_enabled)
            result["qualification_capture_pending"] = int(len(self._g_qualification_capture_records))
            result["qualification_retained_array_refs"] = int(
                self._g_qualification_retained_array_refs
            )
            result["qualification_retained_array_bytes"] = int(
                self._g_qualification_retained_array_bytes
            )
            result["factory_execution_snapshot_initial_only"] = True
            return result

    _ControlledHandoffState.__name__ = "_ControlledHandoffState"
    _ControlledHandoffState.__qualname__ = "_ControlledHandoffState"
    return _ControlledHandoffState


__all__ = [
    "CallsiteBindingError",
    "PROPAGATE_RELATIVE_PATH",
    "PROPAGATE_SOURCE_BLOB",
    "PROPAGATE_CANONICAL_LF_SHA256",
    "bind_propagator",
    "capability_type",
    "make_controlled_state_class",
    "source_binding",
]
