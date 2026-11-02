"""S4-6I OFF/COST factory and scalar residual-cost runtime hook."""

from __future__ import annotations

import ast
import copy
import contextlib
import importlib
import os
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Mapping

from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_5b_steady_cost.timing import TimingRecorder, context
from tools.cpu_s4_6h_combined_parallel.instrument import (
    _code_identity,
    _seal,
    verify_seal,
)

from .instrument import (
    INSTRUMENT_SCHEMA,
    ResidualInstrumentationError,
    instrument_heat_ast,
    instrument_q_ib_ast,
    instrument_residual_cost_ast,
    strip_residual_cost_ast,
)
from .registry import PARENT_LABELS, POWER_VOLUME_UNIT, SCOPE_HELPER, SCOPE_LABEL, registry_snapshot


COST_SCHEMA = "filament.cpu.s4_6i.residual_parallel.cost_factory.v1"
POWER_SCOPE_HELPER = "__cpu_s46i_power_volume_scope__"
Q_ION_HELPER = "__cpu_s46i_q_ion_from_power__"
Q_IB_HELPER = "__cpu_s46i_q_ib_from_power__"
DIRECT_ION_HELPER = "__cpu_s46i_direct_ion_interval_energy__"
DIRECT_IB_HELPER = "__cpu_s46i_direct_ib_interval_energy__"
HEAT_HELPER = "__cpu_s46i_heat_q_per_z__"


class CostFactoryError(RuntimeError):
    """A residual-cost binding, recorder, or source clone failed closed."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _owned_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    repo = _repo_root()
    files: dict[str, dict[str, Any]] = {}
    for name in ("__init__.py", "registry.py", "instrument.py", "cost_factory.py"):
        path = (root / name).resolve(strict=True)
        raw = path.read_bytes()
        record: dict[str, Any] = {
            "path": str(path),
            "size_bytes": int(len(raw)),
            "raw_sha256": pv.raw_sha256(raw),
            "classification": "external",
            "hash_scope": "raw_bytes",
        }
        relative = path.relative_to(repo).as_posix()
        try:
            oid = pv._git_blob_oid(repo, relative)
        except pv.ProvenanceError:
            if os.environ.get("SLURM_JOB_ID"):
                raise CostFactoryError("S46I_COMMITTED_OWNED_SOURCE_REQUIRED")
        else:
            blob = pv._read_git_blob(repo, oid)
            if pv.canonical_lf_bytes(raw) != pv.canonical_lf_bytes(blob):
                if os.environ.get("SLURM_JOB_ID"):
                    raise CostFactoryError("S46I_OWNED_SOURCE_CANONICAL_BLOB_MISMATCH")
                record["classification"] = "local_worktree_text"
                record["hash_scope"] = "raw_bytes"
            else:
                record.update(
                    {
                        "classification": "tracked_text",
                        "hash_scope": pv.TRACKED_HASH_SCOPE,
                        "git_blob_oid": oid,
                        "canonical_lf_sha256": pv.canonical_lf_sha256(blob),
                        "relative_path": relative,
                        "raw_byte_observation": "CURRENT_CHECKOUT_ONLY_NOT_CROSS_PLATFORM_IDENTITY",
                    }
                )
        files[name] = record
    composite = (
        "producer_module:tools.cpu_s4_6i_residual_parallel.cost_factory\n"
        + "".join(f"{name}:{files[name]['raw_sha256']}\n" for name in sorted(files))
    ).encode("utf-8")
    return {
        "schema": COST_SCHEMA + ".owned_sources.v1",
        "producer_module_identity": "tools.cpu_s4_6i_residual_parallel.cost_factory",
        "files": files,
        "composite_raw_sha256": pv.raw_sha256(composite),
    }


def _clone_function(
    function: Callable[..., Any],
    namespace: dict[str, Any],
    *,
    name: str,
) -> FunctionType:
    code = getattr(function, "__code__", None)
    if code is None:
        raise CostFactoryError(f"S46I_ORIGINAL_CODE_MISSING:{name}")
    clone = FunctionType(
        code,
        namespace,
        name=name,
        argdefs=getattr(function, "__defaults__", None),
        closure=getattr(function, "__closure__", None),
    )
    clone.__kwdefaults__ = getattr(function, "__kwdefaults__", None)
    clone.__annotations__ = getattr(function, "__annotations__", {})
    clone.__module__ = getattr(function, "__module__", None)
    clone.__qualname__ = getattr(function, "__qualname__", name)
    clone.__doc__ = getattr(function, "__doc__", None)
    return clone


def _compile_function(
    node: ast.FunctionDef,
    *,
    namespace: dict[str, Any],
    source_path: Path,
) -> FunctionType:
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    local: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec", dont_inherit=True, optimize=0), namespace, local)
    function = local.get(node.name)
    if not isinstance(function, FunctionType) or function.__globals__ is not namespace:
        raise CostFactoryError("S46I_PRIVATE_CLONE_COMPILE_FAILED")
    return function


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise CostFactoryError(f"S46I_SOURCE_FUNCTION_COUNT:{name}:{len(matches)}")
    return matches[0]


def _contains_array(value: Any, *, depth: int = 0) -> bool:
    """Reject scientific arrays in helper closure metadata, not in globals."""

    if depth > 4:
        return False
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return True
    except Exception:
        pass
    if isinstance(value, Mapping):
        return any(_contains_array(item, depth=depth + 1) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_array(item, depth=depth + 1) for item in value)
    return False


class ResidualCostHook:
    """A private scalar hook shared by the C/G2 compile and runtime seams."""

    def __init__(self, recorder: TimingRecorder) -> None:
        if type(recorder) is not TimingRecorder:
            raise CostFactoryError("S46I_RECORDER_EXACT_TYPE_REQUIRED")
        if bool(getattr(recorder, "enabled", True)):
            raise CostFactoryError("S46I_RECORDER_PROFILE_MUST_BE_OFF")
        self.recorder = recorder
        self._power_context: dict[str, str] = {"name": "unknown"}
        self._last_propagation_receipt: dict[str, Any] | None = None

        self._scope_helper = self._make_scope_helper()
        self._power_scope_helper = self._make_power_scope_helper()
        self._build_source_clones()
        self._runtime_helpers = self._make_runtime_helpers()
        self._helper_seals = {
            name: _seal(helper) for name, helper in self._runtime_helpers.items()
        }
        self._private_helper_seals = {
            POWER_SCOPE_HELPER: _seal(self._power_scope_helper),
        }
        self._transform_seals = {
            "propagation_instrument": _seal(instrument_residual_cost_ast),
            "propagation_strip": _seal(strip_residual_cost_ast),
            "q_ib_instrument": _seal(instrument_q_ib_ast),
            "heat_instrument": _seal(instrument_heat_ast),
        }
        self._source_seals = {
            name: _seal(function) for name, function in self._original_functions.items()
        }
        self._clone_seals = {
            name: _seal(function) for name, function in self._source_clones.items()
        }
        self._clone_backend_slots = {
            name: {
                key: function.__globals__.get(key)
                for key in set(function.__code__.co_names) | {"xp", "_np", "np", "Q_CAP"}
                if key in function.__globals__
            }
            for name, function in self._source_clones.items()
        }
        self._backend_functions = {
            name: getattr(self._source_clones["deposition._power_volume"].__globals__["xp"], name)
            for name in ("asarray", "maximum", "nan_to_num", "sum", "clip")
        }
        self._receipt = self._make_receipt()
        self._receipt_fingerprint = copy.deepcopy(self._receipt)
        self._active_globals: dict[str, Any] | None = None

    def _make_scope_helper(self) -> Callable[..., Any]:
        recorder = self.recorder
        power_context = self._power_context

        def scope(
            label: str,
            interval_index: int | None = None,
            occurrence_class: str = "PER_INTERVAL",
            source_site: str = "",
        ):
            if str(label) not in (SCOPE_LABEL,) + PARENT_LABELS:
                raise CostFactoryError("S46I_SCOPE_LABEL_INVALID")
            if source_site in ("deposition.py:42:time_sum", "deposition.py:131:independent_3d_sum"):
                source_site = source_site + ":" + power_context["name"]
            current = recorder.current_context()
            if interval_index is None:
                metadata = recorder.bridged_context(
                    require_companion=False,
                    source_site=str(source_site),
                )
            else:
                metadata = context(
                    int(interval_index),
                    current.phase,
                    current.stage,
                    ordinal=int(interval_index),
                    attempt=1,
                    occurrence_class=str(occurrence_class),
                    source_site=str(source_site),
                )
            # ``common=True`` is the explicit private seam that records COST
            # scopes while the inherited recorder remains profile=False.
            return recorder.scope(label, context=metadata, common=True)

        scope.__name__ = SCOPE_HELPER
        scope.__cpu_s46i_kind__ = "FIXED_RESIDUAL_COST_SCOPE"
        scope.__cpu_s46i_recorder__ = recorder
        return scope

    def _make_power_scope_helper(self) -> Callable[[], Any]:
        scope = self._scope_helper
        state = self._power_context

        def power_scope():
            name = str(state.get("name", "unknown"))
            source_site = f"deposition.py:23:_power_volume:{name}"
            return scope(SCOPE_LABEL, None, "PER_INTERVAL", source_site)

        power_scope.__name__ = POWER_SCOPE_HELPER
        power_scope.__cpu_s46i_kind__ = "FIXED_POWER_VOLUME_SCOPE"
        power_scope.__cpu_s46i_recorder__ = self.recorder
        return power_scope

    @contextlib.contextmanager
    def _set_power_context(self, name: str):
        previous = self._power_context.get("name", "unknown")
        self._power_context["name"] = str(name)
        try:
            yield
        finally:
            self._power_context["name"] = previous

    def _make_context_wrapper(
        self,
        target: Callable[..., Any],
        name: str,
    ) -> Callable[..., Any]:
        owner = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with owner._set_power_context(name):
                return target(*args, **kwargs)

        wrapped.__name__ = name
        wrapped.__qualname__ = name
        wrapped.__cpu_s46i_kind__ = "FIXED_DEPOSITION_CONTEXT_WRAPPER"
        wrapped.__cpu_s46i_target__ = target
        wrapped.__cpu_s46i_context__ = name
        return wrapped

    def _build_source_clones(self) -> None:
        deposition = importlib.import_module("KHz_filament.deposition")
        heat = importlib.import_module("KHz_filament.heat")
        dep_path = Path(str(deposition.__file__)).resolve(strict=True)
        heat_path = Path(str(heat.__file__)).resolve(strict=True)
        dep_tree = ast.parse(dep_path.read_text(encoding="utf-8"), filename=str(dep_path))
        heat_tree = ast.parse(heat_path.read_text(encoding="utf-8"), filename=str(heat_path))
        power_node = _find_function(dep_tree, "_power_volume")
        power_baseline = copy.deepcopy(power_node)
        # The heat instrumenter has the exact generic span machinery needed for
        # heat.py; deposition.py uses a direct source-bound wrapper below.
        power_body = list(power_node.body)
        if not power_body or int(getattr(power_body[0], "lineno", -1)) != 23:
            raise CostFactoryError("S46I_POWER_VOLUME_SOURCE_ANCHOR_CHANGED")
        # _power_volume's return is the sole pointwise expression.  Keep its
        # source statements together under one scalar scope and leave every
        # array operation unchanged.
        from .instrument import _wrap  # private module-local AST primitive

        power_node.body = [
            _wrap(
                power_body,
                source_site="deposition.py:23:_power_volume",
                interval_name=None,
            )
        ]
        power_node.body[0].items[0].context_expr = ast.Call(
            func=ast.Name(id=POWER_SCOPE_HELPER, ctx=ast.Load()), args=[], keywords=[])
        ast.fix_missing_locations(power_node)
        if ast.dump(strip_residual_cost_ast(power_node), include_attributes=False) != ast.dump(power_baseline, include_attributes=False):
            raise CostFactoryError("S46I_POWER_VOLUME_INVERSE_AST_MISMATCH")
        private_dep = dict(getattr(deposition, "__dict__", {}))
        private_dep[SCOPE_HELPER] = self._scope_helper
        private_dep[POWER_SCOPE_HELPER] = self._power_scope_helper
        power_clone = _compile_function(power_node, namespace=private_dep, source_path=dep_path)
        private_dep["_power_volume"] = power_clone
        def reduced_clone(tree, name, line, site, namespace, path):
            node = _find_function(tree, name)
            baseline = copy.deepcopy(node)
            matches = [index for index, statement in enumerate(node.body) if statement.lineno == line]
            if len(matches) != 1:
                raise CostFactoryError("S46I_REDUCTION_ANCHOR_CHANGED:" + name)
            index = matches[0]
            statement = node.body[index]
            if not any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
                       and item.func.attr == "sum" for item in ast.walk(statement)):
                raise CostFactoryError("S46I_REDUCTION_SUM_MISSING:" + name)
            node.body[index:index+1] = [_wrap([statement], source_site=site, interval_name=None)]
            if ast.dump(strip_residual_cost_ast(node), include_attributes=False) != ast.dump(baseline, include_attributes=False):
                raise CostFactoryError("S46I_REDUCTION_INVERSE_AST_MISMATCH:" + name)
            return _compile_function(node, namespace=namespace, source_path=path)

        integrate_clone = reduced_clone(dep_tree, "integrate_power_to_q", 42,
            "deposition.py:42:time_sum", private_dep, dep_path)
        private_dep["integrate_power_to_q"] = integrate_clone
        q_ion_clone = _clone_function(
            deposition.q_ion_from_power,
            private_dep,
            name="__cpu_s46i_q_ion_source_clone__",
        )
        q_ib_node = _find_function(dep_tree, "q_ib_from_power")
        q_ib_receipt = instrument_q_ib_ast(q_ib_node)
        q_ib_clone = _compile_function(q_ib_node, namespace=private_dep, source_path=dep_path)
        direct_clone = reduced_clone(dep_tree, "direct_interval_energy", 131,
            "deposition.py:131:independent_3d_sum", private_dep, dep_path)

        heat_node = _find_function(heat_tree, "heat_Q_per_z")
        heat_baseline = copy.deepcopy(heat_node)
        heat_receipt = instrument_heat_ast(heat_node)
        reduction = [index for index, statement in enumerate(heat_node.body) if statement.lineno == 36]
        if len(reduction) != 1:
            raise CostFactoryError("S46I_HEAT_REDUCTION_ANCHOR_CHANGED")
        index = reduction[0]
        heat_node.body[index:index+1] = [_wrap([heat_node.body[index]],
            source_site="heat.py:36:time_sum", interval_name=None)]
        if ast.dump(strip_residual_cost_ast(heat_node), include_attributes=False) != ast.dump(heat_baseline, include_attributes=False):
            raise CostFactoryError("S46I_HEAT_REDUCTION_INVERSE_AST_MISMATCH")
        private_heat = dict(getattr(heat, "__dict__", {}))
        private_heat[SCOPE_HELPER] = self._scope_helper
        heat_clone = _compile_function(heat_node, namespace=private_heat, source_path=heat_path)

        self._original_functions = {
            "deposition._power_volume": deposition._power_volume,
            "deposition.integrate_power_to_q": deposition.integrate_power_to_q,
            "deposition.q_ion_from_power": deposition.q_ion_from_power,
            "deposition.q_ib_from_power": deposition.q_ib_from_power,
            "deposition.direct_interval_energy": deposition.direct_interval_energy,
            "heat.heat_Q_per_z": heat.heat_Q_per_z,
        }
        self._source_clones = {
            "deposition._power_volume": power_clone,
            "deposition.integrate_power_to_q": integrate_clone,
            "deposition.q_ion_from_power": q_ion_clone,
            "deposition.q_ib_from_power": q_ib_clone,
            "deposition.direct_interval_energy": direct_clone,
            "heat.heat_Q_per_z": heat_clone,
        }
        self._source_paths = {
            "deposition": str(dep_path),
            "heat": str(heat_path),
        }
        self._source_ast_receipts = {
            "deposition_power_volume": {
                "source_site": "deposition.py:23:_power_volume",
                "baseline_code_identity": _code_identity(deposition._power_volume),
                "clone_code_identity": _code_identity(power_clone),
                "inverse_proof": True,
            },
            "heat_Q_per_z": copy.deepcopy(heat_receipt),
            "q_ib_multiplication": copy.deepcopy(q_ib_receipt),
        }
        # Retain the unmodified AST digest as a strict source guard.
        self._source_ast_receipts["deposition_power_volume"]["baseline_ast"] = ast.dump(
            power_baseline, include_attributes=False
        )
        self._source_ast_receipts["heat_Q_per_z"]["baseline_ast"] = ast.dump(
            heat_baseline, include_attributes=False
        )
        self._owned_sources = _owned_source_identity()

    def _make_runtime_helpers(self) -> dict[str, Callable[..., Any]]:
        return {
            SCOPE_HELPER: self._scope_helper,
            Q_ION_HELPER: self._make_context_wrapper(
                self._source_clones["deposition.q_ion_from_power"], "q_ion"
            ),
            Q_IB_HELPER: self._make_context_wrapper(
                self._source_clones["deposition.q_ib_from_power"], "q_ib"
            ),
            DIRECT_ION_HELPER: self._make_context_wrapper(
                self._source_clones["deposition.direct_interval_energy"], "direct_ion"
            ),
            DIRECT_IB_HELPER: self._make_context_wrapper(
                self._source_clones["deposition.direct_interval_energy"], "direct_ib"
            ),
            HEAT_HELPER: self._source_clones["heat.heat_Q_per_z"],
        }

    def _make_receipt(self) -> dict[str, Any]:
        return {
            "schema": COST_SCHEMA,
            "instrumentation_schema": INSTRUMENT_SCHEMA,
            "enabled": True,
            "scope_label": SCOPE_LABEL,
            "scope_helper": SCOPE_HELPER,
            "runtime_helpers": tuple(sorted(self._runtime_helpers)),
            "private_helpers": (POWER_SCOPE_HELPER,),
            "transform_code_identities": {
                name: seal["code_identity"] for name, seal in self._transform_seals.items()
            },
            "registry": registry_snapshot(),
            "power_volume": copy.deepcopy(POWER_VOLUME_UNIT),
            "source_paths": copy.deepcopy(self._source_paths),
            "owned_source_identity": copy.deepcopy(self._owned_sources),
            "source_ast_receipts": copy.deepcopy(self._source_ast_receipts),
            "array_values_retained": False,
            "profile_flag_unchanged": True,
            "common_scope_path": True,
        }

    def runtime_helpers(self) -> dict[str, Callable[..., Any]]:
        return dict(self._runtime_helpers)

    def instrument(self, function_node: ast.FunctionDef) -> dict[str, Any]:
        try:
            receipt = instrument_residual_cost_ast(function_node)
        except ResidualInstrumentationError as exc:
            raise CostFactoryError(str(exc)) from exc
        self._last_propagation_receipt = copy.deepcopy(receipt)
        return receipt

    def strip_generated_nodes(self, node: ast.AST) -> ast.AST:
        try:
            return strip_residual_cost_ast(node)
        except ResidualInstrumentationError as exc:
            raise CostFactoryError(str(exc)) from exc

    def binding(self) -> dict[str, Any]:
        return copy.deepcopy(self._receipt)

    def bind_active(self, active_globals: Mapping[str, Any]) -> None:
        if not isinstance(active_globals, dict):
            raise CostFactoryError("S46I_ACTIVE_GLOBALS_REQUIRED")
        self.verify_binding(active_globals)
        self._active_globals = active_globals

    def verify_binding(self, active_globals: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if type(self.recorder) is not TimingRecorder or bool(self.recorder.enabled):
            raise CostFactoryError("S46I_RECORDER_BINDING_CHANGED")
        if self._receipt != self._receipt_fingerprint:
            raise CostFactoryError("S46I_RECEIPT_CHANGED")
        if _owned_source_identity() != self._receipt.get("owned_source_identity"):
            raise CostFactoryError("S46I_OWNED_SOURCE_IDENTITY_CHANGED")
        for name, seal in self._helper_seals.items():
            helper = self._runtime_helpers.get(name)
            if helper is None:
                raise CostFactoryError(f"S46I_HELPER_MISSING:{name}")
            verify_seal(seal)
            if getattr(helper, "__cpu_s46i_recorder__", self.recorder) is not self.recorder:
                raise CostFactoryError(f"S46I_HELPER_RECORDER_CHANGED:{name}")
            if getattr(helper, "__closure__", None):
                for cell in helper.__closure__ or ():
                    try:
                        value = cell.cell_contents
                    except ValueError as exc:
                        raise CostFactoryError(f"S46I_HELPER_CLOSURE_EMPTY:{name}") from exc
                    if _contains_array(value):
                        raise CostFactoryError(f"S46I_HELPER_CLOSURE_ARRAY:{name}")
        for name, seal in self._private_helper_seals.items():
            verify_seal(seal)
        for seal in self._transform_seals.values():
            verify_seal(seal)
        for name, function in self._original_functions.items():
            verify_seal(self._source_seals[name])
            clone = self._source_clones[name]
            if not callable(clone) or not isinstance(clone.__globals__, dict):
                raise CostFactoryError(f"S46I_SOURCE_CLONE_MISSING:{name}")
            verify_seal(self._clone_seals[name])
            for key, expected in self._clone_backend_slots[name].items():
                if clone.__globals__.get(key) is not expected:
                    raise CostFactoryError(f"S46I_SOURCE_CLONE_GLOBAL_CHANGED:{name}:{key}")
            original_xp = function.__globals__.get("xp")
            if original_xp is not None and clone.__globals__.get("xp") is not original_xp:
                raise CostFactoryError(f"S46I_SOURCE_BACKEND_CHANGED:{name}")
        backend = self._source_clones["deposition._power_volume"].__globals__["xp"]
        for name, expected in self._backend_functions.items():
            if getattr(backend, name) is not expected:
                raise CostFactoryError("S46I_BACKEND_FUNCTION_CHANGED:" + name)
        if active_globals is not None:
            if not isinstance(active_globals, dict):
                raise CostFactoryError("S46I_ACTIVE_GLOBALS_INVALID")
            for name, helper in self._runtime_helpers.items():
                if active_globals.get(name) is not helper:
                    raise CostFactoryError(f"S46I_ACTIVE_HELPER_CHANGED:{name}")
        return {
            "schema": COST_SCHEMA + ".binding.v1",
            "enabled": True,
            "helper_count": len(self._runtime_helpers),
            "active_globals_bound": active_globals is not None,
            "common_scope": True,
            "profile": False,
            "array_values_retained": False,
            "registry_schema": self._receipt["registry"]["schema"],
        }

    def snapshot(self) -> dict[str, Any]:
        snapshot = self.recorder.snapshot()
        if int(snapshot.get("events_dropped", 0)) != 0:
            raise CostFactoryError("S46I_DROPPED_RECORDS")
        if snapshot.get("incomplete_scopes"):
            raise CostFactoryError("S46I_INCOMPLETE_RECORDS")
        return copy.deepcopy(snapshot)


def make_cost_hook(recorder: TimingRecorder) -> ResidualCostHook:
    return ResidualCostHook(recorder)


def source_identity() -> dict[str, Any]:
    return _owned_source_identity()


def _pair_or_default(recorder_pair: Any | None) -> Any:
    adapter = importlib.import_module("tools.cpu_s4_5b_steady_cost.adapter")
    if recorder_pair is None:
        recorder_pair = adapter.make_recorder_pair(False)
    if type(recorder_pair) is not getattr(adapter, "RecorderPair", object):
        raise CostFactoryError("S46I_RECORDER_PAIR_EXACT_TYPE_REQUIRED")
    if bool(getattr(recorder_pair, "profile", True)):
        raise CostFactoryError("S46I_RECORDER_PAIR_PROFILE_MUST_BE_OFF")
    return recorder_pair


def build_role(
    mode: str,
    *,
    capture_root: str | Path,
    cost_enabled: bool = False,
    local_engineering: bool = False,
    pool_override: int | None = None,
    recorder_pair: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], Any, Any]:
    """Build H role X with OFF or scalar COST instrumentation."""

    if not isinstance(cost_enabled, bool):
        raise CostFactoryError("S46I_COST_ENABLED_MUST_BE_BOOL")
    pair = _pair_or_default(recorder_pair)
    hook = make_cost_hook(pair.b) if cost_enabled else None
    combined = importlib.import_module("tools.cpu_s4_6h_combined_parallel.combined_candidate")
    result = combined.build_role(
        mode,
        capture_root=Path(capture_root).resolve(),
        role=combined.ROLE_CANDIDATE,
        local_engineering=bool(local_engineering),
        pool_override=pool_override,
        recorder_pair=pair,
        cost_enabled=bool(cost_enabled),
        residual_cost_hook=hook,
    )
    dispatch = result[1]
    dispatch.cpu_s4_6i_cost_enabled = bool(cost_enabled)
    dispatch.cpu_s4_6i_cost_hook = hook
    dispatch.cpu_s4_6i_cost_binding = (
        None if hook is None else hook.binding()
    )
    try:
        if hook is not None:
            active_globals = getattr(dispatch, "cpu_s4_6e_c_active_globals", None)
            if not isinstance(active_globals, dict):
                raise CostFactoryError("S46I_ACTIVE_PROPAGATOR_GLOBALS_MISSING")
            hook.bind_active(active_globals)
            dispatch.cpu_s4_6i_cost_binding_check = hook.verify_binding(active_globals)
    except BaseException:
        combined.finalize_factory(dispatch, reason="COST_BINDING_FAILURE")
        raise
    return result


def build_candidate(*args: Any, **kwargs: Any):
    return build_role(*args, **kwargs)


def verify_bindings(dispatch: Any) -> dict[str, Any]:
    combined = importlib.import_module("tools.cpu_s4_6h_combined_parallel.combined_candidate")
    check = combined.verify_bindings(dispatch)
    enabled = bool(getattr(dispatch, "cpu_s4_6i_cost_enabled", False))
    hook = getattr(dispatch, "cpu_s4_6i_cost_hook", None)
    if enabled:
        if not isinstance(hook, ResidualCostHook):
            raise CostFactoryError("S46I_COST_HOOK_MISSING")
        active_globals = getattr(dispatch, "cpu_s4_6e_c_active_globals", None)
        check["cost_binding"] = hook.verify_binding(active_globals)
    elif hook is not None:
        raise CostFactoryError("S46I_OFF_COST_HOOK_PRESENT")
    check["cost_enabled"] = enabled
    return check


def metadata(dispatch: Any) -> dict[str, Any]:
    combined = importlib.import_module("tools.cpu_s4_6h_combined_parallel.combined_candidate")
    result = combined.metadata(dispatch)
    result["cost_enabled"] = bool(getattr(dispatch, "cpu_s4_6i_cost_enabled", False))
    hook = getattr(dispatch, "cpu_s4_6i_cost_hook", None)
    result["cost"] = None if hook is None else hook.binding()
    return result


def finalize_factory(dispatch: Any, *, reason: str = "NORMAL") -> dict[str, Any]:
    combined = importlib.import_module("tools.cpu_s4_6h_combined_parallel.combined_candidate")
    return combined.finalize_factory(dispatch, reason=reason)


def take_cost_snapshot(dispatch: Any) -> dict[str, Any]:
    hook = getattr(dispatch, "cpu_s4_6i_cost_hook", None)
    if hook is None:
        return {
            "enabled": False,
            "records": (),
            "events_dropped": 0,
            "incomplete_scopes": (),
        }
    if not isinstance(hook, ResidualCostHook):
        raise CostFactoryError("S46I_COST_HOOK_INVALID")
    return hook.snapshot()


__all__ = [
    "COST_SCHEMA",
    "CostFactoryError",
    "DIRECT_IB_HELPER",
    "DIRECT_ION_HELPER",
    "HEAT_HELPER",
    "Q_IB_HELPER",
    "Q_ION_HELPER",
    "ResidualCostHook",
    "build_candidate",
    "build_role",
    "finalize_factory",
    "make_cost_hook",
    "metadata",
    "source_identity",
    "take_cost_snapshot",
    "verify_bindings",
]
