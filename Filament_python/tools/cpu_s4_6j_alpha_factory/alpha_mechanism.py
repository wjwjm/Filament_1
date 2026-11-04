"""Frozen REF/P1/P24 local mechanism comparison for the alpha chain.

The reference and candidate functions are both compiled from the four
statements extracted from ``propagate_one_pulse``.  Candidate construction
only replaces those four expression nodes with the already-tested alpha
helpers; it does not fuse the assignments or retain the old source in a new
local variable.
"""

from __future__ import annotations

import ast
import copy
import importlib
import statistics
import time
from pathlib import Path
from types import FunctionType
from typing import Any, Mapping

import numpy as np

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_6i_residual_parallel.power_hook import _seal, verify_seal

from . import alpha_capture
from . import alpha_hook
from . import alpha_pool as alpha_kernel
from .alpha_capture import (
    fingerprint,
    load_capture,
    load_state_arrays,
)
from .alpha_pool import AlphaPool
from .factory_timing import source_identity as factory_source_identity


MECHANISM_SCHEMA = "filament.cpu.s4_6j.alpha_mechanism.v1"
ORDERS = (("REF", "P1", "P24"), ("P24", "P1", "REF"))
CONFIGS = ("REF", "P1", "P24")
EXPECTED_LABELS = (
    "ALPHA_CLEAN_SOURCE",
    "ALPHA_DENOM",
    "ALPHA_RATIO",
    "ALPHA_CLEAN_RESULT",
    "ALPHA_CLIP",
)


class AlphaMechanismError(RuntimeError):
    """The local alpha mechanism comparison failed closed."""


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _compile_function(
    node: ast.FunctionDef,
    *,
    namespace: dict[str, Any],
    path: Path,
) -> FunctionType:
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    local: dict[str, Any] = {}
    exec(compile(module, str(path), "exec", dont_inherit=True, optimize=0), namespace, local)
    function = local.get(node.name)
    if not isinstance(function, FunctionType) or function.__globals__ is not namespace:
        raise AlphaMechanismError("S46J_MECHANISM_COMPILE_FAILED")
    return function


def _arguments() -> ast.arguments:
    return ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(arg="ion_source_prop"),
            ast.arg(arg="I"),
            ast.arg(arg="I_floor"),
            ast.arg(arg="pool"),
            ast.arg(arg="interval_index"),
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )


def _helper_call(name: str, arguments: tuple[str, ...]) -> ast.Call:
    call = ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=[ast.Name(id=value, ctx=ast.Load()) for value in arguments],
        keywords=[
            ast.keyword(arg="pool", value=ast.Name(id="pool", ctx=ast.Load())),
            ast.keyword(
                arg="interval_index",
                value=ast.Name(id="interval_index", ctx=ast.Load()),
            ),
        ],
    )
    return call


def _chain_node(
    original_block: list[ast.stmt],
    *,
    name: str,
    candidate: bool,
) -> ast.FunctionDef:
    body = copy.deepcopy(original_block)
    if candidate:
        replacements = (
            (0, "__s46j_mechanism_clean_source__", ("ion_source_prop",)),
            (1, "__s46j_mechanism_ratio__", ("ion_source_prop", "I", "I_floor")),
            (2, "__s46j_mechanism_clean_alpha__", ("alpha_ion",)),
            (3, "__s46j_mechanism_clip_alpha__", ("alpha_ion",)),
        )
        for index, helper, arguments in replacements:
            statement = body[index]
            call = _helper_call(helper, arguments)
            ast.copy_location(call, statement)
            if index < 3:
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    raise AlphaMechanismError("S46J_MECHANISM_ASSIGNMENT_SHAPE")
                statement.value = call
            else:
                if not isinstance(statement, ast.Expr):
                    raise AlphaMechanismError("S46J_MECHANISM_CLIP_SHAPE")
                statement.value = call
    body.append(
        ast.Return(
            value=ast.Tuple(
                elts=[
                    ast.Name(id="ion_source_prop", ctx=ast.Load()),
                    ast.Name(id="alpha_ion", ctx=ast.Load()),
                ],
                ctx=ast.Load(),
            )
        )
    )
    function = ast.FunctionDef(
        name=name,
        args=_arguments(),
        body=body,
        decorator_list=[],
        returns=None,
    )
    ast.copy_location(function, original_block[0])
    ast.fix_missing_locations(function)
    return function


def _inverse_candidate(candidate: ast.FunctionDef, reference: ast.FunctionDef) -> bool:
    restored = copy.deepcopy(candidate)
    if candidate.name != "__s46j_candidate_chain__" or reference.name != "__s46j_reference_chain__":
        return False
    expected = _chain_node(reference.body[:4], name=candidate.name, candidate=True)
    if _dump(candidate) != _dump(expected):
        return False
    restored.name = reference.name
    replacements = (
        (0, "__s46j_mechanism_clean_source__"),
        (1, "__s46j_mechanism_ratio__"),
        (2, "__s46j_mechanism_clean_alpha__"),
        (3, "__s46j_mechanism_clip_alpha__"),
    )
    for index, helper in replacements:
        statement = restored.body[index]
        call = statement.value
        if not isinstance(call, ast.Call) or getattr(call.func, "id", None) != helper:
            return False
        original = reference.body[index]
        if index < 3:
            statement.value = copy.deepcopy(original.value)
        else:
            statement.value = copy.deepcopy(original.value)
    return _dump(restored) == _dump(reference)


def _same_data(actual: np.ndarray, expected: np.ndarray) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    if actual.strides != expected.strides or not actual.flags.c_contiguous:
        return False
    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    if not np.array_equal(actual_nan, expected_nan):
        return False
    actual_inf = np.isinf(actual)
    expected_inf = np.isinf(expected)
    if not np.array_equal(actual_inf, expected_inf):
        return False
    finite = ~(actual_nan | actual_inf)
    if not np.array_equal(actual[finite], expected[finite]):
        return False
    both_inf = actual_inf & expected_inf
    if not np.array_equal(np.signbit(actual[both_inf]), np.signbit(expected[both_inf])):
        return False
    both_zero = finite & (actual == 0.0) & (expected == 0.0)
    if np.any(both_zero) and not np.array_equal(
        actual[both_zero].view(np.uint64), expected[both_zero].view(np.uint64)
    ):
        return False
    return True


def _same_identity(actual: np.ndarray, expected: Mapping[str, Any]) -> bool:
    current = fingerprint(actual, name="IDENTITY")
    fields = ("shape", "dtype", "strides", "c_contiguous", "raw_sha256")
    return all(current.get(field) == expected.get(field) for field in fields)


class AlphaMechanism:
    """Compiled reference/candidate chain and its source-bound guards."""

    def __init__(self) -> None:
        propagate = importlib.import_module("KHz_filament.propagate")
        source_path = Path(str(propagate.__file__)).resolve(strict=True)
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
        function_node = alpha_hook._find_function(tree, "propagate_one_pulse")
        body, start = alpha_hook._alpha_block(function_node)
        self._source_path = source_path
        self._source_function = getattr(propagate, "propagate_one_pulse")
        self._source_seal = _seal(self._source_function)
        self._source_namespace = dict(vars(propagate))
        if self._source_namespace.get("xp") is not np:
            raise AlphaMechanismError("S46J_MECHANISM_SOURCE_BACKEND")
        self._original_block = copy.deepcopy(body[start : start + 4])
        self._source_ast_hash = alpha_hook._ast_hash(
            ast.Module(body=copy.deepcopy(self._original_block), type_ignores=[])
        )
        reference_node = _chain_node(
            self._original_block, name="__s46j_reference_chain__", candidate=False
        )
        self._candidate_helpers = self._make_candidate_helpers()
        self._helper_seals = {
            name: _seal(function) for name, function in self._candidate_helpers.items()
        }
        candidate_namespace = dict(self._source_namespace)
        candidate_namespace.update(self._candidate_helpers)
        reference_namespace = dict(self._source_namespace)
        self._reference_namespace = reference_namespace
        self._candidate_namespace = candidate_namespace
        self._reference_function = _compile_function(
            reference_node, namespace=reference_namespace, path=source_path
        )
        candidate_node = _chain_node(
            self._original_block, name="__s46j_candidate_chain__", candidate=True
        )
        if not _inverse_candidate(candidate_node, reference_node):
            raise AlphaMechanismError("S46J_MECHANISM_INVERSE_AST_MISMATCH")
        self._candidate_ast_hash = alpha_hook._ast_hash(candidate_node)
        self._inverse_proof = {
            "original_block_ast_sha256": self._source_ast_hash,
            "candidate_block_ast_sha256": self._candidate_ast_hash,
            "replacement_count": 4,
            "assignment_order_preserved": True,
            "source_local_lifetime_preserved": True,
            "inverse_ast_equal": True,
        }
        self._candidate_function = _compile_function(
            candidate_node, namespace=candidate_namespace, path=source_path
        )
        self._kernel_functions = {
            name: getattr(alpha_kernel, name)
            for name in ("clean_source", "ratio", "clean_alpha", "clip_alpha")
        }
        self._kernel_seals = {
            name: _seal(function) for name, function in self._kernel_functions.items()
        }
        self._pool_methods = {
            name: getattr(AlphaPool, name)
            for name in (
                "run_round",
                "take_records",
                "snapshot",
                "close",
                "set_failure_cleanup",
            )
        }
        self._pool_method_seals = {
            name: _seal(function) for name, function in self._pool_methods.items()
        }
        self._backend = {
            name: getattr(np, name) for name in ("add", "divide", "nan_to_num", "maximum")
        }
        self._source_identity = factory_source_identity(
            extra_paths=(
                __file__,
                Path(alpha_capture.__file__).resolve(),
                Path(alpha_hook.__file__).resolve(),
                Path(alpha_kernel.__file__).resolve(),
                source_path,
            )
        )
        self._source_identity_fingerprint = copy.deepcopy(self._source_identity)

    @staticmethod
    def _make_candidate_helpers() -> dict[str, Any]:
        def clean_source(source: np.ndarray, *, pool: AlphaPool, interval_index: int) -> np.ndarray:
            return alpha_kernel.clean_source(source, pool=pool, backend=np)

        def ratio(
            source: np.ndarray,
            intensity: np.ndarray,
            I_floor: float,
            *,
            pool: AlphaPool,
            interval_index: int,
        ) -> np.ndarray:
            return alpha_kernel.ratio(
                source, intensity, I_floor, pool=pool, backend=np
            )

        def clean_alpha(source: np.ndarray, *, pool: AlphaPool, interval_index: int) -> np.ndarray:
            return alpha_kernel.clean_alpha(source, pool=pool, backend=np)

        def clip_alpha(source: np.ndarray, *, pool: AlphaPool, interval_index: int) -> np.ndarray:
            return alpha_kernel.clip_alpha(source, pool=pool, backend=np)

        helpers = {
            "__s46j_mechanism_clean_source__": clean_source,
            "__s46j_mechanism_ratio__": ratio,
            "__s46j_mechanism_clean_alpha__": clean_alpha,
            "__s46j_mechanism_clip_alpha__": clip_alpha,
        }
        for name, helper in helpers.items():
            helper.__name__ = name
            helper.__qualname__ = name
            helper.__cpu_s46j_mechanism_kind__ = "SOURCE_BOUND_ALPHA_AST_HELPER"
        return helpers

    def verify_binding(self) -> dict[str, Any]:
        verify_seal(self._source_seal)
        if self._source_seal.get("function") is not self._source_function:
            raise AlphaMechanismError("S46J_MECHANISM_SOURCE_FUNCTION_CHANGED")
        current_source = factory_source_identity(
            extra_paths=(
                __file__,
                Path(alpha_capture.__file__).resolve(),
                Path(alpha_hook.__file__).resolve(),
                Path(alpha_kernel.__file__).resolve(),
                self._source_path,
            )
        )
        if current_source != self._source_identity_fingerprint:
            raise AlphaMechanismError("S46J_MECHANISM_SOURCE_IDENTITY_CHANGED")
        source = self._source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self._source_path))
        function_node = alpha_hook._find_function(tree, "propagate_one_pulse")
        body, start = alpha_hook._alpha_block(function_node)
        current_hash = alpha_hook._ast_hash(
            ast.Module(body=copy.deepcopy(body[start : start + 4]), type_ignores=[])
        )
        if current_hash != self._source_ast_hash:
            raise AlphaMechanismError("S46J_MECHANISM_SOURCE_AST_CHANGED")
        if self._source_namespace.get("xp") is not np:
            raise AlphaMechanismError("S46J_MECHANISM_REFERENCE_BACKEND_CHANGED")
        for name, function in self._kernel_functions.items():
            if getattr(alpha_kernel, name) is not function:
                raise AlphaMechanismError("S46J_MECHANISM_KERNEL_CHANGED:" + name)
            verify_seal(self._kernel_seals[name])
        for name, function in self._pool_methods.items():
            if getattr(AlphaPool, name) is not function:
                raise AlphaMechanismError("S46J_MECHANISM_POOL_METHOD_CHANGED:" + name)
            verify_seal(self._pool_method_seals[name])
        for name, function in self._backend.items():
            if getattr(np, name) is not function:
                raise AlphaMechanismError("S46J_MECHANISM_BACKEND_CHANGED:" + name)
        if self._reference_namespace.get("xp") is not np or self._candidate_namespace.get("xp") is not np:
            raise AlphaMechanismError("S46J_MECHANISM_COMPILED_BACKEND_CHANGED")
        for name, helper in self._candidate_helpers.items():
            if self._candidate_namespace.get(name) is not helper:
                raise AlphaMechanismError("S46J_MECHANISM_HELPER_CHANGED:" + name)
            verify_seal(self._helper_seals[name])
            if helper.__closure__ and any(
                isinstance(cell.cell_contents, np.ndarray) for cell in helper.__closure__
            ):
                raise AlphaMechanismError("S46J_MECHANISM_HELPER_ARRAY_CLOSURE")
        return {
            "schema": MECHANISM_SCHEMA + ".binding.v1",
            "source_identity": copy.deepcopy(self._source_identity),
            "source_ast_hash": self._source_ast_hash,
            "candidate_ast_hash": self._candidate_ast_hash,
            "inverse_proof": copy.deepcopy(self._inverse_proof),
            "backend": "numpy",
            "pool_methods_bound": True,
        }

    def verify_pool(self, pool: AlphaPool) -> None:
        if type(pool) is not AlphaPool:
            raise AlphaMechanismError("S46J_MECHANISM_POOL_TYPE")
        for name, function in self._pool_methods.items():
            bound = getattr(pool, name, None)
            if (
                getattr(bound, "__self__", None) is not pool
                or getattr(bound, "__func__", None) is not function
            ):
                raise AlphaMechanismError("S46J_MECHANISM_POOL_INSTANCE_METHOD:" + name)

    def call(
        self,
        config: str,
        source: np.ndarray,
        intensity: np.ndarray,
        I_floor: float,
        pool: AlphaPool | None,
        interval_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if config == "REF":
            return self._reference_function(source, intensity, I_floor, None, interval_index)
        if config not in ("P1", "P24") or type(pool) is not AlphaPool:
            raise AlphaMechanismError("S46J_MECHANISM_POOL_CONFIGURATION")
        return self._candidate_function(source, intensity, I_floor, pool, interval_index)

    @staticmethod
    def drain(pool: AlphaPool, *, interval_index: int) -> list[dict[str, Any]]:
        rows = pool.take_records()
        if tuple(
            row.get("label") if isinstance(row, Mapping) else None for row in rows
        ) != EXPECTED_LABELS:
            raise AlphaMechanismError("S46J_MECHANISM_ACTIVITY_ORDER")
        labels = {
            "ALPHA_CLEAN_SOURCE": "clean_source",
            "ALPHA_DENOM": "ratio",
            "ALPHA_RATIO": "ratio",
            "ALPHA_CLEAN_RESULT": "clean_alpha",
            "ALPHA_CLIP": "clip_alpha",
        }
        for row in rows:
            if not isinstance(row, dict) or not all(
                not isinstance(value, np.ndarray) for value in row.values()
            ):
                raise AlphaMechanismError("S46J_MECHANISM_ACTIVITY_NONSCALAR")
            row["context"] = labels[row["label"]]
            row["interval_index"] = int(interval_index)
            row["source"] = "propagate.py:830-833:alpha_ion"
        return copy.deepcopy(rows)


def _failure_array_record(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return {
        "path": path.name,
        "size_bytes": int(path.stat().st_size),
        "raw_sha256": pv.raw_sha256_file(path),
        "hash_scope": "raw_bytes",
    }


def _write_partial(
    output: Path,
    *,
    rows: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    failure: Mapping[str, Any] | None,
    error: BaseException,
    execution_commit: str,
) -> None:
    sg.write(
        output.parent / "MECHANISM_PARTIAL.json",
        {
            "schema": MECHANISM_SCHEMA + ".partial.v1",
            "rows": rows,
            "blocks": blocks,
            "failure": copy.deepcopy(dict(failure or {})),
            "error": repr(error),
            "execution_commit": execution_commit,
            "all_exact": False,
        },
    )


def run(
    root: str | Path,
    output: str | Path,
    *,
    shape: tuple[int, int, int],
    execution_commit: str,
    local_engineering: bool = False,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve(strict=True)
    output_path = Path(output).resolve()
    if output_path.exists():
        raise AlphaMechanismError("S46J_MECHANISM_NON_OVERWRITE")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    package = load_capture(root_path, shape=shape, expected_binding=expected_binding)
    stored_binding = package.get("source_binding", {})
    if stored_binding.get("execution_commit") != execution_commit:
        raise AlphaMechanismError("S46J_MECHANISM_EXECUTION_IDENTITY")
    if expected_binding is None and not local_engineering:
        raise AlphaMechanismError("S46J_MECHANISM_BOUND_INPUT_REQUIRED")
    runner = AlphaMechanism()
    runner_binding = runner.verify_binding()
    pools: dict[str, AlphaPool] = {}
    rows: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    started = time.perf_counter()
    try:
        pools["P1"] = AlphaPool(1)
        pools["P24"] = AlphaPool(1 if local_engineering else 24)
        records = package["records"]
        for group, order in enumerate(ORDERS, 1):
            for record in records:
                arrays = load_state_arrays(root_path, record)
                for field in (
                    "source_input",
                    "intensity_input",
                    "cleaned_source_output",
                    "alpha_output",
                ):
                    expected_fingerprint = record.get(field + "_fingerprint")
                    if not isinstance(expected_fingerprint, Mapping) or not _same_identity(
                        arrays[field], expected_fingerprint
                    ):
                        raise AlphaMechanismError(
                            "S46J_MECHANISM_CAPTURE_ARRAY_CHANGED:" + field
                        )
                source_template = arrays["source_input"]
                intensity_template = arrays["intensity_input"]
                expected_clean = arrays["cleaned_source_output"]
                expected_alpha = arrays["alpha_output"]
                for config in order:
                    pool = None if config == "REF" else pools[config]
                    runner.verify_binding()
                    if pool is not None:
                        runner.verify_pool(pool)
                    samples: list[float] = []
                    previous_outputs: tuple[np.ndarray, np.ndarray] | None = None
                    previous_identities: tuple[dict[str, Any], dict[str, Any]] | None = None
                    for index in range(7):
                        source = np.array(source_template, copy=True, order="C")
                        intensity = np.array(intensity_template, copy=True, order="C")
                        source_before = fingerprint(source, name="SOURCE_BEFORE")
                        intensity_before = fingerprint(intensity, name="INTENSITY_BEFORE")
                        tick = time.perf_counter_ns()
                        actual_clean, actual_alpha = runner.call(
                            config,
                            source,
                            intensity,
                            float(record["I_floor"]),
                            pool,
                            int(record["interval_index"]),
                        )
                        activity = (
                            []
                            if config == "REF"
                            else runner.drain(
                                pool,
                                interval_index=int(record["interval_index"]),
                            )
                        )
                        elapsed = (time.perf_counter_ns() - tick) / 1e9
                        if config != "REF":
                            labels = tuple(row.get("label") for row in activity)
                            if labels != EXPECTED_LABELS:
                                raise AlphaMechanismError("S46J_MECHANISM_ACTIVITY_LABELS")
                            if any(
                                row.get("requested_workers") != pool.count for row in activity
                            ):
                                raise AlphaMechanismError("S46J_MECHANISM_ACTIVITY_WORKERS")
                        exact = (
                            _same_data(actual_clean, expected_clean)
                            and _same_data(actual_alpha, expected_alpha)
                            and _same_identity(source, source_before)
                            and _same_identity(intensity, intensity_before)
                            and actual_clean is not source
                            and actual_alpha is not source
                            and not np.shares_memory(actual_clean, source)
                            and not np.shares_memory(actual_alpha, source)
                            and not np.shares_memory(actual_clean, intensity)
                            and not np.shares_memory(actual_alpha, intensity)
                            and (
                                previous_outputs is None
                                or (
                                    _same_identity(previous_outputs[0], previous_identities[0])
                                    and _same_identity(previous_outputs[1], previous_identities[1])
                                    and not np.shares_memory(actual_clean, previous_outputs[0])
                                    and not np.shares_memory(actual_alpha, previous_outputs[1])
                                )
                            )
                        )
                        row = {
                            "group": group,
                            "key": record["key"],
                            "context": record["context"],
                            "interval_index": int(record["interval_index"]),
                            "config": config,
                            "index": index,
                            "warmup": index < 2,
                            "wall_s": elapsed,
                            "exact": bool(exact),
                            "activity": activity,
                        }
                        rows.append(row)
                        if not exact:
                            failure = {
                                "group": group,
                                "key": record["key"],
                                "config": config,
                                "index": index,
                                "input_files": {
                                    field: copy.deepcopy(record[field + "_file"])
                                    for field in ("source_input", "intensity_input")
                                },
                                "expected_files": {
                                    "cleaned_source_output": copy.deepcopy(record["cleaned_source_output_file"]),
                                    "alpha_output": copy.deepcopy(record["alpha_output_file"]),
                                },
                                "actual_cleaned_file": _failure_array_record(
                                    output_path.parent,
                                    f"NONEXACT_{group}_{record['key']}_{config}_{index}_cleaned.npy",
                                    actual_clean,
                                ),
                                "actual_alpha_file": _failure_array_record(
                                    output_path.parent,
                                    f"NONEXACT_{group}_{record['key']}_{config}_{index}_alpha.npy",
                                    actual_alpha,
                                ),
                            }
                            raise AlphaMechanismError("SCIENTIFIC_NONEXACT:S46J_ALPHA_MECHANISM")
                        if index >= 2:
                            samples.append(elapsed)
                        previous_outputs = (actual_clean, actual_alpha)
                        previous_identities = (
                            fingerprint(actual_clean, name="PREVIOUS_CLEAN"),
                            fingerprint(actual_alpha, name="PREVIOUS_ALPHA"),
                        )
                        del source, intensity, actual_clean, actual_alpha
                    medians = statistics.median(samples)
                    # Store the median in a compact side table after all seven
                    # calls for this input/configuration have completed.
                    blocks.append(
                        {
                            "group": group,
                            "key": record["key"],
                            "context": record["context"],
                            "interval_index": int(record["interval_index"]),
                            "config": config,
                            "median_s": medians,
                        }
                    )
                del arrays
            block_group = [
                row
                for row in blocks
                if row["group"] == group and row["key"] == records[0]["key"]
            ]
            # The acceptance block is written below from all six state/config
            # medians; no data is discarded here.
    except BaseException as exc:
        primary_error = exc
        try:
            _write_partial(
                output_path,
                rows=rows,
                blocks=blocks,
                failure=failure,
                error=exc,
                execution_commit=execution_commit,
            )
        except BaseException as recording:
            exc.add_note("S46J_PARTIAL_RECORD_FAILURE:" + repr(recording))
    finally:
        close_errors: list[dict[str, Any]] = []
        snapshots: dict[str, Any] = {}
        terminal_drain: dict[str, Any] = {}
        for name, pool in pools.items():
            try:
                pool.close()
            except BaseException as exc:
                close_errors.append({"config": name, "error": repr(exc)})
            try:
                terminal_drain[name] = pool.take_records()
                snapshots[name] = pool.snapshot()
                if snapshots[name].get("close_error") is not None:
                    close_errors.append(
                        {
                            "config": name,
                            "error": "recorded_close_error:" + str(snapshots[name]["close_error"]),
                        }
                    )
            except BaseException as exc:
                close_errors.append({"config": name, "error": "drain_snapshot:" + repr(exc)})
        retirement_failed = bool(close_errors) or len(snapshots) != len(pools) or any(
            terminal_drain.values()
        ) or any(
            not snapshot.get("closed", False)
            or snapshot.get("active_thread_ids")
            or snapshot.get("pending_scalar_records")
            or snapshot.get("dropped_records", 0)
            or snapshot.get("invalid_records", 0)
            or snapshot.get("close_error") is not None
            for snapshot in snapshots.values()
        )
        try:
            sg.write(
                output_path.parent / "MECHANISM_POOLS.json",
                {
                    "schema": MECHANISM_SCHEMA + ".pools.v1",
                    "pools": snapshots,
                    "errors": close_errors,
                    "terminal_drain": terminal_drain,
                    "retirement_failed": retirement_failed,
                    "primary_error": None if primary_error is None else repr(primary_error),
                },
            )
        except BaseException as exc:
            retirement_failed = True
            close_errors.append({"error": "retirement_record:" + repr(exc)})
        if retirement_failed:
            cleanup = AlphaMechanismError(
                "S46J_MECHANISM_POOL_RETIREMENT:" + repr(close_errors)
            )
            if primary_error is None:
                primary_error = cleanup
            else:
                primary_error.add_note(str(cleanup))
    if primary_error is not None:
        raise primary_error

    if len(rows) != 126 or sum(1 for row in rows if row["warmup"]) != 36:
        raise AlphaMechanismError("S46J_MECHANISM_CALL_COUNTS")
    medians: dict[tuple[int, str, str], float] = {
        (int(row["group"]), str(row["key"]), str(row["config"])): float(row["median_s"])
        for row in blocks
    }
    gain_blocks: list[dict[str, Any]] = []
    for group, order in enumerate(ORDERS, 1):
        for record in package["records"]:
            key = str(record["key"])
            values = {
                config: medians[(group, key, config)]
                for config in CONFIGS
                if (group, key, config) in medians
            }
            if set(values) != set(CONFIGS):
                raise AlphaMechanismError("S46J_MECHANISM_MEDIAN_COUNTS")
            gain = 1.0 - values["P24"] / values["REF"]
            gain_blocks.append(
                {
                    "group": group,
                    "key": key,
                    "context": record["context"],
                    "interval_index": int(record["interval_index"]),
                    "medians_s": values,
                    "p24_gain": gain,
                    "gain_pass": bool(gain >= 0.05),
                }
            )
    gain_pass = all(bool(block["gain_pass"]) for block in gain_blocks)
    status = (
        "LOCAL_ENGINEERING_ONLY"
        if local_engineering
        else "PASS_SELECTED_MECHANISM"
        if gain_pass
        else "VALID_NEGATIVE_LOCAL_GAIN"
    )
    result = {
        "schema": MECHANISM_SCHEMA + ".result.v1",
        "status": status,
        "execution_commit": execution_commit,
        "shape": list(shape),
        "local_engineering": bool(local_engineering),
        "capture_manifest_raw_sha256": pv.raw_sha256_file(root_path / "CAPTURE.json"),
        "source_binding": copy.deepcopy(stored_binding),
        "runner_binding": runner_binding,
        "rows": rows,
        "blocks": gain_blocks,
        "all_exact": True,
        "warmup_calls": 36,
        "formal_calls": 90,
        "inputs": 3,
        "gain_pass": gain_pass,
        "formal14_allowed": bool(gain_pass and not local_engineering),
        "service_s": time.perf_counter() - started,
        "default": "I0",
        "production_qualified": False,
    }
    sg.write(output_path, result)
    return result


__all__ = [
    "AlphaMechanism",
    "AlphaMechanismError",
    "CONFIGS",
    "MECHANISM_SCHEMA",
    "ORDERS",
    "run",
]
