"""Bounded local REF/P1/P24 full-call mechanism comparison."""

from __future__ import annotations

import copy
import hashlib
import statistics
import time
from typing import Any, Mapping

import numpy as np

from KHz_filament.nonlinear import apply_nonlinear

from .c_capture import CAPTURE_SCHEMA, DEFAULT_EXPECTED_TOTAL_CALLS, DEFAULT_INTERVALS as CAPTURE_INTERVALS
from .c_instrument import compile_private_apply
from .c_pool import CPool, NullCPool


MECHANISM_SCHEMA = "filament.cpu_s4_6e.c_mechanism.v2"
CONFIG_ORDER = {
    "G1": ("REF", "P1", "P24"),
    "G2": ("P24", "P1", "REF"),
}
SELECTED_INTERVALS = (0, 3, 7)
WARMUPS = 2
FORMAL_CALLS = 5


class MechanismError(RuntimeError):
    """The local C mechanism comparison contract failed closed."""


class MechanismExactFailure(MechanismError):
    """A warmup or formal call was non-exact; the candidate stopped."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = dict(failure)
        super().__init__("C_MECHANISM_NONEXACT")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise MechanismError(reason)


def _equal(a: Any, b: Any) -> bool:
    return type(a) is np.ndarray and type(b) is np.ndarray and np.array_equal(a, b, equal_nan=True)


def _fingerprint(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not np.ndarray:
        raise MechanismError("C_MECHANISM_ARRAY_TYPE")
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "strides": [int(item) for item in value.strides],
        "c_contiguous": bool(value.flags.c_contiguous),
        "owndata": bool(value.flags.owndata),
        "sha256": hashlib.sha256(memoryview(contiguous).cast("B").tobytes()).hexdigest(),
    }


def _same_layout_and_values(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    # Restored arrays are independent owned copies by contract; source
    # ``owndata`` is retained as evidence but is not expected to survive the
    # restore operation.
    if left is None or right is None:
        return left is right
    keys = ("shape", "dtype", "strides", "c_contiguous", "sha256")
    return all(left.get(key) == right.get(key) for key in keys)


def _nonexact_evidence(actual: Any, reference: Any) -> dict[str, Any]:
    """Return bounded scalar evidence after an out-of-clock mismatch."""

    actual_fp = _fingerprint(actual) if type(actual) is np.ndarray else None
    reference_fp = _fingerprint(reference) if type(reference) is np.ndarray else None
    first_difference_index = None
    difference_count = None
    if type(actual) is np.ndarray and type(reference) is np.ndarray and actual.shape == reference.shape:
        try:
            equal = np.equal(actual, reference)
            if np.issubdtype(actual.dtype, np.inexact) and np.issubdtype(reference.dtype, np.inexact):
                equal = np.logical_or(equal, np.logical_and(np.isnan(actual), np.isnan(reference)))
            different = np.logical_not(equal)
            difference_count = int(np.count_nonzero(different))
            if difference_count:
                first_difference_index = [int(item) for item in np.argwhere(different)[0].tolist()]
        except (TypeError, ValueError):
            difference_count = None
    return {
        "actual": actual_fp,
        "reference": reference_fp,
        "first_difference_index": first_difference_index,
        "difference_count": difference_count,
    }


def _restore(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Restore all arrays and validate their input fingerprints before timing."""

    try:
        E = np.array(record["E_input"], copy=True, order="C")
        phase = np.array(record["phase"], copy=True, order="C")
        alpha = np.array(record["alpha"], copy=True, order="C")
        dz = float(record["dz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MechanismError("C_MECHANISM_CAPTURE_RECORD_INVALID") from exc
    dn_gas = None if record.get("dn_gas") is None else np.array(record["dn_gas"], copy=True, order="C")
    expected = record.get("input_fingerprints")
    if isinstance(expected, Mapping):
        actual = {"E": _fingerprint(E), "phase": _fingerprint(phase), "alpha": _fingerprint(alpha), "dn_gas": _fingerprint(dn_gas)}
        if any(not _same_layout_and_values(actual[name], expected.get(name)) for name in ("E", "phase", "alpha", "dn_gas")):
            raise MechanismError("C_MECHANISM_RESTORED_FINGERPRINT_CHANGED")
    if (
        E.ndim != 3
        or E.dtype != np.dtype("complex128")
        or not E.flags.c_contiguous
        or not E.flags.writeable
        or phase.shape != E.shape
        or alpha.shape != E.shape
        or phase.dtype != np.dtype("float64")
        or alpha.dtype != np.dtype("float64")
        or not phase.flags.c_contiguous
        or not alpha.flags.c_contiguous
        or (dn_gas is not None and (dn_gas.shape != E.shape[1:] or dn_gas.dtype != np.dtype("float64") or not dn_gas.flags.c_contiguous))
    ):
        raise MechanismError("C_MECHANISM_NATIVE_LAYOUT_DOMAIN")
    if np.shares_memory(E, phase) or np.shares_memory(E, alpha) or (dn_gas is not None and np.shares_memory(E, dn_gas)):
        raise MechanismError("C_MECHANISM_RESTORED_ALIAS")
    return E, phase, alpha, dz, {"dn_gas": dn_gas, "k0": record.get("k0")}


def _post_reason(E: np.ndarray, external_view: np.ndarray, phase_array: np.ndarray, alpha: np.ndarray, kwargs: Mapping[str, Any], result: Any, record: Mapping[str, Any], reference: Any) -> str | None:
    if result is not E:
        return "RESULT_NOT_IN_PLACE"
    if not np.shares_memory(external_view, E) or not np.array_equal(external_view, result, equal_nan=True):
        return "EXTERNAL_VIEW_NOT_UPDATED"
    if E.dtype != np.dtype("complex128") or not E.flags.c_contiguous or not E.flags.writeable:
        return "RESULT_LAYOUT_CHANGED"
    if not _same_layout_and_values(_fingerprint(phase_array), record["input_fingerprints"]["phase"]) or not _same_layout_and_values(_fingerprint(alpha), record["input_fingerprints"]["alpha"]) or not _same_layout_and_values(_fingerprint(kwargs["dn_gas"]), record["input_fingerprints"]["dn_gas"]):
        return "INPUT_CHANGED"
    if not _equal(result, reference):
        return "NONEXACT"
    return None


def _invoke_unclocked(function: Any, record: Mapping[str, Any], reference: Any, *, group: str, config: str, interval: int, phase: str) -> None:
    E, phase_array, alpha, dz, kwargs = _restore(record)
    external_view = E.view()
    result = function(E, phase_array, alpha, dz, **kwargs)
    # Result identity, input immutability, and exactness are all post-call
    # checks; none belongs to the timed numerical function body.
    reason = _post_reason(E, external_view, phase_array, alpha, kwargs, result, record, reference)
    if reason is not None:
        payload = {"group": group, "config": config, "interval_index": interval, "phase": phase, "failure": reason}
        if reason == "NONEXACT":
            payload["evidence"] = _nonexact_evidence(result, reference)
        raise MechanismExactFailure(payload)


def _invoke_timed(function: Any, record: Mapping[str, Any], reference: Any, *, group: str, config: str, interval: int, repeat: int) -> tuple[float, bool]:
    E, phase_array, alpha, dz, kwargs = _restore(record)
    external_view = E.view()
    start = time.perf_counter_ns()
    result = function(E, phase_array, alpha, dz, **kwargs)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    # Every check below intentionally runs after the stopwatch stops.
    reason = _post_reason(E, external_view, phase_array, alpha, kwargs, result, record, reference)
    if reason is not None:
        payload = {"group": group, "config": config, "interval_index": interval, "repeat": repeat, "failure": reason}
        if reason == "NONEXACT":
            payload["evidence"] = _nonexact_evidence(result, reference)
        raise MechanismExactFailure(payload)
    return float(elapsed), True


def _validate_groups(groups: Mapping[str, Any]) -> None:
    if not isinstance(groups, Mapping) or set(groups) != {"G1", "G2"}:
        raise MechanismError("C_MECHANISM_GROUPS_EXACTLY_G1_G2")
    for group in ("G1", "G2"):
        bundle = groups[group]
        records = bundle.get("records") if isinstance(bundle, Mapping) else None
        if not isinstance(records, Mapping) or set(int(key) for key in records) != set(SELECTED_INTERVALS):
            raise MechanismError(f"C_MECHANISM_INTERVALS:{group}")
        metadata = bundle.get("metadata") if isinstance(bundle, Mapping) else None
        if not isinstance(metadata, Mapping):
            raise MechanismError(f"C_MECHANISM_CAPTURE_METADATA:{group}")
        if tuple(int(item) for item in metadata.get("selected_interval_indices", ())) != tuple(CAPTURE_INTERVALS):
            raise MechanismError(f"C_MECHANISM_CAPTURE_METADATA_INTERVALS:{group}")
        if int(metadata.get("expected_total_calls", -1)) != DEFAULT_EXPECTED_TOTAL_CALLS:
            raise MechanismError(f"C_MECHANISM_CAPTURE_METADATA_CALL_COUNT:{group}")
        if tuple(int(item) for item in metadata.get("interval_sequence", ())) != tuple(range(DEFAULT_EXPECTED_TOTAL_CALLS)):
            raise MechanismError(f"C_MECHANISM_CAPTURE_METADATA_ORDER:{group}")
        source_role = metadata.get("source_role")
        if not isinstance(source_role, str) or not source_role:
            raise MechanismError(f"C_MECHANISM_CAPTURE_METADATA_SOURCE_ROLE:{group}")
        if str(metadata.get("group")) != group:
            raise MechanismError(f"C_MECHANISM_CAPTURE_METADATA_GROUP:{group}")
        for interval in SELECTED_INTERVALS:
            row = records.get(interval, records.get(str(interval)))
            if not isinstance(row, Mapping) or row.get("output") is None:
                raise MechanismError(f"C_MECHANISM_CAPTURE_INCOMPLETE:{group}:{interval}")
            if str(row.get("group")) != group:
                raise MechanismError(f"C_MECHANISM_CAPTURE_GROUP_BINDING:{group}:{interval}")
            if str(row.get("source_role")) != source_role:
                raise MechanismError(f"C_MECHANISM_CAPTURE_SOURCE_ROLE:{group}:{interval}")
            if int(row.get("interval_index", -1)) != interval or int(row.get("ordinal_1based", -1)) != interval + 1:
                raise MechanismError(f"C_MECHANISM_SOURCE_INTERVAL_BINDING:{group}:{interval}")
            fingerprints = row.get("input_fingerprints")
            if not isinstance(fingerprints, Mapping) or set(fingerprints) != {"E", "phase", "alpha", "dn_gas"}:
                raise MechanismError(f"C_MECHANISM_INPUT_FINGERPRINTS:{group}:{interval}")
            if fingerprints["E"].get("dtype") != "complex128" or not fingerprints["E"].get("c_contiguous"):
                raise MechanismError(f"C_MECHANISM_E_LAYOUT:{group}:{interval}")
            if any(fingerprints[name].get("dtype") != "float64" or not fingerprints[name].get("c_contiguous") for name in ("phase", "alpha")):
                raise MechanismError(f"C_MECHANISM_REAL_LAYOUT:{group}:{interval}")
            output_fingerprint = row.get("output_fingerprint")
            if not isinstance(output_fingerprint, Mapping):
                raise MechanismError(f"C_MECHANISM_OUTPUT_FINGERPRINT:{group}:{interval}")
    # Both reversed blocks deliberately replay the same eight captured states.
    for interval in SELECTED_INTERVALS:
        left = groups["G1"]["records"].get(interval, groups["G1"]["records"].get(str(interval)))
        right = groups["G2"]["records"].get(interval, groups["G2"]["records"].get(str(interval)))
        if left.get("input_fingerprints") != right.get("input_fingerprints"):
            raise MechanismError(f"C_MECHANISM_CAPTURE_STATE_MISMATCH:{interval}")
        if left.get("output_fingerprint") != right.get("output_fingerprint"):
            raise MechanismError(f"C_MECHANISM_CAPTURE_OUTPUT_FINGERPRINT_MISMATCH:{interval}")
        if not _equal(left.get("output"), right.get("output")):
            raise MechanismError(f"C_MECHANISM_CAPTURE_OUTPUT_MISMATCH:{interval}")


def _formal_row(group: str, config: str, interval: int, repeat: int, seconds: float, exact: bool) -> dict[str, Any]:
    return {
        "group": group,
        "config": config,
        "interval_index": int(interval),
        "ordinal_1based": int(interval) + 1,
        "repeat": int(repeat),
        "seconds": float(seconds),
        "exact": bool(exact),
        "restore_outside_clock": True,
        "post_checks_outside_clock": True,
    }


def _base_result(rows: list[dict[str, Any]], exact_by_config: Mapping[str, bool], pool_receipts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": MECHANISM_SCHEMA,
        "capture_schema": CAPTURE_SCHEMA,
        "config_order": copy.deepcopy(CONFIG_ORDER),
        "selected_interval_indices": list(SELECTED_INTERVALS),
        "warmup_calls": 0,
        "formal_calls": len(rows),
        "exact_by_config": dict(exact_by_config),
        "formal_rows": list(rows),
        "pool_receipts": copy.deepcopy(dict(pool_receipts)),
        "production_qualified": False,
    }


def run_local_mechanism(
    groups: Mapping[str, Any],
    *,
    warmups: int = WARMUPS,
    formal_repeats: int = FORMAL_CALLS,
) -> dict[str, Any]:
    """Run two reversed REF/P1/P24 blocks over captured intervals.

    Restoration/fingerprint checks are outside the stopwatch.  Any non-exact
    warmup or formal call returns a preserved failure index immediately; no
    remaining calls for that candidate or later propagation are attempted.
    """

    _validate_groups(groups)
    if int(warmups) != WARMUPS or int(formal_repeats) != FORMAL_CALLS:
        raise MechanismError("C_MECHANISM_FIXED_REPETITIONS")
    rows: list[dict[str, Any]] = []
    exact_by_config: dict[str, bool] = {"REF": True, "P1": True, "P24": True}
    pool_receipts: dict[str, Any] = {}
    times: dict[str, dict[str, dict[int, list[float]]]] = {
        group: {name: {interval: [] for interval in SELECTED_INTERVALS} for name in ("REF", "P1", "P24")}
        for group in ("G1", "G2")
    }
    warmup_count = 0
    try:
        for group in ("G1", "G2"):
            records = {int(key): value for key, value in groups[group]["records"].items()}
            for config in CONFIG_ORDER[group]:
                if config == "REF":
                    function: Any = apply_nonlinear
                    pool: CPool | NullCPool = NullCPool()
                else:
                    pool = CPool(1 if config == "P1" else 24)
                    function, _ = compile_private_apply(apply_nonlinear, pool)
                try:
                    for interval in SELECTED_INTERVALS:
                        record = records[interval]
                        reference = record["output"]
                        for repeat in range(1, WARMUPS + 1):
                            warmup_count += 1
                            _invoke_unclocked(function, record, reference, group=group, config=config, interval=interval, phase="warmup")
                        for repeat in range(1, FORMAL_CALLS + 1):
                            elapsed, exact = _invoke_timed(function, record, reference, group=group, config=config, interval=interval, repeat=repeat)
                            rows.append(_formal_row(group, config, interval, repeat, elapsed, exact))
                            times[group][config][interval].append(elapsed)
                finally:
                    if config != "REF":
                        pool.close()
                        drained = pool.take_records()
                        receipt = pool.snapshot()
                        receipt["records_drained"] = len(drained)
                        pool_receipts[f"{group}_{config}"] = receipt
                    else:
                        pool_receipts[f"{group}_{config}"] = pool.snapshot()
                exact_by_config[config] = True
    except MechanismExactFailure as failure:
        failed_config = failure.failure.get("config")
        if failed_config in exact_by_config:
            exact_by_config[str(failed_config)] = False
        result = _base_result(rows, exact_by_config, pool_receipts)
        result.update(
            status="SCIENTIFIC_NONEXACT",
            warmup_calls=warmup_count,
            stopped_before_remaining=True,
            failure=copy.deepcopy(failure.failure),
            mechanism_gate=False,
        )
        return result

    expected_formal = 2 * 3 * 3 * FORMAL_CALLS
    expected_warm = 2 * 3 * 3 * WARMUPS
    _require(warmup_count == expected_warm, "C_MECHANISM_WARMUP_CALL_COUNT")
    _require(len(rows) == expected_formal, "C_MECHANISM_FORMAL_CALL_COUNT")
    gains: dict[str, dict[str, Any]] = {}
    p24_block_gains: dict[str, float | None] = {}
    for group in ("G1", "G2"):
        gains[group] = {}
        for config in ("P1", "P24"):
            gains[group][config] = {"blocks": {}, "formal_count": 0}
            for interval in SELECTED_INTERVALS:
                ref_median = statistics.median(times[group]["REF"][interval])
                candidate_median = statistics.median(times[group][config][interval])
                gain = None if ref_median <= 0.0 else 1.0 - candidate_median / ref_median
                gains[group][config]["blocks"][str(interval)] = {
                    "reference_median_s": float(ref_median),
                    "candidate_median_s": float(candidate_median),
                    "gain": None if gain is None else float(gain),
                    "formal_count": len(times[group][config][interval]),
                }
                gains[group][config]["formal_count"] += len(times[group][config][interval])
                if config == "P24":
                    p24_block_gains[f"{group}:{interval}"] = None if gain is None else float(gain)
    p24_gate = all(value is not None and value >= 0.05 for value in p24_block_gains.values())
    return {
        "schema": MECHANISM_SCHEMA,
        "capture_schema": CAPTURE_SCHEMA,
        "config_order": copy.deepcopy(CONFIG_ORDER),
        "selected_interval_indices": list(SELECTED_INTERVALS),
        "warmup_calls": expected_warm,
        "formal_calls": expected_formal,
        "exact_by_config": exact_by_config,
        "gains": gains,
        "p24_block_gains": p24_block_gains,
        "p24_block_gate": bool(p24_gate),
        "formal_rows": rows,
        "pool_receipts": pool_receipts,
        "mechanism_gate": bool(p24_gate),
        "production_qualified": False,
        "timer_contract": "restore_and_post_checks_outside_function_clock",
    }


__all__ = [
    "CONFIG_ORDER",
    "FORMAL_CALLS",
    "MECHANISM_SCHEMA",
    "MechanismError",
    "MechanismExactFailure",
    "SELECTED_INTERVALS",
    "WARMUPS",
    "run_local_mechanism",
]
