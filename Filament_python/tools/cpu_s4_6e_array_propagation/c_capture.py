"""Bounded interval capture for the C local mechanism comparison."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CAPTURE_SCHEMA = "filament.cpu_s4_6e.c_capture.v1"
DEFAULT_INTERVALS = (0, 3, 7)
DEFAULT_EXPECTED_TOTAL_CALLS = 8


class CaptureError(RuntimeError):
    """A selected C/R capture was incomplete or retained an unsafe reference."""


def _array_fingerprint(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not np.ndarray:
        raise CaptureError("C_CAPTURE_ARRAY_TYPE")
    contiguous = np.ascontiguousarray(value)
    raw = memoryview(contiguous).cast("B").tobytes()
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "strides": [int(item) for item in value.strides],
        "c_contiguous": bool(value.flags.c_contiguous),
        "writeable": bool(value.flags.writeable),
        "owndata": bool(value.flags.owndata),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nbytes": int(value.nbytes),
    }


def _copy_array(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    if type(value) is not np.ndarray:
        raise CaptureError(f"C_CAPTURE_{name}_TYPE")
    return np.array(value, copy=True, order="C")


class CaptureStore:
    """Capture selected source intervals while auditing the complete call sequence."""

    def __init__(
        self,
        intervals: tuple[int, ...] | None = None,
        *,
        max_records: int | None = None,
        expected_total_calls: int = DEFAULT_EXPECTED_TOTAL_CALLS,
        expected_interval_sequence: tuple[int, ...] | None = None,
        source_role: str = "R",
    ) -> None:
        selected = DEFAULT_INTERVALS if intervals is None else tuple(int(item) for item in intervals)
        if not selected or len(set(selected)) != len(selected) or any(item < 0 for item in selected):
            raise CaptureError("C_CAPTURE_INTERVALS")
        if type(expected_total_calls) is not int or expected_total_calls <= 0 or expected_total_calls > 16:
            raise CaptureError("C_CAPTURE_EXPECTED_CALL_COUNT")
        sequence = tuple(range(expected_total_calls)) if expected_interval_sequence is None else tuple(int(item) for item in expected_interval_sequence)
        if len(sequence) != expected_total_calls or len(set(sequence)) != len(sequence) or not set(selected).issubset(set(sequence)):
            raise CaptureError("C_CAPTURE_EXPECTED_INTERVAL_SEQUENCE")
        if max_records is None:
            max_records = len(selected)
        if type(max_records) is not int or max_records < len(selected) or max_records > 16:
            raise CaptureError("C_CAPTURE_RECORD_CAPACITY")
        if not str(source_role):
            raise CaptureError("C_CAPTURE_SOURCE_ROLE")
        self.intervals = selected
        self.max_records = int(max_records)
        self.expected_total_calls = int(expected_total_calls)
        self.expected_interval_sequence = sequence
        self.source_role = str(source_role)
        self.group = "UNSET"
        self.calls_seen = 0
        self.interval_sequence: list[int] = []
        self._records: dict[int, dict[str, Any]] = {}

    def begin(self, group: str, *, source_role: str | None = None) -> None:
        self.group = str(group)
        if source_role is not None:
            if not str(source_role):
                raise CaptureError("C_CAPTURE_SOURCE_ROLE")
            self.source_role = str(source_role)
        self.calls_seen = 0
        self.interval_sequence = []
        self._records = {}

    def before_call(self, interval_index: int | None, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        if self.calls_seen >= self.expected_total_calls:
            raise CaptureError("C_CAPTURE_EXTRA_CALL")
        if interval_index is None:
            raise CaptureError("C_CAPTURE_INTERVAL_MISSING")
        interval = int(interval_index)
        expected_interval = self.expected_interval_sequence[self.calls_seen]
        if interval != expected_interval:
            raise CaptureError(f"C_CAPTURE_INTERVAL_ORDER:{self.calls_seen}:{interval}:{expected_interval}")
        self.calls_seen += 1
        self.interval_sequence.append(interval)
        if interval not in self.intervals:
            return
        if interval in self._records:
            raise CaptureError(f"C_CAPTURE_DUPLICATE_INTERVAL:{interval}")
        if len(args) != 4:
            raise CaptureError("C_CAPTURE_NONLINEAR_ARGUMENT_COUNT")
        E, phase, alpha, dz = args
        dn_gas = kwargs.get("dn_gas")
        k0 = kwargs.get("k0")
        if type(E) is not np.ndarray or type(phase) is not np.ndarray or type(alpha) is not np.ndarray:
            raise CaptureError("C_CAPTURE_NONLINEAR_ARRAY_DOMAIN")
        if dn_gas is not None and type(dn_gas) is not np.ndarray:
            raise CaptureError("C_CAPTURE_DN_GAS_TYPE")
        if np.shares_memory(E, phase) or np.shares_memory(E, alpha) or (dn_gas is not None and np.shares_memory(E, dn_gas)):
            raise CaptureError("C_CAPTURE_NONLINEAR_ALIAS")
        self._records[interval] = {
            "schema": CAPTURE_SCHEMA + ".record.v1",
            "group": self.group,
            "source_role": self.source_role,
            "interval_index": interval,
            "ordinal_1based": interval + 1,
            "calls_seen_before": int(self.calls_seen - 1),
            "E_input": _copy_array(E, "E_INPUT"),
            "live_E_id": int(id(E)),
            "phase": _copy_array(phase, "PHASE"),
            "alpha": _copy_array(alpha, "ALPHA"),
            "dn_gas": _copy_array(dn_gas, "DN_GAS"),
            "dz": float(dz),
            "k0": None if k0 is None else float(k0),
            "output": None,
            "input_fingerprints": {
                "E": _array_fingerprint(E),
                "phase": _array_fingerprint(phase),
                "alpha": _array_fingerprint(alpha),
                "dn_gas": _array_fingerprint(dn_gas),
            },
        }

    def after_call(
        self,
        interval_index: int | None,
        result: Any,
        args: tuple[Any, ...] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if interval_index is None or int(interval_index) not in self.intervals:
            return
        interval = int(interval_index)
        row = self._records.get(interval)
        if row is None:
            raise CaptureError(f"C_CAPTURE_OUTPUT_WITHOUT_INPUT:{interval}")
        E_input = row["E_input"]
        if result is not None and type(result) is not np.ndarray:
            raise CaptureError("C_CAPTURE_OUTPUT_TYPE")
        if result is None or int(id(result)) != int(row["live_E_id"]):
            raise CaptureError("C_CAPTURE_OUTPUT_NOT_IN_PLACE")
        if args is not None:
            if len(args) != 4:
                raise CaptureError("C_CAPTURE_NONLINEAR_ARGUMENT_COUNT")
            _, phase, alpha, _ = args
            dn_gas = None if kwargs is None else kwargs.get("dn_gas")
            current = {
                "phase": _array_fingerprint(phase),
                "alpha": _array_fingerprint(alpha),
                "dn_gas": _array_fingerprint(dn_gas),
            }
            expected = row["input_fingerprints"]
            if any(current[name] != expected[name] for name in ("phase", "alpha", "dn_gas")):
                raise CaptureError("C_CAPTURE_INPUT_CHANGED")
        # The state helper returns the same live E object.  Copying the result
        # here prevents any later propagation write from changing the capture.
        row["output"] = _copy_array(result, "OUTPUT")
        row["output_fingerprint"] = _array_fingerprint(result)
        if row["output"] is None or E_input is None:
            raise CaptureError("C_CAPTURE_OUTPUT_MISSING")

    def records(self) -> dict[int, dict[str, Any]]:
        return copy.deepcopy(self._records)

    def release_owned_arrays(self) -> None:
        """Drop copied capture arrays after the package is durably written."""

        self._records = {}

    def require_complete(self) -> dict[int, dict[str, Any]]:
        if self.calls_seen != self.expected_total_calls:
            raise CaptureError(f"C_CAPTURE_CALL_COUNT:{self.calls_seen}:{self.expected_total_calls}")
        if tuple(self.interval_sequence) != self.expected_interval_sequence:
            raise CaptureError("C_CAPTURE_INTERVAL_SEQUENCE_INCOMPLETE")
        missing = [item for item in self.intervals if item not in self._records or self._records[item].get("output") is None]
        if missing:
            raise CaptureError(f"C_CAPTURE_INTERVALS_MISSING:{missing}")
        if set(self._records) != set(self.intervals) or len(self._records) > self.max_records:
            raise CaptureError("C_CAPTURE_RECORD_CAPACITY_EXCEEDED")
        return self.records()

    def metadata(self) -> dict[str, Any]:
        rows = self.require_complete()
        return {
            "schema": CAPTURE_SCHEMA + ".metadata",
            "group": self.group,
            "source_role": self.source_role,
            "selected_interval_indices": list(self.intervals),
            "selected_interval_ordinals_1based": [item + 1 for item in self.intervals],
            "expected_total_calls": int(self.expected_total_calls),
            "interval_sequence": list(self.interval_sequence),
            "calls_seen": int(self.calls_seen),
            "record_count": len(rows),
            "live_original_array_refs_retained": False,
            "captured_owned_arrays": True,
            "records": {
                str(key): {
                    "interval_index": int(value["interval_index"]),
                    "ordinal_1based": int(value["ordinal_1based"]),
                    "input_fingerprints": copy.deepcopy(value["input_fingerprints"]),
                    "output_fingerprint": copy.deepcopy(value["output_fingerprint"]),
                }
                for key, value in sorted(rows.items())
            },
        }


def merge_groups(stores: Mapping[str, CaptureStore]) -> dict[str, Any]:
    """Return two independent capture groups with scalar provenance."""

    if not isinstance(stores, Mapping) or set(stores) != {"G1", "G2"}:
        raise CaptureError("C_CAPTURE_GROUPS_EXACTLY_G1_G2")
    result: dict[str, Any] = {"schema": CAPTURE_SCHEMA + ".groups", "groups": {}}
    for group in ("G1", "G2"):
        store = stores[group]
        rows = store.require_complete()
        result["groups"][group] = {
            "metadata": store.metadata(),
            "records": rows,
        }
    return result


__all__ = [
    "CAPTURE_SCHEMA",
    "CaptureError",
    "CaptureStore",
    "DEFAULT_EXPECTED_TOTAL_CALLS",
    "DEFAULT_INTERVALS",
    "merge_groups",
]
