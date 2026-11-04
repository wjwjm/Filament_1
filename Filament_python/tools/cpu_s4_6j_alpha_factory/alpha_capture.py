"""Immediate, scalar-only capture package for the S4-6J alpha chain.

The hook calls this store for every one of the 32 alpha helper calls in an
eight-interval chain.  Only intervals 0, 3 and 7 write arrays.  Tokens and
the in-memory receipt never retain an ndarray; selected arrays are written
before the callback returns and represented thereafter by scalar fingerprints
and file records.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.cpu_s4_3g3_smallgrid import benchmark as sg
from tools.hpc_ops import provenance_v2 as pv
from tools.cpu_s4_6i_residual_parallel.power_hook import _seal

from . import alpha_hook as _alpha_hook
from .alpha_hook import (
    ALPHA_CLEAN_RESULT_HELPER,
    ALPHA_CLEAN_SOURCE_HELPER,
    ALPHA_CLIP_HELPER,
    ALPHA_RATIO_HELPER,
    ALPHA_SOURCE_SITE,
)
from .factory_timing import source_identity as factory_source_identity


CAPTURE_SCHEMA = "filament.cpu.s4_6j.alpha_capture.v1"
CAPTURE_INTERVALS = (0, 3, 7)
CAPTURE_CONTEXTS = (
    ALPHA_CLEAN_SOURCE_HELPER,
    ALPHA_RATIO_HELPER,
    ALPHA_CLEAN_RESULT_HELPER,
    ALPHA_CLIP_HELPER,
)
CONTEXT_LABELS = {
    ALPHA_CLEAN_SOURCE_HELPER: "clean_source",
    ALPHA_RATIO_HELPER: "ratio",
    ALPHA_CLEAN_RESULT_HELPER: "clean_alpha",
    ALPHA_CLIP_HELPER: "clip_alpha",
}
STATE_FIELDS = (
    "source_input",
    "intensity_input",
    "cleaned_source_output",
    "alpha_output",
)


class AlphaCaptureError(RuntimeError):
    """The alpha capture package violated its frozen interface."""


def _validate_array(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise AlphaCaptureError(f"S46J_CAPTURE_{name}_REQUIRES_NDARRAY")
    if value.ndim != 3:
        raise AlphaCaptureError(f"S46J_CAPTURE_{name}_REQUIRES_3D")
    if value.dtype != np.dtype("float64") or not value.dtype.isnative:
        raise AlphaCaptureError(f"S46J_CAPTURE_{name}_REQUIRES_NATIVE_FLOAT64")
    if not value.flags.c_contiguous:
        raise AlphaCaptureError(f"S46J_CAPTURE_{name}_REQUIRES_C_CONTIGUOUS")
    if any(int(size) <= 0 for size in value.shape):
        raise AlphaCaptureError(f"S46J_CAPTURE_{name}_REQUIRES_NONEMPTY")
    return value


def _memory_bounds(value: np.ndarray) -> tuple[int, int]:
    start = int(value.__array_interface__["data"][0])
    return start, start + int(value.nbytes)


def fingerprint(value: Any, *, name: str = "ARRAY") -> dict[str, Any]:
    """Return scalar layout, address-range and raw-byte evidence."""

    array = _validate_array(value, name=name)
    start, end = _memory_bounds(array)
    return {
        "shape": [int(size) for size in array.shape],
        "dtype": str(array.dtype),
        "strides": [int(value) for value in array.strides],
        "c_contiguous": bool(array.flags.c_contiguous),
        "writeable": bool(array.flags.writeable),
        "owns_data": bool(array.flags.owndata),
        "nbytes": int(array.nbytes),
        "memory_bounds": [start, end],
        "raw_sha256": pv.raw_sha256(memoryview(array).cast("B")),
    }


def array_record(path: str | Path) -> dict[str, Any]:
    file_path = Path(path).resolve(strict=True)
    return {
        "path": file_path.name,
        "size_bytes": int(file_path.stat().st_size),
        "classification": "external",
        "hash_scope": "raw_bytes",
        "raw_sha256": pv.raw_sha256_file(file_path),
    }


def _write_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return array_record(path)


def _same_fingerprint(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    fields = (
        "shape",
        "dtype",
        "strides",
        "c_contiguous",
        "writeable",
        "owns_data",
        "nbytes",
        "raw_sha256",
    )
    return all(actual.get(field) == expected.get(field) for field in fields)


def _same_persisted_fingerprint(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Compare array identity fields that survive an NPY round trip.

    The memory address, writability and ownership describe the live capture
    object and necessarily change when an NPY file is reopened.  Shape,
    layout, byte count and raw array bytes remain stable and are the fields
    needed to prove that the receipt points at the captured value.
    """

    fields = (
        "shape",
        "dtype",
        "strides",
        "c_contiguous",
        "nbytes",
        "raw_sha256",
    )
    return all(actual.get(field) == expected.get(field) for field in fields)


def _same_layout_fingerprint(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Compare live layout/ownership without requiring unchanged values."""

    fields = (
        "shape",
        "dtype",
        "strides",
        "c_contiguous",
        "writeable",
        "owns_data",
        "nbytes",
        "memory_bounds",
    )
    return all(actual.get(field) == expected.get(field) for field in fields)


class AlphaCaptureStore:
    """Capture the three selected alpha states during one R8 chain."""

    def __init__(self, root: str | Path, *, shape: tuple[int, int, int]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.shape = tuple(int(value) for value in shape)
        self.calls: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []
        self._states: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, Any] | None = None
        self.closed = False

    def _check_call_domain(
        self,
        *,
        context: str,
        interval_index: Any,
        source: np.ndarray,
    ) -> str:
        if self.closed or self.pending is not None:
            raise AlphaCaptureError("S46J_CAPTURE_LIFECYCLE")
        if type(interval_index) is not int or not 0 <= interval_index < 8:
            raise AlphaCaptureError("S46J_CAPTURE_INTERVAL_DOMAIN")
        if context not in CAPTURE_CONTEXTS:
            raise AlphaCaptureError("S46J_CAPTURE_CONTEXT_DOMAIN")
        call_index = len(self.calls)
        expected = (call_index // 4, CAPTURE_CONTEXTS[call_index % 4])
        if call_index >= 32 or (interval_index, context) != expected:
            raise AlphaCaptureError("S46J_CAPTURE_CALL_ORDER")
        value = _validate_array(source, name="SOURCE")
        if tuple(value.shape) != self.shape:
            raise AlphaCaptureError("S46J_CAPTURE_DOMAIN")
        return CONTEXT_LABELS[context]

    def _state_key(self, interval_index: int) -> str:
        return f"i{interval_index + 1}"

    def _state(self, interval_index: int) -> dict[str, Any]:
        key = self._state_key(interval_index)
        return self._states.setdefault(
            key,
            {
                "key": key,
                "interval_index": int(interval_index),
                "context": "alpha_ion",
                "source_site": ALPHA_SOURCE_SITE,
            },
        )

    @staticmethod
    def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise AlphaCaptureError(f"S46J_CAPTURE_{name}_MAPPING_REQUIRED")
        return value

    def before_call(
        self,
        *,
        context: str,
        interval_index: int,
        source: np.ndarray,
        source_site: str,
        capture_inputs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        label = self._check_call_domain(
            context=context, interval_index=interval_index, source=source
        )
        if source_site != ALPHA_SOURCE_SITE:
            raise AlphaCaptureError("S46J_CAPTURE_SOURCE_SITE")
        inputs = self._require_mapping(capture_inputs, "INPUTS")
        source_value = _validate_array(source, name="SOURCE")
        source_fp = fingerprint(source_value, name="SOURCE")
        token: dict[str, Any] = {
            "call_index": len(self.calls),
            "context": label,
            "helper": context,
            "interval_index": int(interval_index),
            "source_site": source_site,
            "source_before": source_fp,
            "selected": interval_index in CAPTURE_INTERVALS,
        }
        if label == "clean_source":
            named_source = inputs.get("source_input", source_value)
            named_source = _validate_array(named_source, name="SOURCE_INPUT")
            if named_source is not source_value and not np.shares_memory(named_source, source_value):
                raise AlphaCaptureError("S46J_CAPTURE_SOURCE_INPUT_CHANGED")
            if interval_index in CAPTURE_INTERVALS:
                state = self._state(interval_index)
                if "source_input_file" in state:
                    raise AlphaCaptureError("S46J_CAPTURE_DUPLICATE_STATE")
                state["source_input_fingerprint"] = fingerprint(named_source, name="SOURCE_INPUT")
                state["source_input_file"] = _write_array(
                    self.root, f"{self._state_key(interval_index)}_source_input.npy", named_source
                )
                token["source_input_fingerprint"] = copy.deepcopy(state["source_input_fingerprint"])
                token["source_input_file"] = copy.deepcopy(state["source_input_file"])
        elif label == "ratio":
            intensity = inputs.get("intensity_input")
            intensity = _validate_array(intensity, name="INTENSITY_INPUT")
            if intensity.shape != source_value.shape:
                raise AlphaCaptureError("S46J_CAPTURE_INTENSITY_SHAPE")
            floor = inputs.get("I_floor")
            if type(floor) is not float:
                raise AlphaCaptureError("S46J_CAPTURE_IFLOOR_TYPE")
            source_bounds = source_fp["memory_bounds"]
            intensity_fp = fingerprint(intensity, name="INTENSITY_INPUT")
            intensity_bounds = intensity_fp["memory_bounds"]
            if np.shares_memory(source_value, intensity):
                raise AlphaCaptureError("S46J_CAPTURE_SOURCE_INTENSITY_ALIAS")
            token.update(
                {
                    "intensity_input_fingerprint": intensity_fp,
                    "I_floor": floor,
                    "source_intensity_alias": False,
                    "source_memory_bounds": source_bounds,
                    "intensity_memory_bounds": intensity_bounds,
                }
            )
            if interval_index in CAPTURE_INTERVALS:
                state = self._state(interval_index)
                state["I_floor"] = floor
                original_bounds = state["source_input_fingerprint"]["memory_bounds"]
                original_alias = max(original_bounds[0], intensity_bounds[0]) < min(original_bounds[1], intensity_bounds[1])
                state["original_source_intensity_alias"] = original_alias
                if original_alias:
                    raise AlphaCaptureError("S46J_CAPTURE_ORIGINAL_SOURCE_INTENSITY_ALIAS")
                state["intensity_input_fingerprint"] = copy.deepcopy(intensity_fp)
                state["intensity_input_file"] = _write_array(
                    self.root,
                    f"{self._state_key(interval_index)}_intensity_input.npy",
                    intensity,
                )
                token["intensity_input_file"] = copy.deepcopy(state["intensity_input_file"])
        elif label == "clean_alpha":
            token["quotient_input_fingerprint"] = source_fp
        elif label == "clip_alpha":
            if not source_value.flags.writeable:
                raise AlphaCaptureError("S46J_CAPTURE_CLIP_REQUIRES_WRITABLE")
            token["alpha_before_clip_fingerprint"] = source_fp
        self.calls.append(
            {
                "call_index": int(token["call_index"]),
                "interval_index": int(interval_index),
                "context": label,
                "helper": context,
                "source_site": source_site,
            }
        )
        self.pending = token
        return copy.deepcopy(token)

    def after_call(
        self,
        token: Mapping[str, Any] | None,
        *,
        source: np.ndarray,
        output: np.ndarray,
        capture_inputs: Mapping[str, Any] | None = None,
        capture_outputs: Mapping[str, Any] | None = None,
    ) -> None:
        if token is None:
            raise AlphaCaptureError("S46J_CAPTURE_TOKEN_REQUIRED")
        if self.pending is None or dict(token) != self.pending:
            raise AlphaCaptureError("S46J_CAPTURE_TOKEN_CHANGED")
        current = _validate_array(source, name="SOURCE_AFTER")
        result = _validate_array(output, name="OUTPUT")
        label = str(self.pending["context"])
        interval = int(self.pending["interval_index"])
        current_fp = fingerprint(current, name="SOURCE_AFTER")
        if label == "clip_alpha":
            # The final maximum is deliberately in-place.  Its input values
            # may change, but the original object/layout/address range must
            # remain the same so the external-view contract is auditable.
            if not _same_layout_fingerprint(current_fp, self.pending["source_before"]):
                raise AlphaCaptureError("S46J_CAPTURE_CLIP_LAYOUT_CHANGED")
        elif not _same_fingerprint(current_fp, self.pending["source_before"]):
            raise AlphaCaptureError("S46J_CAPTURE_INPUT_CHANGED")
        inputs = self._require_mapping(capture_inputs, "INPUTS")
        outputs = self._require_mapping(capture_outputs, "OUTPUTS")
        state = self._state(interval) if interval in CAPTURE_INTERVALS else None
        if label == "clean_source":
            if result is current or np.shares_memory(result, current):
                raise AlphaCaptureError("S46J_CAPTURE_CLEAN_SOURCE_ALIAS")
            cleaned_fp = fingerprint(result, name="CLEANED_SOURCE_OUTPUT")
            self.pending["cleaned_source_output_fingerprint"] = cleaned_fp
            if state is not None:
                state["cleaned_source_output_fingerprint"] = copy.deepcopy(cleaned_fp)
                state["cleaned_source_output_file"] = _write_array(
                    self.root,
                    f"{self._state_key(interval)}_cleaned_source_output.npy",
                    result,
                )
                named = outputs.get("cleaned_source_output")
                if named is not None and named is not result:
                    raise AlphaCaptureError("S46J_CAPTURE_CLEAN_OUTPUT_TOKEN")
        elif label == "ratio":
            intensity = inputs.get("intensity_input")
            intensity = _validate_array(intensity, name="INTENSITY_AFTER")
            if not _same_fingerprint(fingerprint(intensity), self.pending["intensity_input_fingerprint"]):
                raise AlphaCaptureError("S46J_CAPTURE_INTENSITY_CHANGED")
            if np.shares_memory(current, intensity) or np.shares_memory(result, current) or np.shares_memory(result, intensity):
                raise AlphaCaptureError("S46J_CAPTURE_RATIO_ALIAS")
            ratio_fp = fingerprint(result, name="RATIO_OUTPUT")
            self.pending.update(
                {
                    "ratio_output_fingerprint": ratio_fp,
                    "ratio_source_intensity_alias": False,
                    "ratio_source_output_alias": False,
                    "ratio_output_intensity_alias": False,
                    "ratio_output_memory_bounds": ratio_fp["memory_bounds"],
                }
            )
            if state is not None:
                state["ratio_output_fingerprint"] = copy.deepcopy(ratio_fp)
        elif label == "clean_alpha":
            if result is current or np.shares_memory(result, current):
                raise AlphaCaptureError("S46J_CAPTURE_CLEAN_ALPHA_ALIAS")
            cleaned_alpha_fp = fingerprint(result, name="CLEAN_ALPHA_OUTPUT")
            self.pending["clean_alpha_output_fingerprint"] = cleaned_alpha_fp
            if state is not None:
                state["clean_alpha_output_fingerprint"] = copy.deepcopy(cleaned_alpha_fp)
        elif label == "clip_alpha":
            if result is not current or not np.shares_memory(result, current):
                raise AlphaCaptureError("S46J_CAPTURE_CLIP_NOT_INPLACE")
            final_fp = fingerprint(result, name="ALPHA_OUTPUT")
            self.pending["alpha_output_fingerprint"] = final_fp
            if state is not None:
                state["alpha_output_fingerprint"] = copy.deepcopy(final_fp)
                state["alpha_output_file"] = _write_array(
                    self.root,
                    f"{self._state_key(interval)}_alpha_output.npy",
                    result,
                )
                named = outputs.get("final_alpha")
                if named is not None and named is not result:
                    raise AlphaCaptureError("S46J_CAPTURE_FINAL_OUTPUT_TOKEN")
                state["clip_inplace"] = True
        self.pending = None
        if state is not None and label == "clip_alpha":
            # Ordered immutable names keep the persisted code identity stable
            # across processes with different Python hash seeds.
            required = (
                "source_input_file",
                "intensity_input_file",
                "cleaned_source_output_file",
                "alpha_output_file",
                "I_floor",
            )
            if not all(name in state for name in required):
                raise AlphaCaptureError("S46J_CAPTURE_STATE_INCOMPLETE")
            self.records.append(copy.deepcopy(state))

    def finish(self, *, binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.closed or self.pending is not None:
            raise AlphaCaptureError("S46J_CAPTURE_FINISH_LIFECYCLE")
        if len(self.calls) != 32 or len(self.records) != len(CAPTURE_INTERVALS):
            raise AlphaCaptureError("S46J_CAPTURE_COMPLETENESS")
        expected_keys = [self._state_key(index) for index in CAPTURE_INTERVALS]
        if [row.get("key") for row in self.records] != expected_keys:
            raise AlphaCaptureError("S46J_CAPTURE_STATE_ORDER")
        if sum(
            1
            for row in self.records
            for field in row
            if field.endswith("_file")
        ) != 12:
            raise AlphaCaptureError("S46J_CAPTURE_FILE_COUNT")
        source_binding = copy.deepcopy(dict(binding or {}))
        source_binding.setdefault("capture_source", capture_source_binding())
        receipt = {
            "schema": CAPTURE_SCHEMA,
            "candidate": "ION_ALPHA_CLEANUP_P24",
            "shape": list(self.shape),
            "dtype": "float64",
            "selected_interval_indices": list(CAPTURE_INTERVALS),
            "helper_order": ["clean_source", "ratio", "clean_alpha", "clip_alpha"],
            "total_calls": 32,
            "captured_states": len(CAPTURE_INTERVALS),
            "array_file_count": 12,
            "calls": copy.deepcopy(self.calls),
            "records": copy.deepcopy(self.records),
            "source_binding": source_binding,
            "records_scalar_only": True,
            "retained_array_references": 0,
            "hash_scope": "raw_bytes",
            "array_storage": "NPY_ALLOW_PICKLE_FALSE",
        }
        sg.write(self.root / "CAPTURE.json", receipt)
        self.closed = True
        return receipt


def capture_source_binding() -> dict[str, Any]:
    """Bind the capture implementation and its hook callbacks to the receipt."""

    methods = {}
    for name in ("before_call", "after_call"):
        function = getattr(AlphaCaptureStore, name)
        seal = _seal(function)
        methods[name] = {
            "module": str(function.__module__),
            "qualname": str(function.__qualname__),
            "code_identity": copy.deepcopy(seal["code_identity"]),
        }
    return {
        "schema": CAPTURE_SCHEMA + ".source_binding.v1",
        "source_identity": factory_source_identity(
            extra_paths=(__file__, Path(_alpha_hook.__file__).resolve())
        ),
        "methods": methods,
        "source_site": ALPHA_SOURCE_SITE,
        "helper_order": ["clean_source", "ratio", "clean_alpha", "clip_alpha"],
    }


def _verify_file_record(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise AlphaCaptureError("S46J_CAPTURE_FILE_RECORD")
    path = (root / str(record.get("path", ""))).resolve()
    if path.parent != root or path.suffix != ".npy":
        raise AlphaCaptureError("S46J_CAPTURE_FILE_PATH")
    if record != array_record(path):
        raise AlphaCaptureError("S46J_CAPTURE_FILE_IDENTITY")
    loaded = np.load(path, allow_pickle=False, mmap_mode="r")
    try:
        actual = fingerprint(loaded, name="CAPTURE_FILE")
    finally:
        del loaded
    # Container identity above includes the NPY header. The caller compares
    # this payload fingerprint with the separately saved array fingerprint.
    return actual


def load_capture(
    root: str | Path,
    *,
    shape: tuple[int, int, int],
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and fully validate a completed alpha capture receipt."""

    directory = Path(root).resolve(strict=True)
    receipt = sg.read(directory / "CAPTURE.json")
    if receipt.get("schema") != CAPTURE_SCHEMA:
        raise AlphaCaptureError("S46J_CAPTURE_SCHEMA")
    if tuple(receipt.get("shape", ())) != tuple(shape) or receipt.get("dtype") != "float64":
        raise AlphaCaptureError("S46J_CAPTURE_PACKAGE_DOMAIN")
    if receipt.get("selected_interval_indices") != list(CAPTURE_INTERVALS):
        raise AlphaCaptureError("S46J_CAPTURE_INTERVALS")
    if receipt.get("total_calls") != 32 or receipt.get("captured_states") != 3:
        raise AlphaCaptureError("S46J_CAPTURE_CALL_COUNTS")
    if receipt.get("array_file_count") != 12:
        raise AlphaCaptureError("S46J_CAPTURE_FILE_COUNT")
    source_binding = receipt.get("source_binding")
    capture_source = (
        source_binding.get("capture_source")
        if isinstance(source_binding, Mapping)
        else None
    )
    if not isinstance(capture_source, Mapping):
        raise AlphaCaptureError("S46J_CAPTURE_SOURCE_MANIFEST")
    if dict(capture_source) != capture_source_binding():
        raise AlphaCaptureError("S46J_CAPTURE_SOURCE_IMPLEMENTATION_CHANGED")
    if capture_source.get("schema") != CAPTURE_SCHEMA + ".source_binding.v1":
        raise AlphaCaptureError("S46J_CAPTURE_SOURCE_MANIFEST_SCHEMA")
    methods = capture_source.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != {"before_call", "after_call"}:
        raise AlphaCaptureError("S46J_CAPTURE_METHOD_MANIFEST")
    expected_calls = [
        (interval, CONTEXT_LABELS[context])
        for interval in range(8)
        for context in CAPTURE_CONTEXTS
    ]
    actual_calls = [
        (row.get("interval_index"), row.get("context")) for row in receipt.get("calls", [])
    ]
    if actual_calls != expected_calls:
        raise AlphaCaptureError("S46J_CAPTURE_CALL_ORDER")
    if expected_binding is not None:
        stored_binding = receipt.get("source_binding")
        if not isinstance(stored_binding, Mapping):
            raise AlphaCaptureError("S46J_CAPTURE_SOURCE_BINDING")
        for key, value in expected_binding.items():
            if stored_binding.get(key) != value:
                raise AlphaCaptureError("S46J_CAPTURE_SOURCE_BINDING:" + str(key))
    records = receipt.get("records")
    if not isinstance(records, list) or len(records) != 3:
        raise AlphaCaptureError("S46J_CAPTURE_RECORD_COUNT")
    expected_keys = [f"i{index + 1}" for index in CAPTURE_INTERVALS]
    if [row.get("key") for row in records] != expected_keys:
        raise AlphaCaptureError("S46J_CAPTURE_RECORD_ORDER")
    file_fields = [f"{field}_file" for field in STATE_FIELDS]
    all_files: list[str] = []
    for row in records:
        if type(row.get("I_floor")) is not float:
            raise AlphaCaptureError("S46J_CAPTURE_IFLOOR_RECORD")
        for field in file_fields:
            record = row.get(field)
            actual = _verify_file_record(directory, record)
            fingerprint_name = field.removesuffix("_file") + "_fingerprint"
            expected = row.get(fingerprint_name)
            if not isinstance(expected, Mapping) or not _same_persisted_fingerprint(
                actual, expected
            ):
                raise AlphaCaptureError("S46J_CAPTURE_FILE_FINGERPRINT:" + field)
            all_files.append(str(record["path"]))
        for name in (
            "source_input_fingerprint",
            "intensity_input_fingerprint",
            "cleaned_source_output_fingerprint",
            "alpha_output_fingerprint",
        ):
            if not isinstance(row.get(name), Mapping):
                raise AlphaCaptureError("S46J_CAPTURE_FINGERPRINT_RECORD:" + name)
    if len(all_files) != 12 or len(set(all_files)) != 12:
        raise AlphaCaptureError("S46J_CAPTURE_FILE_COUNT")
    return copy.deepcopy(receipt)


def load_state_arrays(
    root: str | Path,
    record: Mapping[str, Any],
    *,
    copy_arrays: bool = True,
) -> dict[str, np.ndarray]:
    """Restore one state as independent C-contiguous arrays."""

    directory = Path(root).resolve(strict=True)
    values: dict[str, np.ndarray] = {}
    for field in STATE_FIELDS:
        file_record = record[field + "_file"]
        path = (directory / file_record["path"]).resolve(strict=True)
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        if copy_arrays:
            value = np.array(value, copy=True, order="C")
        values[field] = value
    return values


__all__ = [
    "ALPHA_SOURCE_SITE",
    "AlphaCaptureError",
    "AlphaCaptureStore",
    "CAPTURE_CONTEXTS",
    "CAPTURE_INTERVALS",
    "CAPTURE_SCHEMA",
    "STATE_FIELDS",
    "array_record",
    "capture_source_binding",
    "fingerprint",
    "load_capture",
    "load_state_arrays",
]
