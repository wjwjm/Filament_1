"""Minimal E5-1A adapter around the existing Streaming and optical paths.

The module is deliberately an outer glue layer.  ``propagate_one_pulse``,
``build_transverse_input_field``, the HR-3 sinks, and
``StreamingLifecycle`` remain the scientific authorities.  The new code only
binds a pulse to a durable CURRENT generation, supplies an exact schedule
prefix, and provides the non-final/final POST hooks needed by a multi-pulse
driver.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .confio import load_all
from .constants import N0_air, Ui_N2, c0, n2_air
from .device import to_cpu, xp
from .hr4e5_evidence import (atomic_json, compare_arrays_exact, compare_object_sets,
    lineage_binding, sha256_array, sha256_file, validate_paired_exact_report,
    validate_ready_receipt, write_exact_report, _safe_child_path)
from .hr4e5s_streaming import FIELDS, StreamingLifecycle, StreamingLifecycleError, _content_hash, make_post_commit_hook
from .longitudinal import DepositionContract, LongitudinalSchedule, build_deposition_contract
from .propagate import propagate_one_pulse
from .runner import apply_thin_lens_achromatic, build_transverse_input_field
from .slow_state import HR3BDiagnosticSink, validate_hr3b_parameters
from .thermalization import ThermalDiagnosticSink, ThermalSamplePlan
from .hr4e5_storage import StorageBudget


FORMAL_ENTRY_SCHEMA = "khz_filament.hr4e5.e5_1a.formal_entry.v1"
ROOT_METADATA_SCHEMA = "khz_filament.hr4e5.e5_1a.root_metadata.v1"
ADMISSION_SCHEMA = "khz_filament.hr4e5.e5_1a.admission.v1"
BLOCK_SIZE = 8
QUEUE_DEPTH = 16


_ADMISSION_REQUIRED = (
    "campaign_id", "execution_mode", "runtime_or_compatibility", "config_identity",
    "effective_params", "source_identity", "lut_identity", "schedule_identity",
    "grid_identity", "pre0_identity", "n_pulses", "k", "block_size", "queue_depth",
    "f_rep", "dt_hydro", "precision", "r_roots", "c_roots", "pulse_attempt_epoch",
    "budget_identity", "retention_identity",
)
_PLACEHOLDERS = {"", "unknown", "placeholder", "missing", "not_provided", "not_materialized", "none", "null"}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    def safe(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            return {str(key): safe(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(value) for value in item]
        if isinstance(item, np.generic):
            return item.item()
        return item
    return json.dumps(safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def admission_identity_hash(identity: Mapping[str, Any]) -> str:
    value = {str(k): v for k, v in identity.items() if str(k) not in {"identity_sha256", "created_utc"}}
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _reject_placeholder(value: Any, label: str) -> None:
    if value is None:
        raise ValueError(f"formal admission field is missing: {label}")
    if isinstance(value, str) and value.strip().lower() in _PLACEHOLDERS:
        raise ValueError(f"formal admission field is a placeholder: {label}")
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"formal admission field is empty: {label}")
        for key, item in value.items():
            _reject_placeholder(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"formal admission field is empty: {label}")
        for index, item in enumerate(value):
            _reject_placeholder(item, f"{label}[{index}]")


def build_admission_identity(
    *, campaign_id: str, execution_mode: str = "formal", runtime_or_compatibility: Any = None,
    config_identity: Any = None, effective_params: Any = None, source_identity: Any = None,
    lut_identity: Any = None, schedule_identity: Any = None, grid_identity: Any = None,
    pre0_identity: Any = None, n_pulses: int | None = None, k: int | None = None,
    block_size: int = BLOCK_SIZE, queue_depth: int = QUEUE_DEPTH, f_rep: float | None = None,
    dt_hydro: float | None = None, precision: Any = None, r_roots: Any = None,
    c_roots: Any = None, pulse_attempt_epoch: Any = None, budget_identity: Any = None,
    retention_identity: Any = None, scope: str | None = None, **extra: Any,
) -> dict[str, Any]:
    """Create and validate the immutable outer admission identity.

    ``execution_mode=TEST_FIXTURE_ONLY`` is intentionally explicit and cannot
    be accepted by a formal coordinator.  Formal mode rejects absent or
    placeholder evidence before any scientific payload creation.
    """
    mode = str(execution_mode)
    values = {
        "campaign_id": str(campaign_id), "execution_mode": mode,
        "runtime_or_compatibility": runtime_or_compatibility,
        "config_identity": config_identity, "effective_params": effective_params,
        "source_identity": source_identity, "lut_identity": lut_identity,
        "schedule_identity": schedule_identity, "grid_identity": grid_identity,
        "pre0_identity": pre0_identity, "n_pulses": n_pulses, "k": k,
        "block_size": block_size, "queue_depth": queue_depth, "f_rep": f_rep,
        "dt_hydro": dt_hydro, "precision": precision, "r_roots": r_roots,
        "c_roots": c_roots, "pulse_attempt_epoch": pulse_attempt_epoch,
        "budget_identity": budget_identity, "retention_identity": retention_identity,
    }
    if mode == "TEST_FIXTURE_ONLY":
        if str(scope or "TEST_FIXTURE_ONLY") != "TEST_FIXTURE_ONLY":
            raise ValueError("fixture admission identity must declare TEST_FIXTURE_ONLY scope")
        values["scope"] = "TEST_FIXTURE_ONLY"
    else:
        for key in _ADMISSION_REQUIRED:
            _reject_placeholder(values.get(key), key)
        if str(mode).upper() in {"TEST", "FIXTURE", "MOCK"}:
            raise ValueError("formal admission cannot use a test execution mode")
        if int(block_size) != BLOCK_SIZE or int(queue_depth) != QUEUE_DEPTH:
            raise ValueError("formal admission requires block8/queue16")
        values["scope"] = "FORMAL"
    values.update({str(key): value for key, value in extra.items()})
    values["schema"] = ADMISSION_SCHEMA
    values["identity_sha256"] = admission_identity_hash(values)
    return values


def build_fixture_admission_identity(*, campaign_id: str = "TEST_FIXTURE_ONLY", n_pulses: int = 3,
                                     k: int = 32, shape: Sequence[int] = (8, 8),
                                     block_size: int = BLOCK_SIZE, queue_depth: int = QUEUE_DEPTH,
                                     epoch: str = "fixture-epoch") -> dict[str, Any]:
    """Return an explicit, non-qualifying identity for the CPU fixture."""
    return build_admission_identity(
        campaign_id=campaign_id, execution_mode="TEST_FIXTURE_ONLY",
        runtime_or_compatibility={"runtime": "local-cpu", "scope": "TEST_FIXTURE_ONLY"},
        config_identity={"config": "fixture", "scope": "TEST_FIXTURE_ONLY"},
        effective_params={"shape": list(shape)}, source_identity={"source": "fixture"},
        lut_identity={"lut": "fixture-none"}, schedule_identity={"k": int(k), "scope": "fixture"},
        grid_identity={"shape": list(shape), "dtype": "float64"},
        pre0_identity={"rule": "zero_velocity_fixture"}, n_pulses=n_pulses, k=k,
        block_size=block_size, queue_depth=queue_depth, f_rep=5e6, dt_hydro=1e-7,
        precision={"fields": "float64", "optical": "complex128"},
        r_roots={"root": "R"}, c_roots={"root": "C"},
        pulse_attempt_epoch={"pulse": 0, "attempt": 0, "epoch": epoch},
        budget_identity={"max_campaign_live_bytes": 300 * 1024**3, "final_output_budget_bytes": 64 * 1024**3},
        retention_identity={"policy": "fixture"}, scope="TEST_FIXTURE_ONLY",
    )


def validate_admission_identity(identity: Mapping[str, Any], *, formal: bool = True,
                                expected_hash: str | None = None) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or identity.get("schema") != ADMISSION_SCHEMA:
        raise ValueError("admission identity schema is invalid")
    payload = dict(identity)
    mode = str(payload.get("execution_mode", ""))
    if formal:
        if mode.strip().lower() in _PLACEHOLDERS:
            raise ValueError("formal admission field is missing: execution_mode")
        if mode == "TEST_FIXTURE_ONLY" or payload.get("scope") != "FORMAL":
            raise ValueError("fixture-only admission identity cannot qualify formal mode")
        for key in _ADMISSION_REQUIRED:
            _reject_placeholder(payload.get(key), key)
    elif mode == "TEST_FIXTURE_ONLY" and payload.get("scope") != "TEST_FIXTURE_ONLY":
        raise ValueError("fixture admission identity scope is invalid")
    saved = str(payload.get("identity_sha256", ""))
    if not saved or saved != admission_identity_hash(payload) or (expected_hash is not None and saved != str(expected_hash)):
        raise ValueError("admission identity hash mismatch")
    return payload


def persist_admission_identity(path: str | Path, identity: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    value = validate_admission_identity(identity, formal=str(identity.get("execution_mode")) != "TEST_FIXTURE_ONLY")
    destination = Path(path)
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if admission_identity_hash(saved) != admission_identity_hash(value):
            raise ValueError("persisted admission identity is immutable")
        return saved
    atomic_json(destination, value, overwrite=overwrite)
    return value


def load_admission_identity(path: str | Path, *, formal: bool = True, expected_hash: str | None = None) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_admission_identity(value, formal=formal, expected_hash=expected_hash)


create_admission_identity = build_admission_identity
verify_admission_identity = validate_admission_identity


def _require_creation_context(*, admission_identity: Mapping[str, Any] | None,
                              storage_budget: StorageBudget | None,
                              creation_intent: Mapping[str, Any] | str | None,
                              fixture_only: bool) -> tuple[dict[str, Any] | None, StorageBudget | None, dict[str, Any] | None]:
    if fixture_only:
        if admission_identity is None:
            raise ValueError("fixture-only creation requires explicit TEST_FIXTURE_ONLY admission identity")
        value = validate_admission_identity(admission_identity, formal=False)
        if value.get("execution_mode") != "TEST_FIXTURE_ONLY":
            raise ValueError("fixture-only creation requires TEST_FIXTURE_ONLY identity")
        return value, storage_budget, None
    if admission_identity is None or storage_budget is None or creation_intent is None:
        raise ValueError("formal creation requires admission identity, StorageBudget, and creation intent")
    identity = validate_admission_identity(admission_identity, formal=True)
    if storage_budget.admission_hash != identity["identity_sha256"]:
        raise ValueError("StorageBudget admission identity does not match formal entry")
    if not storage_budget.require_quota:
        raise ValueError("formal creation requires fail-closed quota reporting")
    if isinstance(creation_intent, Mapping):
        intent_id = creation_intent.get("intent_id")
        if not intent_id:
            raise ValueError("formal creation intent id is missing")
        intent = storage_budget.validate_intent(str(intent_id))
    else:
        intent = storage_budget.validate_intent(str(creation_intent))
    if str(intent.get("admission_hash", "")) != identity["identity_sha256"]:
        raise ValueError("creation intent admission identity does not match formal entry")
    if intent.get("status") not in {"ACTIVE", "COMPLETED"}:
        raise ValueError("creation intent is not active")
    return identity, storage_budget, intent


def _validate_creation_target(
    *, root: str | Path, identity: Mapping[str, Any], budget: StorageBudget,
    intent: Mapping[str, Any], role: str, trajectory: str, pulse: int,
    attempt: int, generation: str, allow_existing_complete: bool = False,
) -> None:
    """Validate the durable intent before any lifecycle payload is written."""
    target = Path(root).resolve()
    intent_id = str(intent.get("intent_id", ""))
    if not intent_id:
        raise ValueError("formal creation intent id is missing")
    existing_complete = (target / "E5_1A_READY.json").is_file()
    durable = budget.validate_intent(intent_id, path=target, require_active=not existing_complete)
    expected = {
        "role": str(role), "trajectory": str(trajectory), "pulse": int(pulse),
        "attempt": int(attempt), "admission_hash": str(identity["identity_sha256"]),
    }
    for field, value in expected.items():
        if str(durable.get(field)) != str(value):
            raise ValueError(f"creation intent {field} does not match formal entry")
    declared_generation = durable.get("generation")
    if declared_generation is None:
        declared_generation = (durable.get("metadata") or {}).get("generation")
    if declared_generation is None:
        raise ValueError("formal creation intent generation is missing")
    if str(declared_generation) != str(generation):
        raise ValueError("creation intent generation does not match formal entry")
    if existing_complete:
        if not allow_existing_complete or durable.get("status") != "COMPLETED":
            raise ValueError("existing successor is not bound to a completed creation intent")
        saved = json.loads((target / "E5_1A_READY.json").read_text(encoding="utf-8"))
        if (str(saved.get("creation_intent_id", "")) != intent_id
                or str(saved.get("creation_intent_generation", "")) != str(generation)):
            raise ValueError("existing successor creation intent binding changed")
        return
    if target.exists() and any(path.is_file() for path in target.rglob("*")):
        raise ValueError("creation target already contains an unowned or conflicting payload")


def interpulse_worker_parameters(*, f_rep: float, dt_hydro: float, **coefficients: Any) -> dict[str, Any]:
    from .hr4d_pulse_lifecycle import build_interpulse_step_schedule
    schedule = build_interpulse_step_schedule(f_rep=f_rep, dt_hydro=dt_hydro)
    if schedule.remainder_s != 0.0:
        raise ValueError('existing Streaming worker does not support a fractional hydro remainder')
    return dict(coefficients, dt_hydro=schedule.dt_hydro_s, n_hydro_steps=schedule.full_step_count)


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _field_copy(fields: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result = {name: np.array(fields[name], dtype=np.float64, copy=True) for name in FIELDS}
    shape = result["delta_n"].shape
    if any(value.ndim != 2 or value.shape != shape or value.dtype != np.dtype(np.float64) for value in result.values()):
        raise ValueError("Streaming pulse fields require matching finite float64 maps")
    if any(not np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("Streaming pulse fields must be finite")
    for value in result.values():
        value.setflags(write=False)
    return result


class StreamingPulseReadView:
    """Read-only PRE view bound to one Streaming CURRENT generation.

    Each interval is read once and updated once by the optical call.  Updates
    are returned to the caller but never written into CURRENT; the existing
    lifecycle hook persists the resulting POST with PRE velocities intact.
    """

    def __init__(
        self,
        lifecycle: StreamingLifecycle | str | Path,
        *,
        source_indices: Sequence[int] | None = None,
        expected_generation: str | None = None,
        expected_content_sha256: str | None = None,
        expected_admission_hash: str | None = None,
    ):
        self.root = Path(lifecycle.root if isinstance(lifecycle, StreamingLifecycle) else lifecycle).resolve()
        self._closed = False
        initial = StreamingLifecycle.open(self.root)
        if initial._authoritative_namespace != "CURRENT" or (self.root / "authoritative_generation.json").exists():
            raise StreamingLifecycleError("optical PRE view requires an unpromoted CURRENT generation")
        self.current_generation = str(expected_generation or initial.manifest["current_generation"])
        self.current_content_sha256 = str(expected_content_sha256 or initial.manifest["current_content_sha256"])
        if self.current_generation != initial.manifest["current_generation"] or self.current_content_sha256 != initial.manifest["current_content_sha256"]:
            raise StreamingLifecycleError("requested PRE identity differs from CURRENT")
        if expected_admission_hash is not None:
            metadata_path = self.root / "E5_1A_ROOT_METADATA.json"
            if not metadata_path.is_file():
                raise StreamingLifecycleError("PRE admission identity metadata is missing")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if str(metadata.get("admission_identity_sha256", "")) != str(expected_admission_hash):
                raise StreamingLifecycleError("PRE admission identity differs from CURRENT")
        count = int(initial.manifest["expected_screen_count"])
        if source_indices is None:
            self.source_indices = tuple(range(count))
        else:
            values = tuple(int(value) for value in source_indices)
            if values != tuple(range(count)):
                raise ValueError("Streaming CURRENT view requires contiguous local source indices")
            self.source_indices = values
        self.shape = tuple(int(value) for value in initial.manifest["shape"])
        self._read: dict[int, dict[str, np.ndarray]] = {}
        self._updated: set[int] = set()

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("Streaming PRE view is closed")

    def _open_current(self) -> StreamingLifecycle:
        self._assert_open()
        lifecycle = StreamingLifecycle.open(self.root)
        if lifecycle._authoritative_namespace != "CURRENT" or (self.root / "authoritative_generation.json").exists():
            raise StreamingLifecycleError("CURRENT was promoted while optical PRE view was active")
        if str(lifecycle.manifest["current_generation"]) != self.current_generation or str(lifecycle.manifest["current_content_sha256"]) != self.current_content_sha256:
            raise StreamingLifecycleError("CURRENT identity changed while optical PRE view was active")
        return lifecycle

    def read_interval(self, interval_index: int) -> np.ndarray:
        lifecycle = self._open_current()
        index = int(interval_index)
        if index < 0 or index >= len(self.source_indices):
            raise IndexError("Streaming PRE interval is outside the current generation")
        if index in self._read:
            raise ValueError("Streaming PRE interval may be read only once")
        fields = _field_copy(lifecycle.current_fields(index))
        self._read[index] = fields
        return fields["delta_n"].copy()

    def update_interval(self, interval_index: int, delta_n_increment: Any) -> np.ndarray:
        self._open_current()
        index = int(interval_index)
        if index not in self._read or index in self._updated:
            raise ValueError("Streaming PRE view requires one read and one update per interval")
        increment = np.asarray(to_cpu(delta_n_increment), dtype=np.float64)
        before = self._read[index]["delta_n"]
        if increment.shape != before.shape or not np.all(np.isfinite(increment)):
            raise ValueError("Streaming HR-3B increment has invalid shape or values")
        after = np.asarray(before + increment, dtype=np.float64)
        if not np.all(np.isfinite(after)):
            raise ValueError("Streaming POST state is non-finite")
        self._updated.add(index)
        return after

    def post_fields(self, interval_index: int, state_after: Any) -> dict[str, np.ndarray]:
        """Return a detached POST payload with PRE velocities inherited."""
        index = int(interval_index)
        if index not in self._read:
            raise ValueError("POST payload requested before PRE read")
        value = np.asarray(state_after, dtype=np.float64)
        fields = self._read[index]
        if value.shape != self.shape or not np.all(np.isfinite(value)):
            raise ValueError("POST delta_n has invalid shape or values")
        return {
            "delta_n": value.copy(),
            "vx": np.asarray(fields["vx"], dtype=np.float64).copy(),
            "vy": np.asarray(fields["vy"], dtype=np.float64).copy(),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "hr3b_state_schema": "khz_filament.hr4e5.e5_1a.streaming_pre.v1",
            "hr3b_state_filename": "streaming_manifest.json",
            "hr3b_state_dtype": "float64",
            "hr3b_state_shape": (len(self.source_indices), *self.shape),
            "hr3b_state_interval_centered": True,
            "hr3b_state_disk_backed": True,
            "current_generation": self.current_generation,
            "current_content_sha256": self.current_content_sha256,
            "read_count": len(self._read),
            "updated_count": len(self._updated),
        }

    @property
    def complete(self) -> bool:
        return len(self._read) == len(self.source_indices) and self._updated == set(self.source_indices)

    def close(self) -> None:
        self._closed = True


def build_prefix_schedule(full_schedule: LongitudinalSchedule, n_intervals: int, *, block_size: int = BLOCK_SIZE) -> LongitudinalSchedule:
    """Slice an existing schedule without rebuilding or snapping its edges."""
    if not isinstance(full_schedule, LongitudinalSchedule):
        raise TypeError("full_schedule must be a LongitudinalSchedule")
    count = int(n_intervals)
    if count <= 0 or count > full_schedule.n_intervals:
        raise ValueError("prefix interval count is outside the full schedule")
    if int(block_size) <= 0 or count % int(block_size) != 0:
        raise ValueError("prefix interval count must satisfy the frozen block size")
    edges = tuple(full_schedule.z_edges[: count + 1])
    dz_values = tuple(full_schedule.dz_intervals[:count])
    intervals = tuple(full_schedule.intervals[:count])
    result = LongitudinalSchedule(
        z_edges=edges, dz_intervals=dz_values, intervals=intervals,
        z_start=edges[0], z_end=edges[-1],
    )
    result.validate()
    return result


slice_longitudinal_schedule = build_prefix_schedule


def pre0_fields_from_delta_n(delta_n: Any, *, expected_count: int | None = None) -> dict[str, np.ndarray]:
    """Build the engineering PRE0 from an existing delta-n prefix."""
    if isinstance(delta_n, (str, Path)):
        loaded = np.load(Path(delta_n), mmap_mode="r", allow_pickle=False)
        try:
            value = np.asarray(loaded)
            result = np.array(value, dtype=np.float64, copy=True)
        finally:
            close = getattr(loaded, "_mmap", None)
            if close is not None:
                close.close()
    else:
        result = np.array(delta_n, dtype=np.float64, copy=True)
    if result.ndim != 3 or result.dtype != np.dtype(np.float64) or not np.all(np.isfinite(result)):
        raise ValueError("PRE0 delta_n must be a finite float64 [K, Ny, Nx] array")
    if expected_count is not None and result.shape[0] != int(expected_count):
        raise ValueError("PRE0 interval count differs from expected prefix")
    return {"delta_n": result, "vx": np.zeros_like(result), "vy": np.zeros_like(result)}


def _records_from_schedule(schedule: LongitudinalSchedule, *, source_indices: Sequence[int] | None = None) -> list[dict[str, Any]]:
    indices = tuple(range(schedule.n_intervals)) if source_indices is None else tuple(int(value) for value in source_indices)
    if len(indices) != schedule.n_intervals:
        raise ValueError("screen source index count differs from schedule")
    return [
        {"ordinal": index, "screen_id": f"source_index_{source_index:06d}", "source_index": source_index, "z_m": float(0.5 * (schedule.z_edges[index] + schedule.z_edges[index + 1]))}
        for index, source_index in enumerate(indices)
    ]


def _write_root_metadata(root: Path, value: Mapping[str, Any]) -> None:
    payload = {"schema": ROOT_METADATA_SCHEMA, **dict(value), "created_utc": _utc()}
    atomic_json(root / "E5_1A_ROOT_METADATA.json", payload, overwrite=False)


def create_pre0_root(
    *, root: str | Path, delta_n: Any, schedule: LongitudinalSchedule,
    dx_m: float, dy_m: float, current_generation: str = "E5:E5_1A:PRE0",
    source_identity: Mapping[str, Any] | None = None, queue_depth: int = QUEUE_DEPTH,
    admission_identity: Mapping[str, Any] | None = None,
    storage_budget: StorageBudget | None = None,
    creation_intent: Mapping[str, Any] | str | None = None,
    fixture_only: bool = False, trajectory: str = "R", pulse: int = 0,
    attempt: int = 0, role: str = "PRE0",
) -> StreamingLifecycle:
    """Create a private PRE0 Streaming root with deterministic zero velocity."""
    identity, budget, intent = _require_creation_context(
        admission_identity=admission_identity, storage_budget=storage_budget,
        creation_intent=creation_intent, fixture_only=bool(fixture_only),
    )
    if queue_depth != QUEUE_DEPTH:
        raise ValueError('E5-1A queue depth is fixed at 16')
    if not isinstance(schedule, LongitudinalSchedule) or schedule.n_intervals % BLOCK_SIZE:
        raise ValueError("PRE0 schedule must be a valid full-block LongitudinalSchedule")
    if identity is not None:
        if int(identity.get("k", -1)) != int(schedule.n_intervals):
            raise ValueError("PRE0 schedule count differs from admission identity")
        if int(identity.get("block_size", -1)) != BLOCK_SIZE or int(identity.get("queue_depth", -1)) != QUEUE_DEPTH:
            raise ValueError("PRE0 frozen block/queue differs from admission identity")
        grid_identity = identity.get("grid_identity", {})
        if source_identity is not None and dict(source_identity) != dict(identity.get("source_identity", source_identity)):
            raise ValueError("PRE0 source identity differs from admission identity")
    root_path = Path(root).resolve()
    if identity is not None and budget is not None and intent is not None:
        _validate_creation_target(root=root_path, identity=identity, budget=budget, intent=intent,
                                  role=role, trajectory=trajectory, pulse=pulse, attempt=attempt,
                                  generation=str(current_generation))
    fields = pre0_fields_from_delta_n(delta_n, expected_count=schedule.n_intervals)
    if identity is not None and isinstance(grid_identity, Mapping):
        declared_shape = grid_identity.get("shape")
        if declared_shape is not None and list(declared_shape) != list(fields["delta_n"].shape[-2:]):
            raise ValueError("PRE0 grid shape differs from admission identity")
    records = _records_from_schedule(schedule)
    try:
        lifecycle = StreamingLifecycle.create(
            root=root_path, current=fields, screen_records=records,
            current_generation=str(current_generation), dx_m=float(dx_m), dy_m=float(dy_m),
            queue_depth=int(queue_depth), actor="e5_1a_pre0",
        )
        _write_root_metadata(root_path, {
            "entry": "PRE0", "current_generation": str(current_generation),
            "current_content_sha256": lifecycle.manifest["current_content_sha256"],
            "schedule": schedule.as_metadata(), "source_identity": dict(source_identity or (identity or {}).get("source_identity", {})),
            "admission_identity_sha256": None if identity is None else identity["identity_sha256"],
            "creation_intent_id": None if intent is None else intent.get("intent_id"),
            "creation_intent_generation": None if intent is None else intent.get("generation"),
            "scope": "TEST_FIXTURE_ONLY" if fixture_only else "FORMAL",
            "velocity_initialization": "deterministic_positive_float64_zero",
        })
        if budget is not None and intent is not None:
            files = [path for path in root_path.rglob("*") if path.is_file()]
            budget.complete_intent(str(intent["intent_id"]), files=files)
            budget.consume(str(intent["reservation_id"]))
    except Exception as error:
        if budget is not None and intent is not None:
            try:
                budget.interrupt_intent(str(intent["intent_id"]), reason=f"PRE0 creation failed: {type(error).__name__}")
            except Exception:
                pass
        raise
    return lifecycle


create_formal_pre0_root = create_pre0_root


def _next_fields(lifecycle: StreamingLifecycle) -> dict[str, np.ndarray]:
    if lifecycle._authoritative_namespace != "NEXT" or not (lifecycle.root / "authoritative_generation.json").is_file():
        raise StreamingLifecycleError("successor construction requires a promoted authoritative NEXT")
    lifecycle.validate_barrier(actor="e5_1a_successor_validate")
    values = {name: [] for name in FIELDS}
    for ordinal in range(int(lifecycle.manifest["expected_screen_count"])):
        fields = lifecycle.current_fields(ordinal)
        for name in FIELDS:
            values[name].append(np.array(fields[name], dtype=np.float64, copy=True))
    return {name: np.stack(items, axis=0) for name, items in values.items()}


def create_successor_root(
    *, parent_root: str | Path, child_root: str | Path,
    exact_report_path: str | Path | None = None,
    queue_depth: int = QUEUE_DEPTH,
    admission_identity: Mapping[str, Any] | None = None,
    storage_budget: StorageBudget | None = None,
    creation_intent: Mapping[str, Any] | str | None = None,
    fixture_only: bool = False, trajectory: str = "R", pulse: int = 0,
    attempt: int = 0, role: str = "SUCCESSOR",
) -> tuple[StreamingLifecycle, dict[str, Any]]:
    """Copy a promoted parent NEXT into an independent child CURRENT root."""
    identity, budget, intent = _require_creation_context(
        admission_identity=admission_identity, storage_budget=storage_budget,
        creation_intent=creation_intent, fixture_only=bool(fixture_only),
    )
    if queue_depth != QUEUE_DEPTH:
        raise ValueError('E5-1A queue depth is fixed at 16')
    child_path = Path(child_root).resolve()
    parent = StreamingLifecycle.open(parent_root)
    if identity is not None:
        parent_metadata_path = Path(parent_root) / "E5_1A_ROOT_METADATA.json"
        if parent_metadata_path.is_file():
            parent_metadata = json.loads(parent_metadata_path.read_text(encoding="utf-8"))
            if str(parent_metadata.get("admission_identity_sha256", "")) != str(identity["identity_sha256"]):
                raise StreamingLifecycleError("successor parent admission identity differs from requested identity")
        elif not fixture_only:
            raise StreamingLifecycleError("formal successor parent admission metadata is missing")
    schedule_meta_path = Path(parent_root) / "E5_1A_ROOT_METADATA.json"
    schedule_meta = {}
    if schedule_meta_path.is_file():
        schedule_meta = json.loads(schedule_meta_path.read_text(encoding="utf-8"))
    records = [{"ordinal": int(item["ordinal"]), "screen_id": str(item["screen_id"]), "z_m": float(item["z_m"])} for item in parent.manifest["records"]]
    child_generation = str(parent.manifest["next_generation"])
    if identity is not None and budget is not None and intent is not None:
        _validate_creation_target(root=child_path, identity=identity, budget=budget, intent=intent,
                                  role=role, trajectory=trajectory, pulse=pulse, attempt=attempt,
                                  generation=child_generation, allow_existing_complete=True)
    if (child_path / "E5_1A_READY.json").exists():
        receipt = validate_ready_receipt(child_path, expected_parent_root=parent_root)
        return StreamingLifecycle.open(child_path), receipt
    fields = None
    if child_path.exists():
        if not (child_path / "streaming_manifest.json").is_file():
            raise StreamingLifecycleError("partial successor root without complete manifest; retain as failure evidence")
        child = StreamingLifecycle.open(child_path)
        if child.manifest['current_generation'] != child_generation or child._authoritative_namespace != 'CURRENT':
            raise StreamingLifecycleError("conflicting successor generation")
    else:
        fields = _next_fields(parent)
        child = StreamingLifecycle.create(
            root=child_root, current=fields, screen_records=records,
            current_generation=child_generation, dx_m=float(parent.manifest["dx_m"]),
            dy_m=float(parent.manifest["dy_m"]), queue_depth=int(queue_depth), actor="e5_1a_successor",
        )
    if fields is not None:
        del fields
    child_root_path = Path(child_root).resolve()
    # Compare one screen and one field at a time.  The child was created via
    # the existing three-volume ``create`` API, but exact evidence never
    # builds an additional K-by-field object dictionary in memory.
    rows = []
    for ordinal in range(len(records)):
        parent_fields = parent.current_fields(ordinal)
        child_fields = child.current_fields(ordinal)
        for name in FIELDS:
            rows.append(compare_arrays_exact(
                parent_fields[name], child_fields[name],
                name=f"parent_next_to_child_current:{ordinal}:{name}",
                reference_path=parent.root / parent.manifest['records'][ordinal]['next']['artifact'],
                candidate_path=child.root / child.manifest['records'][ordinal]['current']['artifact'],
            ))
    failures = [row for row in rows if row["status"] != "PASS"]
    report = {
        "schema": "khz_filament.hr4e5.e5_1a.evidence.v1",
        "layer": "parent_next_to_child_current", "status": "PASS" if not failures else "FAIL",
        "expected_object_count": len(rows), "candidate_object_count": len(rows),
        "compared_object_count": len(rows), "missing_reference": [], "missing_candidate": [],
        "mismatch_count": len(failures), "rows": rows, "created_utc": _utc(),
    }
    if report["status"] != "PASS":
        raise ValueError("successor CURRENT failed exact parent NEXT binding")
    if exact_report_path is None:
        exact_report_path = child_root_path / "parent_next_child_current_exact.json"
    report_path = Path(exact_report_path)
    if not report_path.is_absolute():
        report_path = child_root_path / report_path
    report_path = _safe_child_path(child_root_path, report_path, label='successor exact report')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report_written = json.loads(report_path.read_text(encoding='utf-8'))
        if report_written.get('status') != 'PASS' or report_written.get('compared_object_count') != len(rows):
            raise StreamingLifecycleError('conflicting existing successor exact report')
    else:
        report_written = write_exact_report(report_path, report)
    fields_identity = {
        f"{ordinal}:{name}": {
            "sha256_array": str(child.manifest["records"][ordinal]["current"]["field_sha256"][name]),
            "shape": list(child.manifest["shape"]), "dtype": child.manifest["dtype"],
        }
        for ordinal in range(len(records)) for name in FIELDS
    }
    parent_next_content = _content_hash([record["next"] for record in parent.manifest["records"]])
    receipt = lineage_binding(
        parent_root=parent.root, child_root=child_root_path,
        parent_generation=str(parent.manifest["next_generation"]),
        parent_content_sha256=parent_next_content,
        child_generation=str(child.manifest["current_generation"]),
        child_content_sha256=str(child.manifest["current_content_sha256"]),
        fields=fields_identity,
        exact_report={"path": str(report_path.relative_to(child_root_path)).replace("\\", "/"), "sha256": sha256_file(report_path), "status": report_written["status"]},
    )
    if identity is not None:
        receipt["admission_identity_sha256"] = identity["identity_sha256"]
    if intent is not None:
        receipt["creation_intent_id"] = intent.get("intent_id")
        receipt["creation_intent_generation"] = intent.get("generation", child_generation)
    lineage_path = child_root_path / 'E5_1A_LINEAGE.json'
    if lineage_path.exists():
        saved = json.loads(lineage_path.read_text(encoding='utf-8'))
        if saved.get('parent_root') != str(parent.root) or saved.get('child_root') != str(child_root_path):
            raise StreamingLifecycleError('conflicting lineage')
    else:
        atomic_json(lineage_path, {"schema": FORMAL_ENTRY_SCHEMA, "status": "READY", "receipt": "E5_1A_READY.json", "parent_root": str(parent.root), "child_root": str(child_root_path), "schedule_metadata": schedule_meta}, overwrite=False)
    if identity is not None and not (child_root_path / "E5_1A_ROOT_METADATA.json").exists():
        _write_root_metadata(child_root_path, {
            "entry": "SUCCESSOR", "current_generation": child.manifest["current_generation"],
            "current_content_sha256": child.manifest["current_content_sha256"],
            "schedule": schedule_meta.get("schedule"),
            "admission_identity_sha256": identity["identity_sha256"],
            "creation_intent_id": None if intent is None else intent.get("intent_id"),
            "creation_intent_generation": None if intent is None else intent.get("generation", child_generation),
            "source_identity": identity.get("source_identity", {}),
            "scope": "TEST_FIXTURE_ONLY" if fixture_only else "FORMAL",
        })
    atomic_json(child_root_path / "E5_1A_READY.json", receipt, overwrite=False)
    archive = parent.root / 'E5_1A_ARCHIVED_AFTER_EXACT.json'
    if not archive.exists():
        atomic_json(archive, {"schema": FORMAL_ENTRY_SCHEMA, "status": "ARCHIVED_AFTER_EXACT", "restartable": False, "successor_root": str(child_root_path), "exact_report": str(report_path), "created_utc": _utc()}, overwrite=False)
    if budget is not None and intent is not None:
        try:
            files = [path for path in child_root_path.rglob("*") if path.is_file()]
            budget.complete_intent(str(intent["intent_id"]), files=files)
            current_intent = budget.validate_intent(str(intent["intent_id"]))
            if current_intent.get("status") == "COMPLETED":
                reservation_id = str(current_intent.get("reservation_id"))
                ledger = budget._read()
                if ledger.get("reservations", {}).get(reservation_id, {}).get("status") == "ACTIVE":
                    budget.consume(reservation_id)
        except Exception as error:
            try:
                budget.interrupt_intent(str(intent["intent_id"]), reason=f"successor creation failed: {type(error).__name__}")
            except Exception:
                pass
            raise
    return child, receipt


create_formal_successor_root = create_successor_root


def validate_successor_ready(root: str | Path, **kwargs: Any) -> dict[str, Any]:
    return validate_ready_receipt(root, **kwargs)


def open_successor_root(root: str | Path, *, allow_archived: bool = False,
                        admission_identity: Mapping[str, Any] | None = None) -> StreamingLifecycle:
    base = Path(root).resolve()
    if (base / "E5_1A_ARCHIVED_AFTER_EXACT.json").is_file() and not allow_archived:
        raise StreamingLifecycleError("archived parent root is not restartable")
    receipt = validate_ready_receipt(base)
    if admission_identity is not None:
        identity = validate_admission_identity(admission_identity, formal=str(admission_identity.get("execution_mode")) != "TEST_FIXTURE_ONLY")
        saved_hash = receipt.get("admission_identity_sha256")
        if saved_hash is not None and str(saved_hash) != str(identity.get("identity_sha256")):
            raise StreamingLifecycleError("successor READY admission identity changed")
    return StreamingLifecycle.open(base)


class StreamingPulseHook:
    """Per-pulse callback preserving existing non-final enqueue semantics."""

    def __init__(self, lifecycle: StreamingLifecycle, *, final: bool = False, resume: bool = False, view: StreamingPulseReadView | None = None):
        self.lifecycle = lifecycle
        self.final = bool(final)
        self.resume = bool(resume)
        self.view = view

    def __call__(self, *, interval, state_after, hr3a_authoritative: bool, hr3b_authoritative: bool) -> None:
        ordinal = int(interval.index)
        fields = self.view.post_fields(ordinal, state_after) if self.view is not None else None
        if self.resume and self.lifecycle.has_authoritative_post(ordinal):
            if fields is None:
                fields = self.lifecycle.current_fields(ordinal)
                fields = {**fields, 'delta_n': np.asarray(state_after)}
            if fields is not None:
                record = self.lifecycle.manifest['records'][ordinal]
                saved = self.lifecycle._artifact_fields(record['post'], namespace='POST')
                if any(compare_arrays_exact(saved[f], fields[f])['status'] != 'PASS' for f in FIELDS):
                    raise StreamingLifecycleError('deterministic replay differs from durable POST')
            return
        self.lifecycle.deposition_finalized(ordinal, actor="e5_1a_optical")
        if fields is None:
            self.lifecycle.commit_post_from_delta_n(ordinal, state_after, actor='e5_1a_optical',
                hr3a_authoritative=hr3a_authoritative, hr3b_authoritative=hr3b_authoritative)
        else:
            self.lifecycle.commit_post(ordinal, fields, actor='e5_1a_optical',
                hr3a_authoritative=hr3a_authoritative, hr3b_authoritative=hr3b_authoritative)
        if not self.final:
            self.lifecycle.enqueue_post(ordinal, actor="e5_1a_optical", wait_for_capacity=True, timeout_s=None)


def _all_interval_sample_plan(schedule: LongitudinalSchedule) -> ThermalSamplePlan:
    edges = np.asarray(schedule.z_edges, dtype=np.float64)
    mids = 0.5 * (edges[:-1] + edges[1:])
    indices = np.arange(schedule.n_intervals, dtype=np.int64)
    return ThermalSamplePlan(
        target_z_m=mids.copy(), interval_index=indices,
        z_left_m=edges[:-1].copy(), z_right_m=edges[1:].copy(), z_mid_m=mids.copy(),
        snap_error_m=np.zeros_like(mids), region=np.full(mids.size, "e5_1a", dtype="U32"),
        reason=np.full(mids.size, "all_prefix_intervals", dtype="U128"),
    )


def _components_from_input(components: Mapping[str, Any] | Sequence[Any] | None, config_path: str | Path | None):
    if config_path is not None:
        return load_all(str(config_path))
    if isinstance(components, Mapping):
        return tuple(components[name] for name in ("grid", "beam", "prop", "ion", "heat", "run", "raman"))
    if components is None:
        raise ValueError("components or config_path is required")
    values = tuple(components)
    if len(values) != 7:
        raise ValueError("components must contain grid, beam, prop, ion, heat, run, raman")
    return values


def _validate_declared_file_identity(declared: Any, *, label: str) -> None:
    if not isinstance(declared, Mapping):
        return
    raw_path = declared.get("path") or declared.get("file")
    expected = declared.get("sha256") or declared.get("sha256_file") or declared.get("raw_sha256")
    if raw_path is None or expected is None:
        return
    path = Path(str(raw_path)).resolve()
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ValueError(f"formal optical {label} identity does not match admission")


def _validate_optical_admission(
    *, lifecycle_root: str | Path, schedule: LongitudinalSchedule, grid: Any,
    identity: Mapping[str, Any], fixture_only: bool, config_path: str | Path | None,
) -> None:
    if int(identity.get("k", -1)) != int(schedule.n_intervals):
        raise ValueError("optical schedule count differs from admission identity")
    if int(identity.get("block_size", -1)) != BLOCK_SIZE or int(identity.get("queue_depth", -1)) != QUEUE_DEPTH:
        raise ValueError("optical block/queue differs from admission identity")
    grid_identity = identity.get("grid_identity", {})
    declared_shape = grid_identity.get("shape") if isinstance(grid_identity, Mapping) else None
    if declared_shape is not None and list(declared_shape) != [int(grid.Ny), int(grid.Nx)]:
        raise ValueError("optical grid shape differs from admission identity")
    metadata_path = Path(lifecycle_root).resolve() / "E5_1A_ROOT_METADATA.json"
    if not metadata_path.is_file():
        if not fixture_only:
            raise ValueError("formal optical root admission metadata is missing")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("admission_identity_sha256", "")) != str(identity["identity_sha256"]):
            raise ValueError("optical root admission identity differs from formal entry")
        if not fixture_only and metadata.get("scope") != "FORMAL":
            raise ValueError("formal optical root scope is invalid")
        declared_source = identity.get("source_identity")
        if isinstance(declared_source, Mapping) and metadata.get("source_identity") not in (None, dict(declared_source)):
            raise ValueError("optical root source identity differs from admission")
    declared_schedule = identity.get("schedule_identity", {})
    if isinstance(declared_schedule, Mapping) and declared_schedule.get("k") is not None:
        if int(declared_schedule["k"]) != int(schedule.n_intervals):
            raise ValueError("optical schedule identity differs from admission")
    if config_path is not None:
        config_identity = identity.get("config_identity")
        _validate_declared_file_identity(config_identity, label="config")
        if isinstance(config_identity, Mapping) and config_identity.get("path"):
            if Path(str(config_identity["path"])).resolve() != Path(str(config_path)).resolve():
                raise ValueError("optical config path differs from admission")
    _validate_declared_file_identity(identity.get("source_identity"), label="source")
    _validate_declared_file_identity(identity.get("lut_identity"), label="LUT")


def run_streaming_optical_pulse(
    *, lifecycle_root: str | Path, schedule: LongitudinalSchedule,
    output_dir: str | Path, components: Mapping[str, Any] | Sequence[Any] | None = None,
    config_path: str | Path | None = None, final: bool = False,
    resume: bool = False, dtype: str = "fp64",
    propagate_fn: Callable[..., Any] | None = None,
    storage_budget: StorageBudget | None = None,
    admission_identity: Mapping[str, Any] | None = None,
    creation_intent: Mapping[str, Any] | str | None = None,
    fixture_only: bool = False, trajectory: str = "C", pulse: int = 0,
    attempt: int = 0, role: str = "OPTICAL",
) -> dict[str, Any]:
    """Run one real optical pulse against an existing Streaming CURRENT root."""
    if not isinstance(schedule, LongitudinalSchedule):
        raise TypeError("schedule must be a LongitudinalSchedule")
    schedule.validate()
    if schedule.n_intervals % BLOCK_SIZE:
        raise ValueError('optical schedule requires complete blocks of eight')
    if dtype != "fp64":
        raise ValueError("E5-1A formal entry requires fp64 fields")
    if admission_identity is None:
        raise ValueError("optical entry requires an explicit admission identity")
    inferred_fixture = str(admission_identity.get("execution_mode", "")) == "TEST_FIXTURE_ONLY"
    if inferred_fixture != bool(fixture_only):
        raise ValueError("optical fixture mode must explicitly match TEST_FIXTURE_ONLY admission")
    identity, budget, intent = _require_creation_context(
        admission_identity=admission_identity, storage_budget=storage_budget,
        creation_intent=creation_intent, fixture_only=bool(fixture_only),
    )
    if propagate_fn is not None and not fixture_only:
        raise ValueError("formal optical entry cannot inject a propagate function")
    destination = Path(output_dir)
    if destination.exists():
        if not resume:
            raise FileExistsError(destination)
        destination = destination / ('replay_' + uuid.uuid4().hex)
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    if lifecycle._authoritative_namespace != "CURRENT" or (Path(lifecycle_root) / "authoritative_generation.json").exists():
        raise StreamingLifecycleError("optical pulse requires CURRENT authority")
    if int(lifecycle.manifest["expected_screen_count"]) != schedule.n_intervals:
        raise ValueError("schedule interval count differs from Streaming root")
    if lifecycle.manifest.get('block_size') != BLOCK_SIZE or lifecycle.manifest.get('queue_depth') != QUEUE_DEPTH:
        raise ValueError('optical root requires frozen block8/queue16')
    for index, record in enumerate(lifecycle.manifest['records']):
        if float(record['z_m']) != float((schedule.z_edges[index] + schedule.z_edges[index+1]) / 2):
            raise ValueError('optical schedule coordinates differ from CURRENT records')
    metadata_path = Path(lifecycle_root) / 'E5_1A_ROOT_METADATA.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('schedule') != json.loads(json.dumps(schedule.as_metadata())):
            raise ValueError('optical schedule differs from root metadata')
    grid, beam, prop, ion, heat, run, raman = _components_from_input(components, config_path)
    if not bool(getattr(heat, 'hr3b_enabled', False)):
        raise ValueError('Streaming optical entry requires configured HR-3B enabled')
    if (tuple(lifecycle.manifest['shape']) != (grid.Ny, grid.Nx)
            or float(lifecycle.manifest['dx_m']) != float(grid.Lx / grid.Nx)
            or float(lifecycle.manifest['dy_m']) != float(grid.Ly / grid.Ny)):
        raise ValueError('Streaming CURRENT grid differs from optical grid')
    _validate_optical_admission(lifecycle_root=lifecycle_root, schedule=schedule, grid=grid,
                                identity=identity, fixture_only=bool(fixture_only),
                                config_path=config_path)
    if budget is None:
        raise ValueError('real optical entry requires the campaign StorageBudget before writing')
    # This reservation covers both the producer and the existing consumer's
    # POST/NEXT writes, six diagnostic maps and the final optical output.
    k = schedule.n_intervals
    screen_bytes = int(grid.Ny) * int(grid.Nx) * 8
    optical_bytes = int(grid.Nt) * int(grid.Ny) * int(grid.Nx) * 16
    peak = (9 if final else 12) * k * screen_bytes + optical_bytes + max(1024**2, k*65536)
    budget.check_final_output_budget(optical_bytes + 9*k*8)
    before_files = {p for base in (destination, Path(lifecycle_root)) for p in base.rglob('*') if p.is_file()}
    if fixture_only:
        reservation = budget.reserve(peak, purpose='fixture_optical_and_streaming_writes',
            allocation_paths=[destination, Path(lifecycle_root)])
    else:
        _validate_creation_target(root=destination, identity=identity, budget=budget,
                                  intent=intent or {}, role=role, trajectory=trajectory,
                                  pulse=pulse, attempt=attempt,
                                  generation=str(lifecycle.manifest["current_generation"]))
        # The optical call writes both the new output tree and POST/NEXT state
        # into the existing lifecycle root.  Both destinations must therefore
        # be present in the same durable intent before propagation starts.
        budget.validate_intent(
            str((intent or {})["intent_id"]),
            path=Path(lifecycle_root).resolve(),
            require_active=True,
        )
        if int((intent or {}).get("expected_bytes", 0)) < int(peak):
            raise ValueError("formal optical creation intent is smaller than write forecast")
        reservation = SimpleNamespace(reservation_id=str((intent or {}).get("reservation_id")))
    destination.mkdir(parents=True)
    axes = __import__("KHz_filament.grids", fromlist=["make_axes"]).make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    if tuple(lifecycle.manifest["shape"]) != (grid.Ny, grid.Nx):
        raise ValueError("Streaming CURRENT shape differs from optical grid")
    source, _ = build_transverse_input_field(axes, beam, xp.complex128)
    source = source.copy()
    omega0 = 2.0 * np.pi * c0 / beam.lam0
    k0 = beam.n0 * omega0 / c0
    if getattr(beam, "focal_length", None):
        if str(getattr(prop, "linear_model", "uppe")).lower() == "uppe":
            source = apply_thin_lens_achromatic(source, axes, beam, prop, chunk_t=getattr(prop, "lens_chunk_t", 0))
        else:
            X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
            source *= xp.exp(xp.asarray(-1j * k0 * (X ** 2 + Y ** 2) / (2.0 * float(beam.focal_length)), dtype=xp.complex128))
    source_for_pulse = source.copy()
    source_sha = sha256_array(np.asarray(to_cpu(source_for_pulse)))
    atomic_json(destination / "source_identity.json", {"schema": FORMAL_ENTRY_SCHEMA, "source_sha256_array": source_sha, "source_alias_checked": not np.shares_memory(np.asarray(to_cpu(source_for_pulse)), np.asarray(to_cpu(source))), "created_utc": _utc()}, overwrite=False)
    plan = _all_interval_sample_plan(schedule)
    thermal_sink = ThermalDiagnosticSink(plan=plan, output_path=str(destination / "sinks"), shape=(grid.Ny, grid.Nx), dtype=np.float64, enabled=True, mode="validation")
    hr3b_sink = HR3BDiagnosticSink(plan=plan, output_path=str(destination / "sinks"), shape=(grid.Ny, grid.Nx), dtype=np.float64, enabled=True)
    beta = validate_hr3b_parameters(rho0=float(heat.rho0), Cv=float(heat.Cv), T0=float(prop.air_T), n0=float(beam.n0))
    view = StreamingPulseReadView(lifecycle)
    hook = StreamingPulseHook(lifecycle, final=final, resume=resume, view=view)
    optical_start = lifecycle.record_telemetry("OPTICAL_START", actor="e5_1a_optical", final_pulse=bool(final))
    call = propagate_one_pulse if propagate_fn is None else propagate_fn
    try:
        final_E, _, diagnostics = call(
            source_for_pulse, kperp2=axes.kperp2, k0=k0, omega0=omega0,
            dz=float(schedule.dz_intervals[0]), z_max=float(schedule.z_end - schedule.z_start), n0=beam.n0,
            n2=float(getattr(prop, "n2", getattr(beam, "n2_air", n2_air))), Ui=Ui_N2, N0=N0_air,
            ion_conf=ion, dn_gas=None, dt=axes.dt, axes=axes, prop_conf=prop, raman_conf=raman,
            record_onaxis_rho_time=True, record_every_z=1, longitudinal_schedule=schedule,
            deposition_contract=build_deposition_contract(schedule, axes=axes), thermal_sink=thermal_sink,
            thermal_slow_state=view, hr3b_parameters={"rho0": float(heat.rho0), "Cv": float(heat.Cv), "T0": float(prop.air_T), "n0": float(beam.n0), "beta_th": beta},
            hr3b_sink=hr3b_sink, post_commit_hook=hook,
        )
    except Exception:
        if final and fixture_only:
            # Final mode has no consumer; the failed producer has unwound.
            # Existing bytes remain charged and no failed artifacts are deleted.
            budget.consume(reservation.reservation_id)
        elif not fixture_only and intent is not None:
            try:
                budget.interrupt_intent(str(intent["intent_id"]), reason="formal optical pulse failed")
            except Exception:
                pass
        raise
    finally:
        view.close()
    if not view.complete:
        raise ValueError("optical path did not read/update every Streaming interval")
    np.save(destination / "final_optical_field.npy", np.asarray(to_cpu(final_E)))
    ledger_fields = tuple(name for name in ("E_dep_ion_interval_J", "E_dep_ib_interval_J", "E_dep_raman_interval_J", "E_dep_plasma_interval_J", "E_thermal_interval_J", "delta_n_increment_min", "delta_n_increment_onaxis", "delta_n_state_min_after_update", "delta_n_state_onaxis_after_update") if name in diagnostics)
    if ledger_fields:
        np.savez(destination / "scientific_ledger.npz", **{name: np.asarray(diagnostics[name]) for name in ledger_fields})
    if len(ledger_fields) != 9:
        raise ValueError('optical path lacks the complete nine-ledger inventory')
    lifecycle.record_telemetry("OPTICAL_COMPLETE", actor="e5_1a_optical", final_pulse=bool(final))
    result = {
        "schema": FORMAL_ENTRY_SCHEMA, "status": "PASS", "final": bool(final),
        "lifecycle_root": str(Path(lifecycle_root).resolve()),
        "output_dir": str(destination.resolve()),
        "schedule_intervals": schedule.n_intervals, "source_sha256_array": source_sha,
        "final_optical_field": "final_optical_field.npy", "final_optical_field_sha256": sha256_array(np.asarray(to_cpu(final_E))),
        "ledger_fields": list(ledger_fields), "diagnostic_keys": sorted(str(key) for key in diagnostics),
        "optical_start_event": optical_start, "current_generation": lifecycle.manifest["current_generation"],
        "current_content_sha256": lifecycle.manifest["current_content_sha256"], "completed_utc": _utc(),
        "admission_identity_sha256": identity["identity_sha256"],
        "creation_intent_id": None if intent is None else intent.get("intent_id"),
    }
    atomic_json(destination / "optical_run.json", result, overwrite=False)
    created_files = [path for base in (destination, Path(lifecycle_root)) for path in base.rglob('*')
                     if path.is_file() and path not in before_files]
    if fixture_only:
        for path in created_files:
            if path.suffix in ('.npy', '.npz'):
                retained = path.name in ('final_optical_field.npy', 'scientific_ledger.npz')
                budget.register_artifact(path, role='final' if retained else 'optical_or_state',
                    reclaimable=not retained, expected_sha256=sha256_file(path),
                    metadata={'reservation_id': reservation.reservation_id}, legacy_test_only=True)
        budget.consume(reservation.reservation_id)
    else:
        budget.complete_intent(str((intent or {})["intent_id"]), files=created_files,
                               metadata={'generation': str(lifecycle.manifest["current_generation"])})
        budget.consume(reservation.reservation_id)
    return result


def commit_final_post(*, lifecycle_root: str | Path, ordinal: int, state_after: Any, resume: bool = False) -> None:
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    StreamingPulseHook(lifecycle, final=True, resume=resume)(
        interval=SimpleNamespace(index=int(ordinal)), state_after=state_after,
        hr3a_authoritative=True, hr3b_authoritative=True,
    )


def validate_final_post(
    *, lifecycle_root: str | Path, receipt_path: str | Path | None = None,
    writer_quiescent: bool = False, expected_optical_dir: str | Path | None = None,
    writer_receipt: str | Path | Mapping[str, Any] | None = None,
    require_durable_writer_receipt: bool = False,
    fixture_only: bool = False,
) -> dict[str, Any]:
    """Validate a terminal POST-only root without enqueueing or promoting."""
    root = Path(lifecycle_root).resolve()
    lifecycle = StreamingLifecycle.open(root)
    failures: list[str] = []
    durable_writer = None
    if require_durable_writer_receipt or not fixture_only:
        raw_writer_path = writer_receipt.get("path", writer_receipt.get("receipt_path")) if isinstance(writer_receipt, Mapping) else writer_receipt
        expected_writer_hash = writer_receipt.get("sha256") if isinstance(writer_receipt, Mapping) else None
        if raw_writer_path is None:
            failures.append("durable_writer_receipt_required")
        else:
            try:
                writer_path = Path(str(raw_writer_path)).resolve()
                if not writer_path.is_relative_to(root) or not writer_path.is_file() or writer_path.is_symlink():
                    raise ValueError("writer receipt is missing or outside lifecycle root")
                if expected_writer_hash is None:
                    raise ValueError("durable writer receipt hash is required")
                if sha256_file(writer_path) != str(expected_writer_hash):
                    raise ValueError("writer receipt hash changed")
                durable_writer = json.loads(writer_path.read_text(encoding="utf-8"))
                if durable_writer.get("status") != "PASS" or durable_writer.get("active_writers") not in ([], ()):
                    failures.append("writer_quiescence_receipt_not_pass")
                if not durable_writer.get("writer_epoch") or not durable_writer.get("coordinator_process_id"):
                    failures.append("writer_quiescence_receipt_identity_missing")
            except (OSError, json.JSONDecodeError):
                failures.append("writer_quiescence_receipt_unreadable")
    elif not bool(writer_quiescent):
        failures.append("writer_quiescence_not_attested")
    if lifecycle._authoritative_namespace != "CURRENT" or (root / "authoritative_generation.json").exists():
        failures.append("current_is_already_promoted")
    if lifecycle.manifest.get("queue") or lifecycle.manifest.get("recovery_backlog"):
        failures.append("queue_or_backlog_not_empty")
    if lifecycle.manifest.get("barrier") is not None or lifecycle.manifest.get("promotion") is not None:
        failures.append("barrier_or_promotion_present")
    try:
        lifecycle._assert_no_staged_or_orphaned_artifacts()
    except Exception as error:
        failures.append(f"artifact_inventory:{type(error).__name__}:{error}")
    for record in lifecycle.manifest["records"]:
        if record.get("state") != "POST_COMMITTED" or record.get("post") is None or record.get("next") is not None:
            failures.append(f"screen_{record.get('ordinal')}_not_terminal_post")
            continue
        try:
            lifecycle._validate_record_provenance(record, require_post=True, require_next=False)
        except Exception as error:
            failures.append(f"screen_{record.get('ordinal')}_provenance:{type(error).__name__}:{error}")
    retained = {}
    if expected_optical_dir is None and receipt_path is not None:
        failures.append('full_optical_evidence_required_for_terminal_receipt')
    if expected_optical_dir is not None:
        optical = Path(expected_optical_dir)
        for name in ("optical_run.json", "scientific_ledger.npz", "final_optical_field.npy"):
            if not (optical / name).is_file():
                failures.append(f"optical_missing:{name}")
        if not failures:
            try:
                run = json.loads((optical/'optical_run.json').read_text(encoding='utf-8'))
                if (run.get('status') != 'PASS' or run.get('final') is not True
                        or run.get('current_generation') != lifecycle.manifest['current_generation']
                        or run.get('current_content_sha256') != lifecycle.manifest['current_content_sha256']
                        or run.get('schedule_intervals') != int(lifecycle.manifest['expected_screen_count'])
                        or Path(str(run.get('lifecycle_root', ''))).resolve() != root):
                    raise ValueError('terminal optical evidence belongs to a different lifecycle or mode')
                e = np.load(optical/'final_optical_field.npy', allow_pickle=False)
                if e.ndim != 3 or e.shape[1:] != tuple(lifecycle.manifest['shape']) or e.dtype != np.complex128 or not np.isfinite(e).all():
                    raise ValueError('final optical layout or finite gate failed')
                if sha256_array(e) != run.get('final_optical_field_sha256', run.get('final_hash')):
                    raise ValueError('final optical hash mismatch')
                with np.load(optical/'scientific_ledger.npz', allow_pickle=False) as ledger:
                    expected_ledgers = {'E_dep_ion_interval_J', 'E_dep_ib_interval_J', 'E_dep_raman_interval_J',
                        'E_dep_plasma_interval_J', 'E_thermal_interval_J', 'delta_n_increment_min',
                        'delta_n_increment_onaxis', 'delta_n_state_min_after_update', 'delta_n_state_onaxis_after_update'}
                    if set(ledger.files) != expected_ledgers or set(ledger.files) != set(run['ledger_fields']):
                        raise ValueError('nine ledger inventory is incomplete')
                    if any(ledger[n].dtype != np.float64 or ledger[n].shape != (int(lifecycle.manifest['expected_screen_count']),) or not np.isfinite(ledger[n]).all() for n in ledger.files):
                        raise ValueError('ledger shape or finite gate failed')
                sink_paths = list(optical.glob('sinks.*samples.npy')) or list(optical.glob('sink_*.npy'))
                if len(sink_paths) != 6:
                    raise ValueError('six sink arrays required for terminal acceptance')
                for path in sink_paths:
                    a = np.load(path, mmap_mode='r', allow_pickle=False)
                    if a.shape != (int(lifecycle.manifest['expected_screen_count']), *lifecycle.manifest['shape']) or a.dtype != np.float64 or not np.isfinite(a).all():
                        raise ValueError('sink layout or finite gate failed')
                    del a
                retained = {str((optical/name).resolve()):sha256_file(optical/name) for name in ('optical_run.json','scientific_ledger.npz','final_optical_field.npy')}
            except (ValueError, KeyError, OSError) as error:
                failures.append(f'optical_evidence:{error}')
    result = {"schema": FORMAL_ENTRY_SCHEMA, "status": "PASS" if not failures else "FAIL", "failures": failures, "expected_post_count": int(lifecycle.manifest["expected_screen_count"]), "completed_post_count": sum(record.get("post") is not None for record in lifecycle.manifest["records"]), "queue_size": len(lifecycle.manifest.get("queue", [])), "backlog_size": len(lifecycle.manifest.get("recovery_backlog", [])), "writer_quiescent": bool(writer_quiescent) or durable_writer is not None, "writer_receipt": None if durable_writer is None else {"path": durable_writer.get("path"), "sha256": durable_writer.get("sha256"), "writer_epoch": durable_writer.get("writer_epoch"), "coordinator_process_id": durable_writer.get("coordinator_process_id")}, "validated_utc": _utc()}
    if result["status"] == "PASS" and receipt_path is not None:
        atomic_json(receipt_path, {**result, "terminal": "POST_FINAL_READY", "retained_optical_hashes": retained, "lifecycle_root": str(root), "current_generation": lifecycle.manifest["current_generation"], "current_content_sha256": lifecycle.manifest["current_content_sha256"]}, overwrite=False)
    return result


def resume_final_post(*, lifecycle_root: str | Path, receipt_path: str | Path, writer_quiescent: bool = True,
                      writer_receipt: str | Path | Mapping[str, Any] | None = None,
                      fixture_only: bool = False,
                      replay_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read and revalidate a durable terminal receipt without bootstrap queue replay."""
    receipt = Path(receipt_path)
    if receipt.is_file():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        lifecycle = StreamingLifecycle.open(lifecycle_root)
        if (saved.get("terminal") != "POST_FINAL_READY" or saved.get("status") != "PASS"
                or saved.get("schema") != FORMAL_ENTRY_SCHEMA
                or Path(str(saved.get("lifecycle_root", ""))).resolve() != Path(lifecycle_root).resolve()
                or saved.get("current_generation") != lifecycle.manifest["current_generation"]
                or saved.get("current_content_sha256") != lifecycle.manifest["current_content_sha256"]
                or len(saved.get("retained_optical_hashes", {})) != 3):
            raise ValueError("terminal receipt identity or retained evidence is invalid")
        for name, digest in saved.get('retained_optical_hashes', {}).items():
            if sha256_file(name) != digest:
                raise ValueError('terminal retained optical evidence changed')
        if writer_receipt is None and isinstance(saved.get("writer_receipt"), Mapping):
            saved_writer = saved["writer_receipt"]
            if saved_writer.get("path") and saved_writer.get("sha256"):
                writer_receipt = {"path": saved_writer["path"], "sha256": saved_writer["sha256"]}
        result = validate_final_post(lifecycle_root=lifecycle_root, writer_quiescent=writer_quiescent,
                                     writer_receipt=writer_receipt, fixture_only=fixture_only)
        if result["status"] != "PASS":
            raise ValueError("terminal receipt no longer validates")
        return {"status": "PASS", "resumed": False, "receipt": saved}
    if replay_kwargs is None:
        raise ValueError('partial terminal recovery requires explicit deterministic replay inputs')
    replay = run_streaming_optical_pulse(lifecycle_root=lifecycle_root, final=True, resume=True, **dict(replay_kwargs))
    result = validate_final_post(lifecycle_root=lifecycle_root, receipt_path=receipt_path, writer_quiescent=writer_quiescent,
                                writer_receipt=writer_receipt, fixture_only=fixture_only,
                                expected_optical_dir=replay['output_dir'])
    if result["status"] != "PASS":
        raise ValueError("terminal POST is incomplete; deterministic optical replay is required")
    return {"status": "PASS", "resumed": True, "result": result}


__all__ = [
    "ADMISSION_SCHEMA", "BLOCK_SIZE", "FORMAL_ENTRY_SCHEMA", "QUEUE_DEPTH", "StreamingPulseHook", "StreamingPulseReadView",
    "admission_identity_hash", "build_admission_identity", "build_fixture_admission_identity", "create_admission_identity",
    "build_prefix_schedule", "commit_final_post", "create_formal_pre0_root", "create_formal_successor_root", "create_pre0_root", "create_successor_root",
    "load_admission_identity", "open_successor_root", "persist_admission_identity", "pre0_fields_from_delta_n",
    "resume_final_post", "run_streaming_optical_pulse", "slice_longitudinal_schedule",
    "validate_admission_identity", "validate_final_post", "validate_successor_ready", "verify_admission_identity",
]
