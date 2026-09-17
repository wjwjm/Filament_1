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
from .hr4e5_evidence import atomic_json, compare_arrays_exact, compare_object_sets, lineage_binding, sha256_array, sha256_file, validate_ready_receipt, write_exact_report, _safe_child_path
from .hr4e5s_streaming import FIELDS, StreamingLifecycle, StreamingLifecycleError, _content_hash, make_post_commit_hook
from .longitudinal import DepositionContract, LongitudinalSchedule, build_deposition_contract
from .propagate import propagate_one_pulse
from .runner import apply_thin_lens_achromatic, build_transverse_input_field
from .slow_state import HR3BDiagnosticSink, validate_hr3b_parameters
from .thermalization import ThermalDiagnosticSink, ThermalSamplePlan
from .hr4e5_storage import StorageBudget


FORMAL_ENTRY_SCHEMA = "khz_filament.hr4e5.e5_1a.formal_entry.v1"
ROOT_METADATA_SCHEMA = "khz_filament.hr4e5.e5_1a.root_metadata.v1"
BLOCK_SIZE = 8
QUEUE_DEPTH = 16


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
) -> StreamingLifecycle:
    """Create a private PRE0 Streaming root with deterministic zero velocity."""
    if queue_depth != QUEUE_DEPTH:
        raise ValueError('E5-1A queue depth is fixed at 16')
    if not isinstance(schedule, LongitudinalSchedule) or schedule.n_intervals % BLOCK_SIZE:
        raise ValueError("PRE0 schedule must be a valid full-block LongitudinalSchedule")
    fields = pre0_fields_from_delta_n(delta_n, expected_count=schedule.n_intervals)
    records = _records_from_schedule(schedule)
    lifecycle = StreamingLifecycle.create(
        root=root, current=fields, screen_records=records,
        current_generation=str(current_generation), dx_m=float(dx_m), dy_m=float(dy_m),
        queue_depth=int(queue_depth), actor="e5_1a_pre0",
    )
    _write_root_metadata(Path(root), {
        "entry": "PRE0", "current_generation": str(current_generation),
        "current_content_sha256": lifecycle.manifest["current_content_sha256"],
        "schedule": schedule.as_metadata(), "source_identity": dict(source_identity or {}),
        "velocity_initialization": "deterministic_positive_float64_zero",
    })
    return lifecycle


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
) -> tuple[StreamingLifecycle, dict[str, Any]]:
    """Copy a promoted parent NEXT into an independent child CURRENT root."""
    if queue_depth != QUEUE_DEPTH:
        raise ValueError('E5-1A queue depth is fixed at 16')
    child_path = Path(child_root).resolve()
    if (child_path / "E5_1A_READY.json").exists():
        receipt = validate_ready_receipt(child_path, expected_parent_root=parent_root)
        return StreamingLifecycle.open(child_path), receipt
    parent = StreamingLifecycle.open(parent_root)
    fields = _next_fields(parent)
    schedule_meta_path = Path(parent_root) / "E5_1A_ROOT_METADATA.json"
    schedule_meta = {}
    if schedule_meta_path.is_file():
        schedule_meta = json.loads(schedule_meta_path.read_text(encoding="utf-8"))
    records = [{"ordinal": int(item["ordinal"]), "screen_id": str(item["screen_id"]), "z_m": float(item["z_m"])} for item in parent.manifest["records"]]
    child_generation = str(parent.manifest["next_generation"])
    if child_path.exists():
        if not (child_path / "streaming_manifest.json").is_file():
            raise StreamingLifecycleError("partial successor root without complete manifest; retain as failure evidence")
        child = StreamingLifecycle.open(child_path)
        if child.manifest['current_generation'] != child_generation or child._authoritative_namespace != 'CURRENT':
            raise StreamingLifecycleError("conflicting successor generation")
    else:
        child = StreamingLifecycle.create(
            root=child_root, current=fields, screen_records=records,
            current_generation=child_generation, dx_m=float(parent.manifest["dx_m"]),
            dy_m=float(parent.manifest["dy_m"]), queue_depth=int(queue_depth), actor="e5_1a_successor",
        )
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
    lineage_path = child_root_path / 'E5_1A_LINEAGE.json'
    if lineage_path.exists():
        saved = json.loads(lineage_path.read_text(encoding='utf-8'))
        if saved.get('parent_root') != str(parent.root) or saved.get('child_root') != str(child_root_path):
            raise StreamingLifecycleError('conflicting lineage')
    else:
        atomic_json(lineage_path, {"schema": FORMAL_ENTRY_SCHEMA, "status": "READY", "receipt": "E5_1A_READY.json", "parent_root": str(parent.root), "child_root": str(child_root_path), "schedule_metadata": schedule_meta}, overwrite=False)
    atomic_json(child_root_path / "E5_1A_READY.json", receipt, overwrite=False)
    archive = parent.root / 'E5_1A_ARCHIVED_AFTER_EXACT.json'
    if not archive.exists():
        atomic_json(archive, {"schema": FORMAL_ENTRY_SCHEMA, "status": "ARCHIVED_AFTER_EXACT", "restartable": False, "successor_root": str(child_root_path), "exact_report": str(report_path), "created_utc": _utc()}, overwrite=False)
    return child, receipt


def validate_successor_ready(root: str | Path, **kwargs: Any) -> dict[str, Any]:
    return validate_ready_receipt(root, **kwargs)


def open_successor_root(root: str | Path, *, allow_archived: bool = False) -> StreamingLifecycle:
    base = Path(root).resolve()
    if (base / "E5_1A_ARCHIVED_AFTER_EXACT.json").is_file() and not allow_archived:
        raise StreamingLifecycleError("archived parent root is not restartable")
    validate_ready_receipt(base)
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


def run_streaming_optical_pulse(
    *, lifecycle_root: str | Path, schedule: LongitudinalSchedule,
    output_dir: str | Path, components: Mapping[str, Any] | Sequence[Any] | None = None,
    config_path: str | Path | None = None, final: bool = False,
    resume: bool = False, dtype: str = "fp64",
    propagate_fn: Callable[..., Any] | None = None,
    storage_budget: StorageBudget | None = None,
) -> dict[str, Any]:
    """Run one real optical pulse against an existing Streaming CURRENT root."""
    if not isinstance(schedule, LongitudinalSchedule):
        raise TypeError("schedule must be a LongitudinalSchedule")
    schedule.validate()
    if schedule.n_intervals % BLOCK_SIZE:
        raise ValueError('optical schedule requires complete blocks of eight')
    if dtype != "fp64":
        raise ValueError("E5-1A formal entry requires fp64 fields")
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
    if storage_budget is None:
        raise ValueError('real optical entry requires the campaign StorageBudget before writing')
    # This reservation covers both the producer and the existing consumer's
    # POST/NEXT writes, six diagnostic maps and the final optical output.
    k = schedule.n_intervals
    screen_bytes = int(grid.Ny) * int(grid.Nx) * 8
    optical_bytes = int(grid.Nt) * int(grid.Ny) * int(grid.Nx) * 16
    peak = (9 if final else 12) * k * screen_bytes + optical_bytes + max(1024**2, k*65536)
    storage_budget.check_final_output_budget(optical_bytes + 9*k*8)
    before_files = {p for base in (destination, Path(lifecycle_root)) for p in base.rglob('*') if p.is_file()}
    reservation = storage_budget.reserve(peak, purpose='real_optical_and_streaming_writes',
        allocation_paths=[destination, Path(lifecycle_root)])
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
        if final:
            # Final mode has no consumer; the failed producer has unwound.
            # Existing bytes remain charged and no failed artifacts are deleted.
            storage_budget.consume(reservation.reservation_id)
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
    }
    atomic_json(destination / "optical_run.json", result, overwrite=False)
    for base in (destination, Path(lifecycle_root)):
        for path in base.rglob('*'):
            if path.is_file() and path not in before_files and path.suffix in ('.npy', '.npz'):
                retained = path.name in ('final_optical_field.npy', 'scientific_ledger.npz')
                storage_budget.register_artifact(path, role='final' if retained else 'optical_or_state',
                    reclaimable=not retained, expected_sha256=sha256_file(path),
                    metadata={'reservation_id': reservation.reservation_id})
    storage_budget.consume(reservation.reservation_id)
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
) -> dict[str, Any]:
    """Validate a terminal POST-only root without enqueueing or promoting."""
    root = Path(lifecycle_root).resolve()
    lifecycle = StreamingLifecycle.open(root)
    failures: list[str] = []
    if not bool(writer_quiescent):
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
    result = {"schema": FORMAL_ENTRY_SCHEMA, "status": "PASS" if not failures else "FAIL", "failures": failures, "expected_post_count": int(lifecycle.manifest["expected_screen_count"]), "completed_post_count": sum(record.get("post") is not None for record in lifecycle.manifest["records"]), "queue_size": len(lifecycle.manifest.get("queue", [])), "backlog_size": len(lifecycle.manifest.get("recovery_backlog", [])), "writer_quiescent": bool(writer_quiescent), "validated_utc": _utc()}
    if result["status"] == "PASS" and receipt_path is not None:
        atomic_json(receipt_path, {**result, "terminal": "POST_FINAL_READY", "retained_optical_hashes": retained, "lifecycle_root": str(root), "current_generation": lifecycle.manifest["current_generation"], "current_content_sha256": lifecycle.manifest["current_content_sha256"]}, overwrite=False)
    return result


def resume_final_post(*, lifecycle_root: str | Path, receipt_path: str | Path, writer_quiescent: bool = True,
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
        result = validate_final_post(lifecycle_root=lifecycle_root, writer_quiescent=writer_quiescent)
        if result["status"] != "PASS":
            raise ValueError("terminal receipt no longer validates")
        return {"status": "PASS", "resumed": False, "receipt": saved}
    if replay_kwargs is None:
        raise ValueError('partial terminal recovery requires explicit deterministic replay inputs')
    replay = run_streaming_optical_pulse(lifecycle_root=lifecycle_root, final=True, resume=True, **dict(replay_kwargs))
    result = validate_final_post(lifecycle_root=lifecycle_root, receipt_path=receipt_path, writer_quiescent=writer_quiescent,
                                expected_optical_dir=replay['output_dir'])
    if result["status"] != "PASS":
        raise ValueError("terminal POST is incomplete; deterministic optical replay is required")
    return {"status": "PASS", "resumed": True, "result": result}


__all__ = [
    "BLOCK_SIZE", "FORMAL_ENTRY_SCHEMA", "QUEUE_DEPTH", "StreamingPulseHook", "StreamingPulseReadView",
    "build_prefix_schedule", "commit_final_post", "create_pre0_root", "create_successor_root",
    "open_successor_root", "pre0_fields_from_delta_n", "resume_final_post", "run_streaming_optical_pulse",
    "slice_longitudinal_schedule", "validate_final_post", "validate_successor_ready",
]
