"""S3-only batch/streaming qualification helpers.

This module is deliberately an execution and evidence adapter.  It reuses the
frozen optical propagation, HR-3 sinks, and ``advance_hr4_single_screen``;
it defines no optical, deposition, or hydro operator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .confio import load_all
from .constants import N0_air, Ui_N2, c0, n2_air
from .device import to_cpu, xp
from .grids import make_axes
from .hr4 import HR4_CHI, HR4_CFL_LIMIT, HR4_GRAVITY_X, HR4_GRAVITY_Y, HR4_NU
from .hr4c_state import HR4CThreeFieldStore, evolve_hr4_full_z
from .hr4e5s_streaming import FIELDS, StreamingLifecycle
from .hr4e_timestep import json_safe, sha256_array, sha256_file
from .longitudinal import build_deposition_contract, build_longitudinal_schedule
from .propagate import propagate_one_pulse
from .runner import apply_thin_lens_achromatic, build_transverse_input_field
from .slow_state import HR3BDiagnosticSink, validate_hr3b_parameters
from .thermalization import ThermalDiagnosticSink, ThermalSamplePlan


S3_SCHEMA = "khz_filament.hr4e5s.s3.v1"
RECOVERY_BOOTSTRAP_SCHEMA = "khz_filament.hr4e5s.s5.recovery_bootstrap.v1"
S3_WINDOW_COUNT = 48
S3_BLOCK_SIZE = 8
DEPOSITION_FIELDS = ("ion", "ib", "raman")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(json_safe(dict(value)), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _recovery_pending_counts(lifecycle: StreamingLifecycle) -> tuple[int, int]:
    """Count durable recovery work without changing the lifecycle root."""
    pending = 0
    stale = 0
    for record in lifecycle.manifest["records"]:
        if record.get("post") is not None and record.get("next") is None:
            pending += 1
            if record.get("state") == "HYDRO_RUNNING" and bool(record.get("hydro_attempt_started")):
                stale += 1
    return pending, stale


def validate_recovery_bootstrap_receipt(
    *, receipt_path: str | Path, lifecycle_root: str | Path,
    runtime_sha: str | None = None, case_id: str | None = None,
    require_current_telemetry_count: bool = True,
) -> dict[str, Any]:
    """Validate the durable serial-recovery bootstrap boundary.

    This is intentionally a structural gate.  It does not reconstruct a
    lifecycle and does not alter any authoritative field.  The producer uses
    it before starting its optical continuation so a recovery pair cannot
    silently acquire a second reconstruction owner.
    """
    receipt_file, root = Path(receipt_path), Path(lifecycle_root)
    if not receipt_file.is_file():
        raise ValueError("recovery bootstrap receipt is missing")
    receipt = _read_json(receipt_file)
    if receipt.get("schema") != RECOVERY_BOOTSTRAP_SCHEMA or receipt.get("status") != "PASS":
        raise ValueError("recovery bootstrap receipt is invalid")
    if receipt.get("bootstrap_event") != "RESTART_RECONSTRUCTED":
        raise ValueError("recovery bootstrap receipt lacks RESTART_RECONSTRUCTED")
    if not isinstance(receipt.get("case_id"), str) or not receipt["case_id"]:
        raise ValueError("recovery bootstrap receipt case is invalid")
    if case_id is not None and receipt["case_id"] != str(case_id):
        raise ValueError("recovery bootstrap receipt case mismatch")
    recorded_sha = receipt.get("runtime_sha")
    if not isinstance(recorded_sha, str) or len(recorded_sha) != 40 or any(char not in "0123456789abcdef" for char in recorded_sha.lower()):
        raise ValueError("recovery bootstrap receipt runtime SHA is invalid")
    if runtime_sha is not None and recorded_sha != str(runtime_sha):
        raise ValueError("recovery bootstrap receipt runtime SHA mismatch")
    try:
        recorded_root = Path(str(receipt["lifecycle_root"])).resolve()
    except (KeyError, TypeError, ValueError):
        raise ValueError("recovery bootstrap receipt lifecycle root is invalid") from None
    if recorded_root != root.resolve():
        raise ValueError("recovery bootstrap receipt lifecycle root mismatch")

    integer_fields = (
        "pending_post_count", "pending_post_count_before", "stale_hydro_running_count",
        "stale_hydro_running_count_before", "reconstructed_pending_post_count",
        "reconstructed_stale_hydro_running_count", "queue_size", "backlog_size",
        "queue_depth", "telemetry_event_index", "bootstrap_event_index",
        "telemetry_event_count",
    )
    for field in integer_fields:
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"recovery bootstrap receipt {field} is invalid")
    if receipt["bootstrap_event_index"] != receipt["telemetry_event_index"]:
        raise ValueError("recovery bootstrap receipt event indexes disagree")
    if receipt["queue_size"] > receipt["queue_depth"]:
        raise ValueError("recovery bootstrap queue exceeds its frozen depth")
    if receipt["pending_post_count"] != receipt["queue_size"] + receipt["backlog_size"]:
        raise ValueError("recovery bootstrap pending count disagrees with queue and backlog")
    if receipt["reconstructed_pending_post_count"] != receipt["pending_post_count"]:
        raise ValueError("recovery bootstrap reconstructed count disagrees with manifest")
    if receipt["reconstructed_stale_hydro_running_count"] != receipt["stale_hydro_running_count_before"]:
        raise ValueError("recovery bootstrap stale count disagrees with pre-bootstrap state")

    lifecycle = StreamingLifecycle.open(root)
    events = lifecycle.manifest.get("telemetry_events")
    if not isinstance(events, list):
        raise ValueError("recovery bootstrap telemetry is invalid")
    if require_current_telemetry_count and receipt["telemetry_event_count"] != len(events):
        raise ValueError("recovery bootstrap telemetry length is invalid")
    if receipt["telemetry_event_count"] > len(events):
        raise ValueError("recovery bootstrap telemetry was truncated")
    index = receipt["telemetry_event_index"]
    if index >= len(events):
        raise ValueError("recovery bootstrap telemetry index is outside the manifest")
    event = events[index]
    if not isinstance(event, Mapping) or event.get("event") != "RESTART_RECONSTRUCTED" or event.get("actor") != "s5_restart":
        raise ValueError("recovery bootstrap telemetry event is invalid")
    if event.get("bootstrap") is not True:
        raise ValueError("recovery bootstrap telemetry event is not marked bootstrap")
    bootstrap_indexes = [
        event_index for event_index, item in enumerate(events)
        if isinstance(item, Mapping) and item.get("event") == "RESTART_RECONSTRUCTED"
    ]
    if bootstrap_indexes != [index]:
        raise ValueError("recovery bootstrap telemetry event count is not exactly one")
    claim_indexes = [
        event_index for event_index, item in enumerate(events)
        if isinstance(item, Mapping)
        and item.get("event") == "HYDRO_CLAIM"
        and item.get("actor") == "hydro_consumer"
    ]
    historical_claim_count = receipt.get("pre_bootstrap_hydro_claim_count")
    if isinstance(historical_claim_count, bool) or not isinstance(historical_claim_count, int) or historical_claim_count < 0:
        raise ValueError("recovery bootstrap receipt historical hydro-claim count is invalid")
    historical_claim_indexes = [event_index for event_index in claim_indexes if event_index < index]
    if len(historical_claim_indexes) != historical_claim_count:
        raise ValueError("recovery bootstrap historical hydro-consumer claim count disagrees with telemetry")
    if any(event_index == index for event_index in claim_indexes):
        raise ValueError("recovery bootstrap telemetry event is a hydro-consumer claim")
    return receipt


def bootstrap_recovery(
    *, lifecycle_root: str | Path, out_path: str | Path,
    runtime_sha: str, case_id: str,
) -> dict[str, Any]:
    """Own the one serial reconstruction transaction for an S5 recovery.

    The receipt is persisted only after ``reconstruct_queue`` and the
    RESTART_RECONSTRUCTED telemetry event are durable.  A second invocation
    against the same lifecycle is rejected rather than heuristically skipped.
    """
    root, receipt_path = Path(lifecycle_root), Path(out_path)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    if not isinstance(runtime_sha, str) or len(runtime_sha) != 40 or any(char not in "0123456789abcdef" for char in runtime_sha.lower()):
        raise ValueError("runtime SHA must be a 40-character hexadecimal commit")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("recovery case id is required")

    lifecycle = StreamingLifecycle.open(root)
    events = lifecycle.manifest.get("telemetry_events", [])
    if any(isinstance(event, Mapping) and event.get("event") == "RESTART_RECONSTRUCTED" for event in events):
        raise ValueError("recovery bootstrap was already recorded")
    historical_hydro_claim_count = sum(
        1 for event in events
        if isinstance(event, Mapping)
        and event.get("event") == "HYDRO_CLAIM"
        and event.get("actor") == "hydro_consumer"
    )
    pending_before, stale_before = _recovery_pending_counts(lifecycle)
    lifecycle.reconstruct_queue(actor="s5_restart")
    lifecycle = StreamingLifecycle.open(root)
    pending_after, _ = _recovery_pending_counts(lifecycle)
    event = lifecycle.record_telemetry(
        "RESTART_RECONSTRUCTED", actor="s5_restart", bootstrap=True,
        reconstructed_pending_post_count=pending_after,
        reconstructed_stale_hydro_running_count=stale_before,
    )
    lifecycle = StreamingLifecycle.open(root)
    telemetry = lifecycle.manifest.get("telemetry_events", [])
    event_index = len(telemetry) - 1
    if event_index < 0 or telemetry[event_index] != event:
        raise RuntimeError("recovery bootstrap telemetry was not durably appended")
    queue = [int(value) for value in lifecycle.manifest["queue"]]
    backlog = [int(value) for value in lifecycle.manifest.get("recovery_backlog", [])]
    receipt = {
        "schema": RECOVERY_BOOTSTRAP_SCHEMA,
        "status": "PASS",
        "bootstrap_event": "RESTART_RECONSTRUCTED",
        "runtime_sha": runtime_sha,
        "case_id": case_id,
        "lifecycle_root": str(root.resolve()),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pending_post_count": len(queue) + len(backlog),
        "pending_post_count_before": pending_before,
        "stale_hydro_running_count": sum(
            1 for record in lifecycle.manifest["records"]
            if record.get("state") == "HYDRO_RUNNING" and record.get("post") is not None and record.get("next") is None
        ),
        "stale_hydro_running_count_before": stale_before,
        "reconstructed_pending_post_count": pending_after,
        "reconstructed_stale_hydro_running_count": stale_before,
        "queue_size": len(queue),
        "backlog_size": len(backlog),
        "queue_depth": int(lifecycle.manifest["queue_depth"]),
        "queue_ordinals": queue,
        "backlog_ordinals": backlog,
        "telemetry_event_index": event_index,
        "bootstrap_event_index": event_index,
        "telemetry_event_count": len(telemetry),
        "pre_bootstrap_hydro_claim_count": historical_hydro_claim_count,
        "manifest_sha256": sha256_file(lifecycle.manifest_path),
    }
    _atomic_json(receipt_path, receipt)
    validate_recovery_bootstrap_receipt(
        receipt_path=receipt_path, lifecycle_root=root,
        runtime_sha=runtime_sha, case_id=case_id,
    )
    return receipt


def _required(value: Mapping[str, Any], name: str) -> Any:
    if name not in value:
        raise ValueError(f"S3 input lacks {name}")
    return value[name]


def _selected_plan(source_z: np.ndarray, indices: Sequence[int]) -> ThermalSamplePlan:
    values = np.asarray(source_z, dtype=np.float64)[list(indices)]
    if values.size != S3_WINDOW_COUNT or not np.all(np.diff(values) > 0.0):
        raise ValueError("S3 screen window must be 48 contiguous increasing source screens")
    return ThermalSamplePlan(
        target_z_m=values.copy(), interval_index=np.asarray(indices, dtype=np.int64),
        z_left_m=values.copy(), z_right_m=values.copy(), z_mid_m=values.copy(),
        snap_error_m=np.zeros(values.size, dtype=np.float64),
        region=np.full(values.size, "s3_window", dtype="U32"),
        reason=np.full(values.size, "frozen_peak_centered_48_contiguous", dtype="U128"),
    )


def prepare_input_manifest(*, source_manifest_path: str | Path, source_state_path: str | Path,
                           config_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Freeze the single objective 48-screen S3 case without copying the source."""
    source_manifest_file, state_file, config_file, output = map(Path, (source_manifest_path, source_state_path, config_path, out_path))
    if output.exists():
        raise FileExistsError(output)
    source = _read_json(source_manifest_file)
    expected_file_sha = str(_required(source, "hr3b_state_file_sha256"))
    expected_array_sha = str(_required(source, "hr3b_state_sha256"))
    if sha256_file(state_file) != expected_file_sha:
        raise ValueError("S3 source state file SHA256 disagrees with frozen E1B manifest")
    if source.get("config_sha256") and sha256_file(config_file) != str(source["config_sha256"]):
        raise ValueError("S3 config SHA256 disagrees with frozen E1B manifest")
    state = np.load(state_file, mmap_mode="r", allow_pickle=False)
    try:
        z = np.asarray(_required(source, "source_z_positions_m"), dtype=np.float64)
        if state.dtype != np.dtype(np.float64) or state.ndim != 3 or z.shape != (state.shape[0],):
            raise ValueError("S3 source must be the frozen float64 [K,Ny,Nx] E1B state")
        if sha256_array(state) != expected_array_sha:
            raise ValueError("S3 source canonical array SHA256 disagrees with frozen E1B manifest")
        peak = int(np.argmax(np.maximum(-np.min(state, axis=(1, 2)), 0.0)))
        first, stop = peak - S3_WINDOW_COUNT // 2, peak + S3_WINDOW_COUNT // 2
        if first < 0 or stop > state.shape[0]:
            raise ValueError("frozen E1B peak cannot support an interior 48-screen S3 window")
        indices = list(range(first, stop))
        selected = np.asarray(state[indices], dtype=np.float64)
        if not np.all(np.isfinite(selected)):
            raise ValueError("S3 selected CURRENT field is non-finite")
    finally:
        close = getattr(state, "_mmap", None)
        if close is not None:
            close.close()
    _selected_plan(z, indices)
    grid, beam, prop, ion, heat, run, raman = load_all(str(config_file))
    if int(run.Npulses) != 1 or not bool(getattr(heat, "hr3b_enabled", False)) or bool(getattr(heat, "hr3c_enabled", False)):
        raise ValueError("S3 requires the frozen one-pulse HR-3B, non-HR-3C E1B configuration")
    schedule = build_longitudinal_schedule(
        dz=float(prop.dz), z_max=float(prop.z_max), z_start=0.0,
        focus_window_step=bool(getattr(prop, "focus_window_step", False)),
        focus_center_m=getattr(prop, "focus_center_m", None),
        focus_halfwidth_m=float(getattr(prop, "focus_halfwidth_m", 0.0)),
        dz_focus=float(getattr(prop, "dz_focus", prop.dz)),
    )
    schedule_mids = 0.5 * (np.asarray(schedule.z_edges[:-1], dtype=np.float64) + np.asarray(schedule.z_edges[1:], dtype=np.float64))
    if schedule.n_intervals != len(z) or not np.array_equal(schedule_mids, z):
        raise ValueError("S3 source z identity does not exactly match the frozen optical schedule")
    result = {
        "schema": S3_SCHEMA, "stage": "HR-4E-5S-S3", "source_manifest": str(source_manifest_file),
        "source_manifest_sha256": sha256_file(source_manifest_file), "source_state": str(state_file),
        "source_state_file_sha256": expected_file_sha, "source_state_array_sha256": expected_array_sha,
        "config": str(config_file), "config_sha256": sha256_file(config_file),
        "current_generation": "E1B_hr3b_source:current", "pulse_source_identity": "E1B_frozen_source_then_one_identical_pulse",
        "window_selection_rule": "first argmax(-min(delta_n)); indices peak-24 through peak+23; 48 contiguous screens",
        "peak_source_index": peak, "screen_indices": indices,
        "screen_records": [{"ordinal": ordinal, "screen_id": f"source_index_{index:05d}", "source_index": index, "z_m": float(z[index]), "current_delta_n_sha256": sha256_array(selected[ordinal]), "current_velocity_initialization": "exact_zero_float64"} for ordinal, index in enumerate(indices)],
        "shape": list(selected.shape[1:]), "dtype": "float64", "dx_m": float(3.01e-3 / 301), "dy_m": float(3.51e-3 / 351),
        "hydro": {"dt_hydro": 1.0e-6, "n_hydro_steps": 1000, "chi": HR4_CHI, "nu": HR4_NU, "n0": float(beam.n0), "gravity_x": HR4_GRAVITY_X, "gravity_y": HR4_GRAVITY_Y, "cfl_limit": HR4_CFL_LIMIT, "block_size": S3_BLOCK_SIZE, "queue_depth": 16},
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(output, result)
    return result


class S3ReadOnlyCurrentState:
    """Read frozen CURRENT slices and form one POST slice without mutating it."""

    def __init__(self, state_path: str | Path, *, expected_shape: Sequence[int], expected_count: int):
        self.path = Path(state_path)
        self._state = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if self._state.dtype != np.dtype(np.float64) or self._state.shape != (int(expected_count), *tuple(expected_shape)):
            raise ValueError("S3 CURRENT source layout differs from frozen input manifest")

    def read_interval(self, interval_index: int):
        return self._state[int(interval_index)]

    def update_interval(self, interval_index: int, delta_n_increment):
        before = np.asarray(self._state[int(interval_index)], dtype=np.float64)
        increment = np.asarray(to_cpu(delta_n_increment), dtype=np.float64)
        if increment.shape != before.shape or not np.all(np.isfinite(increment)):
            raise ValueError("S3 HR-3B increment is invalid")
        return before + increment

    def metadata(self) -> dict[str, Any]:
        return {"hr3b_state_schema": "khz_filament.hr4e5s.s3.readonly_current.v1", "hr3b_state_filename": self.path.name, "hr3b_state_dtype": "float64", "hr3b_state_shape": list(self._state.shape), "hr3b_state_interval_centered": True, "hr3b_state_disk_backed": True, "read_only_current": True}

    def close(self) -> None:
        close = getattr(self._state, "_mmap", None)
        if close is not None:
            close.close()


class _SelectedStreamingHook:
    def __init__(self, lifecycle: StreamingLifecycle, records: Sequence[Mapping[str, Any]], *, resume: bool = False):
        self.lifecycle = lifecycle
        self.ordinals = {int(item["source_index"]): int(item["ordinal"]) for item in records}
        self.resume = bool(resume)

    def __call__(self, *, interval, state_after, hr3a_authoritative: bool, hr3b_authoritative: bool) -> None:
        ordinal = self.ordinals.get(int(interval.index))
        if ordinal is None:
            return
        # A recovery replays the deterministic optical propagation but never
        # replaces a durable POST.  The lifecycle validates the saved POST
        # before this skip, so only unfinished records may be committed.
        if self.resume and self.lifecycle.has_authoritative_post(ordinal):
            return
        self.lifecycle.deposition_finalized(ordinal, actor="optical")
        self.lifecycle.commit_post_from_delta_n(ordinal, state_after, actor="optical", hr3a_authoritative=hr3a_authoritative, hr3b_authoritative=hr3b_authoritative)
        self.lifecycle.enqueue_post(ordinal, actor="optical", wait_for_capacity=True, timeout_s=None)


def create_streaming_lifecycle(*, input_manifest: Mapping[str, Any], root: str | Path) -> StreamingLifecycle:
    manifest = dict(input_manifest)
    source = np.load(str(manifest["source_state"]), mmap_mode="r", allow_pickle=False)
    try:
        indices = [int(item["source_index"]) for item in manifest["screen_records"]]
        selected = np.asarray(source[indices], dtype=np.float64)
    finally:
        close = getattr(source, "_mmap", None)
        if close is not None:
            close.close()
    zero = np.zeros_like(selected)
    return StreamingLifecycle.create(root=root, current={"delta_n": selected, "vx": zero, "vy": zero}, screen_records=manifest["screen_records"], current_generation=str(manifest["current_generation"]), dx_m=float(manifest["dx_m"]), dy_m=float(manifest["dy_m"]), queue_depth=int(manifest["hydro"]["queue_depth"]), actor="s3_initializer")


def _ledger_payload(diag: Mapping[str, Any]) -> dict[str, np.ndarray]:
    names = ("E_dep_ion_interval_J", "E_dep_ib_interval_J", "E_dep_raman_interval_J", "E_dep_plasma_interval_J", "E_thermal_interval_J", "delta_n_increment_min", "delta_n_increment_onaxis", "delta_n_state_min_after_update", "delta_n_state_onaxis_after_update")
    payload = {name: np.asarray(diag[name]) for name in names if name in diag}
    if not payload:
        raise ValueError("S3 optical diagnostic lacks authoritative scalar ledgers")
    return payload


def run_optical_path(*, input_manifest_path: str | Path, out_dir: str | Path,
                     streaming_root: str | Path | None = None, dtype: str = "fp64", resume: bool = False,
                     bootstrap_receipt_path: str | Path | None = None,
                     bootstrap_receipt_validator: Callable[..., None] | None = None) -> dict[str, Any]:
    """Run the complete frozen optical trajectory, capturing only S3 screens."""
    manifest, destination = _read_json(Path(input_manifest_path)), Path(out_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    grid, beam, prop, ion, heat, run, raman = load_all(str(manifest["config"]))
    if dtype != "fp64":
        raise ValueError("S3 frozen field evidence requires fp64")
    axes = make_axes(grid.Nx, grid.Ny, grid.Nt, grid.Lx, grid.Ly, grid.Twin)
    if tuple(manifest["shape"]) != (grid.Ny, grid.Nx) or not np.isclose(axes.dx, float(manifest["dx_m"]), rtol=0.0, atol=1e-18) or not np.isclose(axes.dy, float(manifest["dy_m"]), rtol=0.0, atol=1e-18):
        raise ValueError("S3 grid differs from frozen input manifest")
    E, _ = build_transverse_input_field(axes, beam, xp.complex128)
    omega0 = 2.0 * np.pi * c0 / beam.lam0
    k0 = beam.n0 * omega0 / c0
    if getattr(beam, "focal_length", None):
        if str(getattr(prop, "linear_model", "uppe")).lower() == "uppe":
            E = apply_thin_lens_achromatic(E, axes, beam, prop, chunk_t=getattr(prop, "lens_chunk_t", 0))
        else:
            X, Y = xp.meshgrid(axes.x, axes.y, indexing="xy")
            E *= xp.exp(xp.asarray(-1j * k0 * (X ** 2 + Y ** 2) / (2.0 * float(beam.focal_length)), dtype=xp.complex128))
    schedule = build_longitudinal_schedule(dz=float(prop.dz), z_max=float(prop.z_max), z_start=0.0, focus_window_step=bool(getattr(prop, "focus_window_step", False)), focus_center_m=getattr(prop, "focus_center_m", None), focus_halfwidth_m=float(getattr(prop, "focus_halfwidth_m", 0.0)), dz_focus=float(getattr(prop, "dz_focus", prop.dz)))
    records = manifest["screen_records"]
    if schedule.n_intervals != int(np.load(str(manifest["source_state"]), mmap_mode="r").shape[0]):
        raise ValueError("S3 optical schedule differs from frozen CURRENT schedule")
    schedule_mids = 0.5 * (np.asarray(schedule.z_edges[:-1], dtype=np.float64) + np.asarray(schedule.z_edges[1:], dtype=np.float64))
    plan = _selected_plan(schedule_mids, [int(item["source_index"]) for item in records])
    thermal_sink = ThermalDiagnosticSink(plan=plan, output_path=str(destination / "s3_optical"), shape=(grid.Ny, grid.Nx), dtype=np.float64, enabled=True, mode="validation")
    hr3b_sink = HR3BDiagnosticSink(plan=plan, output_path=str(destination / "s3_optical"), shape=(grid.Ny, grid.Nx), dtype=np.float64, enabled=True)
    beta = validate_hr3b_parameters(rho0=float(heat.rho0), Cv=float(heat.Cv), T0=float(prop.air_T), n0=float(beam.n0))
    current = S3ReadOnlyCurrentState(manifest["source_state"], expected_shape=(grid.Ny, grid.Nx), expected_count=schedule.n_intervals)
    hook = None
    ownership_before = None
    if streaming_root is not None:
        streaming_before = StreamingLifecycle.open(streaming_root)
        ownership_before = {
            "authoritative_namespace": streaming_before._authoritative_namespace,
            "authoritative_generation": streaming_before._authoritative_generation,
            "next_pointer_exists": (Path(streaming_root) / "authoritative_generation.json").is_file(),
        }
        if ownership_before["next_pointer_exists"] or ownership_before["authoritative_namespace"] != "CURRENT":
            raise ValueError("S3 optical producer refuses a pre-promoted or NEXT authoritative generation")
        if resume:
            if bootstrap_receipt_path is None:
                raise ValueError("S3 recovery resume requires an explicit bootstrap receipt")
            validator = validate_recovery_bootstrap_receipt if bootstrap_receipt_validator is None else bootstrap_receipt_validator
            validator(
                receipt_path=bootstrap_receipt_path,
                lifecycle_root=streaming_root,
                runtime_sha=os.environ.get("EXPECTED_GIT_SHA"),
                # The launcher performed the strict no-new-event validation
                # before either GPU process was started.  Once the pair is
                # live, a consumer claim is legitimate and must not make the
                # producer reject that already-durable bootstrap boundary.
                require_current_telemetry_count=False,
            )
        elif bootstrap_receipt_path is not None:
            raise ValueError("bootstrap receipt is only valid for recovery resume")
        streaming_before.record_telemetry("OPTICAL_START", actor="optical")
        hook = _SelectedStreamingHook(streaming_before, records, resume=resume)
    try:
        final_E, _, diag = propagate_one_pulse(E, kperp2=axes.kperp2, k0=k0, omega0=omega0, dz=prop.dz, z_max=prop.z_max, n0=beam.n0, n2=float(getattr(prop, "n2", getattr(beam, "n2_air", n2_air))), Ui=Ui_N2, N0=N0_air, ion_conf=ion, dn_gas=None, dt=axes.dt, axes=axes, prop_conf=prop, raman_conf=raman, record_onaxis_rho_time=True, record_every_z=1, longitudinal_schedule=schedule, deposition_contract=build_deposition_contract(schedule, axes=axes), thermal_sink=thermal_sink, thermal_slow_state=current, hr3b_parameters={"rho0": float(heat.rho0), "Cv": float(heat.Cv), "T0": float(prop.air_T), "n0": float(beam.n0), "beta_th": beta}, hr3b_sink=hr3b_sink, post_commit_hook=hook)
    finally:
        current.close()
    np.save(destination / "final_optical_field.npy", np.asarray(to_cpu(final_E)))
    ledger = _ledger_payload(diag)
    np.savez(destination / "scientific_ledger.npz", **ledger)
    ownership_after = None
    if streaming_root is not None:
        streaming_before.record_telemetry("OPTICAL_COMPLETE", actor="optical")
        streaming_after = StreamingLifecycle.open(streaming_root)
        ownership_after = {
            "authoritative_namespace": streaming_after._authoritative_namespace,
            "authoritative_generation": streaming_after._authoritative_generation,
            "next_pointer_exists": (Path(streaming_root) / "authoritative_generation.json").is_file(),
        }
        if ownership_after["next_pointer_exists"] or ownership_after["authoritative_namespace"] != "CURRENT":
            raise ValueError("S3 NEXT was promoted before optical completion")
    result = {"schema": S3_SCHEMA, "input_manifest_sha256": sha256_file(input_manifest_path), "backend": getattr(xp, "__name__", "unknown"), "final_optical_field": "final_optical_field.npy", "final_optical_field_sha256": sha256_array(np.asarray(to_cpu(final_E))), "ledger": "scientific_ledger.npz", "ledger_fields": sorted(ledger), "deposition_archives": {name: f"s3_optical.hr3a_q{name}_samples.npy" for name in DEPOSITION_FIELDS}, "post_delta_n_archive": "s3_optical.hr3b_delta_n_state_after_update_samples.npy", "optical_current_ownership_before": ownership_before, "optical_current_ownership_after": ownership_after, "optical_complete_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _atomic_json(destination / "optical_run.json", result)
    return result


def run_batch_hydro(*, input_manifest_path: str | Path, optical_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    manifest, optical, destination = _read_json(Path(input_manifest_path)), Path(optical_dir), Path(out_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    post_delta = np.load(optical / "s3_optical.hr3b_delta_n_state_after_update_samples.npy", mmap_mode="r", allow_pickle=False)
    if post_delta.shape != (S3_WINDOW_COUNT, *manifest["shape"]) or post_delta.dtype != np.float64:
        raise ValueError("S3 batch POST archive layout is invalid")
    mids = np.asarray([item["z_m"] for item in manifest["screen_records"]], dtype=np.float64)
    edges = np.empty(mids.size + 1, dtype=np.float64); edges[1:-1] = 0.5 * (mids[:-1] + mids[1:]); edges[0] = mids[0] - (edges[1] - mids[0]); edges[-1] = mids[-1] + (mids[-1] - edges[-2])
    state_path = destination / "batch_post_state"
    store = HR4CThreeFieldStore(output_path=str(state_path), n_intervals=S3_WINDOW_COUNT, shape=tuple(manifest["shape"]), dtype=np.float64, z_edges=edges, dx=float(manifest["dx_m"]), dy=float(manifest["dy_m"]), authoritative_metadata={"schema": S3_SCHEMA, "path": "S3_BATCH_CANONICAL_REFERENCE", "input_manifest_sha256": sha256_file(input_manifest_path)})
    try:
        store.begin_staging()
        store.write_staging_batch(0, {"delta_n": np.asarray(post_delta), "vx": np.zeros_like(post_delta), "vy": np.zeros_like(post_delta)})
        store.commit_staging({"operation": "s3_batch_post_finalize", "batch_intervals": S3_BLOCK_SIZE})
        hydro = manifest["hydro"]
        evolution = evolve_hr4_full_z(store, dt_hydro=float(hydro["dt_hydro"]), n_hydro_steps=int(hydro["n_hydro_steps"]), batch_intervals=S3_BLOCK_SIZE, chi=float(hydro["chi"]), nu=float(hydro["nu"]), n0=float(hydro["n0"]), gravity_x=float(hydro["gravity_x"]), gravity_y=float(hydro["gravity_y"]), cfl_limit=float(hydro["cfl_limit"]))
        next_fields = store.read_authoritative_batch(0, S3_WINDOW_COUNT)
        for field in FIELDS:
            np.save(destination / f"next_{field}.npy", np.asarray(next_fields[field]))
    finally:
        store.close()
    result = {"schema": S3_SCHEMA, "path": "S3_BATCH_CANONICAL_REFERENCE", "input_manifest_sha256": sha256_file(input_manifest_path), "post_delta_n": str(optical / "s3_optical.hr3b_delta_n_state_after_update_samples.npy"), "next": {field: f"next_{field}.npy" for field in FIELDS}, "evolution": evolution, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _atomic_json(destination / "batch_run.json", result)
    return result


def consume_streaming(*, lifecycle_root: str | Path, hydro: Mapping[str, Any], producer_complete: str | Path, poll_s: float = 0.05, actor: str = "hydro_consumer") -> dict[str, Any]:
    """One actual hydro process: wait for full blocks while optical continues."""
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    completed: list[list[int]] = []
    marker = Path(producer_complete)
    idle = False
    while True:
        block = lifecycle.run_one_hydro_block(dt_hydro=float(hydro["dt_hydro"]), n_hydro_steps=int(hydro["n_hydro_steps"]), chi=float(hydro["chi"]), nu=float(hydro["nu"]), n0=float(hydro["n0"]), gravity_x=float(hydro["gravity_x"]), gravity_y=float(hydro["gravity_y"]), cfl_limit=float(hydro["cfl_limit"]), actor=actor)
        if block:
            if idle:
                lifecycle.record_telemetry("CONSUMER_IDLE_END", actor=actor)
                idle = False
            completed.append(block)
            continue
        lifecycle = StreamingLifecycle.open(lifecycle_root)
        if marker.is_file() and not lifecycle.manifest["queue"]:
            incomplete = [item["ordinal"] for item in lifecycle.manifest["records"] if item["next"] is None]
            if not incomplete:
                lifecycle.record_telemetry("CONSUMER_COMPLETE", actor=actor)
                return {"completed_blocks": completed, "status": "PASS", "actor": actor}
        if not idle:
            lifecycle.record_telemetry("CONSUMER_IDLE_BEGIN", actor=actor)
            idle = True
        time.sleep(float(poll_s))


def finalize_streaming(*, lifecycle_root: str | Path, out_path: str | Path) -> dict[str, Any]:
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    barrier = lifecycle.validate_barrier(actor="s3_barrier")
    lifecycle.inject_s5_fault(
        "F06_BARRIER_PASS_PRE_PROMOTION",
        ordinal=int(lifecycle.manifest["expected_screen_count"]) - 1,
        lifecycle_stage="BARRIER_PASS_PRE_PROMOTION",
    )
    promotion = lifecycle.promote_next_to_current(actor="s3_barrier")
    result = {"schema": S3_SCHEMA, "barrier": barrier, "promotion": promotion, "streaming_manifest_sha256": sha256_file(lifecycle.manifest_path)}
    _atomic_json(Path(out_path), result)
    return result


def _load_stream_field(root: Path, ordinal: int, field: str) -> np.ndarray:
    lifecycle = StreamingLifecycle.open(root)
    entry = lifecycle.manifest["records"][ordinal]["next"]
    if entry is None:
        raise ValueError("streaming NEXT is incomplete")
    return lifecycle._artifact_fields(entry, namespace="NEXT")[field]


def compare_exact(*, input_manifest_path: str | Path, batch_optical_dir: str | Path, batch_hydro_dir: str | Path, streaming_optical_dir: str | Path, streaming_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Perform S3's exact, field-level scientific comparison and persist CSV/JSON."""
    manifest = _read_json(Path(input_manifest_path)); batch_optical, batch_hydro, stream_optical, stream_root, destination = map(Path, (batch_optical_dir, batch_hydro_dir, streaming_optical_dir, streaming_root, out_dir))
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    def check(layer: str, ordinal: int, name: str, left: np.ndarray, right: np.ndarray) -> None:
        rows.append({"layer": layer, "ordinal": ordinal, "screen_id": manifest["screen_records"][ordinal]["screen_id"], "field": name, "shape_equal": bool(left.shape == right.shape), "dtype_equal": bool(left.dtype == right.dtype), "reference_sha256": sha256_array(left), "candidate_sha256": sha256_array(right), "hash_equal": bool(sha256_array(left) == sha256_array(right)), "array_equal": bool(np.array_equal(left, right))})
    for name in DEPOSITION_FIELDS:
        left, right = np.load(batch_optical / f"s3_optical.hr3a_q{name}_samples.npy", mmap_mode="r"), np.load(stream_optical / f"s3_optical.hr3a_q{name}_samples.npy", mmap_mode="r")
        for ordinal in range(S3_WINDOW_COUNT): check("deposition", ordinal, name, left[ordinal], right[ordinal])
    batch_post = np.load(batch_optical / "s3_optical.hr3b_delta_n_state_after_update_samples.npy", mmap_mode="r")
    lifecycle = StreamingLifecycle.open(stream_root)
    for ordinal in range(S3_WINDOW_COUNT):
        post = lifecycle._artifact_fields(lifecycle.manifest["records"][ordinal]["post"], namespace="POST")
        check("post", ordinal, "delta_n", batch_post[ordinal], post["delta_n"])
        zero = np.zeros_like(batch_post[ordinal]); check("post", ordinal, "vx", zero, post["vx"]); check("post", ordinal, "vy", zero, post["vy"])
    for field in FIELDS:
        left = np.load(batch_hydro / f"next_{field}.npy", mmap_mode="r")
        for ordinal in range(S3_WINDOW_COUNT): check("next", ordinal, field, left[ordinal], _load_stream_field(stream_root, ordinal, field))
    with (destination / "hr4e5s_s3_scientific_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    failures = [row for row in rows if not (row["shape_equal"] and row["dtype_equal"] and row["hash_equal"] and row["array_equal"])]
    batch_optical_run, stream_optical_run = _read_json(batch_optical / "optical_run.json"), _read_json(stream_optical / "optical_run.json")
    optical_left = np.load(batch_optical / str(batch_optical_run["final_optical_field"]), mmap_mode="r")
    optical_right = np.load(stream_optical / str(stream_optical_run["final_optical_field"]), mmap_mode="r")
    optical = {"shape_equal": bool(optical_left.shape == optical_right.shape), "dtype_equal": bool(optical_left.dtype == optical_right.dtype), "reference_sha256": sha256_array(optical_left), "candidate_sha256": sha256_array(optical_right), "array_equal": bool(np.array_equal(optical_left, optical_right))}
    optical["hash_equal"] = optical["reference_sha256"] == optical["candidate_sha256"]
    _atomic_json(destination / "hr4e5s_s3_optical_equivalence.json", {"schema": S3_SCHEMA, **optical, "status": "PASS" if all(optical[key] for key in ("shape_equal", "dtype_equal", "hash_equal", "array_equal")) else "FAIL"})
    with np.load(batch_optical / str(batch_optical_run["ledger"]), allow_pickle=False) as left_ledger, np.load(stream_optical / str(stream_optical_run["ledger"]), allow_pickle=False) as right_ledger:
        ledger_rows = []
        for name in sorted(set(left_ledger.files) | set(right_ledger.files)):
            if name not in left_ledger.files or name not in right_ledger.files:
                ledger_rows.append({"field": name, "present_both": False, "equal": False})
                continue
            left, right = np.asarray(left_ledger[name]), np.asarray(right_ledger[name])
            ledger_rows.append({"field": name, "present_both": True, "shape_equal": bool(left.shape == right.shape), "dtype_equal": bool(left.dtype == right.dtype), "reference_sha256": sha256_array(left), "candidate_sha256": sha256_array(right), "equal": bool(left.shape == right.shape and left.dtype == right.dtype and sha256_array(left) == sha256_array(right) and np.array_equal(left, right))})
    ledger = {"schema": S3_SCHEMA, "rows": ledger_rows, "status": "PASS" if ledger_rows and all(row["equal"] for row in ledger_rows) else "FAIL"}
    _atomic_json(destination / "hr4e5s_s3_ledger_equivalence.json", ledger)
    normalized_batch = []
    normalized_stream = []
    for ordinal, record in enumerate(manifest["screen_records"]):
        zeros = np.zeros(tuple(manifest.get("shape", batch_post.shape[1:])), dtype=np.float64)
        normalized_batch.append({"ordinal": ordinal, "screen_id": record["screen_id"], "z_m": record["z_m"], "shape": list(batch_post[ordinal].shape), "dtype": batch_post.dtype.name, "current_delta_n_sha256": record.get("current_delta_n_sha256"), "post": {"delta_n": sha256_array(batch_post[ordinal]), "vx": sha256_array(zeros), "vy": sha256_array(zeros)}, "next": {field: sha256_array(np.load(batch_hydro / f"next_{field}.npy", mmap_mode="r")[ordinal]) for field in FIELDS}})
        stream_record = lifecycle.manifest["records"][ordinal]
        normalized_stream.append({"ordinal": ordinal, "screen_id": stream_record["screen_id"], "z_m": stream_record["z_m"], "shape": list(lifecycle.manifest["shape"]), "dtype": lifecycle.manifest["dtype"], "current_delta_n_sha256": stream_record["current"]["field_sha256"]["delta_n"], "post": dict(stream_record["post"]["field_sha256"]), "next": dict(stream_record["next"]["field_sha256"])})
    normalized = {"schema": S3_SCHEMA, "excluded_runtime_fields": ["job_id", "timestamps", "queue_sequence", "worker_pid", "temporary_path", "runtime_generation_uuid"], "batch": normalized_batch, "streaming": normalized_stream, "status": "PASS" if normalized_batch == normalized_stream else "FAIL"}
    _atomic_json(destination / "hr4e5s_s3_normalized_manifest_equivalence.json", normalized)
    barrier = lifecycle.manifest.get("barrier", {})
    promotion = lifecycle.manifest.get("promotion", {})
    ownership = {"schema": S3_SCHEMA, "before": stream_optical_run.get("optical_current_ownership_before"), "after_optical": stream_optical_run.get("optical_current_ownership_after"), "promotion": promotion, "status": "PASS" if stream_optical_run.get("optical_current_ownership_before", {}).get("authoritative_namespace") == "CURRENT" and stream_optical_run.get("optical_current_ownership_after", {}).get("authoritative_namespace") == "CURRENT" and barrier.get("status") == "PASS" and promotion.get("authoritative_namespace") == "NEXT" else "FAIL"}
    _atomic_json(destination / "hr4e5s_s3_current_next_ownership_audit.json", ownership)
    barrier_audit = {"schema": S3_SCHEMA, "expected_screen_count": S3_WINDOW_COUNT, "barrier": barrier, "promotion": promotion, "status": "PASS" if barrier.get("status") == "PASS" and promotion.get("authoritative_namespace") == "NEXT" else "FAIL"}
    _atomic_json(destination / "hr4e5s_s3_barrier_promotion_audit.json", barrier_audit)
    all_pass = not failures and len(rows) == S3_WINDOW_COUNT * 9 and all(optical[key] for key in ("shape_equal", "dtype_equal", "hash_equal", "array_equal")) and ledger["status"] == "PASS" and normalized["status"] == "PASS" and ownership["status"] == "PASS" and barrier_audit["status"] == "PASS"
    result = {"schema": S3_SCHEMA, "expected_field_comparisons": S3_WINDOW_COUNT * 9, "completed_field_comparisons": len(rows), "mismatch_count": len(failures), "comparisons": rows, "optical_status": "PASS" if all(optical[key] for key in ("shape_equal", "dtype_equal", "hash_equal", "array_equal")) else "FAIL", "ledger_status": ledger["status"], "normalized_manifest_status": normalized["status"], "ownership_status": ownership["status"], "barrier_promotion_status": barrier_audit["status"], "status": "PASS" if all_pass else "FAIL"}
    _atomic_json(destination / "hr4e5s_s3_scientific_comparisons.json", result)
    return result


__all__ = ["RECOVERY_BOOTSTRAP_SCHEMA", "S3_BLOCK_SIZE", "S3_SCHEMA", "S3_WINDOW_COUNT", "S3ReadOnlyCurrentState", "bootstrap_recovery", "compare_exact", "consume_streaming", "create_streaming_lifecycle", "finalize_streaming", "prepare_input_manifest", "run_batch_hydro", "run_optical_path", "validate_recovery_bootstrap_receipt"]
