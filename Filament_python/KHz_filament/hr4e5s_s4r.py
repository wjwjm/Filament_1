"""S4R long-replay qualification utilities; the HR-4 solver remains frozen."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hr4e_timestep import sha256_array, sha256_file
from .hr4 import HR4_CHI, HR4_CFL_LIMIT, HR4_GRAVITY_X, HR4_GRAVITY_Y, HR4_NU
from .hr4e5s_s3 import S3_BLOCK_SIZE, create_streaming_lifecycle
from .hr4e5s_s4 import enqueue_hydro_replay
from .hr4e5s_streaming import FIELDS, StreamingLifecycle


S4R_SCHEMA = "khz_filament.hr4e5s.s4r.v1"
S4R_SCREEN_COUNT = 384
R_OPT = 0.756190444
R_HYDRO_REQUIRED = 0.831809488


def _read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _source_window(source: Mapping[str, Any], state: np.ndarray) -> list[int]:
    z = np.asarray(source["source_z_positions_m"], dtype=np.float64)
    if state.dtype != np.dtype(np.float64) or state.ndim != 3 or z.shape != (state.shape[0],):
        raise ValueError("S4R source is not the frozen float64 [K,Ny,Nx] HR-3B state")
    peak = int(np.argmax(np.maximum(-np.min(state, axis=(1, 2)), 0.0)))
    first, stop = peak - S4R_SCREEN_COUNT // 2, peak + S4R_SCREEN_COUNT // 2
    if first < 0 or stop > state.shape[0]:
        raise ValueError("frozen E1B peak cannot support the required interior 384-screen window")
    indices = list(range(first, stop))
    if indices[0] > 7998 or indices[-1] < 8045 or len(indices) != S4R_SCREEN_COUNT:
        raise ValueError("S4R deterministic window does not include the S3/S4 qualification region")
    if not np.all(np.diff(z[indices]) > 0.0) or not np.all(np.isfinite(state[indices])):
        raise ValueError("S4R selected source window is non-finite or non-monotone")
    return indices


def prepare_input_manifest(*, source_manifest_path: str | Path, source_state_path: str | Path,
                           config_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Freeze one genuine, peak-centred 384-screen HR-3B POST replay input."""
    source_manifest_file, state_file, config_file, output = map(Path, (source_manifest_path, source_state_path, config_path, out_path))
    if output.exists():
        raise FileExistsError(output)
    source = _read(source_manifest_file)
    if sha256_file(state_file) != str(source["hr3b_state_file_sha256"]):
        raise ValueError("S4R source state file SHA256 disagrees with the frozen E1B manifest")
    if sha256_file(config_file) != str(source["config_sha256"]):
        raise ValueError("S4R config SHA256 disagrees with the frozen E1B manifest")
    state = np.load(state_file, mmap_mode="r", allow_pickle=False)
    try:
        if sha256_array(state) != str(source["hr3b_state_sha256"]):
            raise ValueError("S4R source canonical array SHA256 disagrees with the frozen E1B manifest")
        indices = _source_window(source, state)
        z = np.asarray(source["source_z_positions_m"], dtype=np.float64)
        records = []
        for ordinal, index in enumerate(indices):
            current = np.asarray(state[index], dtype=np.float64)
            records.append({"ordinal": ordinal, "screen_id": f"source_index_{index:05d}", "source_index": index,
                            "z_m": float(z[index]), "current_delta_n_sha256": sha256_array(current),
                            "post_delta_n_sha256": sha256_array(current), "post_source": "E1B_hr3b_authoritative_state",
                            "current_velocity_initialization": "exact_zero_float64", "post_velocity_initialization": "exact_zero_float64"})
        shape = list(state.shape[1:])
    finally:
        close = getattr(state, "_mmap", None)
        if close is not None:
            close.close()
    result = {"schema": S4R_SCHEMA, "stage": "HR-4E-5S-S4R", "source_manifest": str(source_manifest_file),
              "source_manifest_sha256": sha256_file(source_manifest_file), "source_state": str(state_file),
              "source_state_file_sha256": str(source["hr3b_state_file_sha256"]), "source_state_array_sha256": str(source["hr3b_state_sha256"]),
              "config": str(config_file), "config_sha256": sha256_file(config_file),
              "current_generation": "E1B_hr3b_source:current", "post_generation": "E1B_hr3b_authoritative:post",
              "window_selection_rule": "first argmax(-min(delta_n)); indices peak-192 through peak+191; 384 genuine contiguous screens",
              "peak_source_index": indices[S4R_SCREEN_COUNT // 2], "screen_indices": indices, "screen_records": records,
              "shape": shape, "dtype": "float64", "dx_m": float(3.01e-3 / 301), "dy_m": float(3.51e-3 / 351),
              "hydro": {"dt_hydro": 1.0e-6, "n_hydro_steps": 1000, "chi": HR4_CHI, "nu": HR4_NU, "n0": float(source["n0"]),
                         "gravity_x": HR4_GRAVITY_X, "gravity_y": HR4_GRAVITY_Y, "cfl_limit": HR4_CFL_LIMIT, "block_size": S3_BLOCK_SIZE, "queue_depth": 16},
              "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(output, result)
    return result


def prepare_hydro_replay(*, input_manifest_path: str | Path, root: str | Path) -> dict[str, Any]:
    """Materialize immutable CURRENT and authoritative HR-3B POST artifacts."""
    manifest = _read(Path(input_manifest_path))
    if len(manifest["screen_records"]) != S4R_SCREEN_COUNT or int(manifest["hydro"]["block_size"]) != S3_BLOCK_SIZE:
        raise ValueError("S4R input does not meet the frozen 384-screen/block-8 contract")
    lifecycle = create_streaming_lifecycle(input_manifest=manifest, root=root)
    source = np.load(str(manifest["source_state"]), mmap_mode="r", allow_pickle=False)
    try:
        for record in manifest["screen_records"]:
            ordinal, source_index = int(record["ordinal"]), int(record["source_index"])
            post = np.asarray(source[source_index], dtype=np.float64)
            if sha256_array(post) != str(record["post_delta_n_sha256"]):
                raise ValueError(f"S4R POST provenance mismatch for source index {source_index}")
            lifecycle.deposition_finalized(ordinal, actor="s4r_replay_prepare")
            lifecycle.commit_post_from_delta_n(ordinal, post, actor="s4r_replay_prepare", hr3a_authoritative=True, hr3b_authoritative=True)
    finally:
        close = getattr(source, "_mmap", None)
        if close is not None:
            close.close()
    lifecycle.record_telemetry("S4R_REPLAY_POST_READY", actor="s4r_replay_prepare", prepared_screen_count=S4R_SCREEN_COUNT,
                               post_source="E1B_hr3b_authoritative_state")
    return {"schema": S4R_SCHEMA, "status": "PASS", "root": str(lifecycle.root), "prepared_screen_count": S4R_SCREEN_COUNT,
            "post_source": "E1B_hr3b_authoritative_state", "input_manifest_sha256": sha256_file(input_manifest_path)}


def enqueue_replay(*, lifecycle_root: str | Path, producer_complete: str | Path) -> dict[str, Any]:
    return enqueue_hydro_replay(lifecycle_root=lifecycle_root, producer_complete=producer_complete, actor="s4r_replay_enqueue")


def compare_replay_next(*, reference_lifecycle_root: str | Path, lifecycle_root: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Require all 384 × 3 NEXT fields to be bitwise-identical to N=1."""
    reference, candidate = StreamingLifecycle.open(reference_lifecycle_root), StreamingLifecycle.open(lifecycle_root)
    rows: list[dict[str, Any]] = []
    if reference.manifest["expected_screen_count"] != S4R_SCREEN_COUNT or candidate.manifest["expected_screen_count"] != S4R_SCREEN_COUNT:
        raise ValueError("S4R exact reference/candidate screen count is invalid")
    for ordinal in range(S4R_SCREEN_COUNT):
        left_record, right_record = reference.manifest["records"][ordinal], candidate.manifest["records"][ordinal]
        for field in FIELDS:
            left = reference._artifact_fields(left_record["next"], namespace="NEXT")[field]
            right = candidate._artifact_fields(right_record["next"], namespace="NEXT")[field]
            rows.append({"ordinal": ordinal, "screen_id": left_record["screen_id"], "field": field,
                         "screen_identity_equal": bool(left_record["screen_id"] == right_record["screen_id"] and left_record["z_m"] == right_record["z_m"]),
                         "shape_equal": bool(left.shape == right.shape), "dtype_equal": bool(left.dtype == right.dtype),
                         "reference_sha256": sha256_array(left), "candidate_sha256": sha256_array(right),
                         "hash_equal": bool(sha256_array(left) == sha256_array(right)), "array_equal": bool(np.array_equal(left, right))})
    failures = [row for row in rows if not all(bool(row[key]) for key in ("screen_identity_equal", "shape_equal", "dtype_equal", "hash_equal", "array_equal"))]
    destination = Path(out_path)
    _write(destination, {"schema": S4R_SCHEMA, "reference_lifecycle": str(reference_lifecycle_root), "candidate_lifecycle": str(lifecycle_root),
                         "expected_field_comparisons": S4R_SCREEN_COUNT * len(FIELDS), "completed_field_comparisons": len(rows),
                         "mismatch_count": len(failures), "status": "PASS" if not failures else "FAIL", "comparisons": rows})
    csv_path = destination.with_suffix(".csv")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return _read(destination)


def _durations(timing_dir: Path) -> list[dict[str, Any]]:
    rows, open_events = [], {}
    for path in sorted(timing_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("schema") != "khz_filament.hr4e5s.s4r.timing.v1":
                continue
            key = (event.get("pid"), event.get("worker_id"), event.get("phase"), event.get("ordinal"), event.get("artifact"))
            if event.get("boundary") == "begin":
                open_events[key] = event
            elif event.get("boundary") == "end" and key in open_events:
                start = open_events.pop(key)
                duration = float(event["monotonic_s"]) - float(start["monotonic_s"])
                if duration < 0.0:
                    raise ValueError("S4R timing clock regressed")
                rows.append({"worker_id": event["worker_id"], "pid": event["pid"], "gpu_visible_devices": event.get("gpu_visible_devices", ""),
                             "phase": event["phase"], "ordinal": event.get("ordinal"), "artifact": event.get("artifact"),
                             "start_s": start["monotonic_s"], "end_s": event["monotonic_s"], "duration_s": duration})
    if open_events:
        raise ValueError("S4R timing events have unmatched begin records")
    return rows


def summarize_replay(*, lifecycle_root: str | Path, timing_dir: str | Path, out_dir: str | Path, hydro_workers: int) -> dict[str, Any]:
    """Persist non-invasive timing decomposition and strict service-rate evidence."""
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    events = list(lifecycle.manifest.get("telemetry_events", []))
    claims = [float(item["monotonic_s"]) for item in events if item.get("event") == "HYDRO_CLAIM"]
    commits = [float(item["monotonic_s"]) for item in events if item.get("event") == "HYDRO_SCREEN_END"]
    if len(claims) != S4R_SCREEN_COUNT // S3_BLOCK_SIZE or len(commits) != S4R_SCREEN_COUNT:
        raise ValueError("S4R hydro claim/commit telemetry is incomplete")
    span = commits[-1] - claims[0]
    if span <= 0.0:
        raise ValueError("S4R service span is invalid")
    rows = _durations(Path(timing_dir))
    destination = Path(out_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with (destination / "hr4e5s_s4r_block_timings.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["worker_id", "pid", "gpu_visible_devices", "phase", "ordinal", "artifact", "start_s", "end_s", "duration_s"])
        writer.writeheader(); writer.writerows(rows)
    hydro_rows = [row for row in rows if str(row["worker_id"]).startswith("hydro_consumer_")]
    phase_seconds: dict[str, float] = defaultdict(float)
    for row in hydro_rows:
        phase = str(row["phase"])
        if phase == "ARTIFACT_WRITE" and "/next/" not in str(row.get("artifact", "")):
            continue
        if phase == "ARTIFACT_ATOMIC_RENAME" and "/next/" not in str(row.get("artifact", "")):
            continue
        phase_seconds[phase] += float(row["duration_s"])
    total_measured = sum(phase_seconds.values())
    budget = [{"hydro_workers": int(hydro_workers), "component": name, "cumulative_seconds": seconds,
               "fraction_of_measured_cumulative": None if total_measured == 0.0 else seconds / total_measured,
               "measurement_note": "HR4_SOLVER_ENVELOPE includes frozen host-to-device transfer plus GPU kernels; no new device synchronization was introduced" if name == "HR4_SOLVER_ENVELOPE" else ""}
              for name, seconds in sorted(phase_seconds.items())]
    with (destination / "hr4e5s_s4r_overhead_budget.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(budget[0]) if budget else ["hydro_workers", "component", "cumulative_seconds", "fraction_of_measured_cumulative", "measurement_note"])
        writer.writeheader(); writer.writerows(budget)
    lock_rows = [row for row in hydro_rows if row["phase"] in {"MANIFEST_LOCK_WAIT", "MANIFEST_LOCK_HOLD"}]
    with (destination / "hr4e5s_s4r_lock_wait.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["worker_id", "pid", "gpu_visible_devices", "phase", "ordinal", "artifact", "start_s", "end_s", "duration_s"])
        writer.writeheader(); writer.writerows(lock_rows)
    workers: dict[str, dict[str, Any]] = {}
    for row in hydro_rows:
        item = workers.setdefault(str(row["worker_id"]), {"worker_id": row["worker_id"], "gpu_visible_devices": row["gpu_visible_devices"], "measured_cumulative_seconds": 0.0})
        item["measured_cumulative_seconds"] += float(row["duration_s"])
    worker_rows = list(workers.values())
    with (destination / "hr4e5s_s4r_worker_utilization.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["worker_id", "gpu_visible_devices", "measured_cumulative_seconds"])
        writer.writeheader(); writer.writerows(worker_rows)
    service = {"schema": S4R_SCHEMA, "status": "PASS", "hydro_workers": int(hydro_workers), "screen_count": S4R_SCREEN_COUNT,
               "block_size": S3_BLOCK_SIZE, "block_count": S4R_SCREEN_COUNT // S3_BLOCK_SIZE, "first_hydro_claim_s": claims[0],
               "last_next_commit_s": commits[-1], "hydro_service_span_s": span, "hydro_screens_per_s": S4R_SCREEN_COUNT / span,
               "capacity_ratio": (S4R_SCREEN_COUNT / span) / R_OPT, "capacity_headroom": (S4R_SCREEN_COUNT / span) / R_OPT - 1.0,
               "required_hydro_screens_per_s": R_HYDRO_REQUIRED, "device_sync_introduced_by_telemetry": 0,
               "timing_limitations": "No device-event split was added inside frozen advance_hr4_single_screen; its envelope includes host-to-device transfer and GPU compute."}
    _write(destination / "hr4e5s_s4r_service_rate.json", service)
    return service


__all__ = ["R_HYDRO_REQUIRED", "R_OPT", "S4R_SCHEMA", "S4R_SCREEN_COUNT", "compare_replay_next", "enqueue_replay", "prepare_hydro_replay", "prepare_input_manifest", "summarize_replay"]
