"""Non-scientific S4 replay and timing utilities for the frozen S3 case."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hr4e5s_s3 import FIELDS, S3_SCHEMA, S3_WINDOW_COUNT, create_streaming_lifecycle
from .hr4e5s_streaming import StreamingLifecycle
from .hr4e_timestep import sha256_array


def _read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def prepare_hydro_replay(*, input_manifest_path: str | Path, batch_optical_dir: str | Path, root: str | Path) -> dict[str, Any]:
    """Materialize the exact batch POST set before the timed hydro replay."""
    manifest = _read(Path(input_manifest_path))
    lifecycle = create_streaming_lifecycle(input_manifest=manifest, root=root)
    post = np.load(Path(batch_optical_dir) / "s3_optical.hr3b_delta_n_state_after_update_samples.npy", mmap_mode="r", allow_pickle=False)
    if post.shape != (S3_WINDOW_COUNT, *tuple(manifest["shape"])) or post.dtype != np.float64:
        raise ValueError("S4 replay POST layout differs from the frozen batch artifact")
    for ordinal in range(S3_WINDOW_COUNT):
        lifecycle.deposition_finalized(ordinal, actor="s4_replay_prepare")
        lifecycle.commit_post_from_delta_n(ordinal, post[ordinal], actor="s4_replay_prepare")
    lifecycle.record_telemetry("REPLAY_POST_READY", actor="s4_replay_prepare", prepared_screen_count=S3_WINDOW_COUNT)
    return {"schema": S3_SCHEMA, "status": "PASS", "root": str(lifecycle.root), "prepared_screen_count": S3_WINDOW_COUNT,
            "post_source": str(Path(batch_optical_dir) / "s3_optical.hr3b_delta_n_state_after_update_samples.npy")}


def enqueue_hydro_replay(*, lifecycle_root: str | Path, producer_complete: str | Path, actor: str = "s4_replay_producer") -> dict[str, Any]:
    """Feed already-persisted POST artifacts into the bounded queue."""
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    lifecycle.record_telemetry("REPLAY_ENQUEUE_START", actor=actor)
    for ordinal in range(int(lifecycle.manifest["expected_screen_count"])):
        lifecycle.enqueue_post(ordinal, actor=actor, wait_for_capacity=True, timeout_s=None)
    marker = Path(producer_complete)
    if marker.exists():
        raise FileExistsError(marker)
    marker.touch()
    lifecycle.record_telemetry("REPLAY_ENQUEUE_COMPLETE", actor=actor)
    return {"schema": S3_SCHEMA, "status": "PASS", "enqueued_screen_count": int(lifecycle.manifest["expected_screen_count"])}


def compare_replay_next(*, batch_hydro_dir: str | Path, lifecycle_root: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Strict NEXT-only exact guard for the S4 hydro-service matrix."""
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    rows: list[dict[str, Any]] = []
    for field in FIELDS:
        reference = np.load(Path(batch_hydro_dir) / f"next_{field}.npy", mmap_mode="r", allow_pickle=False)
        if reference.shape[0] != S3_WINDOW_COUNT or reference.dtype != np.float64:
            raise ValueError("S4 replay batch NEXT layout is invalid")
        for ordinal in range(S3_WINDOW_COUNT):
            candidate = lifecycle._artifact_fields(lifecycle.manifest["records"][ordinal]["next"], namespace="NEXT")[field]
            rows.append({"ordinal": ordinal, "field": field, "shape_equal": bool(reference[ordinal].shape == candidate.shape),
                         "dtype_equal": bool(reference[ordinal].dtype == candidate.dtype), "reference_sha256": sha256_array(reference[ordinal]),
                         "candidate_sha256": sha256_array(candidate), "hash_equal": bool(sha256_array(reference[ordinal]) == sha256_array(candidate)),
                         "array_equal": bool(np.array_equal(reference[ordinal], candidate))})
    failures = [row for row in rows if not all(row[key] for key in ("shape_equal", "dtype_equal", "hash_equal", "array_equal"))]
    result = {"schema": S3_SCHEMA, "status": "PASS" if not failures and len(rows) == S3_WINDOW_COUNT * len(FIELDS) else "FAIL",
              "expected_field_comparisons": S3_WINDOW_COUNT * len(FIELDS), "completed_field_comparisons": len(rows),
              "mismatch_count": len(failures), "comparisons": rows}
    _atomic_json(Path(out_path), result)
    return result


def _event_times(events: list[Mapping[str, Any]], name: str) -> list[float]:
    return [float(item["monotonic_s"]) for item in events if item.get("event") == name]


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_telemetry(*, lifecycle_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Derive S4 timing/queue/worker metrics from persisted monotonic events."""
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    events = list(lifecycle.manifest.get("telemetry_events", []))
    if not events:
        raise ValueError("S4 telemetry is absent")
    events = sorted(events, key=lambda item: float(item["monotonic_s"]))
    destination = Path(out_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    columns = sorted({key for event in events for key in event})
    with (destination / "hr4e5s_s4_runtime_events.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(events)
    queue_events = [event for event in events if "queue_occupancy" in event]
    with (destination / "hr4e5s_s4_queue_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(queue_events)
    optical_start, optical_end = _event_times(events, "OPTICAL_START"), _event_times(events, "OPTICAL_COMPLETE")
    post = _event_times(events, "POST_COMMITTED")
    hydro_claim = _event_times(events, "HYDRO_CLAIM")
    hydro_start, hydro_end = _event_times(events, "HYDRO_BLOCK_START"), _event_times(events, "HYDRO_BLOCK_END")
    next_commit = _event_times(events, "HYDRO_SCREEN_END")
    barrier = _event_times(events, "BARRIER_PASS")
    streaming_mode = bool(optical_start or optical_end)
    if bool(optical_start) != bool(optical_end):
        raise ValueError("S4 optical telemetry has an unmatched start or completion event")
    required = {"post": post, "hydro_claim": hydro_claim, "hydro_start": hydro_start, "hydro_end": hydro_end, "next": next_commit, "barrier": barrier}
    if streaming_mode:
        required.update({"optical_start": optical_start, "optical_end": optical_end})
    if any(not value for value in required.values()):
        raise ValueError("S4 telemetry is incomplete")
    opt_active = optical_end[-1] - optical_start[0] if streaming_mode else None
    hydro_span = next_commit[-1] - hydro_start[0]
    overlap = max(0.0, min(optical_end[-1], next_commit[-1]) - max(optical_start[0], hydro_start[0])) if streaming_mode else None
    backpressure_begin = _event_times(events, "PRODUCER_BACKPRESSURE_BEGIN")
    backpressure_end = _event_times(events, "PRODUCER_BACKPRESSURE_END")
    blocked = sum(max(0.0, right - left) for left, right in zip(backpressure_begin, backpressure_end))
    occupancy = [float(event["queue_occupancy"]) for event in queue_events]
    capacity = int(lifecycle.manifest["queue_depth"])
    workers: dict[str, dict[str, Any]] = {}
    for event in events:
        actor = str(event.get("actor", ""))
        if not actor.startswith("hydro_consumer"):
            continue
        entry = workers.setdefault(actor, {"actor": actor, "claims": 0, "blocks_completed": 0, "work_s": 0.0, "idle_s": 0.0, "active": {}})
        if event.get("event") == "HYDRO_CLAIM": entry["claims"] += 1
        if event.get("event") == "HYDRO_BLOCK_START": entry["active"]["work"] = float(event["monotonic_s"])
        if event.get("event") == "HYDRO_BLOCK_END":
            entry["blocks_completed"] += 1; entry["work_s"] += max(0.0, float(event["monotonic_s"]) - float(entry["active"].pop("work", event["monotonic_s"])))
        if event.get("event") == "CONSUMER_IDLE_BEGIN": entry["active"]["idle"] = float(event["monotonic_s"])
        if event.get("event") == "CONSUMER_IDLE_END": entry["idle_s"] += max(0.0, float(event["monotonic_s"]) - float(entry["active"].pop("idle", event["monotonic_s"])))
    worker_rows = [{key: value for key, value in row.items() if key != "active"} for row in workers.values()]
    with (destination / "hr4e5s_s4_worker_utilization.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actor", "claims", "blocks_completed", "work_s", "idle_s"]); writer.writeheader(); writer.writerows(worker_rows)
    active_blocks: dict[tuple[str, tuple[int, ...]], float] = {}
    block_durations: list[float] = []
    for event in events:
        block = event.get("block")
        if not isinstance(block, list):
            continue
        key = (str(event.get("actor", "")), tuple(int(value) for value in block))
        if event.get("event") == "HYDRO_BLOCK_START":
            active_blocks[key] = float(event["monotonic_s"])
        elif event.get("event") == "HYDRO_BLOCK_END" and key in active_blocks:
            block_durations.append(max(0.0, float(event["monotonic_s"]) - active_blocks.pop(key)))
    if active_blocks or len(block_durations) != len(hydro_end):
        raise ValueError("S4 hydro block telemetry is unpaired")
    result = {"schema": "khz_filament.hr4e5s.s4.telemetry.v1", "status": "PASS", "mode": "stream" if streaming_mode else "replay", "event_count": len(events),
              "clock": "time.perf_counter; no explicit device synchronization", "optical_start_s": optical_start[0] if streaming_mode else None, "optical_complete_s": optical_end[-1] if streaming_mode else None,
              "first_post_commit_s": post[0], "last_post_commit_s": post[-1], "first_hydro_claim_s": hydro_claim[0], "first_hydro_start_s": hydro_start[0], "last_next_commit_s": next_commit[-1],
              "barrier_pass_s": barrier[-1], "T_stream_s": barrier[-1] - optical_start[0] if streaming_mode else None, "T_opt_active_s": opt_active,
              "T_hydro_span_s": hydro_span, "T_tail_s": max(0.0, next_commit[-1] - optical_end[-1]) if streaming_mode else None,
              "T_startup_to_hydro_s": hydro_start[0] - post[0], "T_overlap_s": overlap,
              "overlap_fraction_of_optical": overlap / opt_active if streaming_mode and opt_active and opt_active > 0 else None,
              "overlap_fraction_of_hydro_span": overlap / hydro_span if streaming_mode and hydro_span > 0 else None,
              "R_opt_screens_per_s": S3_WINDOW_COUNT / (post[-1] - post[0]) if streaming_mode and post[-1] > post[0] else None,
              "R_hydro_screens_per_s": S3_WINDOW_COUNT / (next_commit[-1] - hydro_claim[0]) if next_commit[-1] > hydro_claim[0] else None,
              "queue": {"capacity": capacity, "max": max(occupancy), "mean": float(np.mean(occupancy)), "median": float(np.median(occupancy)), "p95": _percentile(occupancy, 95),
                        "sample_fraction_empty": sum(value == 0 for value in occupancy) / len(occupancy), "sample_fraction_full": sum(value == capacity for value in occupancy) / len(occupancy),
                        "producer_block_events": len(backpressure_begin), "producer_blocked_s": blocked, "producer_blocked_fraction": blocked / opt_active if streaming_mode and opt_active and opt_active > 0 else None},
              "median_hydro_block_service_s": _percentile(block_durations, 50), "worker_rows": worker_rows}
    _atomic_json(destination / "hr4e5s_s4_streaming_metrics.json", result)
    return result


__all__ = ["compare_replay_next", "enqueue_hydro_replay", "prepare_hydro_replay", "summarize_telemetry"]
