"""Conservative encoded storage bound for the fixed N3/K8048 journal path.

This module is admission-side accounting.  It uses the journal's canonical
encoder and the streaming checkpoint/HEAD JSON layout, but does not alter the
journal protocol or lifecycle writer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

from .hr4e5_streaming_journal import (
    HEAD_SCHEMA,
    JOURNAL_SCHEMA,
    MAX_TRANSACTION_BYTES,
    _canonical,
)


N3_K = 8048
N3_PULSES = 3
N3_EXACT_FIELD_OBJECT_PAIRS = 338016
N3_EXACT_METADATA_KEYS = 30
N3_BASE_TRANSACTIONS = 233420
N3_BACKPRESSURE_TRANSACTIONS = 249516
N3_BACKPRESSURE_EXTRA = N3_BACKPRESSURE_TRANSACTIONS - N3_BASE_TRANSACTIONS
N3_ROOT_COUNT = 2 * N3_PULSES
N3_BLOCK_SIZE = 8
N3_QUEUE_DEPTH = 16
N3_STREAMING_SCHEMA = "khz_filament.hr4e5s.streaming.v1"

MAX_PATH_UTF8_BYTES = 512
MAX_CAMPAIGN_ID_BYTES = 128
MAX_SCREEN_ID_BYTES = 64
MAX_GENERATION_ID_BYTES = 128
MAX_ACTOR_BYTES = 64
MAX_EVENT_BYTES = 64
MAX_WORKER_ID_BYTES = 64
MAX_CUDA_TOKEN_BYTES = 64
MAX_CUDA_VISIBLE_DEVICES_BYTES = 6 * MAX_CUDA_TOKEN_BYTES + 5
MAX_HOST_IDENTITY_BYTES = 292  # POSIX hostname (255) + ':' + Linux boot UUID (36)
MAX_PROCESS_START_TOKEN_BYTES = 20
MAX_PID = 2**31 - 1
MAX_DYNAMIC_STRING_BYTES = 768
MAX_EXACT_RELATIVE_PATH_BYTES = 512
MAX_EXACT_IDENTIFIER_BYTES = 128
MAX_EXACT_ARRAY_RANK = 3
MAX_EXACT_ARRAY_DIMENSION = 384
TERMINAL_REFERENCE_RESERVATION_BYTES = 16 * 1024**2  # production _run_terminal contract

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SAFE_EVENT = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SAFE_PROCESS_TOKEN = re.compile(r"^[0-9]+$")
_MAX_FLOAT = 1.7976931348623157e308
_NEG_MAX_FLOAT = -_MAX_FLOAT
_HASH = "f" * 64
_ACTOR = "A" * MAX_ACTOR_BYTES
_WORKER = "W" * MAX_WORKER_ID_BYTES
_CUDA = ",".join("G" * MAX_CUDA_TOKEN_BYTES for _ in range(6))
_HOST = "H" * 255 + ":" + "B" * 36
_PROCESS_TOKEN = "9" * MAX_PROCESS_START_TOKEN_BYTES
_GENERATION = "G" * MAX_GENERATION_ID_BYTES
_NEXT_GENERATION = "N" * MAX_GENERATION_ID_BYTES


def _utf8_text(value: Any, *, label: str, maximum: int, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string without NUL")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    if len(encoded) > maximum:
        raise ValueError(f"{label} exceeds its {maximum}-byte admission bound")
    if pattern is not None and not pattern.fullmatch(value):
        raise ValueError(f"{label} contains characters outside the fixed journal alphabet")
    return value


def _absolute_path(value: Any, *, label: str) -> str:
    raw = _utf8_text(str(value), label=label, maximum=MAX_PATH_UTF8_BYTES)
    if ".." in raw.replace("\\", "/").split("/"):
        raise ValueError(f"{label} contains parent traversal")
    if not (Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute()):
        raise ValueError(f"{label} must be absolute")
    return raw


def validate_n3_journal_entry_strings(
    *,
    campaign_root: str | Path,
    campaign_id: str,
    paths: Mapping[str, str | Path] | None = None,
    actor: str,
    event: str,
    worker_id: str,
    cuda_visible_devices: str | None,
    process_identity: Mapping[str, Any],
    screen_ids: Sequence[str] = (),
    current_generation: str,
    next_generation: str,
    prospective_cuda_tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate every caller-controlled string before N3 journal writes.

    The official path uses fixed event/actor templates, bounded environment
    values and OS process identity.  The check rejects overlong values instead
    of shortening them, preserving the bytes that would otherwise be recorded.
    """
    checked_root = _absolute_path(campaign_root, label="campaign_root")
    checked_paths = {
        str(name): _absolute_path(value, label=f"path[{name}]")
        for name, value in dict(paths or {}).items()
    }
    _utf8_text(campaign_id, label="campaign_id", maximum=MAX_CAMPAIGN_ID_BYTES, pattern=_SAFE_ID)
    _utf8_text(actor, label="actor", maximum=MAX_ACTOR_BYTES, pattern=_SAFE_ID)
    _utf8_text(event, label="event", maximum=MAX_EVENT_BYTES, pattern=_SAFE_EVENT)
    _utf8_text(worker_id, label="worker_id", maximum=MAX_WORKER_ID_BYTES, pattern=_SAFE_ID)
    cuda_status = "actual"
    if cuda_visible_devices is None or cuda_visible_devices == "":
        if prospective_cuda_tokens is None:
            raise ValueError("missing CUDA_VISIBLE_DEVICES requires explicit prospective tokens")
        tokens = list(prospective_cuda_tokens)
        cuda_status = "prospective"
    else:
        _utf8_text(cuda_visible_devices, label="CUDA_VISIBLE_DEVICES",
                   maximum=MAX_CUDA_VISIBLE_DEVICES_BYTES)
        tokens = cuda_visible_devices.split(",")
    if not 1 <= len(tokens) <= 6:
        raise ValueError("CUDA_VISIBLE_DEVICES must contain one to six bounded device tokens")
    for token in tokens:
        _utf8_text(token, label="CUDA_VISIBLE_DEVICES token", maximum=MAX_CUDA_TOKEN_BYTES,
                   pattern=_SAFE_ID)
    if cuda_status == "prospective" and len(tokens) != 6:
        raise ValueError("prospective CUDA_VISIBLE_DEVICES must describe all six N3 devices")
    checked_cuda = ",".join(tokens)
    _utf8_text(checked_cuda, label="CUDA_VISIBLE_DEVICES",
               maximum=MAX_CUDA_VISIBLE_DEVICES_BYTES)
    _utf8_text(current_generation, label="current_generation",
               maximum=MAX_GENERATION_ID_BYTES, pattern=_SAFE_ID)
    _utf8_text(next_generation, label="next_generation",
               maximum=MAX_GENERATION_ID_BYTES, pattern=_SAFE_ID)
    if current_generation == next_generation:
        raise ValueError("current and next generation identities must differ")
    for index, screen_id in enumerate(screen_ids):
        _utf8_text(screen_id, label=f"screen_ids[{index}]", maximum=MAX_SCREEN_ID_BYTES,
                   pattern=_SAFE_ID)
    if not isinstance(process_identity, Mapping):
        raise ValueError("process identity must be a mapping")
    pid = process_identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or not 1 <= pid <= MAX_PID:
        raise ValueError("process identity PID is outside its fixed integer range")
    _utf8_text(process_identity.get("host_identity"), label="process host identity",
               maximum=MAX_HOST_IDENTITY_BYTES, pattern=_SAFE_ID)
    _utf8_text(process_identity.get("process_start_token"), label="process start token",
               maximum=MAX_PROCESS_START_TOKEN_BYTES, pattern=_SAFE_PROCESS_TOKEN)
    return {
        "campaign_root": checked_root,
        "paths": checked_paths,
        "campaign_id": campaign_id,
        "actor": actor,
        "event": event,
        "worker_id": worker_id,
        "cuda_visible_devices": checked_cuda if cuda_status == "actual" else None,
        "cuda_identity_status": cuda_status,
        "prospective_cuda_tokens": tokens if cuda_status == "prospective" else None,
        "screen_id_count": len(screen_ids),
        "process_identity": {
            "pid": pid,
            "host_identity": process_identity["host_identity"],
            "process_start_token": process_identity["process_start_token"],
        },
    }


def _screen_id(ordinal: int) -> str:
    return f"S{ordinal:05d}" + "X" * (MAX_SCREEN_ID_BYTES - 6)


def _field_hashes() -> dict[str, str]:
    return {name: _HASH for name in ("delta_n", "vx", "vy")}


def _transition(ordinal: int, state: str, *, extra: Mapping[str, Any] = ()) -> dict[str, Any]:
    event: dict[str, Any] = {
        "state": state,
        "status": "PASS",
        "timestamp_utc": "9" * 20,
        "actor": _ACTOR,
        "ordinal": ordinal,
        "screen_id": _screen_id(ordinal),
        "current_generation": _GENERATION,
        "next_generation": _NEXT_GENERATION,
        "retry_count": 1,
        "current_content_sha256": _HASH,
        "source_file_sha256": _HASH,
    }
    event.update(extra)
    return event


def _artifact(ordinal: int, namespace: str, *, next_state: bool = False) -> dict[str, Any]:
    hashes = _field_hashes()
    metadata: dict[str, Any] = {
        "schema": N3_STREAMING_SCHEMA,
        "namespace": namespace,
        "ordinal": ordinal,
        "screen_id": _screen_id(ordinal),
        "z_m": _NEG_MAX_FLOAT,
        "current_generation": _GENERATION,
        "current_content_sha256": _HASH,
        "field_sha256": hashes,
        "shape": [351, 301],
        "dtype": "float64",
    }
    if next_state:
        metadata.update({
            "next_generation": _NEXT_GENERATION,
            "post_file_sha256": _HASH,
            "post_field_sha256": hashes,
        })
    else:
        metadata.update({
            "current_field_sha256": hashes,
            "hr3a_authoritative": True,
            "hr3b_authoritative": True,
        })
    return {
        "artifact": f"{namespace.lower()}/screen_{ordinal:06d}.npz",
        "file_sha256": _HASH,
        "field_sha256": hashes,
        "metadata": metadata,
    }


def _telemetry(ordinal: int | None = None, *, event: str = "E" * MAX_EVENT_BYTES,
               block: Sequence[int] | None = None, producer_blocked: bool = False,
               final_pulse: bool | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event": event,
        "monotonic_s": _MAX_FLOAT,
        "actor": _ACTOR,
        "worker_id": _WORKER,
        "gpu_visible_devices": _CUDA,
        "queue_occupancy": N3_QUEUE_DEPTH,
        "queue_capacity": N3_QUEUE_DEPTH,
        "producer_blocked": producer_blocked,
    }
    if ordinal is not None:
        row.update({"ordinal": ordinal, "screen_id": _screen_id(ordinal), "z_m": _NEG_MAX_FLOAT})
    if block is not None:
        row["block"] = list(block)
    if final_pulse is not None:
        row["final_pulse"] = final_pulse
    return row


def _process() -> dict[str, Any]:
    return {"pid": MAX_PID, "host_identity": _HOST, "process_start_token": _PROCESS_TOKEN}


def _journal_line_bytes(*, sequence: int, operation: str,
                        operations: Sequence[Mapping[str, Any]]) -> int:
    body = {
        "schema": JOURNAL_SCHEMA,
        "lineage_id": "L" * 32,
        "sequence": sequence,
        "previous_transaction_sha256": _HASH,
        "operation": operation,
        "actor": _ACTOR,
        "process": _process(),
        "committed_utc": "9" * 20,
        "operations": [dict(item) for item in operations],
    }
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    return len(_canonical({**body, "transaction_sha256": digest}) + b"\n")


def _set(path: list[Any], value: Any, *, existed: bool = True, previous: Any = None) -> dict[str, Any]:
    return {"op": "set", "path": path, "value": value,
            "existed": existed, "previous": previous}


def _append(path: list[Any], value: Any) -> dict[str, Any]:
    return {"op": "append", "path": path, "value": value}


def _ordinary_operation_templates() -> dict[str, list[dict[str, Any]]]:
    last = N3_K - 1
    ordinal = last
    post = _artifact(ordinal, "POST")
    next_entry = _artifact(ordinal, "NEXT", next_state=True)
    old_queue = list(range(N3_QUEUE_DEPTH))
    new_queue = list(range(N3_QUEUE_DEPTH - 1)) + [last]

    optical_stage = [
        _set(["records", ordinal, "state"], "OPTICAL_IN_PROGRESS", previous="CURRENT_READY"),
        _append(["records", ordinal, "transitions"], _transition(ordinal, "OPTICAL_IN_PROGRESS")),
        _set(["records", ordinal, "state"], "DEPOSITION_FINALIZED", previous="OPTICAL_IN_PROGRESS"),
        _append(["records", ordinal, "transitions"], _transition(
            ordinal, "DEPOSITION_FINALIZED", extra={"optical_finalized_s": _MAX_FLOAT})),
    ]
    commit_post = [
        _set(["records", ordinal, "post"], post, existed=False),
        _append(["rate_events"], {
            "ordinal": ordinal, "screen_id": _screen_id(ordinal), "z_m": _NEG_MAX_FLOAT,
            "event": "POST_COMMITTED", "time_s": _MAX_FLOAT,
            "optical_finalized_s": _MAX_FLOAT, "queue_depth": N3_QUEUE_DEPTH,
            "backpressure": False,
        }),
        _append(["telemetry_events"], _telemetry(ordinal, event="POST_COMMITTED")),
        _set(["records", ordinal, "state"], "POST_COMMITTED", previous="DEPOSITION_FINALIZED"),
        _append(["records", ordinal, "transitions"], _transition(
            ordinal, "POST_COMMITTED", extra={
                "source_file_sha256": _HASH, "output_file_sha256": _HASH,
                "post_committed_s": _MAX_FLOAT,
            })),
    ]
    enqueue_post = [
        _append(["queue"], last),
        _set(["queue"], new_queue, previous=old_queue),
        _append(["rate_events"], {
            "ordinal": ordinal, "screen_id": _screen_id(ordinal), "z_m": _NEG_MAX_FLOAT,
            "event": "ENQUEUED", "time_s": _MAX_FLOAT,
            "queue_depth": N3_QUEUE_DEPTH, "backpressure": False,
        }),
        _append(["telemetry_events"], _telemetry(
            ordinal, event="ENQUEUE", producer_blocked=False)),
        _append(["telemetry_events"], _telemetry(
            ordinal, event="PRODUCER_BACKPRESSURE_END")),
        _set(["records", ordinal, "state"], "HYDRO_QUEUED", previous="POST_COMMITTED"),
        _append(["records", ordinal, "transitions"], _transition(
            ordinal, "HYDRO_QUEUED", extra={
                "source_file_sha256": _HASH, "enqueue_s": _MAX_FLOAT,
                "queue_depth": N3_QUEUE_DEPTH, "backpressure": False,
            })),
    ]
    begin_hydro = [
        _set(["records", ordinal, "hydro_attempt_started"], True, previous=False),
        _set(["records", ordinal, "state"], "HYDRO_RUNNING", previous="HYDRO_RUNNING"),
        _append(["records", ordinal, "transitions"], _transition(
            ordinal, "HYDRO_RUNNING", extra={
                "source_file_sha256": _HASH, "hydro_attempt_started": True,
            })),
    ]
    commit_next = [
        _set(["records", ordinal, "next"], next_entry, existed=False),
        _set(["records", ordinal, "hydro_attempt_started"], False, previous=True),
        _set(["records", ordinal, "hydro_owner"], None, previous={
            "actor": _ACTOR, "process": _process(), "claim_token": "C" * 32,
        }),
        _set(["records", ordinal, "state"], "NEXT_COMMITTED", previous="HYDRO_RUNNING"),
        _append(["records", ordinal, "transitions"], _transition(
            ordinal, "NEXT_COMMITTED", extra={
                "source_file_sha256": _HASH, "output_file_sha256": _HASH,
            })),
    ]
    claim_block: list[dict[str, Any]] = []
    for index in range(N3_BLOCK_SIZE):
        block_ordinal = N3_K - N3_BLOCK_SIZE + index
        owner = {"actor": _ACTOR, "process": _process(), "claim_token": "C" * 32}
        claim_block.extend((
            _set(["records", block_ordinal, "hydro_attempt_started"], False, previous=False),
            _set(["records", block_ordinal, "hydro_owner"], owner, previous=None),
            _set(["records", block_ordinal, "state"], "HYDRO_RUNNING", previous="HYDRO_QUEUED"),
            _append(["records", block_ordinal, "transitions"], _transition(
                block_ordinal, "HYDRO_RUNNING", extra={"source_file_sha256": _HASH})),
        ))
    claim_block.extend((
        _set(["queue"], list(range(N3_BLOCK_SIZE)), previous=old_queue),
        _append(["telemetry_events"], _telemetry(
            event="HYDRO_CLAIM", block=range(N3_K - N3_BLOCK_SIZE, N3_K))),
    ))
    backpressure_begin = [
        _append(["rate_events"], {
            "ordinal": ordinal, "screen_id": _screen_id(ordinal), "z_m": _NEG_MAX_FLOAT,
            "event": "BACKPRESSURE", "time_s": _MAX_FLOAT,
            "queue_depth": N3_QUEUE_DEPTH, "backpressure": True,
        }),
        _append(["telemetry_events"], _telemetry(
            ordinal, event="PRODUCER_BACKPRESSURE_BEGIN", producer_blocked=True)),
    ]
    return {
        "optical_stage": optical_stage,
        "commit_post": commit_post,
        "enqueue_post": enqueue_post,
        "begin_hydro_screen": begin_hydro,
        "commit_next": commit_next,
        "claim_block": claim_block,
        "hydro_screen_start_end": [_append(["telemetry_events"], _telemetry(
            ordinal, event="HYDRO_SCREEN_START", block=range(N3_K - N3_BLOCK_SIZE, N3_K)))],
        "hydro_block_start_end": [_append(["telemetry_events"], _telemetry(
            event="HYDRO_BLOCK_START", block=range(N3_K - N3_BLOCK_SIZE, N3_K)))],
        "optical_start_end": [_append(["telemetry_events"], _telemetry(
            event="OPTICAL_COMPLETE", final_pulse=False))],
        "barrier_start": [_append(["telemetry_events"], _telemetry(event="BARRIER_START"))],
        "barrier_pass": [_append(["telemetry_events"], _telemetry(event="BARRIER_START"))],
        "backpressure_begin": backpressure_begin,
    }

def _barrier_template() -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = [
        _set(["barrier"], {
            "schema": N3_STREAMING_SCHEMA,
            "status": "PASS",
            "failures": [],
            "expected_screen_count": N3_K,
            "validated_utc": "9" * 20,
            "current_generation": _GENERATION,
            "next_generation": _NEXT_GENERATION,
            "current_content_sha256": _HASH,
        }, previous=None),
    ]
    for ordinal in range(N3_K):
        operations.extend((
            _set(["records", ordinal, "state"], "BARRIER_VALIDATED", previous="NEXT_COMMITTED"),
            _append(["records", ordinal, "transitions"], _transition(
                ordinal, "BARRIER_VALIDATED", extra={
                    "source_file_sha256": _HASH, "output_file_sha256": _HASH,
                })),
        ))
    return operations


def _promotion_pointer() -> dict[str, Any]:
    return {
        "schema": N3_STREAMING_SCHEMA,
        "authoritative_namespace": "NEXT",
        "source_current_generation": _GENERATION,
        "authoritative_generation": _NEXT_GENERATION,
        "barrier": {
            "schema": N3_STREAMING_SCHEMA,
            "status": "PASS",
            "failures": [],
            "expected_screen_count": N3_K,
            "validated_utc": "9" * 20,
            "current_generation": _GENERATION,
            "next_generation": _NEXT_GENERATION,
            "current_content_sha256": _HASH,
        },
        "promoted_utc": "9" * 20,
        "actor": _ACTOR,
    }


def _checkpoint_bytes() -> int:
    records: list[dict[str, Any]] = []
    current_records: list[dict[str, Any]] = []
    for ordinal in range(N3_K):
        screen_id = _screen_id(ordinal)
        current = {
            "ordinal": ordinal,
            "screen_id": screen_id,
            "z_m": _NEG_MAX_FLOAT,
            "artifact": f"current/screen_{ordinal:06d}.npz",
            "file_sha256": _HASH,
            "field_sha256": _field_hashes(),
        }
        current_records.append(current)
        records.append({
            "ordinal": ordinal,
            "screen_id": screen_id,
            "z_m": _NEG_MAX_FLOAT,
            "state": "CURRENT_READY",
            "retry_count": 1,
            "hydro_attempt_started": False,
            "hydro_owner": None,
            "current": current,
            "post": None,
            "next": None,
            "transitions": [_transition(ordinal, "CURRENT_READY", extra={
                "output_file_sha256": _HASH,
            })],
        })
    current_hash = hashlib.sha256(_canonical(current_records)).hexdigest()
    for record in records:
        for item in record["transitions"]:
            item["current_content_sha256"] = current_hash
    manifest = {
        "schema": N3_STREAMING_SCHEMA,
        "current_generation": _GENERATION,
        "current_content_sha256": current_hash,
        "next_generation": _NEXT_GENERATION,
        "shape": [351, 301],
        "dtype": "float64",
        "dx_m": _NEG_MAX_FLOAT,
        "dy_m": _NEG_MAX_FLOAT,
        "expected_screen_count": N3_K,
        "queue_depth": N3_QUEUE_DEPTH,
        "block_size": N3_BLOCK_SIZE,
        "queue": [],
        "recovery_active": False,
        "recovery_backlog": [],
        "recovery_backlog_cursor": 0,
        "recovery_attempts": [],
        "records": records,
        "hydro_worker": {
            "qualification": "HR-4E-5P/P5",
            "screen_solver": "advance_hr4_single_screen",
            "block_size": N3_BLOCK_SIZE,
        },
        "barrier": None,
        "promotion": None,
        "rate_events": [],
        "telemetry_events": [],
        "created_utc": "9" * 20,
        "s5_recovery_events": [],
        "persistence_format": "journal_v2",
    }
    return len(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")

def _head_bytes(*, sequence: int, offset: int) -> int:
    head = {
        "schema": HEAD_SCHEMA,
        "lineage_id": "L" * 32,
        "checkpoint_sha256": _HASH,
        "sequence": sequence,
        "committed_offset": offset,
        "transaction_sha256": _HASH,
    }
    return len(json.dumps(head, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def _exact_descriptor(*, side: str, namespace: str, ordinal: int,
                      campaign_id: str) -> dict[str, Any]:
    member = "M" * MAX_EXACT_IDENTIFIER_BYTES
    path = "P" * MAX_EXACT_RELATIVE_PATH_BYTES
    return {
        "trajectory": side,
        "campaign_id": campaign_id,
        "pulse": N3_PULSES - 1,
        "attempt": 0,
        "path": path,
        "relative_path": path,
        "locator": {"member": member},
        "shape": [351, 301, 384],
        "dtype": "complex128",
        "finite": False,
        "canonical_array_hash": _HASH,
        "file_sha256": _HASH,
        "file_identity": {
            "st_dev": 2**64 - 1,
            "st_ino": 2**64 - 1,
            "st_nlink": 2**64 - 1,
            "size": 2**64 - 1,
            "mtime_ns": 2**64 - 1,
        },
        "creation_record": {
            "intent_id": "I" * MAX_EXACT_IDENTIFIER_BYTES,
            "generation": "G" * MAX_EXACT_IDENTIFIER_BYTES,
            "created_utc": "9" * 20,
        },
        "role": "R" * MAX_EXACT_IDENTIFIER_BYTES,
        "namespace": "N" * MAX_EXACT_IDENTIFIER_BYTES,
        "source_index": ordinal,
        "name": "N" * MAX_EXACT_IDENTIFIER_BYTES,
    }


def _exact_report_encoded_upper_bound(campaign_id: str) -> tuple[int, int, int]:
    row_count = N3_EXACT_FIELD_OBJECT_PAIRS + N3_EXACT_METADATA_KEYS
    row = {
        "status": "PASS",
        "comparison_key": "K" * MAX_EXACT_IDENTIFIER_BYTES,
        "reference": _exact_descriptor(
            side="R", namespace="next", ordinal=N3_K - 1, campaign_id=campaign_id),
        "candidate": _exact_descriptor(
            side="C", namespace="next", ordinal=N3_K - 1, campaign_id=campaign_id),
        "reference_sha256_array": _HASH,
        "candidate_sha256_array": _HASH,
    }
    payload = {
        "schema": "S" * MAX_EXACT_IDENTIFIER_BYTES,
        "status": "PASS",
        "admission_identity_sha256": _HASH,
        "campaign_id": campaign_id,
        "pulse": N3_PULSES - 1,
        "attempt": 0,
        "screen_count": 999999,
        "ledger_count": 999999,
        "optical_count": 999999,
        "expected_object_count": row_count,
        "compared_object_count": row_count,
        "mismatch_count": 0,
        "rows": [row],
    }
    single = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    marker = '"rows": [\n'
    marker_at = single.index(marker) + len(marker)
    row_end = single.index("\n  ]", marker_at)
    encoded_row_at_depth = single[marker_at:row_end].encode("utf-8")
    encoded_total = len(single.encode("utf-8")) + (row_count - 1) * (
        len(encoded_row_at_depth) + 2)
    if len(encoded_row_at_depth) > 8192:
        raise ValueError("formal exact descriptor row exceeds its production 8192-byte reservation")
    header_bytes = len(single.encode("utf-8")) - len(encoded_row_at_depth)
    # All three retained reports, their newline, and per-report headers.
    encoded_total += 2 * header_bytes + 3
    return encoded_total, len(encoded_row_at_depth), row_count


def build_n3_journal_capacity_bound(*, campaign_root: str | Path,
                                    campaign_id: str) -> dict[str, Any]:
    # Build N3/K8048 bounds from production encoders and per-call-class counts.
    root = _absolute_path(campaign_root, label="campaign_root")
    _utf8_text(campaign_id, label="campaign_id", maximum=MAX_CAMPAIGN_ID_BYTES,
               pattern=_SAFE_ID)
    templates = _ordinary_operation_templates()
    barrier_operations = _barrier_template()
    promotion_pointer = _promotion_pointer()
    promotion_operations = [
        _set(["promotion"], promotion_pointer, previous=None),
        _append(["telemetry_events"], _telemetry(event="PROMOTION")),
    ]
    operation_names = {
        "optical_stage": "deposition_finalized",
        "commit_post": "commit_post",
        "enqueue_post": "enqueue_post",
        "begin_hydro_screen": "begin_hydro_screen",
        "commit_next": "commit_next",
        "claim_block": "claim_block",
        "hydro_screen_start_end": "record_telemetry",
        "hydro_block_start_end": "record_telemetry",
        "optical_start_end": "record_telemetry",
        "barrier_start": "record_telemetry",
        "barrier_pass": "record_telemetry",
        "backpressure_begin": "enqueue_post",
        "barrier_transition": "validate_barrier",
        "promotion": "promote_next_to_current",
    }
    template_operations = {**templates, "barrier_transition": barrier_operations,
                           "promotion": promotion_operations}
    transaction_sizes = {
        name: _journal_line_bytes(
            sequence=N3_BACKPRESSURE_TRANSACTIONS,
            operation=operation_names[name],
            operations=operations,
        )
        for name, operations in template_operations.items()
    }
    for name, size in transaction_sizes.items():
        if size > MAX_TRANSACTION_BYTES:
            raise ValueError(f"formal {name} transaction exceeds the journal size limit")

    k_per_block = N3_K // N3_BLOCK_SIZE
    common_nonfinal = {
        "optical_stage": N3_K,
        "commit_post": N3_K,
        "enqueue_post": N3_K,
        "begin_hydro_screen": N3_K,
        "commit_next": N3_K,
        "claim_block": k_per_block,
        "optical_start_end": 2,
        "barrier_start": 1,
        "barrier_transition": 1,
        "barrier_pass": 1,
    }
    root_classes: dict[str, dict[str, int]] = {}
    transaction_base_per_root: dict[str, int] = {}
    transaction_stress_per_root: dict[str, int] = {}
    journal_by_root: dict[str, int] = {}
    for name, is_candidate, is_nonfinal in (
        ("R_p0", False, True), ("C_p0", True, True),
        ("R_p1", False, True), ("C_p1", True, True),
        ("R_p2", False, False), ("C_p2", True, False),
    ):
        counts: dict[str, int] = {}
        if is_nonfinal:
            counts.update(common_nonfinal)
            if is_candidate:
                counts["hydro_screen_start_end"] = 2 * N3_K
                counts["hydro_block_start_end"] = 2 * k_per_block
            counts["promotion"] = 1
        else:
            counts["optical_stage"] = N3_K
            counts["commit_post"] = N3_K
            counts["optical_start_end"] = 2
        base_count = sum(counts.values())
        expected_base = (41252 if is_nonfinal and not is_candidate else
                         59360 if is_nonfinal else 16098)
        if base_count != expected_base:
            raise AssertionError(f"formal N3 transaction classes do not close for {name}")
        stress_count = base_count
        root_bytes = sum(count * transaction_sizes[class_name]
                         for class_name, count in counts.items())
        if is_candidate and is_nonfinal:
            counts["backpressure_begin"] = N3_K
            stress_count += N3_K
            root_bytes += N3_K * transaction_sizes["backpressure_begin"]
        root_classes[name] = counts
        transaction_base_per_root[name] = base_count
        transaction_stress_per_root[name] = stress_count
        journal_by_root[name] = root_bytes

    base_total = sum(transaction_base_per_root.values())
    stress_total = sum(transaction_stress_per_root.values())
    if base_total != N3_BASE_TRANSACTIONS or stress_total != N3_BACKPRESSURE_TRANSACTIONS:
        raise AssertionError("fixed N3 transaction counts do not close")
    class_totals: dict[str, int] = {}
    for counts in root_classes.values():
        for class_name, count in counts.items():
            class_totals[class_name] = class_totals.get(class_name, 0) + count
    expected_class_totals = {
        "optical_stage": 48288,
        "commit_post": 48288,
        "enqueue_post": 32192,
        "begin_hydro_screen": 32192,
        "commit_next": 32192,
        "claim_block": 4024,
        "hydro_screen_start_end": 32192,
        "hydro_block_start_end": 4024,
        "optical_start_end": 12,
        "barrier_start": 4,
        "barrier_transition": 4,
        "barrier_pass": 4,
        "promotion": 4,
        "backpressure_begin": N3_BACKPRESSURE_EXTRA,
    }
    if class_totals != expected_class_totals:
        raise AssertionError("N3 per-class transaction totals differ from the audited call-path counts")

    checkpoint_each = _checkpoint_bytes()
    checkpoint_total = N3_ROOT_COUNT * checkpoint_each
    checkpoint_temporary_peak = N3_ROOT_COUNT * checkpoint_each
    journal_total = sum(journal_by_root.values())
    head_each = max(_head_bytes(sequence=transaction_stress_per_root[name],
                                offset=journal_by_root[name])
                    for name in transaction_stress_per_root)
    head_persisted = N3_ROOT_COUNT * head_each
    head_temporary_peak = N3_ROOT_COUNT * head_each
    pointer_bytes = len(json.dumps(promotion_pointer, sort_keys=True, indent=2,
                                   allow_nan=False).encode("utf-8") + b"\n")
    pointer_persisted = 4 * pointer_bytes
    pointer_temporary_peak = 2 * pointer_bytes

    exact_report_bytes, exact_row_bytes, exact_row_count = _exact_report_encoded_upper_bound(campaign_id)
    terminal_reference_bytes = 2*(TERMINAL_REFERENCE_RESERVATION_BYTES + 4*1024**2)
    replay_reference_bytes = 0
    disk_objects = {
        "n3_checkpoints_persisted": checkpoint_total,
        "n3_checkpoint_publish_temporaries_peak": checkpoint_temporary_peak,
        "n3_cumulative_journal": journal_total,
        "n3_heads_persisted": head_persisted,
        "n3_head_publish_temporaries_peak": head_temporary_peak,
        "n3_promotion_pointers_persisted": pointer_persisted,
        "n3_promotion_publish_temporaries_peak": pointer_temporary_peak,
        "n3_exact_reports_persisted_upper_bound": exact_report_bytes,
        "n3_exact_report_publish_temporary_peak": (15*N3_K+10)*(exact_row_bytes+2)+65536,
        # Existing commit enforces 64 MiB before append. On natural failure the
        # entry stops; reserve one maximum terminal transaction per root.
        "n3_natural_failure_terminal_transactions": N3_ROOT_COUNT*MAX_TRANSACTION_BYTES,
        "n3_replay_reference_disk_objects": replay_reference_bytes,
        "n3_terminal_references_reserved_upper_bound": terminal_reference_bytes,
    }
    coexisting_peak = sum(disk_objects.values())
    return {
        "schema": "khz_filament.hr4e5.n3_journal_capacity.v1",
        "campaign_root": root,
        "campaign_id": campaign_id,
        "n_pulses": N3_PULSES,
        "k": N3_K,
        "lifecycle_roots": N3_ROOT_COUNT,
        "transaction_counts": {
            "base": base_total,
            "backpressure_max": stress_total,
            "backpressure_extra": N3_BACKPRESSURE_EXTRA,
            "per_root_base": transaction_base_per_root,
            "per_root_backpressure_max": transaction_stress_per_root,
            "per_root_classes": root_classes,
            "per_class_total": class_totals,
        },
        "encoded_transaction_upper_bounds": {
            name: {
                "bytes": size,
                "operation": operation_names[name],
                "encoder": "hr4e5_streaming_journal._canonical",
            }
            for name, size in transaction_sizes.items()
        } | {"max_transaction_bytes": MAX_TRANSACTION_BYTES},
        "journal_bytes_by_root": journal_by_root,
        "journal_bytes_total": journal_total,
        "checkpoint": {
            "bytes_each": checkpoint_each,
            "count": N3_ROOT_COUNT,
            "persisted_bytes": checkpoint_total,
            "publication_temporary_peak_bytes": checkpoint_temporary_peak,
            "encoder": "hr4e5s_streaming._atomic_json(indent=2, sort_keys=True, newline)",
        },
        "head": {
            "bytes_each_max": head_each,
            "persisted_bytes": head_persisted,
            "atomic_temporary_peak_bytes": head_temporary_peak,
            "encoder": "hr4e5_streaming_journal._atomic_head(indent=2, sort_keys=True, newline)",
        },
        "promotion_pointer": {
            "bytes_each": pointer_bytes,
            "persisted_count": 4,
            "persisted_bytes": pointer_persisted,
            "atomic_temporary_peak_bytes": pointer_temporary_peak,
            "encoder": "hr4e5s_streaming._atomic_json(indent=2, sort_keys=True, newline)",
        },
        "reference_views": {
            "exact_field_object_pairs": N3_EXACT_FIELD_OBJECT_PAIRS,
            "exact_metadata_keys": N3_EXACT_METADATA_KEYS,
            "exact_report_row_count": exact_row_count,
            "exact_row_encoded_upper_bound_bytes": exact_row_bytes,
            "exact_report_encoded_upper_bound_bytes": exact_report_bytes,
            "exact_reports_peak_bytes": exact_report_bytes,
            "exact_encoder": "hr4e5_evidence.atomic_json(json.dump(indent=2, sort_keys=True))",
            "exact_row_reservation_bytes": 8192,
            "exact_row_bound_within_reservation": True,
            "replay_reference_bytes": replay_reference_bytes,
            "replay_reuses_exact_descriptors": True,
            "terminal_reference_reservation_bytes": terminal_reference_bytes,
            "terminal_intent_enforces_reservation": True,
        },
        "disk_objects": disk_objects,
        "coexisting_peak_bytes": coexisting_peak,
        "additional_capacity_charge_bytes": coexisting_peak,
        "existing_metadata_allowance_bytes_deducted": 0,
        "policy": {
            "max_campaign_live_bytes": 300 * 1024**3,
            "final_output_budget_bytes": 64 * 1024**3,
            "safety_margin_bytes": 8 * 1024**3,
            "existing_metadata_allowance_deducted_again": False,
        },
    }


__all__ = [
    "N3_BACKPRESSURE_TRANSACTIONS",
    "N3_BASE_TRANSACTIONS",
    "N3_EXACT_FIELD_OBJECT_PAIRS",
    "N3_EXACT_METADATA_KEYS",
    "N3_K",
    "N3_PULSES",
    "build_n3_journal_capacity_bound",
    "validate_n3_journal_entry_strings",
]
