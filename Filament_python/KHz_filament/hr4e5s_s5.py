"""Read-only S5 lifecycle evidence helpers.

This module deliberately contains no optical, deposition, or HR-4 numerical
operator.  It compares two completed Streaming lifecycle roots using their
canonical field hashes and exact arrays, and snapshots an interrupted root
without attempting recovery.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .hr4e5s_streaming import FIELDS, RECOVERY_ATTEMPT_SCHEMA, S5_FAULT_SPECS, StreamingLifecycle
from .hr4e_timestep import json_safe, sha256_array, sha256_file


S5_SCHEMA = "khz_filament.hr4e5s.s5.v1"
S5_CONTRACT_SCHEMA = "khz_filament.hr4e5s.s5.contract_check.v1"
DEPOSITION_FIELDS = ("ion", "ib", "raman")


_S5_CONTRACT_TARGETS = {
    "F01_OPTICAL_PRE_POST_COMMIT": {"state": "DEPOSITION_FINALIZED", "post": False, "next": False, "barrier": "NOT_PASS"},
    "F02_POST_COMMITTED_PRE_HYDRO": {"state": "POST_COMMITTED", "post": True, "next": False, "barrier": "NOT_PASS"},
    "F03_HYDRO_PRE_NEXT_COMMIT": {"state": "HYDRO_RUNNING", "post": True, "next": False, "barrier": "NOT_PASS"},
    "F04_NEXT_TEMP_PRE_ATOMIC_RENAME": {"state": "HYDRO_RUNNING", "post": True, "next": False, "barrier": "NOT_PASS"},
    "F05_NEXT_COMMITTED_PRE_BARRIER": {"state": "NEXT_COMMITTED", "post": True, "next": True, "barrier": "NOT_PASS"},
    "F06_BARRIER_PASS_PRE_PROMOTION": {"state": "BARRIER_VALIDATED", "post": True, "next": True, "barrier": "PASS"},
}

_S5_RECOVERY_RETRY_DELTAS = {
    "F01_OPTICAL_PRE_POST_COMMIT": 0,
    "F02_POST_COMMITTED_PRE_HYDRO": 0,
    "F03_HYDRO_PRE_NEXT_COMMIT": 1,
    "F04_NEXT_TEMP_PRE_ATOMIC_RENAME": 1,
    "F05_NEXT_COMMITTED_PRE_BARRIER": 0,
    "F06_BARRIER_PASS_PRE_PROMOTION": 0,
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
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


def _artifact_inventory(lifecycle: StreamingLifecycle) -> dict[str, list[str]]:
    expected = {
        namespace: {
            Path(str(record[namespace.lower()]["artifact"])).as_posix()
            for record in lifecycle.manifest["records"]
            if namespace == "CURRENT" or record[namespace.lower()] is not None
        }
        for namespace in ("CURRENT", "POST", "NEXT")
    }
    inventory: dict[str, list[str]] = {}
    for namespace in ("current", "post", "next"):
        actual = sorted(path.relative_to(lifecycle.root).as_posix() for path in (lifecycle.root / namespace).glob("*.npz"))
        inventory[namespace] = sorted(set(actual) - expected[namespace.upper()])
    return inventory


def _semantic_barrier(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    return {
        "status": value.get("status"),
        "failures": list(value.get("failures", [])),
        "expected_screen_count": value.get("expected_screen_count"),
        "current_generation": value.get("current_generation"),
        "next_generation": value.get("next_generation"),
        "current_content_sha256": value.get("current_content_sha256"),
    }


def _semantic_promotion(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    return {
        "schema": value.get("schema"),
        "authoritative_namespace": value.get("authoritative_namespace"),
        "source_current_generation": value.get("source_current_generation"),
        "authoritative_generation": value.get("authoritative_generation"),
        "barrier": _semantic_barrier(value.get("barrier")),
    }


def normalized_lifecycle(lifecycle: StreamingLifecycle) -> dict[str, Any]:
    """Return the final durable semantics, excluding only named runtime history."""
    manifest = lifecycle.manifest
    records = []
    for record in manifest["records"]:
        records.append({
            "ordinal": int(record["ordinal"]),
            "screen_id": str(record["screen_id"]),
            "z_m": float(record["z_m"]),
            "state": str(record["state"]),
            "retry_count": int(record["retry_count"]),
            "current": dict(record["current"]["field_sha256"]),
            "post": None if record["post"] is None else dict(record["post"]["field_sha256"]),
            "next": None if record["next"] is None else dict(record["next"]["field_sha256"]),
        })
    pointer_path = lifecycle.root / "authoritative_generation.json"
    pointer = None if not pointer_path.is_file() else _semantic_promotion(_read_json(pointer_path))
    return {
        "schema": manifest["schema"],
        "current_generation": manifest["current_generation"],
        "next_generation": manifest["next_generation"],
        "current_content_sha256": manifest["current_content_sha256"],
        "shape": list(manifest["shape"]),
        "dtype": manifest["dtype"],
        "dx_m": float(manifest["dx_m"]),
        "dy_m": float(manifest["dy_m"]),
        "expected_screen_count": int(manifest["expected_screen_count"]),
        "queue_depth": int(manifest["queue_depth"]),
        "block_size": int(manifest["block_size"]),
        "queue": [int(value) for value in manifest["queue"]],
        "recovery_backlog": [int(value) for value in manifest.get("recovery_backlog", [])],
        "records": records,
        "barrier": _semantic_barrier(manifest.get("barrier")),
        "promotion": _semantic_promotion(manifest.get("promotion")),
        "authoritative_pointer": pointer,
        "excluded_runtime_fields": [
            "created_utc", "transitions timestamp_utc/actor/monotonic values",
            "rate_events", "telemetry_events", "s5_recovery_events",
            "fault provenance files", "Slurm job id", "process id",
        ],
    }


def _authoritative_lifecycle(lifecycle: StreamingLifecycle) -> dict[str, Any]:
    """Return final state without recovery-history metadata.

    This is intentionally narrower than ``normalized_lifecycle`` only after
    ``validate_recovery_provenance`` has made retry history an independently
    strict contract. Array hashes, record identity/state, queue/backlog,
    barrier, promotion, and the authoritative pointer remain exact.
    """
    value = normalized_lifecycle(lifecycle)
    for record in value["records"]:
        record.pop("retry_count")
    value.pop("excluded_runtime_fields")
    return value


def inspect_lifecycle(*, lifecycle_root: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    """Persist a read-only interrupted-state snapshot before S5 recovery."""
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    root = lifecycle.root
    records = []
    for record in lifecycle.manifest["records"]:
        records.append({
            "ordinal": int(record["ordinal"]), "screen_id": str(record["screen_id"]), "state": str(record["state"]),
            "post_authoritative": record["post"] is not None,
            "next_authoritative": record["next"] is not None,
            "post_artifact": None if record["post"] is None else str(record["post"]["artifact"]),
            "next_artifact": None if record["next"] is None else str(record["next"]["artifact"]),
        })
    fault_paths = (root / "s5_fault_provenance.json", root / ".s5_fault_consumed.json")
    result = {
        "schema": S5_SCHEMA,
        "lifecycle_root": str(root),
        "manifest_sha256": sha256_file(lifecycle.manifest_path),
        "records": records,
        "queue": [int(value) for value in lifecycle.manifest["queue"]],
        "recovery_active": bool(lifecycle.manifest.get("recovery_active", False)),
        "recovery_backlog": [int(value) for value in lifecycle.manifest.get("recovery_backlog", [])],
        "recovery_attempts": list(lifecycle.manifest.get("recovery_attempts", [])),
        "barrier": _semantic_barrier(lifecycle.manifest.get("barrier")),
        "promotion": _semantic_promotion(lifecycle.manifest.get("promotion")),
        "pointer_present": (root / "authoritative_generation.json").is_file(),
        "temporary_artifacts": sorted(path.relative_to(root).as_posix() for path in root.rglob("*.tmp")),
        "unreferenced_final_artifacts": _artifact_inventory(lifecycle),
        "fault_provenance": [None if not path.is_file() else _read_json(path) for path in fault_paths],
        "normalized_lifecycle": normalized_lifecycle(lifecycle),
    }
    if out_path is not None:
        _atomic_json(Path(out_path), result)
    return result


def _fault_event_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fields which describe one deterministic S5 fault, excluding runtime time."""
    return {
        "schema": value.get("schema"),
        "fault_id": value.get("fault_id"),
        "target_screen": value.get("target_screen"),
        "ordinal": value.get("ordinal"),
        "lifecycle_stage": value.get("lifecycle_stage"),
        "expected_authoritative_state": value.get("expected_authoritative_state"),
        "fault_once": value.get("fault_once"),
        "temporary_artifact": value.get("temporary_artifact"),
    }


def validate_fault_contract(*, lifecycle_root: str | Path, fault_id: str, target_screen: str,
                            out_path: str | Path | None = None) -> dict[str, Any]:
    """Validate one frozen S5 fault boundary without changing lifecycle state.

    This is deliberately a test-only gate.  It validates the persisted crash
    state immediately before recovery; it never calls reconstruction, writes a
    lifecycle manifest, or relaxes ownership/provenance validation.
    """
    root = Path(lifecycle_root)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **observed: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), **observed})

    result: dict[str, Any] = {
        "schema": S5_CONTRACT_SCHEMA,
        "lifecycle_root": str(root),
        "fault_id": str(fault_id),
        "target_screen": str(target_screen),
        "checks": checks,
    }
    expected = _S5_CONTRACT_TARGETS.get(str(fault_id))
    check("supported_fault_id", expected is not None)
    if expected is None:
        result.update({"contract_match": False, "status": "FAIL"})
        if out_path is not None:
            _atomic_json(Path(out_path), result)
        return result

    try:
        lifecycle = StreamingLifecycle.open(root)
    except Exception as exc:
        check("lifecycle_open_and_pointer_validation", False, error_type=type(exc).__name__)
        result.update({"contract_match": False, "status": "FAIL"})
        if out_path is not None:
            _atomic_json(Path(out_path), result)
        return result

    records = list(lifecycle.manifest["records"])
    target_records = [record for record in records if str(record["screen_id"]) == str(target_screen)]
    check("target_screen_exists_once", len(target_records) == 1, count=len(target_records))
    target = target_records[0] if len(target_records) == 1 else None
    target_ordinal = None if target is None else int(target["ordinal"])

    provenance_path = root / "s5_fault_provenance.json"
    consumed_path = root / ".s5_fault_consumed.json"
    provenance = None
    consumed = None
    try:
        provenance = _read_json(provenance_path) if provenance_path.is_file() else None
        consumed = _read_json(consumed_path) if consumed_path.is_file() else None
    except Exception as exc:
        check("fault_provenance_parse", False, error_type=type(exc).__name__)
    else:
        check("fault_provenance_exists", isinstance(provenance, Mapping))
        check("fault_consumed_marker_exists", isinstance(consumed, Mapping))
        if isinstance(provenance, Mapping) and isinstance(consumed, Mapping):
            check("provenance_consumed_semantically_consistent", _fault_event_semantics(provenance) == _fault_event_semantics(consumed))
            specification = S5_FAULT_SPECS.get(str(fault_id), {})
            required = {
                "schema": "khz_filament.hr4e5s.s5.fault_provenance.v1",
                "fault_id": str(fault_id),
                "target_screen": str(target_screen),
                "lifecycle_stage": specification.get("stage"),
                "expected_authoritative_state": specification.get("expected_authoritative_state"),
                "fault_once": True,
            }
            event = _fault_event_semantics(provenance)
            check("fault_provenance_matches_selected_fault", all(event.get(key) == value for key, value in required.items()))
            check("fault_provenance_target_ordinal_matches_record", target_ordinal is not None and event.get("ordinal") == target_ordinal)

    if target is not None:
        check("target_state", str(target["state"]) == expected["state"], observed=str(target["state"]), required=expected["state"])
        check("target_post_authority", (target["post"] is not None) == expected["post"], observed=target["post"] is not None, required=expected["post"])
        check("target_next_authority", (target["next"] is not None) == expected["next"], observed=target["next"] is not None, required=expected["next"])
        try:
            lifecycle._validate_record_provenance(target, require_post=target["post"] is not None, require_next=target["next"] is not None)
        except Exception as exc:
            check("target_authoritative_artifact_provenance", False, error_type=type(exc).__name__)
        else:
            check("target_authoritative_artifact_provenance", True)

    try:
        for record in records:
            lifecycle._validate_record_provenance(record, require_post=record["post"] is not None, require_next=record["next"] is not None)
    except Exception as exc:
        check("global_authoritative_artifact_provenance", False, error_type=type(exc).__name__)
    else:
        check("global_authoritative_artifact_provenance", True)

    for namespace in ("current", "post", "next"):
        paths = [str(record[namespace]["artifact"]) for record in records if record[namespace] is not None]
        check(f"no_duplicate_{namespace}_authoritative_identity", len(paths) == len(set(paths)), count=len(paths))

    inventory = _artifact_inventory(lifecycle)
    check("no_unexpected_unreferenced_final_artifact", not any(inventory.values()), inventory=inventory)
    temporary = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.tmp"))
    if str(fault_id) == "F04_NEXT_TEMP_PRE_ATOMIC_RENAME" and target_ordinal is not None and isinstance(provenance, Mapping):
        expected_temp = provenance.get("temporary_artifact")
        expected_temp_normalized = expected_temp.replace("\\", "/") if isinstance(expected_temp, str) else expected_temp
        valid_temp_name = (
            isinstance(expected_temp_normalized, str)
            and expected_temp_normalized.startswith(f"next/screen_{target_ordinal:06d}.npz.")
            and expected_temp_normalized.endswith(".tmp")
        )
        check("f04_expected_temp_provenance", valid_temp_name, expected_temp=expected_temp_normalized)
        check("f04_exactly_one_matching_next_temp", valid_temp_name and temporary == [expected_temp_normalized], temporary_artifacts=temporary)
    else:
        check("no_unauthorized_temporary_artifact", temporary == [], temporary_artifacts=temporary)

    barrier = lifecycle.manifest.get("barrier")
    barrier_status = barrier.get("status") if isinstance(barrier, Mapping) else None
    if expected["barrier"] == "PASS":
        check("barrier_pass", barrier_status == "PASS", observed=barrier_status)
        check("all_records_barrier_validated", all(str(record["state"]) == "BARRIER_VALIDATED" for record in records))
    else:
        check("barrier_not_pass", barrier_status != "PASS", observed=barrier_status)

    promotion = lifecycle.manifest.get("promotion")
    pointer_path = root / "authoritative_generation.json"
    check("no_premature_promotion", promotion is None)
    check("no_premature_authoritative_pointer", not pointer_path.exists())

    result["contract_match"] = all(bool(entry["pass"]) for entry in checks)
    result["status"] = "PASS" if result["contract_match"] else "FAIL"
    if out_path is not None:
        _atomic_json(Path(out_path), result)
    return result


def validate_recovery_provenance(*, reference_lifecycle_root: str | Path,
                                 candidate_lifecycle_root: str | Path,
                                 fault_id: str | None = None,
                                 target_screen: str | None = None,
                                 out_path: str | Path | None = None) -> dict[str, Any]:
    """Strictly audit the recovery-only metadata before exact comparison.

    A recovery may re-execute a stale HYDRO_RUNNING record for F03/F04. The
    resulting retry counter is not science state, but it is never ignored:
    the selected fault, target identity, count delta, input provenance, and
    absence of unrelated retries must all match this contract.
    """
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **observed: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), **observed})

    result: dict[str, Any] = {
        "schema": S5_SCHEMA,
        "reference_lifecycle_root": str(reference_lifecycle_root),
        "candidate_lifecycle_root": str(candidate_lifecycle_root),
        "fault_id": fault_id,
        "target_screen": target_screen,
        "checks": checks,
    }
    try:
        reference = StreamingLifecycle.open(reference_lifecycle_root)
        candidate = StreamingLifecycle.open(candidate_lifecycle_root)
    except Exception as exc:
        check("open_and_validate_lifecycles", False, error_type=type(exc).__name__)
        result.update({"recovery_provenance_exact": False, "status": "FAIL"})
        if out_path is not None:
            _atomic_json(Path(out_path), result)
        return result

    reference_records = list(reference.manifest["records"])
    candidate_records = list(candidate.manifest["records"])
    identities_match = [
        (int(item["ordinal"]), str(item["screen_id"]), float(item["z_m"]))
        for item in reference_records
    ] == [
        (int(item["ordinal"]), str(item["screen_id"]), float(item["z_m"]))
        for item in candidate_records
    ]
    check("reference_candidate_record_identity_exact", identities_match)
    selected_delta = 0
    target_ordinal: int | None = None
    if fault_id is None and target_screen is None:
        check("clean_reference_mode", True)
    elif fault_id is None or target_screen is None:
        check("fault_selector_is_complete", False)
    else:
        selected_delta = _S5_RECOVERY_RETRY_DELTAS.get(str(fault_id), -1)
        check("supported_fault_id", selected_delta >= 0)
        targets = [record for record in candidate_records if str(record["screen_id"]) == str(target_screen)]
        check("target_screen_exists_once", len(targets) == 1, count=len(targets))
        if len(targets) == 1:
            target_ordinal = int(targets[0]["ordinal"])

    retry_deltas: dict[int, int] = {}
    if identities_match:
        retry_deltas = {
            int(candidate_record["ordinal"]): int(candidate_record["retry_count"]) - int(reference_record["retry_count"])
            for reference_record, candidate_record in zip(reference_records, candidate_records)
        }
        expected_deltas = {
            ordinal: (selected_delta if target_ordinal is not None and ordinal == target_ordinal else 0)
            for ordinal in retry_deltas
        }
        check("retry_count_delta_exact", retry_deltas == expected_deltas,
              observed=retry_deltas, expected=expected_deltas)
    else:
        check("retry_count_delta_exact", False)

    attempts = list(candidate.manifest.get("recovery_attempts", []))
    expected_attempt_ordinals = ([] if selected_delta <= 0 or target_ordinal is None else [target_ordinal] * selected_delta)
    attempt_ordinals = [int(item.get("ordinal", -1)) for item in attempts if isinstance(item, Mapping)]
    check("recovery_attempt_ordinal_exact", attempt_ordinals == expected_attempt_ordinals,
          observed=attempt_ordinals, expected=expected_attempt_ordinals)
    attempts_valid = len(attempt_ordinals) == len(attempts)
    if attempts_valid:
        for attempt in attempts:
            ordinal = int(attempt["ordinal"])
            candidate_record = candidate_records[ordinal]
            reference_record = reference_records[ordinal]
            attempts_valid = attempts_valid and (
                attempt.get("schema") == RECOVERY_ATTEMPT_SCHEMA
                and attempt.get("screen_id") == candidate_record["screen_id"]
                and attempt.get("reason") == "STALE_HYDRO_RUNNING_RECONSTRUCTED"
                and int(attempt.get("retry_count", -1)) == int(candidate_record["retry_count"])
                and attempt.get("post_file_sha256") == candidate_record["post"]["file_sha256"]
                and dict(attempt.get("post_field_sha256", {})) == dict(candidate_record["post"]["field_sha256"])
                and candidate_record["post"]["file_sha256"] == reference_record["post"]["file_sha256"]
                and dict(candidate_record["post"]["field_sha256"]) == dict(reference_record["post"]["field_sha256"])
            )
    check("recovery_attempt_input_provenance_exact", attempts_valid)
    check("terminal_queue_and_backlog_empty", not candidate.manifest["queue"] and not candidate.manifest.get("recovery_backlog", []),
          queue=list(candidate.manifest["queue"]), backlog=list(candidate.manifest.get("recovery_backlog", [])))

    result["retry_deltas"] = retry_deltas
    result["recovery_attempts"] = attempts
    result["recovery_provenance_exact"] = all(bool(entry["pass"]) for entry in checks)
    result["status"] = "PASS" if result["recovery_provenance_exact"] else "FAIL"
    if out_path is not None:
        _atomic_json(Path(out_path), result)
    return result


def _check(rows: list[dict[str, Any]], *, layer: str, ordinal: int, field: str, reference: np.ndarray, candidate: np.ndarray) -> None:
    reference_hash, candidate_hash = sha256_array(reference), sha256_array(candidate)
    rows.append({
        "layer": layer, "ordinal": int(ordinal), "field": field,
        "shape_equal": bool(reference.shape == candidate.shape),
        "dtype_equal": bool(reference.dtype == candidate.dtype),
        "reference_sha256": reference_hash, "candidate_sha256": candidate_hash,
        "hash_equal": bool(reference_hash == candidate_hash),
        "array_equal": bool(np.array_equal(reference, candidate)),
    })


def _field(lifecycle: StreamingLifecycle, ordinal: int, namespace: str, field: str) -> np.ndarray:
    entry = lifecycle.manifest["records"][ordinal][namespace.lower()]
    if entry is None:
        raise ValueError(f"{namespace} is incomplete for screen {ordinal}")
    return lifecycle._artifact_fields(entry, namespace=namespace)[field]


def _final_optical(directory: Path) -> np.ndarray:
    run = _read_json(directory / "optical_run.json")
    return np.load(directory / str(run["final_optical_field"]), mmap_mode="r", allow_pickle=False)


def compare_clean_reference(*, reference_lifecycle_root: str | Path, reference_optical_dir: str | Path,
                            candidate_lifecycle_root: str | Path, candidate_optical_dir: str | Path,
                            out_dir: str | Path, recovery_fault_id: str | None = None,
                            recovery_target_screen: str | None = None,
                            recovery_provenance_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare a recovered S5 stream to its fault-off streaming reference exactly."""
    reference = StreamingLifecycle.open(reference_lifecycle_root)
    candidate = StreamingLifecycle.open(candidate_lifecycle_root)
    ref_optical, cand_optical, destination = map(Path, (reference_optical_dir, candidate_optical_dir, out_dir))
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    if recovery_provenance_override is None:
        recovery_provenance = validate_recovery_provenance(
            reference_lifecycle_root=reference_lifecycle_root,
            candidate_lifecycle_root=candidate_lifecycle_root,
            fault_id=recovery_fault_id,
            target_screen=recovery_target_screen,
            out_path=destination / "s5_1_recovery_provenance.json",
        )
    else:
        # S5-FINAL has a pre-frozen multi-worker retry contract.  It is
        # validated by its scenario adapter before this unchanged field,
        # ledger, ownership, barrier, promotion, and artifact comparison core
        # is entered.  F01--F06 callers retain the strict default above.
        recovery_provenance = dict(recovery_provenance_override)
    if int(reference.manifest["expected_screen_count"]) != 48 or int(candidate.manifest["expected_screen_count"]) != 48:
        raise ValueError("S5 requires the frozen 48-screen S3 window")
    rows: list[dict[str, Any]] = []
    count = int(reference.manifest["expected_screen_count"])
    if count != int(candidate.manifest["expected_screen_count"]):
        raise ValueError("reference and candidate screen counts differ")
    for ordinal in range(count):
        reference._validate_record_provenance(reference.manifest["records"][ordinal], require_post=True, require_next=True)
        candidate._validate_record_provenance(candidate.manifest["records"][ordinal], require_post=True, require_next=True)
        for field in FIELDS:
            _check(rows, layer="POST", ordinal=ordinal, field=field, reference=_field(reference, ordinal, "POST", field), candidate=_field(candidate, ordinal, "POST", field))
            _check(rows, layer="NEXT", ordinal=ordinal, field=field, reference=_field(reference, ordinal, "NEXT", field), candidate=_field(candidate, ordinal, "NEXT", field))
    for field in DEPOSITION_FIELDS:
        left = np.load(ref_optical / f"s3_optical.hr3a_q{field}_samples.npy", mmap_mode="r", allow_pickle=False)
        right = np.load(cand_optical / f"s3_optical.hr3a_q{field}_samples.npy", mmap_mode="r", allow_pickle=False)
        if left.shape[0] != count or right.shape[0] != count:
            raise ValueError("S5 deposition sample count differs from lifecycle window")
        for ordinal in range(count):
            _check(rows, layer="DEPOSITION", ordinal=ordinal, field=field, reference=left[ordinal], candidate=right[ordinal])
    with (destination / "s5_1_field_exact_comparisons.csv").open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    field_failures = [row for row in rows if not (row["shape_equal"] and row["dtype_equal"] and row["hash_equal"] and row["array_equal"])]
    left_optical, right_optical = _final_optical(ref_optical), _final_optical(cand_optical)
    optical_rows: list[dict[str, Any]] = []
    _check(optical_rows, layer="FINAL_OPTICAL", ordinal=-1, field="E", reference=left_optical, candidate=right_optical)
    optical = optical_rows[0]
    ref_run, cand_run = _read_json(ref_optical / "optical_run.json"), _read_json(cand_optical / "optical_run.json")
    ledger_rows: list[dict[str, Any]] = []
    with np.load(ref_optical / str(ref_run["ledger"]), allow_pickle=False) as left_ledger, np.load(cand_optical / str(cand_run["ledger"]), allow_pickle=False) as right_ledger:
        for field in sorted(set(left_ledger.files) | set(right_ledger.files)):
            if field not in left_ledger.files or field not in right_ledger.files:
                ledger_rows.append({"field": field, "present_both": False, "shape_equal": False, "dtype_equal": False, "hash_equal": False, "array_equal": False})
                continue
            entry_rows: list[dict[str, Any]] = []
            _check(entry_rows, layer="LEDGER", ordinal=-1, field=field, reference=np.asarray(left_ledger[field]), candidate=np.asarray(right_ledger[field]))
            ledger_rows.append({"field": field, "present_both": True, **{key: entry_rows[0][key] for key in ("shape_equal", "dtype_equal", "reference_sha256", "candidate_sha256", "hash_equal", "array_equal")}})
    ref_semantic, cand_semantic = _authoritative_lifecycle(reference), _authoritative_lifecycle(candidate)
    manifest_exact = ref_semantic == cand_semantic
    completion_reference = [(record["ordinal"], record["screen_id"], record["state"]) for record in ref_semantic["records"]]
    completion_candidate = [(record["ordinal"], record["screen_id"], record["state"]) for record in cand_semantic["records"]]
    ownership_pass = (
        ref_semantic["promotion"] is not None and cand_semantic["promotion"] is not None
        and ref_semantic["promotion"] == cand_semantic["promotion"]
        and ref_semantic["authoritative_pointer"] == cand_semantic["authoritative_pointer"]
        and isinstance(ref_semantic["authoritative_pointer"], Mapping)
        and ref_semantic["authoritative_pointer"].get("authoritative_namespace") == "NEXT"
    )
    final_clean = not ref_semantic["queue"] and not cand_semantic["queue"] and not any(_artifact_inventory(reference).values()) and not any(_artifact_inventory(candidate).values()) and not list(reference.root.rglob("*.tmp")) and not list(candidate.root.rglob("*.tmp"))
    result = {
        "schema": S5_SCHEMA,
        "expected_field_comparisons": count * 9,
        "completed_field_comparisons": len(rows),
        "mismatch_count": len(field_failures),
        "field_rows": rows,
        "final_optical": optical,
        "ledger_rows": ledger_rows,
        "authoritative_manifest_exact": manifest_exact,
        "authoritative_reference_manifest": ref_semantic,
        "authoritative_candidate_manifest": cand_semantic,
        "recovery_provenance": recovery_provenance,
        "completion_map_exact": completion_reference == completion_candidate,
        "ownership_exact": ownership_pass,
        "barrier_exact": ref_semantic["barrier"] == cand_semantic["barrier"] and ref_semantic["barrier"] is not None and ref_semantic["barrier"].get("status") == "PASS",
        "promotion_generation_exact": ownership_pass,
        "final_artifact_inventory_clean": final_clean,
    }
    result["status"] = "PASS" if (
        not field_failures and all(optical[key] for key in ("shape_equal", "dtype_equal", "hash_equal", "array_equal"))
        and ledger_rows and all(row["present_both"] and row["shape_equal"] and row["dtype_equal"] and row["hash_equal"] and row["array_equal"] for row in ledger_rows)
        and result["authoritative_manifest_exact"] and recovery_provenance["status"] == "PASS"
        and result["completion_map_exact"] and result["ownership_exact"]
        and result["barrier_exact"] and result["promotion_generation_exact"] and result["final_artifact_inventory_clean"]
    ) else "FAIL"
    _atomic_json(destination / "s5_1_exact_comparison.json", result)
    return result


__all__ = [
    "S5_CONTRACT_SCHEMA",
    "S5_SCHEMA",
    "compare_clean_reference",
    "inspect_lifecycle",
    "normalized_lifecycle",
    "validate_fault_contract",
    "validate_recovery_provenance",
]
