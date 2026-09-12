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

from .hr4e5s_streaming import FIELDS, StreamingLifecycle
from .hr4e_timestep import json_safe, sha256_array, sha256_file


S5_SCHEMA = "khz_filament.hr4e5s.s5.v1"
DEPOSITION_FIELDS = ("ion", "ib", "raman")


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


def inspect_lifecycle(*, lifecycle_root: str | Path, out_path: str | Path) -> dict[str, Any]:
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
        "barrier": _semantic_barrier(lifecycle.manifest.get("barrier")),
        "promotion": _semantic_promotion(lifecycle.manifest.get("promotion")),
        "pointer_present": (root / "authoritative_generation.json").is_file(),
        "temporary_artifacts": sorted(path.relative_to(root).as_posix() for path in root.rglob("*.tmp")),
        "unreferenced_final_artifacts": _artifact_inventory(lifecycle),
        "fault_provenance": [None if not path.is_file() else _read_json(path) for path in fault_paths],
        "normalized_lifecycle": normalized_lifecycle(lifecycle),
    }
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
                            out_dir: str | Path) -> dict[str, Any]:
    """Compare a recovered S5 stream to its fault-off streaming reference exactly."""
    reference = StreamingLifecycle.open(reference_lifecycle_root)
    candidate = StreamingLifecycle.open(candidate_lifecycle_root)
    ref_optical, cand_optical, destination = map(Path, (reference_optical_dir, candidate_optical_dir, out_dir))
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
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
    ref_semantic, cand_semantic = normalized_lifecycle(reference), normalized_lifecycle(candidate)
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
        "normalized_manifest_exact": manifest_exact,
        "normalized_reference_manifest": ref_semantic,
        "normalized_candidate_manifest": cand_semantic,
        "completion_map_exact": completion_reference == completion_candidate,
        "ownership_exact": ownership_pass,
        "barrier_exact": ref_semantic["barrier"] == cand_semantic["barrier"] and ref_semantic["barrier"] is not None and ref_semantic["barrier"].get("status") == "PASS",
        "promotion_generation_exact": ownership_pass,
        "final_artifact_inventory_clean": final_clean,
    }
    result["status"] = "PASS" if (
        not field_failures and all(optical[key] for key in ("shape_equal", "dtype_equal", "hash_equal", "array_equal"))
        and ledger_rows and all(row["present_both"] and row["shape_equal"] and row["dtype_equal"] and row["hash_equal"] and row["array_equal"] for row in ledger_rows)
        and result["normalized_manifest_exact"] and result["completion_map_exact"] and result["ownership_exact"]
        and result["barrier_exact"] and result["promotion_generation_exact"] and result["final_artifact_inventory_clean"]
    ) else "FAIL"
    _atomic_json(destination / "s5_1_exact_comparison.json", result)
    return result


__all__ = ["S5_SCHEMA", "compare_clean_reference", "inspect_lifecycle", "normalized_lifecycle"]
