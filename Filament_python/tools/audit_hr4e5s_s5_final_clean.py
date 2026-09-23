#!/usr/bin/env python3
"""Read-only terminal audit for an existing S5-FINAL fault-OFF clean result.

This is a post-run tool.  It does not change lifecycle state or reinterpret a
Slurm terminal status; a historical failed job remains failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KHz_filament.hr4e_timestep import sha256_array, sha256_file  # noqa: E402
from KHz_filament.hr4e5s_s5 import inspect_lifecycle  # noqa: E402
from KHz_filament.hr4e5s_streaming import StreamingLifecycle  # noqa: E402

FIELDS = ("delta_n", "vx", "vy")
DEPOSITION_ARCHIVES = (
    "s3_optical.hr3a_qib_samples.npy",
    "s3_optical.hr3a_qion_samples.npy",
    "s3_optical.hr3a_qraman_samples.npy",
    "s3_optical.hr3a_qthermal_samples.npy",
    "s3_optical.hr3b_delta_n_increment_samples.npy",
    "s3_optical.hr3b_delta_n_state_after_update_samples.npy",
)


class CleanAuditError(ValueError):
    """One required clean-result gate failed closed."""


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanAuditError(message)


def _finite_array(path: Path, *, expected_count: int | None = None) -> np.ndarray:
    _require(path.is_file(), f"required artifact missing: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    _require(isinstance(value, np.ndarray) and value.size > 0, f"array is empty: {path}")
    if expected_count is not None:
        _require(value.ndim > 0 and value.shape[0] == expected_count, f"screen count differs: {path}")
    flat = value.reshape(-1)
    for start in range(0, flat.size, 1_000_000):
        _require(bool(np.isfinite(flat[start:start + 1_000_000]).all()), f"non-finite array: {path}")
    return value


def verify_final_optical(optical: Path) -> dict[str, Any]:
    """The JSON field is an array-content digest, not a raw NPY-file digest."""
    run = _read_json(optical / "optical_run.json")
    _require(run.get("final_optical_field") == "final_optical_field.npy", "final optical path is invalid")
    path = optical / "final_optical_field.npy"
    value = _finite_array(path)
    _require(value.flags.c_contiguous, "final optical array is not C contiguous")
    canonical = sha256_array(value)
    _require(canonical == run.get("final_optical_field_sha256"), "final optical canonical array hash differs")
    return {"canonical_array_sha256": canonical, "raw_npy_sha256": sha256_file(path),
            "shape": list(value.shape), "dtype": str(value.dtype)}


def _verify_record_artifact(root: Path, manifest: dict[str, Any], record: dict[str, Any], namespace: str) -> None:
    entry = record.get(namespace.lower())
    _require(isinstance(entry, dict), f"{namespace} artifact entry is missing for screen {record['ordinal']}")
    ordinal = int(record["ordinal"])
    relative = f"{namespace.lower()}/screen_{ordinal:06d}.npz"
    _require(str(entry.get("artifact", "")).replace("\\", "/") == relative,
             f"{namespace} artifact path is invalid for screen {ordinal}")
    path = root / relative
    _require(path.is_file(), f"{namespace} artifact is missing for screen {ordinal}")
    _require(sha256_file(path) == entry.get("file_sha256"), f"{namespace} file hash differs for screen {ordinal}")
    with np.load(path, allow_pickle=False) as stored:
        _require(set(stored.files) == {*FIELDS, "metadata_json"}, f"{namespace} fields are invalid for screen {ordinal}")
        metadata = json.loads(str(stored["metadata_json"].item()))
        hashes = {}
        for field in FIELDS:
            value = np.asarray(stored[field])
            _require(value.shape == tuple(manifest["shape"]) and value.dtype == np.float64,
                     f"{namespace} layout is invalid for screen {ordinal}")
            _require(bool(np.isfinite(value).all()), f"{namespace} has non-finite values for screen {ordinal}")
            hashes[field] = sha256_array(value)
    _require(hashes == entry.get("field_sha256") == metadata.get("field_sha256"),
             f"{namespace} canonical hashes differ for screen {ordinal}")
    _require(metadata.get("schema") == manifest["schema"] and metadata.get("namespace") == namespace
             and metadata.get("ordinal") == ordinal and metadata.get("screen_id") == record["screen_id"]
             and metadata.get("z_m") == record["z_m"] and metadata.get("shape") == manifest["shape"]
             and metadata.get("dtype") == manifest["dtype"], f"{namespace} metadata differs for screen {ordinal}")
    if namespace == "CURRENT":
        _require(metadata.get("generation") == manifest["current_generation"], "CURRENT generation differs")
    elif namespace == "POST":
        _require(metadata.get("current_generation") == manifest["current_generation"]
                 and metadata.get("current_content_sha256") == manifest["current_content_sha256"]
                 and metadata.get("current_field_sha256") == record["current"]["field_sha256"]
                 and metadata.get("hr3a_authoritative") is True and metadata.get("hr3b_authoritative") is True,
                 f"POST provenance differs for screen {ordinal}")
    else:
        _require(metadata.get("current_generation") == manifest["current_generation"]
                 and metadata.get("current_content_sha256") == manifest["current_content_sha256"]
                 and metadata.get("next_generation") == manifest["next_generation"]
                 and metadata.get("post_file_sha256") == record["post"]["file_sha256"]
                 and metadata.get("post_field_sha256") == record["post"]["field_sha256"],
                 f"NEXT provenance differs for screen {ordinal}")


def audit_clean_reference(clean_root: str | Path, *, expected_screens: int = 48) -> dict[str, Any]:
    """Verify one already-created clean result without writing its source tree."""
    clean = Path(clean_root)
    lifecycle_root = clean / "lifecycle"
    manifest_path = lifecycle_root / "streaming_manifest.json"
    manifest_sha_before = sha256_file(manifest_path)
    lifecycle = StreamingLifecycle.open(lifecycle_root)
    manifest = lifecycle.manifest
    records = manifest["records"]
    _require(len(records) == expected_screens and manifest["expected_screen_count"] == expected_screens,
             "clean lifecycle screen count differs")
    _require([record["ordinal"] for record in records] == list(range(expected_screens))
             and all(record["state"] == "BARRIER_VALIDATED" for record in records),
             "clean lifecycle completion map is invalid")
    _require(not manifest["queue"] and not manifest["recovery_backlog"]
             and not manifest.get("recovery_active", False), "clean lifecycle queue or backlog remains")
    _require(not manifest.get("recovery_attempts") and all(record.get("retry_count") == 0 for record in records),
             "clean reference contains retry history")
    barrier, promotion = manifest.get("barrier"), manifest.get("promotion")
    _require(isinstance(barrier, dict) and barrier.get("status") == "PASS" and not barrier.get("failures"),
             "clean barrier is not PASS")
    _require(isinstance(promotion, dict) and promotion.get("authoritative_namespace") == "NEXT"
             and promotion.get("authoritative_generation") == manifest["next_generation"],
             "clean promotion is not authoritative NEXT")
    inventory = inspect_lifecycle(lifecycle_root=lifecycle_root)
    pointer = inventory["normalized_lifecycle"].get("authoritative_pointer")
    _require(isinstance(pointer, dict) and pointer == inventory["normalized_lifecycle"]["promotion"],
             "authoritative pointer and promotion differ")
    _require(not inventory["temporary_artifacts"] and not any(inventory["unreferenced_final_artifacts"].values()),
             "temporary or orphan lifecycle artifact remains")
    _require(not list(clean.rglob("*.tmp")) and not (clean / "recovery").exists()
             and not (clean / "interrupted_state_inventory.json").exists(),
             "clean result contains temporary or recovery state")
    for record in records:
        for namespace in ("CURRENT", "POST", "NEXT"):
            _verify_record_artifact(lifecycle_root, manifest, record, namespace)
        # Public API independently checks that the promoted NEXT is readable.
        _require(set(lifecycle.current_fields(record["ordinal"])) == set(FIELDS),
                 "authoritative NEXT fields are incomplete")

    final = _read_json(clean / "final.json")
    snapshot = _read_json(clean / "final_lifecycle_audit.json")
    _require(final.get("streaming_manifest_sha256") == manifest_sha_before
             and snapshot.get("manifest_sha256") == manifest_sha_before,
             "final receipt does not bind the lifecycle manifest")
    _require(final.get("barrier") == barrier and final.get("promotion") == promotion,
             "final receipt and lifecycle state differ")
    normalized = inventory["normalized_lifecycle"]
    _require(snapshot.get("normalized_lifecycle") == normalized
             and snapshot.get("barrier") == normalized["barrier"]
             and snapshot.get("promotion") == normalized["promotion"],
             "final snapshot and normalized lifecycle state differ")
    _require(not snapshot.get("active_hydro_claims") and not snapshot.get("queue")
             and not snapshot.get("recovery_backlog") and not snapshot.get("temporary_artifacts")
             and not any(snapshot.get("unreferenced_final_artifacts", {}).values()),
             "final lifecycle snapshot is not closed")

    optical = clean / "optical"
    optical_run = _read_json(optical / "optical_run.json")
    input_path = clean.parent / "s5_final_input_manifest.json"
    _require(optical_run.get("input_manifest_sha256") == sha256_file(input_path),
             "optical input manifest hash differs")
    optical_identity = verify_final_optical(optical)
    _require(optical_run.get("ledger") == "scientific_ledger.npz", "ledger path is invalid")
    ledger_path = optical / "scientific_ledger.npz"
    _require(ledger_path.is_file(), "scientific ledger is missing")
    with np.load(ledger_path, allow_pickle=False) as ledger:
        _require(set(ledger.files) == set(optical_run.get("ledger_fields", [])) and len(ledger.files) == 9,
                 "scientific ledger fields differ")
        for field in ledger.files:
            _require(bool(np.isfinite(ledger[field]).all()), f"scientific ledger field is non-finite: {field}")
    for name in DEPOSITION_ARCHIVES:
        _finite_array(optical / name, expected_count=expected_screens)
    _require(sha256_file(manifest_path) == manifest_sha_before, "manifest changed during read-only audit")
    return {
        "schema": "khz_filament.hr4e5s.s5_final.clean_terminal_audit.v2",
        "status": "PASS", "expected_screen_count": expected_screens,
        "completed_screens": len(records), "queue_depth": 0, "recovery_backlog_depth": 0,
        "barrier": "PASS", "promotion": "PASS", "authoritative_namespace": "NEXT",
        "manifest_sha256": manifest_sha_before, "validated_artifacts": expected_screens * 3,
        "final_optical": optical_identity, "ledger_fields": sorted(optical_run["ledger_fields"]),
        "deposition_archives": list(DEPOSITION_ARCHIVES), "fault_injection": "OFF",
        "lifecycle_open": "PASS", "artifact_inventory": "PASS",
    }


def qualify_exact_summary(exact: dict[str, Any]) -> dict[str, Any]:
    """Classify existing comparison evidence; never rerun or relax comparator."""
    optical = exact.get("final_optical", {})
    ledger = exact.get("ledger_rows", [])
    provenance = exact.get("recovery_provenance", {})
    required = {
        "comparison_status": exact.get("status") == "PASS",
        "field_count": exact.get("expected_field_comparisons") == 432
                       and exact.get("completed_field_comparisons") == 432
                       and len(exact.get("field_rows", [])) == 432,
        "zero_mismatches": exact.get("mismatch_count") == 0,
        "final_optical_exact": all(optical.get(key) is True for key in
                                   ("shape_equal", "dtype_equal", "hash_equal", "array_equal")),
        "ledger_exact": len(ledger) == 9 and all(
            row.get("present_both") is True
            and all(row.get(key) is True for key in ("shape_equal", "dtype_equal", "hash_equal", "array_equal"))
            for row in ledger),
        "authoritative_lifecycle_exact": exact.get("authoritative_manifest_exact") is True
                                         and exact.get("completion_map_exact") is True,
        "ownership_exact": exact.get("ownership_exact") is True,
        "barrier_exact": exact.get("barrier_exact") is True,
        "promotion_exact": exact.get("promotion_generation_exact") is True,
        "final_artifact_inventory_clean": exact.get("final_artifact_inventory_clean") is True,
        "recovery_provenance_exact": provenance.get("status") == "PASS"
                                     and provenance.get("recovery_provenance_exact") is True
                                     and len(provenance.get("checks", [])) == 4
                                     and all(item.get("pass") is True for item in provenance["checks"]),
    }
    return {"schema": "khz_filament.hr4e5s.s5_final.scientific_qualification.v1",
            "status": "PASS" if all(required.values()) else "FAIL", "checks": required}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.out.exists(), "audit report path already exists")
    _require(not args.out.is_relative_to(args.clean_root), "audit report must be outside the clean source root")
    try:
        report = audit_clean_reference(args.clean_root)
    except Exception as error:
        report = {"schema": "khz_filament.hr4e5s.s5_final.clean_terminal_audit.v2",
                  "status": "FAIL", "first_error": f"{type(error).__name__}: {error}"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.out),
                      "first_error": report.get("first_error")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
