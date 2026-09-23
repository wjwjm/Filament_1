"""CPU-only regressions for the local S5-FINAL post-run audit."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from KHz_filament.hr4e_timestep import sha256_array, sha256_file
from KHz_filament.hr4e5s_s3 import finalize_streaming
from KHz_filament.hr4e5s_s5_final import snapshot_interrupted_state
from KHz_filament.hr4e5s_streaming import StreamingLifecycle


def _audit_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "audit_hr4e5s_s5_final_clean.py"
    spec = importlib.util.spec_from_file_location("s5_final_clean_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def clean_case(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real 48-screen terminal lifecycle with tiny arrays and no science run."""
    run_root = tmp_path_factory.mktemp("s5-final-clean-audit")
    clean = run_root / "clean"
    clean.mkdir()
    current = {
        "delta_n": np.full((48, 2, 2), -1e-6, dtype=np.float64),
        "vx": np.zeros((48, 2, 2), dtype=np.float64),
        "vy": np.zeros((48, 2, 2), dtype=np.float64),
    }
    records = [{"ordinal": i, "screen_id": f"z{i:05d}", "z_m": i * 1e-4} for i in range(48)]
    lifecycle = StreamingLifecycle.create(root=clean / "lifecycle", current=current,
                                          screen_records=records, current_generation="audit-test-current",
                                          dx_m=1e-4, dy_m=1e-4, queue_depth=8, actor="test")
    for ordinal in range(48):
        lifecycle.deposition_finalized(ordinal, actor="optical")
        post = lifecycle.current_fields(ordinal)
        post["delta_n"] -= 1e-8
        lifecycle.commit_post(ordinal, post, actor="optical")
        lifecycle.enqueue_post(ordinal, actor="optical")
        if (ordinal + 1) % 8 == 0:
            block = lifecycle.claim_block(actor="hydro_consumer_0")
            assert block == list(range(ordinal - 7, ordinal + 1))
            for claimed in block:
                fields = lifecycle.current_fields(claimed)
                fields["delta_n"] -= 1e-8
                lifecycle.commit_next(claimed, fields, actor="hydro_consumer_0")
    finalize_streaming(lifecycle_root=lifecycle.root, out_path=clean / "final.json")
    snapshot_interrupted_state(lifecycle_root=lifecycle.root, out_path=clean / "final_lifecycle_audit.json")

    input_path = run_root / "s5_final_input_manifest.json"
    _write_json(input_path, {"schema": "test", "screen_count": 48})
    optical = clean / "optical"
    optical.mkdir()
    final = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex128)
    np.save(optical / "final_optical_field.npy", final)
    ledger_fields = [f"ledger_{index}" for index in range(9)]
    np.savez(optical / "scientific_ledger.npz",
             **{name: np.array([index], dtype=np.float64) for index, name in enumerate(ledger_fields)})
    for name in _audit_module().DEPOSITION_ARCHIVES:
        np.save(optical / name, np.zeros((48, 1, 1), dtype=np.float64))
    _write_json(optical / "optical_run.json", {
        "final_optical_field": "final_optical_field.npy",
        "final_optical_field_sha256": sha256_array(final),
        "ledger": "scientific_ledger.npz", "ledger_fields": ledger_fields,
        "input_manifest_sha256": sha256_file(input_path),
    })
    return clean


def test_s5_final_audit_uses_existing_public_api_only(clean_case: Path):
    module = _audit_module()
    assert not hasattr(StreamingLifecycle, "_load_artifact")
    with pytest.raises(AttributeError, match="_load_artifact"):
        getattr(StreamingLifecycle.open(clean_case / "lifecycle"), "_load_artifact")
    batch = Path(__file__).resolve().parents[2] / "artifacts/hr4e5s_s5_final/same_runtime_clean_closeout/hr4e5s_s5_final_clean.sbatch"
    assert "_load_artifact" not in batch.read_text(encoding="utf-8")
    assert "audit_hr4e5s_s5_final_clean.py" in batch.read_text(encoding="utf-8")
    source = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    unsupported = {"_load_artifact", "_validate_committed_artifacts"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in unsupported for node in ast.walk(source))


def test_s5_final_audit_array_hash_is_not_raw_npy_hash(tmp_path: Path):
    module = _audit_module()
    optical = tmp_path / "optical"
    optical.mkdir()
    values = np.array([1.0, 2.0], dtype=np.float64)
    np.save(optical / "final_optical_field.npy", values)
    _write_json(optical / "optical_run.json", {
        "final_optical_field": "final_optical_field.npy",
        "final_optical_field_sha256": sha256_array(values),
    })
    result = module.verify_final_optical(optical)
    assert result["canonical_array_sha256"] == sha256_array(values)
    assert result["raw_npy_sha256"] == sha256_file(optical / "final_optical_field.npy")
    assert result["canonical_array_sha256"] != result["raw_npy_sha256"]


def test_s5_final_audit_optical_tamper_fails(tmp_path: Path):
    module = _audit_module()
    optical = tmp_path / "optical"
    optical.mkdir()
    values = np.array([1.0, 2.0], dtype=np.float64)
    path = optical / "final_optical_field.npy"
    np.save(path, values)
    _write_json(optical / "optical_run.json", {
        "final_optical_field": path.name, "final_optical_field_sha256": sha256_array(values),
    })
    np.save(path, np.array([1.0, 2.1], dtype=np.float64))
    with pytest.raises(module.CleanAuditError, match="canonical array hash differs"):
        module.verify_final_optical(optical)


def test_s5_final_audit_48_screen_terminal_pass(clean_case: Path):
    result = _audit_module().audit_clean_reference(clean_case)
    assert result["status"] == "PASS" and result["completed_screens"] == 48
    assert result["queue_depth"] == result["recovery_backlog_depth"] == 0
    assert result["barrier"] == result["promotion"] == "PASS"
    assert result["authoritative_namespace"] == "NEXT"


@pytest.mark.parametrize("missing", [
    "lifecycle/streaming_manifest.json",
    "lifecycle/post/screen_000000.npz",
    "optical/optical_run.json",
    "optical/final_optical_field.npy",
    "optical/scientific_ledger.npz",
    "optical/s3_optical.hr3a_qion_samples.npy",
    "final.json",
    "final_lifecycle_audit.json",
])
def test_s5_final_audit_missing_required_artifact_fails_closed(
        clean_case: Path, tmp_path: Path, missing: str):
    module = _audit_module()
    copied = tmp_path / "copied"
    shutil.copytree(clean_case.parent, copied)
    (copied / "clean" / missing).unlink()
    with pytest.raises((module.CleanAuditError, FileNotFoundError)):
        module.audit_clean_reference(copied / "clean")


def test_s5_final_audit_exact_summary_scientific_pass():
    module = _audit_module()
    row = {"present_both": True, "shape_equal": True, "dtype_equal": True,
           "hash_equal": True, "array_equal": True}
    exact = {
        "status": "PASS", "expected_field_comparisons": 432,
        "completed_field_comparisons": 432, "field_rows": [{}] * 432,
        "mismatch_count": 0, "final_optical": dict(row), "ledger_rows": [dict(row) for _ in range(9)],
        "authoritative_manifest_exact": True, "completion_map_exact": True,
        "ownership_exact": True, "barrier_exact": True,
        "promotion_generation_exact": True, "final_artifact_inventory_clean": True,
        "recovery_provenance": {"status": "PASS", "recovery_provenance_exact": True,
                                "checks": [{"pass": True}] * 4},
    }
    assert module.qualify_exact_summary(exact)["status"] == "PASS"
    exact["mismatch_count"] = 1
    assert module.qualify_exact_summary(exact)["status"] == "FAIL"
