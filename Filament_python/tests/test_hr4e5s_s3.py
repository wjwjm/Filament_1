from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_s3_exact_comparison_covers_all_48_screen_scientific_fields(tmp_path):
    from KHz_filament.hr4e5s_s3 import compare_exact
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    records = [
        {"ordinal": index, "screen_id": f"source_index_{index:05d}", "source_index": index, "z_m": 0.8 + index * 1.0e-4}
        for index in range(48)
    ]
    input_manifest = {"screen_records": records, "shape": [1, 1]}
    input_path = tmp_path / "input.json"
    current = np.zeros((48, 1, 1), dtype=np.float64)
    from KHz_filament.hr4e_timestep import sha256_array
    for record in records:
        record["current_delta_n_sha256"] = sha256_array(current[record["ordinal"]])
    input_path.write_text(json.dumps(input_manifest), encoding="utf-8")
    lifecycle = StreamingLifecycle.create(root=tmp_path / "stream", current={"delta_n": current, "vx": current.copy(), "vy": current.copy()}, screen_records=records, current_generation="s3-current", dx_m=1.0e-5, dy_m=1.0e-5, queue_depth=16)
    for ordinal in range(48):
        lifecycle.deposition_finalized(ordinal)
        lifecycle.commit_post_from_delta_n(ordinal, current[ordinal])
        lifecycle.enqueue_post(ordinal)
        if ordinal % 8 == 7:
            for claimed in lifecycle.claim_block():
                fields = lifecycle._artifact_fields(lifecycle.manifest["records"][claimed]["post"], namespace="POST")
                lifecycle.commit_next(claimed, fields)
    batch_optical, stream_optical, batch_hydro = (tmp_path / "batch_optical", tmp_path / "stream_optical", tmp_path / "batch_hydro")
    for directory in (batch_optical, stream_optical, batch_hydro):
        directory.mkdir()
    for name in ("ion", "ib", "raman"):
        value = np.zeros((48, 1, 1), dtype=np.float64)
        np.save(batch_optical / f"s3_optical.hr3a_q{name}_samples.npy", value)
        np.save(stream_optical / f"s3_optical.hr3a_q{name}_samples.npy", value)
    np.save(batch_optical / "s3_optical.hr3b_delta_n_state_after_update_samples.npy", current)
    for field in ("delta_n", "vx", "vy"):
        np.save(batch_hydro / f"next_{field}.npy", current)
    final = np.zeros((2, 1, 1), dtype=np.complex128)
    ledger = {"E_dep_ion_interval_J": np.zeros(48, dtype=np.float64)}
    for directory, ownership in (
        (batch_optical, None),
        (stream_optical, {"authoritative_namespace": "CURRENT", "next_pointer_exists": False}),
    ):
        np.save(directory / "final_optical_field.npy", final)
        np.savez(directory / "scientific_ledger.npz", **ledger)
        (directory / "optical_run.json").write_text(json.dumps({"final_optical_field": "final_optical_field.npy", "ledger": "scientific_ledger.npz", "optical_current_ownership_before": ownership, "optical_current_ownership_after": ownership}), encoding="utf-8")
    lifecycle.validate_barrier()
    lifecycle.promote_next_to_current()
    result = compare_exact(input_manifest_path=input_path, batch_optical_dir=batch_optical, batch_hydro_dir=batch_hydro, streaming_optical_dir=stream_optical, streaming_root=tmp_path / "stream", out_dir=tmp_path / "comparison")
    assert result["status"] == "PASS"
    assert result["expected_field_comparisons"] == result["completed_field_comparisons"] == 432


def test_s3_launcher_preflights_the_private_lut_workspace_and_submits_in_two_phases():
    root = Path(__file__).resolve().parents[1]
    batch = (root / "tools" / "hr4e5s_s3.sbatch").read_text(encoding="utf-8")
    preflight = (root / "tools" / "hpc_ops" / "run_hr4e5s_s3_preflight.sh").read_text(encoding="utf-8")
    submit = (root / "tools" / "hpc_ops" / "submit_hr4e5s_s3.sh").read_text(encoding="utf-8")

    assert 'readonly LUT_WORKSPACE="$RUN_ROOT/lut_workspace"' in preflight
    assert "audit_hr4e5s_s3_lut_workspace.py" in preflight
    assert 'cd "$LUT_WORKSPACE"' in batch
    assert 'mkdir -m 700 -- "$stream_root/consumer"' in batch
    assert 'mkdir -m 700 -- "$stream_root/optical" "$stream_root/consumer"' not in batch
    assert 'if [[ "$producer_status" -ne 0 ]]; then' in batch
    assert 'kill "$consumer_pid" 2>/dev/null || true' in batch
    assert 'exit "$producer_status"' in batch
    assert 'BATCH_REFERENCE_ROOT:?}' in batch
    assert '"$BATCH_REFERENCE_ROOT/batch_optical"' in batch
    assert "LUT_WORKSPACE=$RUN_ROOT/lut_workspace" in submit
    assert 'case "$SUBMIT_PHASE" in' in submit
    assert "afterok:" not in submit
    assert "sacct -j" in submit
    assert "BATCH_REFERENCE_ROOT=$BATCH_REFERENCE_ROOT" in submit
    assert 'BATCH_REFERENCE_JOB="${10}"' in submit
    assert 'batch_job="$BATCH_REFERENCE_JOB"' in submit
