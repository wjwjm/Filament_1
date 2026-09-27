"""Control-plane identity checks for cross-epoch WP2 recovery."""

from pathlib import Path
import json

import pytest

from KHz_filament.hr4e5_recovery_identity import (
    build_scientific_pair_identity,
    capture_implementation_identity,
    require_same_implementation,
)
from KHz_filament.hr4e5_recovery_output import (
    activate_output_attempt, qualify_output_attempt, validate_active_output,
)


def _source_tree(root: Path) -> Path:
    package = root / "KHz_filament"
    package.mkdir()
    (package / "hr4e5_production_worker.py").write_bytes(b"worker = 1\r\n")
    (package / "hr4e5_production_optical_worker.py").write_bytes(b"optical = 1\n")
    helper = root / "tools" / "hpc_ops" / "provenance_v2.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"helper = 1\n")
    return package


def test_implementation_identity_reads_all_current_source_bytes(tmp_path):
    package = _source_tree(tmp_path)
    first = capture_implementation_identity(python_root=tmp_path, include_runtime=False)
    (package / "hr4e5_production_worker.py").write_bytes(b"worker = 1\n")
    line_ending_only = capture_implementation_identity(python_root=tmp_path, include_runtime=False)
    require_same_implementation(first, line_ending_only)

    (package / "hr4e5_production_worker.py").write_bytes(b"worker = 2\n")
    changed = capture_implementation_identity(python_root=tmp_path, include_runtime=False)
    with pytest.raises(ValueError, match="IMPLEMENTATION_IDENTITY_MISMATCH"):
        require_same_implementation(first, changed)

    (package / "new_untracked_production_module.py").write_bytes(b"new = True\n")
    added = capture_implementation_identity(python_root=tmp_path, include_runtime=False)
    assert added["identity_sha256"] != changed["identity_sha256"]
    assert any(row["path"] == "KHz_filament/new_untracked_production_module.py" for row in added["files"])


def test_pair_identity_is_stable_across_execution_epochs_and_scoped_to_science():
    admission = {
        "campaign_id": "campaign-one",
        "identity_sha256": "a" * 64,
        "config_identity": {"sha256": "b" * 64},
        "coordinator_epoch": "epoch-zero",
    }
    first = build_scientific_pair_identity(
        admission_identity=admission, pulse=0,
        reference_generation="R:p0:PRE0", candidate_generation="C:p0:PRE0",
    )
    later = build_scientific_pair_identity(
        admission_identity={**admission, "coordinator_epoch": "epoch-one"}, pulse=0,
        reference_generation="R:p0:PRE0", candidate_generation="C:p0:PRE0",
    )
    assert first == later
    assert first["scientific_pair_id"] != build_scientific_pair_identity(
        admission_identity=admission, pulse=1,
        reference_generation="R:p1:PRE", candidate_generation="C:p1:PRE",
    )["scientific_pair_id"]
    assert first["scientific_pair_id"] != build_scientific_pair_identity(
        admission_identity=admission, pulse=0,
        reference_generation="R:p0:PRE0", candidate_generation="C:p0:changed",
    )["scientific_pair_id"]
    assert first["scientific_pair_id"] != build_scientific_pair_identity(
        admission_identity={**admission, "config_identity": {"sha256": "c" * 64}}, pulse=0,
        reference_generation="R:p0:PRE0", candidate_generation="C:p0:PRE0",
    )["scientific_pair_id"]


def _completed_output(folder: Path, *, admission_hash: str, generation: str):
    folder.mkdir(parents=True)
    names = (
        "final_optical_field.npy", "scientific_ledger.npz",
        "sinks.hr3a_qion_samples.npy", "sinks.hr3a_qib_samples.npy",
        "sinks.hr3a_qraman_samples.npy", "sinks.hr3a_qthermal_samples.npy",
        "sinks.hr3b_delta_n_increment_samples.npy",
        "sinks.hr3b_delta_n_state_after_update_samples.npy",
    )
    for name in names:
        (folder / name).write_bytes(name.encode("ascii"))
    (folder / "optical_run.json").write_text(json.dumps({
        "status": "PASS", "output_dir": str(folder.resolve()),
        "admission_identity_sha256": admission_hash, "current_generation": generation,
    }), encoding="utf-8")


def test_active_output_locator_preserves_failed_bytes_and_rejects_tamper(tmp_path):
    state = tmp_path / "C" / "p0" / "state"
    state.mkdir(parents=True)
    original = tmp_path / "C" / "p0" / "optical"
    _completed_output(original, admission_hash="a" * 64, generation="C:p0:PRE0")
    first = qualify_output_attempt(
        campaign_root=tmp_path, state_root=state, output_dir=original,
        attempt_id="producer_epoch0", campaign_id="campaign", scientific_pair_id="pair",
        trajectory="C", pulse=0, generation="C:p0:PRE0", coordinator_epoch=0,
        writer_grant_id="grant0", admission_identity_sha256="a" * 64,
        implementation_identity_sha256="b" * 64,
    )
    active = activate_output_attempt(
        campaign_root=tmp_path, state_root=state,
        record_path=first["record_path"], expected_record_sha256=first["record_sha256"],
    )
    assert active["output_dir"] == str(original.resolve())

    recovered = original / "replay_epoch1"
    _completed_output(recovered, admission_hash="a" * 64, generation="C:p0:PRE0")
    second = qualify_output_attempt(
        campaign_root=tmp_path, state_root=state, output_dir=recovered,
        attempt_id="producer_epoch1", campaign_id="campaign", scientific_pair_id="pair",
        trajectory="C", pulse=0, generation="C:p0:PRE0", coordinator_epoch=1,
        writer_grant_id="grant1", admission_identity_sha256="a" * 64,
        implementation_identity_sha256="b" * 64,
    )
    active = activate_output_attempt(
        campaign_root=tmp_path, state_root=state,
        record_path=second["record_path"], expected_record_sha256=second["record_sha256"],
    )
    assert active["output_dir"] == str(recovered.resolve())
    assert (original / "final_optical_field.npy").is_file()
    (recovered / "scientific_ledger.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity changed|hash changed"):
        validate_active_output(campaign_root=tmp_path, state_root=state)
