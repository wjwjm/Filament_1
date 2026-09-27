from __future__ import annotations

from copy import deepcopy
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import json

from KHz_filament.hr4e5_gpu_binding import (
    build_assignment_record,
    build_cpu_schema_receipt,
    build_gpu_final_receipt,
    build_gpu_startup_receipt,
    assign_roles,
    validate_assignment_record,
    validate_startup_final_identities,
)
from KHz_filament.hr4e5_gpu_bootstrap import (
    probe_gpu_runtime,
    resolve_receipt_paths,
    validate_child_device_assignment,
    write_durable_receipt,
)


def _startup(role: str, token: str, scheduler_visible_devices: str, scheduler_index: int, gpu_uuid: str):
    return build_gpu_startup_receipt(
        campaign_id="campaign-1",
        scientific_pair_id="pair-1",
        coordinator_epoch="epoch-1",
        implementation_identity_sha256="a" * 64,
        role=role,
        assigned_token=token,
        scheduler_visible_devices=scheduler_visible_devices,
        visible_devices=token,
        local_index=0,
        pid=1000 + scheduler_index,
        ppid=100,
        process_start_token=f"start-{scheduler_index}",
        parent_start_token="parent-start",
        gpu_uuid=gpu_uuid,
        gpu_model="GPU test fixture",
        gpu_total_memory_bytes=80_000_000_000,
        gpu_used_memory_bytes=1_000_000,
        cupy_version="test-only",
        cuda_runtime_version=12090,
        representative_allocation_bytes=1024,
        scheduler_job_id="12345",
        hostname="test-host",
        started_utc="2026-09-27T00:00:00Z",
    )


def test_explicit_role_assignment_supports_primary_and_fallback_topologies():
    assert assign_roles("4,5,6,7,8,9", 6) == {
        "R": "4",
        "C_OPT": "5",
        "H0": "6",
        "H1": "7",
        "H2": "8",
        "H3": "9",
    }
    assert assign_roles("GPU-a,GPU-b,MIG-c,MIG-d", 4) == {
        "R": "GPU-a",
        "C_OPT": "GPU-b",
        "H0": "MIG-c",
        "H1": "MIG-d",
    }


@pytest.mark.parametrize(
    "visible,count",
    [
        (None, 4),
        ("", 4),
        ("0,1,2", 4),
        ("0,1,2,3,4", 4),
        ("0,1,1,3", 4),
        ("GPU-Ab,gpu-ab,2,3", 4),
        ("0,,2,3", 4),
        ("-1", 4),
        ("0,1,2,3", 5),
    ],
)
def test_role_assignment_rejects_missing_duplicate_or_wrong_count(visible, count):
    with pytest.raises(ValueError):
        assign_roles(visible, count)


def test_assignment_record_is_schema_validated_as_one_contract():
    record = build_assignment_record("0,1,2,3", 4)
    assert validate_assignment_record(record, 4)
    altered = deepcopy(record)
    altered["role_to_token"]["H0"] = "3"
    with pytest.raises(ValueError, match="role-to-token"):
        validate_assignment_record(altered)


def test_child_bootstrap_requires_exact_role_token_and_single_visible_gpu():
    assert validate_child_device_assignment("H1", "7", "7", "4,5,6,7") == (3, 0)
    with pytest.raises(ValueError, match="exactly one"):
        validate_child_device_assignment("H1", "0,1", "0", "4,5,6,7")
    with pytest.raises(ValueError, match="disagrees"):
        validate_child_device_assignment("H1", "4", "4", "4,5,6,7")


def test_gpu_probe_imports_device_submodule_when_package_exports_a_string(monkeypatch):
    import KHz_filament

    fake_cupy = ModuleType("cupy")
    fake_cupy.cuda = SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
    fake_device = ModuleType("KHz_filament.device")
    fake_device.USE_GPU = True
    fake_device.xp = fake_cupy
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setitem(sys.modules, "KHz_filament.device", fake_device)
    monkeypatch.setattr(KHz_filament, "device", "device")

    with pytest.raises(RuntimeError, match="single-token CUDA mask"):
        probe_gpu_runtime()


def test_cpu_schema_fixture_has_no_synthetic_gpu_identity():
    receipt = build_cpu_schema_receipt("R")
    assert receipt["status"] == "SCHEMA_ONLY"
    assert receipt["identity"]["gpu"] == {"uuid": None, "model": None, "total_memory_bytes": None}
    assert receipt["identity"]["assigned_cuda_visible_device"] is None


def test_terminal_candidate_optical_can_close_without_a_child_grant_but_hydro_cannot():
    candidate = _startup("C_OPT", "1", "0,1,2,3", 1, "GPU-candidate")
    hydro = _startup("H0", "2", "0,1,2,3", 2, "GPU-hydro")
    metadata = {"science_output_status": "PASS",
                "writer_grant_status": "NO_CHILD_GRANT_WRITERS_CLOSED"}
    assert build_gpu_final_receipt(candidate, exit_code=0, elapsed_seconds=1,
                                   final_metadata=metadata)["status"] == "PASS"
    assert build_gpu_final_receipt(hydro, exit_code=0, elapsed_seconds=1,
                                   final_metadata=metadata)["status"] == "FAIL"


def test_receipt_paths_keep_audit_root_outside_campaign_storage(tmp_path):
    campaign = tmp_path / "campaign"
    audit = tmp_path / "audit" / "pulse-001" / "epoch-001"
    resolved_audit, startup, final = resolve_receipt_paths(
        audit_root=audit,
        campaign_root=campaign,
        role="R",
    )
    assert resolved_audit == audit.resolve()
    assert startup == resolved_audit / "R.startup.json"
    assert final == resolved_audit / "R.final.json"
    with pytest.raises(ValueError, match="separate"):
        resolve_receipt_paths(
            audit_root=campaign / "audit",
            campaign_root=campaign,
            role="R",
        )


def test_durable_receipt_is_no_overwrite(tmp_path):
    path = tmp_path / "audit" / "receipt.json"
    write_durable_receipt(path, {"status": "PASS", "sequence": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"sequence": 1, "status": "PASS"}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_durable_receipt(path, {"status": "FAIL", "sequence": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["sequence"] == 1


def test_startup_and_final_receipts_match_full_process_and_gpu_identity():
    tokens = "0,1,2,3"
    starts = [
        _startup(role, token, tokens, index, f"GPU-{index:04d}")
        for index, (role, token) in enumerate(assign_roles(tokens, 4).items())
    ]
    finals = [
        build_gpu_final_receipt(
            receipt,
            exit_code=0,
            elapsed_seconds=12.5,
            final_metadata={"science_output_status": "PASS", "writer_grant_status": (
                "NO_CHILD_GRANT_WRITERS_CLOSED" if receipt["role"] == "R" else "CLOSED")},
        )
        for receipt in starts
    ]
    assert all(receipt["status"] == "PASS" for receipt in finals)
    assert validate_startup_final_identities(starts, finals)

    changed = deepcopy(finals)
    changed[0]["identity"]["process"]["start_token"] = "replacement-process"
    with pytest.raises(ValueError, match="identity changed"):
        validate_startup_final_identities(starts, changed)


def test_duplicate_gpu_uuid_and_missing_final_status_do_not_pass():
    tokens = "0,1,2,3"
    roles = assign_roles(tokens, 4)
    starts = [
        _startup(role, token, tokens, index, "GPU-duplicate" if index < 2 else f"GPU-{index}")
        for index, (role, token) in enumerate(roles.items())
    ]
    finals = [
        build_gpu_final_receipt(
            receipt,
            exit_code=0,
            elapsed_seconds=1.0,
            final_metadata={"science_output_status": "PASS", "writer_grant_status": (
                "NO_CHILD_GRANT_WRITERS_CLOSED" if receipt["role"] == "R" else "CLOSED")},
        )
        for receipt in starts
    ]
    with pytest.raises(ValueError, match="duplicate GPU UUID"):
        validate_startup_final_identities(starts, finals)

    starts[0] = _startup("R", "0", tokens, 0, "GPU-unique-0")
    finals = [
        build_gpu_final_receipt(receipt, exit_code=0, elapsed_seconds=1.0)
        for receipt in starts
    ]
    assert all(receipt["status"] == "FAIL" for receipt in finals)


def test_formal_gpu_admission_fails_before_campaign_creation_without_cpu_coordinator_or_slurm(tmp_path, monkeypatch):
    from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign

    spec = E5AProductionSpec(
        root=tmp_path / "campaign", campaign_id="gpu-gate",
        config_path="unused", source_manifest_path="unused",
        lut_manifest_path="unused", pre0_delta_n_path="unused",
        require_gpu_backend=True, candidate_hydro_workers=2,
    )
    monkeypatch.setenv("UPPE_USE_GPU", "1")
    with pytest.raises(RuntimeError, match="coordinator"):
        open_e5_1a_production_campaign(spec)
    monkeypatch.setenv("UPPE_USE_GPU", "0")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        open_e5_1a_production_campaign(spec)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="expected 4"):
        open_e5_1a_production_campaign(spec)
    assert not (tmp_path / "campaign").exists()


def test_gpu_pair_barrier_validates_new_roles_and_prior_optical_reuse(tmp_path, monkeypatch):
    from KHz_filament import hr4e5_production as production

    spec = production.E5AProductionSpec(
        root=tmp_path / "campaign", campaign_id="campaign-1",
        config_path="unused", source_manifest_path="unused",
        lut_manifest_path="unused", pre0_delta_n_path="unused",
        n_pulses=2, candidate_hydro_workers=2, require_gpu_backend=True,
    )
    runner = object.__new__(production._E5AProductionRunner)
    runner.spec = spec
    runner.root = spec.root
    runner.identity = {"campaign_id": "campaign-1"}
    runner.implementation_identity = {"identity_sha256": "a" * 64}
    runner.require_gpu_backend = True
    runner.driver = SimpleNamespace(campaign=SimpleNamespace(active_coordinator_epoch=lambda: "2"))
    audit = runner._gpu_audit_root(0)
    audit.mkdir(parents=True)
    assignment = build_assignment_record("0,1,2,3", 4)
    assignment.update({"campaign_id": "campaign-1", "scientific_pair_id": "pair-1", "coordinator_epoch": "2",
                       "implementation_identity_sha256": "a" * 64})
    assignment_path = audit / "GPU_ROLE_ASSIGNMENT.json"
    assignment_path.write_text(json.dumps(assignment), encoding="utf-8")
    runner._gpu_assignment_paths = {0: assignment_path}
    prior_assignment = dict(assignment, coordinator_epoch="1")
    prior_folder = audit.parent / "epoch_1"
    prior_folder.mkdir(parents=True)
    (prior_folder / "GPU_ROLE_ASSIGNMENT.json").write_text(
        json.dumps(prior_assignment), encoding="utf-8")

    def persist_role(role, index, epoch):
        folder = audit if epoch == "2" else audit.parent / f"epoch_{epoch}"
        folder.mkdir(parents=True, exist_ok=True)
        receipt = _startup(role, str(index), "0,1,2,3", index, f"GPU-{index}")
        receipt["coordinator_epoch"] = epoch
        receipt["identity"]["process"]["ppid"] = os.getpid()
        metadata = {"science_output_status": "PASS", "writer_grant_status": (
            "NO_CHILD_GRANT_WRITERS_CLOSED" if role == "R" else "CLOSED")}
        final = build_gpu_final_receipt(receipt, exit_code=0, elapsed_seconds=1,
                                        final_metadata=metadata)
        (folder / f"{role}.startup.json").write_text(json.dumps(receipt), encoding="utf-8")
        (folder / f"{role}.final.json").write_text(json.dumps(final), encoding="utf-8")

    persist_role("R", 0, "2")
    persist_role("H0", 2, "2")
    persist_role("H1", 3, "2")
    persist_role("C_OPT", 1, "1")
    monkeypatch.setattr(production, "validate_active_output", lambda **_kwargs: {
        "qualified_attempt": {"producer_coordinator_epoch": "1"}})
    runner.validate_gpu_pulse_receipts(0)
    validated = json.loads((audit / "GPU_ROLE_RECEIPTS_VALIDATED.json").read_text(encoding="utf-8"))
    assert validated["current_roles"] == ["H0", "H1", "R"]
    assert set(validated["reused_roles"]) == {"C_OPT"}
