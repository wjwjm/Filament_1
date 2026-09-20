"""Focused E5-1A-R admission, intent, evidence and coordinator checks."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest


def _formal_identity(campaign_id: str = "formal-r", *, n_pulses: int = 1):
    from KHz_filament.hr4e5_formal_entry import build_admission_identity

    return build_admission_identity(
        campaign_id=campaign_id, runtime_or_compatibility={"runtime": "local-test", "sha": "runtime"},
        config_identity={"path": "config.json", "sha256": "config"},
        effective_params={"dz": 1e-4, "z_max": 8e-4}, source_identity={"path": "source.npy", "sha256": "source"},
        lut_identity={"path": "lut", "sha256": "lut"}, schedule_identity={"edges": [0, 1], "sha256": "schedule"},
        grid_identity={"Nx": 4, "Ny": 4, "Nt": 4, "dtype": "complex128"},
        pre0_identity={"parent_sha256": "pre0", "derived_sha256": "prefix"}, n_pulses=n_pulses, k=8,
        f_rep=5e6, dt_hydro=1e-7, precision={"fields": "float64", "optical": "complex128"},
        r_roots={"root": "R"}, c_roots={"root": "C"},
        pulse_attempt_epoch={"pulse": 0, "attempt": 0, "epoch": "epoch-0"},
        budget_identity={"max_campaign_live_bytes": 322122547200, "final_output_budget_bytes": 68719476736},
        retention_identity={"roles": ["final", "ledger"]},
    )


def _production_runner_shell():
    """Return a typed, non-running shell for private formal-gate unit tests."""
    from KHz_filament.hr4e5_production import _E5AProductionRunner

    return _E5AProductionRunner.__new__(_E5AProductionRunner)


def test_r01_formal_identity_rejects_missing_and_tamper():
    from KHz_filament.hr4e5_formal_entry import validate_admission_identity

    with pytest.raises(ValueError, match="missing|placeholder"):
        validate_admission_identity({"schema": "khz_filament.hr4e5.e5_1a.admission.v1"}, formal=True)
    identity = _formal_identity()
    changed = dict(identity, k=16)
    with pytest.raises(ValueError, match="hash"):
        validate_admission_identity(changed, formal=True)
    fixture = dict(identity, execution_mode="TEST_FIXTURE_ONLY", scope="TEST_FIXTURE_ONLY")
    fixture["identity_sha256"] = "bad"
    with pytest.raises(ValueError, match="hash|fixture"):
        validate_admission_identity(fixture, formal=True)


def test_r01_same_driver_reaches_real_cpu_optical_path(tmp_path):
    """The controlled driver reaches the real optical call; quota stays fixture-only."""
    from test_hr4e5_e5_1a import _fields, _records, _schedule
    from KHz_filament.config import (
        BeamConfig, GridConfig, HeatConfig, IonizationConfig, PropagationConfig,
        RamanConfig, RunConfig,
    )
    from KHz_filament.hr4e5_formal_entry import (
        build_fixture_admission_identity, run_streaming_optical_pulse,
    )
    from KHz_filament.hr4e5_paired_campaign import FormalPairedDriver
    from KHz_filament.hr4e5s_streaming import StreamingLifecycle

    root = tmp_path / "controlled-real-optical"
    identity = build_fixture_admission_identity(
        campaign_id="real-optical-driver-fixture", n_pulses=1, k=8, shape=(8, 8),
    )
    grid = GridConfig(Nx=8, Ny=8, Nt=8, Lx=8e-4, Ly=8e-4, Twin=80e-15)
    components = {
        "grid": grid,
        "beam": BeamConfig(w0=1.5e-4, tau_fwhm=40e-15, energy_J=1e-10, focal_length=None),
        "prop": PropagationConfig(
            z_max=8e-4, dz=1e-4, linear_model="paraxial", auto_substep=False,
            focus_window_step=False, limit_focus_window=False, progress_every_z=0,
            energy_probe_every=0, diag_extra=False, use_electronic_kerr=False,
            use_raman_phase=False, use_raman_absorption=False, use_plasma_phase=False,
            use_ionization_loss=False, use_ionization_solver=False,
        ),
        "ion": IonizationConfig(species=[]), "heat": HeatConfig(hr3b_enabled=True),
        "run": RunConfig(Npulses=1), "raman": RamanConfig(enabled=False, absorption=False),
    }
    lifecycle = StreamingLifecycle.create(
        root=root / "candidate_state", current=_fields(shape=(8, 8)),
        screen_records=_records(), current_generation="fixture:pre",
        dx_m=grid.Lx / grid.Nx, dy_m=grid.Ly / grid.Ny,
    )

    class Runner:
        def __init__(self):
            self.budget = None
            self.steps = []

        def _report(self, name):
            self.steps.append(name)
            return _write_formal_simple_report(
                root, root / f"{name}.json", identity,
                "R" if name == "R" else "C",
            )

        def run_reference(self, pulse):
            return self._report("R")

        def run_candidate(self, pulse):
            result = run_streaming_optical_pulse(
                lifecycle_root=lifecycle.root, schedule=_schedule(),
                output_dir=root / "candidate_optical", components=components,
                final=True, storage_budget=self.budget,
                admission_identity=identity, fixture_only=True,
                trajectory="C", pulse=int(pulse),
            )
            assert result["status"] == "PASS" and result["schedule_intervals"] == 8
            return self._report("C")

        def run_exact(self, pulse):
            return self._report("exact")

        def run_terminal(self, pulse):
            return self._report("terminal")

    runner = Runner()
    driver = FormalPairedDriver(
        root, admission_identity=identity, runner=runner, n_pulses=1,
        campaign_id=identity["campaign_id"], fixture_only=True,
        require_quota=False, safety_margin_bytes=1024**2,
    )
    runner.budget = driver.campaign.storage
    result = driver.run()
    assert result["status"] == "COMPLETE"
    assert runner.steps == ["R", "C", "exact", "terminal"]
    assert (root / "candidate_optical" / "final_optical_field.npy").is_file()
    reopened = StreamingLifecycle.open(lifecycle.root)
    assert all(record["state"] == "POST_COMMITTED" for record in reopened.manifest["records"])
    assert not list((lifecycle.root / "next").glob("*.npz"))


def test_r03_intent_preexisting_file_is_not_retroactively_owned(tmp_path):
    from KHz_filament.hr4e5_storage import MockQuotaProvider, StorageBudget, StorageIntegrityError

    identity = _formal_identity()
    root = tmp_path / "campaign"
    target = root / "payload"
    target.mkdir(parents=True)
    old = target / "old.npy"
    np.save(old, np.ones(2, dtype=np.float64))
    budget = StorageBudget(root, campaign_id=identity["campaign_id"], require_intents=True,
                           admission_hash=identity["identity_sha256"], require_quota=True,
                           provider=MockQuotaProvider(free_bytes=2**30, quota_bytes=2**30), safety_margin_bytes=128)
    reservation = budget.reserve(4096, purpose="payload", allocation_paths=[target])
    intent = budget.create_intent(reservation_id=reservation.reservation_id, trajectory="R", pulse=0,
                                  attempt=0, role="PRE0", allowed_paths=[target], expected_bytes=4096,
                                  admission_hash=identity["identity_sha256"])
    new = target / "new.npy"
    np.save(new, np.zeros(2, dtype=np.float64))
    with pytest.raises(StorageIntegrityError, match="pre-existing"):
        budget.complete_intent(intent["intent_id"])
    assert budget.intents()[intent["intent_id"]]["status"] == "ACTIVE"


def test_r03_terminal_inventory_requires_quiescent_storage_ledger(tmp_path):
    from KHz_filament.hr4e5_storage import (
        MockQuotaProvider, StorageBudget, StorageIntegrityError,
    )

    identity = _formal_identity(campaign_id="terminal-quiescence-r")
    root = tmp_path / "terminal-quiescence"
    target = root / "payload.npy"
    budget = StorageBudget(
        root, campaign_id=identity["campaign_id"], require_intents=True,
        admission_hash=identity["identity_sha256"], require_quota=True,
        provider=MockQuotaProvider(free_bytes=2**30, quota_bytes=2**30),
        safety_margin_bytes=128,
    )
    reservation = budget.reserve(4096, purpose="unfinished", allocation_paths=[target])
    intent = budget.create_intent(
        reservation_id=reservation.reservation_id, trajectory="C", pulse=0,
        attempt=0, role="FINAL", allowed_paths=[target], expected_bytes=4096,
        admission_hash=identity["identity_sha256"], epoch="epoch-0",
        generation="terminal:test",
    )
    writer = budget.open_writer(
        reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
        coordinator_epoch="epoch-0", trajectory="C", pulse=0, attempt=0,
        generation="terminal:test",
    )
    with pytest.raises(StorageIntegrityError, match="not quiescent") as active:
        budget.validate_terminal_inventory({})
    assert "active_reservations" in str(active.value)
    assert "unfinished_intents" in str(active.value)
    assert "active_writers" in str(active.value)
    budget.close_writer(writer["writer_id"], coordinator_epoch="epoch-0", status="INTERRUPTED")
    budget.interrupt_intent(intent["intent_id"], reason="test interruption")
    with pytest.raises(StorageIntegrityError, match="unfinished_intents"):
        budget.validate_terminal_inventory({})


def test_r03_formal_pre0_refuses_without_intent_and_fixture_is_explicit(tmp_path):
    from KHz_filament.hr4e5_formal_entry import build_fixture_admission_identity, create_pre0_root
    from KHz_filament.hr4e5_storage import StorageBudget
    from KHz_filament.longitudinal import build_longitudinal_schedule

    schedule = build_longitudinal_schedule(dz=1e-4, z_max=8e-4)
    fields = np.zeros((8, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="admission identity"):
        create_pre0_root(root=tmp_path / "formal", delta_n=fields, schedule=schedule, dx_m=1e-4, dy_m=1e-4)
    fixture = build_fixture_admission_identity(k=8, shape=(2, 2))
    lifecycle = create_pre0_root(root=tmp_path / "fixture", delta_n=fields, schedule=schedule,
                                 dx_m=1e-4, dy_m=1e-4, fixture_only=True,
                                 admission_identity=fixture)
    metadata = json.loads((lifecycle.root / "E5_1A_ROOT_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["scope"] == "TEST_FIXTURE_ONLY"

    identity = _formal_identity(campaign_id="unknown-quota-r")
    campaign_root = tmp_path / "formal_unknown_quota"
    target = campaign_root / "R" / "PRE0"
    budget = StorageBudget(
        campaign_root,
        campaign_id=identity["campaign_id"],
        require_intents=True,
        admission_hash=identity["identity_sha256"],
        require_quota=False,
        safety_margin_bytes=128,
    )
    reservation = budget.reserve(1024**2, purpose="PRE0", allocation_paths=[target])
    intent = budget.create_intent(
        reservation_id=reservation.reservation_id,
        trajectory="R",
        pulse=0,
        attempt=0,
        role="PRE0",
        allowed_paths=[target],
        expected_bytes=1024**2,
        admission_hash=identity["identity_sha256"],
        generation="E5:E5_1A:PRE0",
        campaign_id=identity["campaign_id"],
    )
    with pytest.raises(ValueError, match="quota"):
        create_pre0_root(
            root=target,
            delta_n=fields,
            schedule=schedule,
            dx_m=1e-4,
            dy_m=1e-4,
            admission_identity=identity,
            storage_budget=budget,
            creation_intent=intent,
        )
    assert not target.exists()


def test_r02_object_binding_rejects_replaced_array(tmp_path):
    from KHz_filament.hr4e5_evidence import build_expected_object_set, validate_object_set

    root = tmp_path / "evidence"
    root.mkdir()
    path = root / "array.npy"
    np.save(path, np.arange(4, dtype=np.float64))
    rows = build_expected_object_set({"a": path}, campaign_id="c", trajectory="R", pulse=0,
                                    attempt=0, namespace="POST", root=root, source_indices={"a": 0})
    assert validate_object_set(rows, campaign_id="c", root=root)["status"] == "PASS"
    np.save(path, np.arange(4, dtype=np.float64) + 1)
    with pytest.raises(ValueError, match="identity|hash"):
        validate_object_set(rows, campaign_id="c", root=root)


def test_r10_budget_plan_is_scalar_and_has_headroom():
    from KHz_filament.hr4e5_storage import plan_campaign_budget

    plan = plan_campaign_budget(n_pulses=3, k=8048, ny=351, nx=301, nt=384)
    assert plan["peak_bytes"] <= 322122547200
    assert plan["final_output_bytes"] <= 68719476736
    assert plan["headroom_bytes"] > 0 and plan["final_headroom_bytes"] > 0
    assert plan["k8048_arrays_created"] is False


def _write_formal_simple_report(root, path, identity, trajectory, *, writer=None):
    from KHz_filament.hr4e5_evidence import bind_exact_report, build_expected_object_set, write_exact_report

    array = root / f"{Path(path).stem}_{trajectory.lower()}_summary.npy"
    np.save(array, np.arange(4, dtype=np.float64))
    rows = build_expected_object_set(
        {"a": array}, campaign_id=identity["campaign_id"], trajectory=trajectory,
        pulse=0, attempt=0, namespace="POST", root=root, source_indices={"a": 0},
    )
    result = bind_exact_report(path, {"status": "PASS", "mismatch_count": 0}, rows,
                               campaign_id=identity["campaign_id"], root=root)
    if writer is not None:
        payload = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
        payload.update(writer)
        result = write_exact_report(result["report_path"], payload)
    return result


def _formal_pair_report(root, path, identity, budget):
    from KHz_filament.hr4e5_evidence import (
        PAIRED_EXACT_SCHEMA, _file_identity, _paired_exact_keys, atomic_json,
        sha256_array, sha256_file,
    )

    n_pulses = int(identity["n_pulses"])
    expected_keys = sorted(_paired_exact_keys(k=int(identity["k"]), n_pulses=n_pulses, pulse=0))
    descriptors = {"R": {}, "C": {}}
    for side in ("R", "C"):
        object_root = root / "objects" / side
        reservation = budget.reserve(
            len(expected_keys) * 1024,
            purpose=f"formal exact {side} objects",
            allocation_paths=[object_root],
        )
        intent = budget.create_intent(
            reservation_id=reservation.reservation_id,
            trajectory=side,
            pulse=0,
            attempt=0,
            role="exact_object",
            allowed_paths=[object_root],
            expected_bytes=len(expected_keys) * 1024,
            admission_hash=identity["identity_sha256"],
            generation="generation-0",
            campaign_id=identity["campaign_id"],
        )
        created = []
        for key in expected_keys:
            target = object_root / (key.replace(":", "_") + ".npy")
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, np.asarray([1.0], dtype=np.float64))
            created.append(target)
        completed = budget.complete_intent(
            intent["intent_id"], files=created, role="exact_object",
            metadata={"reclaimable": True},
        )
        budget.consume(reservation.reservation_id, actual_bytes=int(completed["actual_bytes"]))
        artifacts = budget.artifacts()
        for key, target in zip(expected_keys, created):
            value = np.load(target, allow_pickle=False)
            ownership = artifacts[target.relative_to(root).as_posix()]
            descriptors[side][key] = {
                "trajectory": side, "campaign_id": identity["campaign_id"], "pulse": 0,
                "attempt": 0, "path": target.relative_to(root).as_posix(),
                "relative_path": target.relative_to(root).as_posix(), "locator": {},
                "shape": list(value.shape), "dtype": value.dtype.name, "finite": True,
                "canonical_array_hash": sha256_array(value), "file_sha256": sha256_file(target),
                "file_identity": _file_identity(target),
                "creation_record": dict(ownership["creation_record"]),
            }

    rows = []
    for key in expected_keys:
        sides = {}
        key_parts = key.split(":")
        if key_parts[0] == "screen":
            descriptor_fields = {"role": "screen", "namespace": key_parts[1], "source_index": int(key_parts[2]), "name": key_parts[3]}
        elif key_parts[0] == "sink":
            descriptor_fields = {"role": "sink", "namespace": "sink", "source_index": int(key_parts[2]), "name": key_parts[1]}
        elif key_parts[0] == "ledger":
            descriptor_fields = {"role": "ledger", "namespace": "ledger", "source_index": None, "name": key_parts[1]}
        else:
            descriptor_fields = {"role": "final_optical", "namespace": "final", "source_index": None, "name": "final_optical_field"}
        for side in ("R", "C"):
            sides[side] = {**descriptors[side][key], **descriptor_fields}
        rows.append({"status": "PASS", "comparison_key": key,
                     "reference": sides["R"], "candidate": sides["C"],
                     "reference_sha256_array": sides["R"]["canonical_array_hash"],
                     "candidate_sha256_array": sides["C"]["canonical_array_hash"]})
    payload = {
        "schema": PAIRED_EXACT_SCHEMA, "status": "PASS",
        "admission_identity_sha256": identity["identity_sha256"],
        "campaign_id": identity["campaign_id"], "pulse": 0, "attempt": 0,
        "screen_count": (12 if n_pulses == 1 else 15) * int(identity["k"]), "ledger_count": 9, "optical_count": 1,
        "expected_object_count": len(rows), "compared_object_count": len(rows),
        "mismatch_count": 0, "rows": rows,
    }
    atomic_json(path, payload)
    return {**payload, "report_path": str(path), "report_sha256": sha256_file(path)}


def _complete_formal_gc(root, budget, identity, *, side, pulse, targets, writer_epoch):
    from KHz_filament.hr4e5_evidence import (
        atomic_json, bind_exact_report, build_expected_object_set, sha256_file,
    )
    from KHz_filament.hr4e5_storage import (
        RECLAIM_PREREQUISITE_SCHEMA, WRITER_RECEIPT_SCHEMA, StorageIntegrityError,
    )

    writer = root / f"writer_{side}{pulse}.json"
    reservation = budget.reserve(
        1024**2, purpose=f"gc-writer-{side}-{pulse}",
        allocation_paths=[writer],
    )
    intent = budget.create_intent(
        reservation_id=reservation.reservation_id, trajectory=side,
        pulse=int(pulse), attempt=0, role="GC_EVIDENCE",
        allowed_paths=[writer], expected_bytes=1024**2,
        admission_hash=identity["identity_sha256"], epoch=writer_epoch,
        generation=f"gc:{side}:p{pulse}",
    )
    writer_record = budget.open_writer(
        reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
        coordinator_epoch=writer_epoch, trajectory=side, pulse=int(pulse),
        attempt=0, generation=f"gc:{side}:p{pulse}",
    )
    writer_receipt = budget.write_quiescence_receipt(
        writer, coordinator_epoch=writer_epoch, trajectory=side,
        pulse=int(pulse), attempt=0,
    )
    assert writer_receipt["schema"] == WRITER_RECEIPT_SCHEMA
    assert writer_receipt["registry_backed"] is True
    completed_writer = budget.complete_intent(intent["intent_id"], files=[writer])
    budget.consume(reservation.reservation_id, actual_bytes=completed_writer["actual_bytes"])
    budget.close_writer(writer_record["writer_id"], coordinator_epoch=writer_epoch)
    assert budget.active_writers(coordinator_epoch=writer_epoch, include_management=False) == []
    evidence = root / f"evidence_formal_{side}{pulse}"
    evidence.mkdir(parents=True, exist_ok=True)
    exact = evidence / "exact.json"
    object_rows = build_expected_object_set(
        {f"target_{index}": target for index, target in enumerate(targets)},
        campaign_id=identity["campaign_id"], trajectory=side,
        pulse=int(pulse), attempt=0, namespace="gc_target", root=root,
    )
    bind_exact_report(
        exact, {
            "status": "PASS", "mismatch_count": 0,
            "missing_reference": [], "missing_candidate": [],
        }, object_rows, campaign_id=identity["campaign_id"], root=root,
    )
    child = evidence / "child"
    child.mkdir(parents=True, exist_ok=True)
    child_field = child / "field.bin"
    child_field.write_bytes(b"self-contained-successor")
    child_binding = child / "binding.json"
    atomic_json(child_binding, {"status": "PASS", "rows": [{"status": "PASS"}]})
    ready = evidence / "READY.json"
    atomic_json(ready, {
        "status": "READY", "self_contained": True,
        "parent_payload_required": False,
        "child_root": str(child),
        "fields": {"field": {"path": "field.bin", "sha256": sha256_file(child_field)}},
        "binding": {"path": str(child_binding), "sha256": sha256_file(child_binding)},
    })
    dependency = evidence / "dependency.json"
    atomic_json(dependency, {"status": "PASS", "no_future_dependency": True})
    artifacts = budget.artifacts()
    target_bindings = []
    for target in targets:
        relative = target.resolve().relative_to(root).as_posix()
        item = artifacts[relative]
        target_bindings.append({
            "relative_path": relative, "identity": dict(item["identity"]),
            "sha256": item["sha256"],
        })
    prerequisite = evidence / "prerequisites.json"
    atomic_json(prerequisite, {
        "schema": RECLAIM_PREREQUISITE_SCHEMA, "status": "PASS",
        "campaign_id": identity["campaign_id"],
        "gates": {
            "exact_complete": True, "successor_ready": True,
            "no_active_writers": True, "no_future_dependency": True,
            "receipts_durable": True,
        },
        "target_bindings": target_bindings,
        "evidence": {
            "exact_complete": {"path": str(exact), "sha256": sha256_file(exact)},
            "successor_ready": {"path": str(ready), "sha256": sha256_file(ready)},
            "no_future_dependency": {
                "path": str(dependency), "sha256": sha256_file(dependency),
            },
        },
    })
    plan = budget.plan_reclaim(
        targets, exact_complete=True, successor_ready=True,
        no_active_writers=True, no_future_dependency=True,
        receipts_durable=True, writer_receipt=writer_receipt,
        prerequisite_receipt=prerequisite, plan_id=f"formal-{side}{pulse}",
        trajectory=side, pulse=int(pulse), attempt=0,
        admission_hash=identity["identity_sha256"],
    )
    wrong_side = "C" if side == "R" else "R"
    with pytest.raises(StorageIntegrityError, match="trajectory"):
        budget.apply_reclaim(
            plan["plan_id"], expected_trajectory=wrong_side,
            expected_pulse=int(pulse), expected_attempt=0,
            expected_admission_hash=identity["identity_sha256"],
        )
    assert all(target.is_file() for target in targets)
    completed = budget.apply_reclaim(
        plan["plan_id"], expected_trajectory=side,
        expected_pulse=int(pulse), expected_attempt=0,
        expected_admission_hash=identity["identity_sha256"],
    )
    assert completed["status"] == "COMPLETED"
    verified = budget.verify_reclaim(
        plan["plan_id"], expected_trajectory=side, expected_pulse=int(pulse),
        expected_attempt=0, expected_admission_hash=identity["identity_sha256"],
    )
    assert verified["status"] == "PASS"
    report = root / f"gc_report_{side}{pulse}.json"
    atomic_json(report, {
        "status": "PASS", "campaign_id": identity["campaign_id"],
        "gc_plan_id": plan["plan_id"],
        "writer_receipt_path": str(writer),
        "writer_receipt_sha256": sha256_file(writer),
    })
    return {
        "status": "PASS", "report_path": str(report),
        "report_sha256": sha256_file(report),
        "gc_plan_id": plan["plan_id"],
        "writer_receipt_path": str(writer),
        "writer_receipt_sha256": sha256_file(writer),
    }


def test_r06_formal_exact_rejects_r_only_and_accepts_a_complete_paired_report(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import _open_production_driver

    identity = _formal_identity()

    def run_case(root):
        root.mkdir(exist_ok=True)
        driver = _open_production_driver(root, admission_identity=identity, runner=_production_runner_shell(), n_pulses=1,
                                         campaign_id=identity["campaign_id"])
        from KHz_filament.hr4e5_storage import MockQuotaProvider
        driver.campaign.storage.provider = MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40)
        exact = _formal_pair_report(root, root / "paired_exact.json", identity, driver.campaign.storage)
        result = driver._validate_receipt(
            {"status": "PASS", "report_path": exact["report_path"]},
            pulse=0, step="exact",
        )
        driver.close()
        return result

    bad_root = tmp_path / "r_only"
    bad_root.mkdir()
    report_r = _write_formal_simple_report(bad_root, bad_root / "reference.json", identity, "R")
    report_c = _write_formal_simple_report(bad_root, bad_root / "candidate.json", identity, "C")

    class BadRunner:
        def run_reference(self, pulse): return {"status": "PASS", "report_path": report_r["report_path"]}
        def run_candidate(self, pulse): return {"status": "PASS", "report_path": report_c["report_path"]}
        def run_exact(self, pulse): return {"status": "PASS", "report_path": report_r["report_path"]}

    bad_runner = _production_runner_shell()
    bad = BadRunner()
    bad_runner.run_reference = bad.run_reference
    bad_runner.run_candidate = bad.run_candidate
    bad_runner.run_exact = bad.run_exact
    with pytest.raises(Exception, match="formal exact|schema|object"):
        _open_production_driver(bad_root, admission_identity=identity, runner=bad_runner, n_pulses=1,
                           campaign_id=identity["campaign_id"]).run()

    good_root = tmp_path / "paired"
    result = run_case(good_root)
    assert result["status"] == "PASS"
    assert result["validated_object_count"] == 12 * 8 + 9 + 1


def test_r07_crash_takeover_requires_hash_bound_stale_exit_receipt(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import PairedCampaign
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    identity = _formal_identity(campaign_id="takeover-r")
    root = tmp_path / "takeover"
    campaign = PairedCampaign.create(root, n_pulses=1, campaign_id=identity["campaign_id"],
                                     admission_identity=identity, formal=True,
                                     require_intents=True, require_quota=False)
    campaign.register_process_start(process_id="old-coordinator", source="test")
    reopened = PairedCampaign.open(root, n_pulses=1, campaign_id=identity["campaign_id"],
                                   admission_identity=identity, formal=True,
                                   require_intents=True, require_quota=False)
    with pytest.raises(StorageIntegrityError, match="live ACTIVE coordinator"):
        reopened.register_process_start(process_id="new-coordinator", source="takeover", takeover=True)
    with pytest.raises(StorageIntegrityError, match="external crash takeover receipts are forbidden"):
        reopened.register_process_start(
            process_id="new-coordinator", source="takeover", takeover=True,
            takeover_receipt={"path": str(root / "forged.json"), "sha256": "0" * 64},
        )
    campaign.register_process_exit(process_id="old-coordinator")


def test_r07_independent_process_crash_stale_takeover_and_resume(tmp_path):
    identity = _formal_identity(campaign_id="takeover-subprocess-r")
    root = tmp_path / "subprocess-takeover"
    identity_path = tmp_path / "admission.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    child_one = textwrap.dedent(
        """
        import json, os, sys
        from pathlib import Path
        from KHz_filament.hr4e5_paired_campaign import _open_production_driver
        from KHz_filament.hr4e5_evidence import atomic_json
        from KHz_filament.hr4e5_production import _E5AProductionRunner

        root = Path(sys.argv[1]).resolve()
        identity = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        runner = _E5AProductionRunner.__new__(_E5AProductionRunner)
        driver = _open_production_driver(root, admission_identity=identity, runner=runner,
                                    n_pulses=1, campaign_id=identity["campaign_id"])
        atomic_json(root / "child_one_started.json", {
            "process_id": driver.epoch, "campaign_id": identity["campaign_id"],
            "next_pair_index": driver.campaign.next_pair_index,
        })
        os._exit(17)
        """
    )
    first = subprocess.run(
        [sys.executable, "-s", "-B", "-c", child_one, str(root), str(identity_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert first.returncode == 17, first.stdout + "\n" + first.stderr
    state_path = root / "E5_1A_CAMPAIGN_STATE.json"
    state_before = json.loads(state_path.read_text(encoding="utf-8"))
    active = state_before.get("active_coordinator")
    assert isinstance(active, dict) and active.get("status") == "ACTIVE"
    assert state_before["admission_hash"] == identity["identity_sha256"]
    assert state_before["next_pair_index"] == 0
    assert state_before["pairs"][0]["pulse_index"] == 0
    assert state_before["pairs"][0].get("formal_steps") == {}

    child_two = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from KHz_filament.hr4e5_paired_campaign import _open_production_driver
        from KHz_filament.hr4e5_evidence import atomic_json
        from KHz_filament.hr4e5_production import _E5AProductionRunner

        root = Path(sys.argv[1]).resolve()
        identity = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        runner = _E5AProductionRunner.__new__(_E5AProductionRunner)
        driver = _open_production_driver(root, admission_identity=identity, runner=runner,
                                    n_pulses=1, campaign_id=identity["campaign_id"],
                                    takeover=True)
        new_epoch = driver.epoch
        report = driver.resume(run=False)
        assert report["admission_hash"] == identity["identity_sha256"]
        assert report["next_pair_index"] == 0
        assert report["pairs"][0]["pulse_index"] == 0
        assert report["pairs"][0].get("formal_steps") == {}
        driver.close()
        atomic_json(root / "child_two_resumed.json", {
            "process_id": new_epoch, "campaign_id": identity["campaign_id"],
            "admission_hash": report["admission_hash"],
            "next_pair_index": report["next_pair_index"],
            "pulse_index": report["pairs"][0]["pulse_index"],
        })
        """
    )
    second = subprocess.run(
        [sys.executable, "-s", "-B", "-c", child_two, str(root), str(identity_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert second.returncode == 0, second.stdout + "\n" + second.stderr
    resumed = json.loads((root / "child_two_resumed.json").read_text(encoding="utf-8"))
    state_after = json.loads(state_path.read_text(encoding="utf-8"))
    epochs = state_after["process_epochs"]
    assert len(epochs) == 2
    assert len({str(item["process_id"]) for item in epochs}) == 2
    assert epochs[0]["status"] == "EXITED_STALE"
    assert epochs[0]["exit_reason"] == "durable_crash_takeover_receipt"
    takeover_path = Path(epochs[0]["exit_receipt"]["path"])
    assert takeover_path == root / ".takeover" / f"epoch_{active['epoch']}.json"
    takeover = json.loads(takeover_path.read_text(encoding="utf-8"))
    assert takeover["registry_backed"] is True
    assert takeover["stale"] is True and takeover["exit_observed"] is True
    assert takeover["liveness_evidence"]["status"] == "DEAD"
    assert epochs[1]["status"] == "EXITED"
    assert resumed["process_id"] == epochs[1]["process_id"]
    assert resumed["process_id"] != epochs[0]["process_id"]
    assert state_after["active_coordinator"] is None
    assert state_after["admission_hash"] == identity["identity_sha256"]
    assert state_after["next_pair_index"] == 0
    assert state_after["pairs"][0]["pulse_index"] == 0
    assert state_after["pairs"][0].get("formal_steps") == {}


def test_r08_formal_old_epoch_cannot_record_or_commit(tmp_path):
    from KHz_filament.hr4e5_evidence import atomic_json, sha256_file
    from KHz_filament.hr4e5_paired_campaign import PairedCampaign
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    identity = _formal_identity(campaign_id="stale-epoch-r")
    root = tmp_path / "stale-epoch"
    campaign = PairedCampaign.create(
        root, n_pulses=1, campaign_id=identity["campaign_id"],
        admission_identity=identity, formal=True, require_intents=True, require_quota=False,
    )
    old = campaign.register_process_start(process_id="old-epoch", source="test")
    receipt = root / "step.json"
    atomic_json(receipt, {"status": "PASS", "campaign_id": identity["campaign_id"]})
    durable = {"path": str(receipt), "sha256": sha256_file(receipt)}
    campaign.formal_step(0, "reference", status="COMMITTED", receipt=durable, epoch=old["epoch"])
    campaign.register_process_exit(process_id="old-epoch")
    new = campaign.register_process_start(process_id="new-epoch", source="resume")
    with pytest.raises(StorageIntegrityError, match="current coordinator epoch"):
        campaign.formal_step(0, "candidate", status="IN_PROGRESS")
    with pytest.raises(StorageIntegrityError, match="stale coordinator epoch"):
        campaign.formal_step(0, "candidate", status="IN_PROGRESS", epoch=old["epoch"])
    with pytest.raises(StorageIntegrityError, match="stale coordinator epoch"):
        campaign.commit_formal_pair(0, epoch=old["epoch"])
    assert new["epoch"] != old["epoch"]
    campaign.register_process_exit(process_id="new-epoch")


def test_r08_exact_resume_after_gc_uses_metadata_only_pair_validator(tmp_path):
    from KHz_filament.hr4e5_evidence import sha256_file
    from KHz_filament.hr4e5_paired_campaign import _open_production_driver
    from KHz_filament.hr4e5_storage import MockQuotaProvider, StorageIntegrityError

    identity = _formal_identity(campaign_id="exact-after-gc-r", n_pulses=2)
    root = tmp_path / "exact-after-gc"
    driver = _open_production_driver(
        root, admission_identity=identity, runner=_production_runner_shell(), n_pulses=2,
        campaign_id=identity["campaign_id"], safety_margin_bytes=1024**2,
    )
    driver.campaign.storage.provider = MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40)
    report = _formal_pair_report(root, root / "paired_exact.json", identity, driver.campaign.storage)
    durable = {"path": report["report_path"], "sha256": sha256_file(report["report_path"])}
    epoch = driver.campaign.active_coordinator_epoch()
    driver.campaign.formal_step(0, "exact", status="COMMITTED", receipt=durable, epoch=epoch)
    artifacts = driver.campaign.storage.artifacts()
    targets = {
        side: [root / relative for relative in artifacts if relative.startswith(f"objects/{side}/")]
        for side in ("R", "C")
    }
    reference_gc = _complete_formal_gc(
        root, driver.campaign.storage, identity, side="R", pulse=0,
        targets=targets["R"], writer_epoch=epoch,
    )
    driver.campaign.formal_step(
        0, "reference_gc", status="COMMITTED", receipt=reference_gc, epoch=epoch,
    )
    with pytest.raises(StorageIntegrityError, match="trajectory"):
        driver.campaign.formal_step(
            0, "candidate_gc", status="COMMITTED", receipt=reference_gc, epoch=epoch,
        )
    candidate_gc = _complete_formal_gc(
        root, driver.campaign.storage, identity, side="C", pulse=0,
        targets=targets["C"], writer_epoch=epoch,
    )
    driver.campaign.formal_step(
        0, "candidate_gc", status="COMMITTED", receipt=candidate_gc, epoch=epoch,
    )
    resumed = driver._saved_receipt(0, "exact")
    assert resumed["status"] == "PASS"
    assert resumed["metadata_only"] is True
    assert all(row.get("metadata_only") is True for row in resumed["rows"])
    driver.close()


def test_r09_formal_successor_requires_ready_receipt(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import _open_production_driver
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    identity = _formal_identity(campaign_id="missing-ready-r", n_pulses=2)
    root = tmp_path / "missing-ready"
    root.mkdir()
    report = _write_formal_simple_report(root, root / "successor_report.json", identity, "R")
    parent = root / "parent"
    parent.mkdir(parents=True)
    child = root / "child"
    child.mkdir(parents=True)
    driver = _open_production_driver(
        root, admission_identity=identity, runner=_production_runner_shell(), n_pulses=2,
        campaign_id=identity["campaign_id"], safety_margin_bytes=1024**2,
    )
    with pytest.raises(StorageIntegrityError, match="READY"):
        driver._validate_receipt(
            {
                "status": "PASS",
                "report_path": report["report_path"],
                "child_root": str(child),
                "parent_root": str(parent),
                "parent_generation": "generation-0",
            },
            pulse=0, step="reference_successor",
        )
    driver.close()


def test_r09_formal_terminal_requires_inventory_contract(tmp_path):
    from KHz_filament.hr4e5_evidence import sha256_file
    from KHz_filament.hr4e5_paired_campaign import _open_production_driver
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    identity = _formal_identity(campaign_id="missing-terminal-inventory-r")
    root = tmp_path / "missing-terminal-inventory"
    writer_path = root / "writer.json"
    writer_path.parent.mkdir(parents=True)
    writer_path.write_text(json.dumps({
        "status": "PASS", "campaign_id": identity["campaign_id"],
        "active_writers": [], "writer_epoch": "writer", "coordinator_process_id": "runner",
    }), encoding="utf-8")
    report = _write_formal_simple_report(
        root, root / "terminal.json", identity, "C",
        writer={"writer_receipt_path": str(writer_path), "writer_receipt_sha256": sha256_file(writer_path)},
    )
    class Runner:
        def expected_terminal_roles(self, pulse): return {"final": []}
    runner = _production_runner_shell()
    runner.expected_terminal_roles = Runner().expected_terminal_roles
    driver = _open_production_driver(
        root, admission_identity=identity, runner=runner, n_pulses=1,
        campaign_id=identity["campaign_id"], safety_margin_bytes=1024**2,
    )
    value = {
        "status": "PASS", "report_path": report["report_path"],
        "writer_receipt_path": str(writer_path), "writer_receipt_sha256": sha256_file(writer_path),
    }
    with pytest.raises(StorageIntegrityError, match="registry-backed|writer receipt"):
        driver._validate_receipt(value, pulse=0, step="terminal")
    driver.close()


def _local_production_spec(tmp_path, *, campaign_id="local-production", n_pulses=1, k=8):
    from KHz_filament.hr4e5_evidence import sha256_file
    from KHz_filament.hr4e5_production import E5AProductionSpec, _source_hash

    inputs = tmp_path / "inputs"; inputs.mkdir(parents=True, exist_ok=True)
    config = inputs / "config.json"
    value = {
        "grid": {"Nx": 8, "Ny": 8, "Nt": 8, "Lx": 8e-4, "Ly": 8e-4, "Twin": 80e-15},
        "beam": {"w0": 1.5e-4, "tau_fwhm": 40e-15, "energy_J": 1e-10,
                 "focal_length": None},
        "propagation": {"z_max": k*1e-4, "dz": 1e-4, "linear_model": "paraxial",
            "auto_substep": False, "focus_window_step": False, "limit_focus_window": False,
            "progress_every_z": 0, "energy_probe_every": 0, "diag_extra": False,
            "use_electronic_kerr": False, "use_raman_phase": False,
            "use_raman_absorption": False, "use_plasma_phase": False,
            "use_ionization_loss": False, "use_ionization_solver": False},
        "ionization": {"species": []},
        "heat": {"hr3b_enabled": True, "f_rep": 5e6, "dt_hydro": 1e-7,
                 "chi": 2.17e-5, "D_th": 2.17e-5, "nu": 1.5e-5,
                 "gravity_x": 0.0, "gravity_y": -9.81},
        "run": {"Npulses": n_pulses},
        "raman": {"enabled": False, "absorption": False},
    }
    config.write_text(json.dumps(value), encoding="utf-8")
    source = inputs / "source.json"
    source.write_text(json.dumps({"schema": "local-source.v1",
                                  "canonical_array_hash": _source_hash(config)}), encoding="utf-8")
    lut = inputs / "lut.json"
    lut.write_text(json.dumps({"schema": "local-lut.v1", "mode": "DISABLED_BY_CONFIG",
                               "config_sha256": sha256_file(config)}), encoding="utf-8")
    pre0 = inputs / "pre0.npy"; np.save(pre0, np.zeros((k, 8, 8), dtype=np.float64))
    return E5AProductionSpec(root=tmp_path/"campaign", campaign_id=campaign_id,
        config_path=config, source_manifest_path=source, lut_manifest_path=lut,
        pre0_delta_n_path=pre0, n_pulses=n_pulses, safety_margin_bytes=1024**2)


def test_production_factory_is_unique_and_n1_real_cpu_path(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import FormalPairedDriver, _open_production_driver
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament.hr4e5_storage import StorageIntegrityError

    with pytest.raises(StorageIntegrityError, match="production factory"):
        FormalPairedDriver(tmp_path/"forbidden", admission_identity=_formal_identity(), runner=object(), n_pulses=1)
    with pytest.raises(StorageIntegrityError, match="fixed E5-1A production runner"):
        _open_production_driver(
            tmp_path/"forbidden-private", admission_identity=_formal_identity(),
            runner=object(), n_pulses=1,
        )
    campaign = open_e5_1a_production_campaign(_local_production_spec(tmp_path))
    result = campaign.run()
    assert result["status"] == "COMPLETE"
    assert result["admission_identity"]["execution_mode"] == "LOCAL_ORCHESTRATION_QUALIFICATION"


def test_production_admission_recomputes_source_lut_and_budget_identity(tmp_path):
    from dataclasses import replace
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    source_spec = _local_production_spec(tmp_path / "source", campaign_id="tamper-source")
    Path(source_spec.source_manifest_path).write_text(
        json.dumps({"schema": "local-source.v1", "canonical_array_hash": "bad"}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source manifest differs"):
        open_e5_1a_production_campaign(source_spec)

    lut_spec = _local_production_spec(tmp_path / "lut", campaign_id="tamper-lut")
    Path(lut_spec.lut_manifest_path).write_text(
        json.dumps({"schema": "local-lut.v1", "mode": "DISABLED_BY_CONFIG",
                    "config_sha256": "bad"}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disabled LUT manifest is not bound"):
        open_e5_1a_production_campaign(lut_spec)

    budget_spec = _local_production_spec(tmp_path / "budget", campaign_id="tamper-budget")
    campaign = open_e5_1a_production_campaign(budget_spec)
    campaign.close()
    with pytest.raises((ValueError, RuntimeError), match="identity|safety margin|conflicts"):
        open_e5_1a_production_campaign(
            replace(budget_spec, safety_margin_bytes=2 * 1024**2)
        )


def test_production_n2_k16_multiblock_overlap_happy_path(tmp_path):
    from KHz_filament.hr4e5_paired_campaign import validate_overlap_events
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign

    campaign = open_e5_1a_production_campaign(
        _local_production_spec(tmp_path, campaign_id="local-production-k16", n_pulses=2, k=16)
    )
    result = campaign.run()
    assert result["status"] == "COMPLETE"
    assert result["next_pair_index"] == 2
    exact0 = json.loads((tmp_path / "campaign" / "exact" / "p0.json").read_text(encoding="utf-8"))
    exact1 = json.loads((tmp_path / "campaign" / "exact" / "p1.json").read_text(encoding="utf-8"))
    assert exact0["screen_count"] == 15 * 16
    assert exact1["screen_count"] == 12 * 16
    state0 = json.loads((tmp_path / "campaign" / "C" / "p0" / "state" / "streaming_manifest.json").read_text(encoding="utf-8"))
    assert state0["barrier"]["status"] == "PASS"
    assert state0["promotion"]["authoritative_namespace"] == "NEXT"
    assert len(state0["records"]) == 16
    events = state0["telemetry_events"]
    overlap = validate_overlap_events(events)
    assert overlap["status"] == "PASS"
    first_hydro_index, first_hydro = next(
        (index, event) for index, event in enumerate(events)
        if event["event"] == "HYDRO_BLOCK_START"
    )
    optical_complete_index, optical_complete = next(
        (index, event) for index, event in enumerate(events)
        if event["event"] == "OPTICAL_COMPLETE"
    )
    assert first_hydro["block"] == list(range(8))
    assert first_hydro_index < optical_complete_index
    assert first_hydro["monotonic_s"] < optical_complete["monotonic_s"]
    ledger = json.loads((tmp_path / "campaign" / "storage_ledger.json").read_text(encoding="utf-8"))
    for side in ("R", "C"):
        relative = f"{side}/p0/state/E5_1A_ARCHIVED_AFTER_EXACT.json"
        assert relative in ledger["artifacts"]
        assert ledger["artifacts"][relative]["role"] == "SUCCESSOR"


def test_production_n3_k8_split_process_matches_uninterrupted(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament.hr4e5_storage import plan_campaign_budget

    split_base = tmp_path / "split"
    split_spec = _local_production_spec(
        split_base, campaign_id="local-production-n3-split", n_pulses=3, k=8,
    )
    child = textwrap.dedent(
        """
        import sys
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
        spec = E5AProductionSpec(
            root=sys.argv[1], campaign_id=sys.argv[2], config_path=sys.argv[3],
            source_manifest_path=sys.argv[4], lut_manifest_path=sys.argv[5],
            pre0_delta_n_path=sys.argv[6], n_pulses=3, safety_margin_bytes=1024**2,
        )
        result = open_e5_1a_production_campaign(spec).run(stop_after=1)
        assert result["next_pair_index"] == 1
        """
    )
    first = subprocess.run(
        [sys.executable, "-s", "-B", "-c", child, str(split_spec.root),
         split_spec.campaign_id, str(split_spec.config_path), str(split_spec.source_manifest_path),
         str(split_spec.lut_manifest_path), str(split_spec.pre0_delta_n_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert first.returncode == 0, first.stdout + "\n" + first.stderr
    split_result = open_e5_1a_production_campaign(split_spec).run()
    assert split_result["status"] == "COMPLETE" and split_result["next_pair_index"] == 3

    baseline_base = tmp_path / "baseline"
    baseline_seed = _local_production_spec(
        baseline_base, campaign_id="local-production-n3-baseline", n_pulses=3, k=8,
    )
    baseline_result = open_e5_1a_production_campaign(baseline_seed).run()
    assert baseline_result["status"] == "COMPLETE"

    def exact_signature(root):
        reports = [json.loads((root / "exact" / f"p{pulse}.json").read_text(encoding="utf-8"))
                   for pulse in range(3)]
        return reports, [
            (row["comparison_key"], row["reference_sha256_array"], row["candidate_sha256_array"])
            for report in reports for row in report["rows"]
        ]

    split_root = Path(split_spec.root)
    baseline_root = Path(baseline_seed.root)
    split_exact, split_signature = exact_signature(split_root)
    baseline_exact, baseline_signature = exact_signature(baseline_root)
    assert [item["screen_count"] for item in split_exact] == [120, 120, 96]
    assert sum(item["screen_count"] for item in split_exact) == 336
    assert sum(item["ledger_count"] for item in split_exact) == 27
    assert sum(item["optical_count"] for item in split_exact) == 3
    assert split_signature == baseline_signature

    def handoff_count(root):
        total = 0
        for side in ("R", "C"):
            for pulse in (1, 2):
                report = json.loads((root / side / f"p{pulse}" / "state" /
                                     "parent_next_child_current_exact.json").read_text(encoding="utf-8"))
                total += int(report["compared_object_count"])
        return total

    assert handoff_count(split_root) == handoff_count(baseline_root) == 96

    def durable_signature(root):
        ledger = json.loads((root / "storage_ledger.json").read_text(encoding="utf-8"))
        state = json.loads((root / "E5_1A_CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
        roles = json.loads((root / "E5_1A_TERMINAL_ROLE_CONTRACT.json").read_text(encoding="utf-8"))["roles"]
        return {
            "gc": sorted((plan["trajectory"], plan["pulse"], plan["status"], len(plan["targets"]))
                         for plan in ledger["gc_plans"].values()),
            "writers": sorted((item["role"], item["status"], item["trajectory"], item["pulse"])
                              for item in ledger["writers"].values()),
            "reservations": sorted(item["status"] for item in ledger["reservations"].values()),
            "roles": {key: len(value) for key, value in roles.items()},
            "pairs": [(pair["pulse_index"], pair["status"]) for pair in state["pairs"]],
        }, state

    split_durable, split_state = durable_signature(split_root)
    baseline_durable, baseline_state = durable_signature(baseline_root)
    assert split_durable == baseline_durable
    assert set(split_durable["reservations"]) == {"CONSUMED"}
    assert len(split_state["process_epochs"]) == 2
    assert len(baseline_state["process_epochs"]) == 1
    assert all(item["status"] == "EXITED" for item in split_state["process_epochs"])
    local_plan = plan_campaign_budget(n_pulses=3, k=8, ny=8, nx=8, nt=8,
                                      safety_margin_bytes=1024**2)
    assert local_plan["peak_bytes"] < local_plan["max_campaign_live_bytes"]
    admission_report = json.loads(
        (split_root / "R" / "p0" / "reports" / "admission.json").read_text(encoding="utf-8")
    )
    assert admission_report["capacity_snapshot"]["scope"] == "LOCAL_POLICY_CAP_NOT_SITE_QUOTA"
    assert admission_report["capacity_snapshot"]["filesystem_free_bytes"] > 0
    assert admission_report["capacity_snapshot"]["site_quota_measured"] is False

    readonly = textwrap.dedent(
        """
        import sys
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
        spec = E5AProductionSpec(root=sys.argv[1], campaign_id=sys.argv[2], config_path=sys.argv[3],
            source_manifest_path=sys.argv[4], lut_manifest_path=sys.argv[5], pre0_delta_n_path=sys.argv[6],
            n_pulses=3, safety_margin_bytes=1024**2)
        campaign = open_e5_1a_production_campaign(spec)
        assert campaign.report()["status"] == "COMPLETE"
        campaign.close()
        """
    )
    third = subprocess.run(
        [sys.executable, "-s", "-B", "-c", readonly, str(split_spec.root), split_spec.campaign_id,
         str(split_spec.config_path), str(split_spec.source_manifest_path), str(split_spec.lut_manifest_path),
         str(split_spec.pre0_delta_n_path)], capture_output=True, text=True, timeout=30,
    )
    assert third.returncode == 0, third.stdout + "\n" + third.stderr
    assert not (split_root / "R" / "p3").exists()
    assert not (split_root / "C" / "p3").exists()


def test_production_writer_live_refusal_and_dead_stale_takeover(tmp_path):
    from KHz_filament.hr4e5_production import open_e5_1a_production_campaign
    from KHz_filament.hr4e5_storage import StorageBudgetError, StorageIntegrityError

    spec = _local_production_spec(
        tmp_path, campaign_id="local-production-writer-takeover", n_pulses=2, k=8,
    )
    coordinator = textwrap.dedent(
        """
        import os, sys
        from KHz_filament.hr4e5_production import E5AProductionSpec, open_e5_1a_production_campaign
        spec = E5AProductionSpec(root=sys.argv[1], campaign_id=sys.argv[2], config_path=sys.argv[3],
            source_manifest_path=sys.argv[4], lut_manifest_path=sys.argv[5], pre0_delta_n_path=sys.argv[6],
            n_pulses=2, safety_margin_bytes=1024**2)
        open_e5_1a_production_campaign(spec)
        os._exit(17)
        """
    )
    first = subprocess.run(
        [sys.executable, "-s", "-B", "-c", coordinator, str(spec.root), spec.campaign_id,
         str(spec.config_path), str(spec.source_manifest_path), str(spec.lut_manifest_path),
         str(spec.pre0_delta_n_path)], capture_output=True, text=True, timeout=30,
    )
    assert first.returncode == 17

    writer_script = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from KHz_filament.hr4e5_storage import StorageBudget, MockQuotaProvider
        root = Path(sys.argv[1])
        identity = json.loads((root / "E5_1A_ADMISSION_IDENTITY.json").read_text(encoding="utf-8"))
        state = json.loads((root / "E5_1A_CAMPAIGN_STATE.json").read_text(encoding="utf-8"))
        epoch = state["active_coordinator"]["epoch"]
        budget = StorageBudget(root, safety_margin_bytes=1024**2,
            provider=MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40), require_quota=True,
            campaign_id=identity["campaign_id"], require_intents=True,
            admission_hash=identity["identity_sha256"], admission_identity=identity)
        residual = root / "crash_writer" / "residual.bin"
        reservation = budget.reserve(1024**2, purpose="crash-writer", allocation_paths=[residual])
        intent = budget.create_intent(reservation_id=reservation.reservation_id, trajectory="C",
            pulse=0, attempt=0, role="CRASH_TEST", allowed_paths=[residual], expected_bytes=1024**2,
            admission_hash=identity["identity_sha256"], epoch=epoch, generation="crash:test")
        writer = budget.open_writer(reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
            coordinator_epoch=epoch, trajectory="C", pulse=0, attempt=0, generation="crash:test",
            writer_id="crash-writer")
        residual.parent.mkdir(parents=True, exist_ok=True)
        residual.write_bytes(b"stale-residual")
        (root / "writer_started.json").write_text(json.dumps({"reservation_id": reservation.reservation_id,
            "intent_id": intent["intent_id"], "writer_id": writer["writer_id"], "epoch": epoch}), encoding="utf-8")
        sys.stdin.read()
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-s", "-B", "-c", writer_script, str(spec.root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    marker = Path(spec.root) / "writer_started.json"
    for _ in range(100):
        if marker.is_file():
            break
        time.sleep(0.05)
    assert marker.is_file()
    with pytest.raises(StorageBudgetError, match="live ACTIVE writers block takeover"):
        open_e5_1a_production_campaign(spec)
    process.terminate()
    process.wait(timeout=15)

    recovered = open_e5_1a_production_campaign(spec)
    storage = recovered._driver.campaign.storage
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    ledger = json.loads((Path(spec.root) / "storage_ledger.json").read_text(encoding="utf-8"))
    stale = ledger["writers"][marker_value["writer_id"]]
    assert stale["status"] == "INTERRUPTED_STALE"
    assert stale["liveness_evidence"]["status"] == "DEAD"
    assert ledger["reservations"][marker_value["reservation_id"]]["status"] == "ACTIVE"
    assert (Path(spec.root) / "crash_writer" / "residual.bin").is_file()

    new_epoch = recovered._driver.coordinator_epoch
    replacement = storage.open_writer(
        reservation_id=marker_value["reservation_id"], intent_id=marker_value["intent_id"],
        coordinator_epoch=new_epoch, trajectory="C", pulse=0, attempt=0,
        generation="crash:test", writer_id="replacement-writer",
    )
    with pytest.raises(StorageIntegrityError, match="stale epoch"):
        storage.close_writer(replacement["writer_id"], coordinator_epoch=marker_value["epoch"])
    completed = storage.complete_intent(
        marker_value["intent_id"], files=[Path(spec.root) / "crash_writer" / "residual.bin"],
    )
    storage.consume(marker_value["reservation_id"], actual_bytes=completed["actual_bytes"])
    storage.close_writer(replacement["writer_id"], coordinator_epoch=new_epoch)
    recovered.close()

    identity = storage.active_writers()
    assert identity == []


def test_writer_pid_reuse_and_unknown_liveness_fail_closed(tmp_path):
    from KHz_filament.hr4e5_storage import (
        MockQuotaProvider, StorageBudget, StorageIntegrityError, process_identity,
    )

    def opened(root, process, writer_id):
        budget = StorageBudget(
            root, provider=MockQuotaProvider(free_bytes=2**40, quota_bytes=2**40),
            require_quota=True, campaign_id=writer_id, require_intents=True,
            admission_hash="a" * 64,
        )
        target = root / "residual.bin"
        reservation = budget.reserve(1024**2, purpose=writer_id, allocation_paths=[target])
        intent = budget.create_intent(
            reservation_id=reservation.reservation_id, trajectory="C", pulse=0, attempt=0,
            role="CRASH_TEST", allowed_paths=[target], expected_bytes=1024**2,
            admission_hash="a" * 64, epoch=0, generation="reuse:test",
        )
        budget.open_writer(
            reservation_id=reservation.reservation_id, intent_id=intent["intent_id"],
            coordinator_epoch=0, trajectory="C", pulse=0, attempt=0,
            generation="reuse:test", process=process, writer_id=writer_id,
        )
        target.write_bytes(b"residual")
        return budget, reservation

    reused = process_identity()
    reused["process_start_token"] += "-reused"
    reused_budget, reused_reservation = opened(tmp_path / "reused", reused, "pid-reused")
    result = reused_budget.interrupt_stale_writers(coordinator_epoch=0)
    assert result["probes"]["pid-reused"]["reason"] == "pid_reused"
    reused_ledger = json.loads((tmp_path / "reused" / "storage_ledger.json").read_text(encoding="utf-8"))
    assert reused_ledger["writers"]["pid-reused"]["status"] == "INTERRUPTED_STALE"
    assert reused_ledger["reservations"][reused_reservation.reservation_id]["status"] == "ACTIVE"

    unknown = process_identity()
    unknown["host_identity"] = "unknown-remote-host"
    unknown_budget, unknown_reservation = opened(tmp_path / "unknown", unknown, "unknown-host")
    with pytest.raises(StorageIntegrityError, match="unverifiable ACTIVE writers block takeover"):
        unknown_budget.interrupt_stale_writers(coordinator_epoch=0)
    unknown_ledger = json.loads((tmp_path / "unknown" / "storage_ledger.json").read_text(encoding="utf-8"))
    assert unknown_ledger["writers"]["unknown-host"]["status"] == "ACTIVE"
    assert unknown_ledger["reservations"][unknown_reservation.reservation_id]["status"] == "ACTIVE"
